use crate::{
    INTERSECTS_UPPER_BOUND, MainBackendBase,
    camera::Camera,
    dim_check::DimCheck,
    get_tile_offset::{CHECKS_PER_ITER, get_tile_offsets},
    render_aux::RenderAux,
    sh::sh_degree_from_coeffs,
    shaders::{self, Project, MapGaussiansToIntersect, Rasterize},
};
use brush_kernel::create_dispatch_buffer;
use brush_kernel::create_tensor;
use brush_kernel::create_uniform_buffer;
use brush_kernel::{CubeCount, calc_cube_count};
use brush_prefix_sum::prefix_sum;
use brush_sort::radix_argsort;
use burn::tensor::{DType, IntDType};
use burn::tensor::ops::IntTensorOps;
use burn_cubecl::cubecl::server::Bindings;

use burn_cubecl::kernel::into_contiguous;
use burn_wgpu::WgpuRuntime;
use burn_wgpu::{CubeDim, CubeTensor};
use glam::{Vec3, uvec2, vec4};
use std::mem::offset_of;

pub(crate) fn calc_tile_bounds(img_size: glam::UVec2) -> glam::UVec2 {
    uvec2(
        img_size.x.div_ceil(shaders::helpers::TILE_WIDTH),
        img_size.y.div_ceil(shaders::helpers::TILE_WIDTH),
    )
}

// On wasm, we cannot do a sync readback at all.
// Instead, can just estimate a max number of intersects. All the kernels only handle the actual
// number of intersects, and spin up empty threads for the rest atm. In the future, could use indirect
// dispatch to avoid this.
// Estimating the max number of intersects can be a bad hack though... The worst case scenario is so massive
// that it's easy to run out of memory... How do we actually properly deal with this :/
pub fn max_intersections(tile_bounds: glam::UVec2, num_splats: u32) -> u32 {
    // Assume on average each splat is maximally covering half x half the screen,
    // and adjust for the variance such that we're fairly certain we have enough intersections.
    let num_tiles = tile_bounds[0] * tile_bounds[1];
    let max_possible = num_tiles.saturating_mul(num_splats);
    // clamp to max nr. of dispatches.
    max_possible.min(INTERSECTS_UPPER_BOUND)
}

const PROJ_SIZE: usize = size_of::<shaders::helpers::TransformedSplat>() / size_of::<f32>();
const BOUNDS_SIZE: usize = size_of::<shaders::helpers::SplatBounds>() / size_of::<f32>();

const NEAR_PLANE: f32 = 0.2; // don't set too close to zero to avoid precision issues
const FAR_PLANE: f32  = 1000.0; // don't set too high to avoid precision issues
const VP33: f32 = (FAR_PLANE + NEAR_PLANE) / (FAR_PLANE - NEAR_PLANE);
const VP34: f32 = -2.0 * FAR_PLANE * NEAR_PLANE / (FAR_PLANE - NEAR_PLANE);

pub(crate) fn render_forward(
    camera: &Camera,
    img_size: glam::UVec2,
    means: CubeTensor<WgpuRuntime>,
    log_scales: CubeTensor<WgpuRuntime>,
    quats: CubeTensor<WgpuRuntime>,
    sh_coeffs: CubeTensor<WgpuRuntime>,
    raw_opacities: CubeTensor<WgpuRuntime>,
    background: Vec3,
    bwd_info: bool,
) -> (CubeTensor<WgpuRuntime>, RenderAux<MainBackendBase>) {
    assert!(!bwd_info, "this implementation is inference-only");
    assert!(
        img_size[0] > 0 && img_size[1] > 0,
        "Can't render images with 0 size."
    );

    // Tensor params might not be contiguous, convert them to contiguous tensors.
    let means = into_contiguous(means);
    let log_scales = into_contiguous(log_scales);
    let quats = into_contiguous(quats);
    let sh_coeffs = into_contiguous(sh_coeffs);
    let raw_opacities = into_contiguous(raw_opacities);

    let device = &means.device.clone();
    let client = &means.client.clone();

    let _span = tracing::trace_span!("render_forward").entered();

    // Check whether input dimensions are valid.
    DimCheck::new()
        .check_dims("means", &means, &["D".into(), 3.into()])
        .check_dims("log_scales", &log_scales, &["D".into(), 3.into()])
        .check_dims("quats", &quats, &["D".into(), 4.into()])
        .check_dims("sh_coeffs", &sh_coeffs, &["D".into(), "C".into(), 3.into()])
        .check_dims("raw_opacities", &raw_opacities, &["D".into()]);

    // Divide screen into tiles.
    let tile_bounds = calc_tile_bounds(img_size);

    // A note on some confusing naming that'll be used throughout this function:
    // Gaussians are stored in various states of buffers, eg. at the start they're all in one big buffer,
    // then we sparsely store some results, then sort gaussian based on depths, etc.
    // Overall this means there's lots of indices flying all over the place, and it's hard to keep track
    // what is indexing what. So, for some sanity, try to match a few "gaussian ids" (gid) variable names.
    // - Global Gaussian ID - global_gid
    // - Compacted Gaussian ID - compact_gid
    // - Per tile intersection depth sorted ID - tiled_gid
    // - Sorted by tile per tile intersection depth sorted ID - sorted_tiled_gid
    // Then, various buffers map between these, which are named x_from_y_gid, eg.
    //  global_from_compact_gid.

    // Tile rendering setup.
    let sh_degree = sh_degree_from_coeffs(sh_coeffs.shape.dims[1] as u32);
    let total_splats = means.shape.dims[0];
    let max_intersects = max_intersections(tile_bounds, total_splats as u32);
    let focal = camera.focal(img_size);
    let center = camera.center(img_size);

    // splat transform uniforms.
    let vp = glam::Mat4::from_cols(
        vec4(focal.x, 0.0, 0.0, 0.0),
        vec4(0.0, focal.y, 0.0, 0.0),
        vec4(center.x, center.y, VP33, 1.0),
        vec4(0.0, 0.0, VP34, 0.0),
    );
    let w2c = glam::Mat4::from(camera.world_to_local());
    let vpm = vp * w2c;
    let vpm_t = vpm.transpose();
    let m_z = w2c.row(2);

    let uniforms = shaders::helpers::RenderUniforms {
        vpm_t: vpm_t.to_cols_array_2d(),
        m_z: m_z.into(),
        camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
        focal: focal.into(),
        pixel_center: center.into(),
        img_size: img_size.into(),
        tile_bounds: tile_bounds.into(),
        sh_degree,
        total_splats: total_splats as u32,
        max_intersects,
        background: [background.x, background.y, background.z, 1.0],
        // Nb: Bit of a hack as these aren't _really_ uniforms but are written to by the shaders.
        num_visible: 0,
        num_intersections: 0,
        near_plane: NEAR_PLANE,
        far_plane: FAR_PLANE,
        padding: 0.0, // padding for 16 byte alignment
    };

    // Nb: This contains both static metadata and some dynamic data so can't pass this as metadata to execute. In the future
    // should separate the two.
    let uniforms_buffer = create_uniform_buffer(uniforms, device, client);

    let projected_splats = create_tensor([total_splats, PROJ_SIZE], device, DType::F32);
    let splat_bounds = create_tensor([total_splats, BOUNDS_SIZE], device, DType::F32);

    let (cum_tiles_hit, num_visible, num_intersections) = {
        let splat_intersect_counts =
            MainBackendBase::int_zeros([total_splats + 1].into(), device, IntDType::U32);

        tracing::trace_span!("Project").in_scope(||
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
            client.launch_unchecked(
                Project::task(),
                calc_cube_count([total_splats as u32], Project::WORKGROUP_SIZE),
                Bindings::new().with_buffers(
                vec![
                    uniforms_buffer.handle.clone().binding(),
                    means.handle.binding(),
                    log_scales.handle.binding(),
                    quats.handle.binding(),
                    sh_coeffs.handle.binding(),
                    raw_opacities.handle.binding(),
                    projected_splats.handle.clone().binding(),
                    splat_bounds.handle.clone().binding(),
                    splat_intersect_counts.handle.clone().binding(),
                ]),
            ).expect("Failed to render splats");
        });

        // TODO: Only need to do this up to num_visible gaussians really.
        let cum_tiles_hit = tracing::trace_span!("PrefixSumGaussHits")
            .in_scope(|| prefix_sum(splat_intersect_counts));

        // Get the number of visible splats and instances from the uniforms buffer.
        let num_vis_field_offset = offset_of!(shaders::helpers::RenderUniforms, num_visible) / 4;
        let num_visible = MainBackendBase::int_slice(
            uniforms_buffer.clone(),
            &[(num_vis_field_offset..num_vis_field_offset + 1).into()],
        );
        let num_hits_field_offset = offset_of!(shaders::helpers::RenderUniforms, num_intersections) / 4;
        let num_intersections = MainBackendBase::int_slice(
            uniforms_buffer.clone(),
            &[(num_hits_field_offset..num_hits_field_offset + 1).into()],
        );

        (cum_tiles_hit, num_visible, num_intersections)
    };

    // Each intersection maps to a gaussian.
    let (tile_offsets, compact_gid_from_isect) = {
        let num_vis_map_wg =
            create_dispatch_buffer(num_visible, MapGaussiansToIntersect::WORKGROUP_SIZE);

        let tile_id_from_isect = create_tensor([max_intersects as usize], device, DType::U32);
        let compact_gid_from_isect = create_tensor([max_intersects as usize], device, DType::U32);

        tracing::trace_span!("MapGaussiansToIntersect").in_scope(|| {
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
            client.launch_unchecked(
                MapGaussiansToIntersect::task(false),
                CubeCount::Dynamic(num_vis_map_wg.handle.clone().binding()),
                Bindings::new().with_buffers(
                vec![
                    uniforms_buffer.handle.clone().binding(),
                    splat_bounds.handle.binding(),
                    cum_tiles_hit.handle.binding(),
                    tile_id_from_isect.handle.clone().binding(),
                    compact_gid_from_isect.handle.clone().binding(),
                ]),
            ).expect("Failed to render splats");}
        });

        // We're sorting by tile ID, but we know beforehand what the maximum value
        // can be. We don't need to sort all the leading 0 bits!
        let num_tiles = tile_bounds.x * tile_bounds.y;
        let bits = u32::BITS - num_tiles.leading_zeros();

        let (tile_id_from_isect, compact_gid_from_isect) = tracing::trace_span!("Tile sort")
            .in_scope(|| {
                radix_argsort(
                    tile_id_from_isect,
                    compact_gid_from_isect,
                    &num_intersections,
                    bits,
                )
            });

        let cube_dim = CubeDim::new_1d(256);
        let num_vis_map_wg =
            create_dispatch_buffer(num_intersections.clone(), [256 * CHECKS_PER_ITER, 1, 1]);
        let cube_count = CubeCount::Dynamic(num_vis_map_wg.handle.binding());

        // Tiles without splats will be written as having a range of [0, 0].
        let tile_offsets = MainBackendBase::int_zeros(
            [tile_bounds.y as usize, tile_bounds.x as usize, 2].into(),
            device,
            IntDType::U32,
        );

        // SAFETY: Safe kernel.
        unsafe {
            get_tile_offsets::launch_unchecked::<WgpuRuntime>(
                client,
                cube_count,
                cube_dim,
                tile_id_from_isect.as_tensor_arg(1),
                tile_offsets.as_tensor_arg(1),
                num_intersections.as_tensor_arg(1),
            )
            .expect("Failed to render splats");
        }

        (tile_offsets, compact_gid_from_isect)
    };

    let _span = tracing::trace_span!("Rasterize").entered();

    let out_img = create_tensor(
        [img_size.y as usize, img_size.x as usize, 1], // Channels are packed into 4 bytes, aka one float.
        device,
        DType::F32,
    );

    let bindings = Bindings::new().with_buffers(vec![
        uniforms_buffer.handle.clone().binding(),
        compact_gid_from_isect.handle.clone().binding(),
        tile_offsets.handle.clone().binding(),
        projected_splats.handle.clone().binding(),
        out_img.handle.clone().binding(),
    ]);

    // Compile the kernel.
    let raster_task = Rasterize::task(cfg!(target_family = "wasm"));

    // SAFETY: Kernel checked to have no OOB, bounded loops.
    unsafe {
        client
            .launch_unchecked(
                raster_task,
                CubeCount::Static(tile_bounds.x * tile_bounds.y, 1, 1),
                bindings,
            )
            .expect("Failed to render splats");
    }

    // these are not needed for inference
    let visible = create_tensor([1], device, DType::F32);
    let global_from_compact_gid =
        create_tensor([1], device, DType::U32); // int_zeros([total_splats].into(), device, IntDType::U32);

    // Sanity check the buffers.
    assert!(
        uniforms_buffer.is_contiguous(),
        "Uniforms must be contiguous"
    );
    assert!(
        tile_offsets.is_contiguous(),
        "Tile offsets must be contiguous"
    );
    assert!(
        global_from_compact_gid.is_contiguous(),
        "Global from compact gid must be contiguous"
    );
    assert!(visible.is_contiguous(), "Visible must be contiguous");
    assert!(
        projected_splats.is_contiguous(),
        "Projected splats must be contiguous"
    );
    assert!(
        num_intersections.is_contiguous(),
        "Num intersections must be contiguous"
    );

    (
        out_img,
        RenderAux {
            uniforms_buffer,
            tile_offsets,
            num_intersections,
            projected_splats,
            compact_gid_from_isect,
            global_from_compact_gid,
            visible,
            img_size,
        },
    )
}
