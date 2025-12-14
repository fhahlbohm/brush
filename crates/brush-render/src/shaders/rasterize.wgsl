#import helpers

const K: u32 = 16u;

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> compact_gid_from_isect: array<u32>;
@group(0) @binding(2) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> projected: array<helpers::ProjectedSplat>;

@group(0) @binding(4) var<storage, read_write> out_img: array<u32>;

var<workgroup> range_uniform: vec2u;

var<workgroup> local_batch: array<helpers::ProjectedSplat, helpers::TILE_SIZE>;

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(helpers::TILE_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let pix_loc = helpers::map_1d_to_2d(global_id.x, uniforms.tile_bounds.x);
    let pix_id = pix_loc.x + pix_loc.y * uniforms.img_size.x;
    let pixel_coord = vec2f(pix_loc) + 0.5f;
    let tile_loc = vec2u(pix_loc.x / helpers::TILE_WIDTH, pix_loc.y / helpers::TILE_WIDTH);

    let tile_id = tile_loc.x + tile_loc.y * uniforms.tile_bounds.x;
    let inside = pix_loc.x < uniforms.img_size.x && pix_loc.y < uniforms.img_size.y;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between the bin counts.
    range_uniform = vec2u(
        tile_offsets[tile_id * 2],
        tile_offsets[tile_id * 2 + 1],
    );

    // Stupid hack as Chrome isn't convinced the range variable is uniform, which it better be.
    let range = workgroupUniformLoad(&range_uniform);

    var core_depths: array<f32, K>;
    var core_infos: array<vec4f, K>;
    for (var i = 0u; i < K; i++) {
        core_depths[i] = 123456789.0f;
        core_infos[i] = vec4f(0.0);
    }
    var T_tail = 1.0f;
    var rgba_premultiplied_tail = vec4f(0.0);

    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    for (var batch_start = range.x; batch_start < range.y; batch_start += helpers::TILE_SIZE) {
        // process gaussians in the current batch for this pixel
        let remaining = min(helpers::TILE_SIZE, range.y - batch_start);

        let load_isect_id = batch_start + local_idx;
        let compact_gid = compact_gid_from_isect[load_isect_id];

        workgroupBarrier();
        if local_idx < remaining {
            local_batch[local_idx] = projected[compact_gid];
        }
        workgroupBarrier();

        for (var t = 0u; t < remaining; t++) {
            let proj = local_batch[t];

            let VPMT1 = vec4f(proj.VPMT1_x, proj.VPMT1_y, proj.VPMT1_z, proj.VPMT1_w);
            let VPMT2 = vec4f(proj.VPMT2_x, proj.VPMT2_y, proj.VPMT2_z, proj.VPMT2_w);
            let VPMT4 = vec4f(proj.VPMT4_x, proj.VPMT4_y, proj.VPMT4_z, proj.VPMT4_w);
            let MT3 = vec4f(proj.MT3_x, proj.MT3_y, proj.MT3_z, proj.MT3_w);
            let color = vec4f(proj.color_r, proj.color_g, proj.color_b, proj.color_a);

            let plane_x_diag = VPMT1 - VPMT4 * pixel_coord.x;
            let plane_y_diag = VPMT2 - VPMT4 * pixel_coord.y;
            let m = plane_x_diag.w * plane_y_diag.xyz - plane_x_diag.xyz * plane_y_diag.w;
            let d = cross(plane_x_diag.xyz, plane_y_diag.xyz);
            let numerator_rho2 = dot(m, m);
            let denominator = dot(d, d);
            if (numerator_rho2 > 11.0825270903f * denominator) {
                continue;
            }
            let denominator_rcp = 1.0f / denominator;
            let sigma = 0.5f * numerator_rho2 * denominator_rcp;
            let alpha = color.a * exp(-sigma);
            if (alpha < 1.0f / 255.0f) {
                continue;
            }
            let eval_point_diag = cross(d, m) * denominator_rcp;
            var depth = dot(MT3.xyz, eval_point_diag) + MT3.w;
            var rgba_premultiplied = vec4f(color.rgb * alpha, alpha);

            if (depth < core_depths[K - 1u] && alpha >= 0.05f) {
                for (var core_idx = 0u; core_idx < K; core_idx++) {
                    if (depth < core_depths[core_idx]) {
                        let temp_depth = depth;
                        depth = core_depths[core_idx];
                        core_depths[core_idx] = temp_depth;
                        let temp_rgba_premultiplied = rgba_premultiplied;
                        rgba_premultiplied = core_infos[core_idx];
                        core_infos[core_idx] = temp_rgba_premultiplied;
                    }
                }
            }
            rgba_premultiplied_tail += rgba_premultiplied;
            T_tail *= (1.0f - rgba_premultiplied.a);
        }
    }

    if inside {
        var pix_out = vec3f(0.0);
        var T = 1.0f;
        for (var core_idx = 0u; core_idx < K; core_idx++) {
            let rgba_premultiplied = core_infos[core_idx];
            pix_out += T * rgba_premultiplied.rgb;
            T *= (1.0f - rgba_premultiplied.a);
            if (T < 1e-4f) {
                T = 0.0f;
                break;
            }
        }
        if (T > 0.0f && rgba_premultiplied_tail.a >= 1.0f / 255.0f) {
            let weight_tail = T * (1.0f - T_tail);
            pix_out += weight_tail * (1.0f / rgba_premultiplied_tail.a) * rgba_premultiplied_tail.rgb;
            T *= T_tail;
        }
        // Compose with background. Nb that color is already pre-multiplied
        // by definition.
        let final_color = vec4f(pix_out + T * uniforms.background.rgb, 1.0f - T);

        let colors_u = vec4u(clamp(final_color * 255.0f, vec4f(0.0), vec4f(255.0)));
        let packed: u32 = colors_u.x | (colors_u.y << 8u) | (colors_u.z << 16u) | (colors_u.w << 24u);
        out_img[pix_id] = packed;
    }
}
