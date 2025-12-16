#define UNIFORM_WRITE

#import helpers;

@group(0) @binding(0) var<storage, read_write> uniforms: helpers::RenderUniforms;

@group(0) @binding(1) var<storage, read> means: array<helpers::PackedVec3>;
@group(0) @binding(2) var<storage, read> log_scales: array<helpers::PackedVec3>;
@group(0) @binding(3) var<storage, read> quats: array<vec4f>;
@group(0) @binding(4) var<storage, read> coeffs: array<helpers::PackedVec3>;
@group(0) @binding(5) var<storage, read> raw_opacities: array<f32>;

@group(0) @binding(6) var<storage, read_write> transformed: array<helpers::TransformedSplat>;
@group(0) @binding(7) var<storage, read_write> splat_bounds: array<helpers::SplatBounds>;
@group(0) @binding(8) var<storage, read_write> splat_intersect_counts: array<u32>;

struct ShCoeffs {
    b0_c0: vec3f,

    b1_c0: vec3f,
    b1_c1: vec3f,
    b1_c2: vec3f,

    b2_c0: vec3f,
    b2_c1: vec3f,
    b2_c2: vec3f,
    b2_c3: vec3f,
    b2_c4: vec3f,

    b3_c0: vec3f,
    b3_c1: vec3f,
    b3_c2: vec3f,
    b3_c3: vec3f,
    b3_c4: vec3f,
    b3_c5: vec3f,
    b3_c6: vec3f,

    b4_c0: vec3f,
    b4_c1: vec3f,
    b4_c2: vec3f,
    b4_c3: vec3f,
    b4_c4: vec3f,
    b4_c5: vec3f,
    b4_c6: vec3f,
    b4_c7: vec3f,
    b4_c8: vec3f,
}

const SH_C0: f32 = 0.2820947917738781f;

// Evaluate spherical harmonics bases at unit direction for high orders using approach described by
// Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
// See https://jcgt.org/published/0002/02/06/ for reference implementation
fn sh_coeffs_to_color(
    degree: u32,
    viewdir: vec3f,
    sh: ShCoeffs,
) -> vec3f {
    var colors = SH_C0 * sh.b0_c0;

    if (degree == 0) {
        return colors;
    }

    let x = viewdir.x;
    let y = viewdir.y;
    let z = viewdir.z;

    let fTmp0A = 0.48860251190292f;
    colors += fTmp0A *
                    (-y * sh.b1_c0 +
                    z * sh.b1_c1 -
                    x * sh.b1_c2);

    if (degree == 1) {
        return colors;
    }
    let z2 = z * z;

    let fTmp0B = -1.092548430592079f * z;
    let fTmp1A = 0.5462742152960395f;
    let fC1 = x * x - y * y;
    let fS1 = 2.f * x * y;
    let pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    let pSH7 = fTmp0B * x;
    let pSH5 = fTmp0B * y;
    let pSH8 = fTmp1A * fC1;
    let pSH4 = fTmp1A * fS1;

    colors +=
        pSH4 * sh.b2_c0 +
        pSH5 * sh.b2_c1 +
        pSH6 * sh.b2_c2 +
        pSH7 * sh.b2_c3 +
        pSH8 * sh.b2_c4;

    if (degree == 2) {
        return colors;
    }

    let fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    let fTmp1B = 1.445305721320277f * z;
    let fTmp2A = -0.5900435899266435f;
    let fC2 = x * fC1 - y * fS1;
    let fS2 = x * fS1 + y * fC1;
    let pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    let pSH13 = fTmp0C * x;
    let pSH11 = fTmp0C * y;
    let pSH14 = fTmp1B * fC1;
    let pSH10 = fTmp1B * fS1;
    let pSH15 = fTmp2A * fC2;
    let pSH9  = fTmp2A * fS2;
    colors +=   pSH9  * sh.b3_c0 +
                pSH10 * sh.b3_c1 +
                pSH11 * sh.b3_c2 +
                pSH12 * sh.b3_c3 +
                pSH13 * sh.b3_c4 +
                pSH14 * sh.b3_c5 +
                pSH15 * sh.b3_c6;

    if (degree == 3) {
        return colors;
    }

    let fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    let fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    let fTmp2B = -1.770130769779931f * z;
    let fTmp3A = 0.6258357354491763f;
    let fC3 = x * fC2 - y * fS2;
    let fS3 = x * fS2 + y * fC2;
    let pSH20 = (1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6);
    let pSH21 = fTmp0D * x;
    let pSH19 = fTmp0D * y;
    let pSH22 = fTmp1C * fC1;
    let pSH18 = fTmp1C * fS1;
    let pSH23 = fTmp2B * fC2;
    let pSH17 = fTmp2B * fS2;
    let pSH24 = fTmp3A * fC3;
    let pSH16 = fTmp3A * fS3;
    colors += pSH16 * sh.b4_c0 +
                pSH17 * sh.b4_c1 +
                pSH18 * sh.b4_c2 +
                pSH19 * sh.b4_c3 +
                pSH20 * sh.b4_c4 +
                pSH21 * sh.b4_c5 +
                pSH22 * sh.b4_c6 +
                pSH23 * sh.b4_c7 +
                pSH24 * sh.b4_c8;
    return colors;
}

fn num_sh_coeffs(degree: u32) -> u32 {
    return (degree + 1) * (degree + 1);
}

fn read_coeffs(base_id: ptr<function, u32>) -> vec3f {
    let ret = helpers::as_vec(coeffs[*base_id]);
    *base_id += 1u;
    return ret;
}

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let global_gid = gid.x;

    if global_gid >= uniforms.total_splats {
        return;
    }

    // early near/far culling based on view-space mean.
    let mean = helpers::as_vec(means[global_gid]);
    let M3 = uniforms.m_z;
    let M3_xyz = vec3f(M3.xyz);
    let depth = dot(M3_xyz, mean) + M3.w;
    if depth < uniforms.near_plane || depth > uniforms.far_plane {
        return;
    }

    // compute VPMT transform
    let rot = helpers::quat_to_mat(normalize(quats[global_gid]));
    let scale = exp(helpers::as_vec(log_scales[global_gid]));
    let u = rot[0] * scale.x;
    let v = rot[1] * scale.y;
    let w = rot[2] * scale.z;
    let VPM_T = uniforms.vpm_t;
    let VPM1 = VPM_T[0];
    let VPM2 = VPM_T[1];
    let VPM3 = VPM_T[2];
    let VPM4 = VPM_T[3];
    let VPMT1 = vec4f(dot(VPM1.xyz, u), dot(VPM1.xyz, v), dot(VPM1.xyz, w), dot(VPM1.xyz, mean) + VPM1.w);
    let VPMT2 = vec4f(dot(VPM2.xyz, u), dot(VPM2.xyz, v), dot(VPM2.xyz, w), dot(VPM2.xyz, mean) + VPM2.w);
    let VPMT3 = vec4f(dot(VPM3.xyz, u), dot(VPM3.xyz, v), dot(VPM3.xyz, w), dot(VPM3.xyz, mean) + VPM3.w);
    let VPMT4 = vec4f(dot(VPM4.xyz, u), dot(VPM4.xyz, v), dot(VPM4.xyz, w), dot(VPM4.xyz, mean) + VPM4.w);
    let MT3 = vec4f(dot(M3_xyz, u), dot(M3_xyz, v), dot(M3_xyz, w), depth);

    // compute screen-space bounding box and cull if outside
    let opac = helpers::sigmoid(raw_opacities[global_gid]);
    let rho_cutoff = 2.0f * log(opac * 255.0f); // corresponds to blending threshold of (1 / 255)
    let t = vec4f(rho_cutoff, rho_cutoff, rho_cutoff, -1.0f);
    let d = dot(t, VPMT4 * VPMT4);
    if (d == 0.0f) {
        return;
    }
    let f = (1.0f / d) * t;
    let center = vec3f(dot(f, VPMT1 * VPMT4), dot(f, VPMT2 * VPMT4), dot(f, VPMT3 * VPMT4));
    let extent = sqrt(max(center * center - vec3f(dot(f, VPMT1 * VPMT1), dot(f, VPMT2 * VPMT2), dot(f, VPMT3 * VPMT3)), vec3f(1e-12f)));
    let min_bounds = center - extent;
    let max_bounds = center + extent;
    if max_bounds.x <= 0.0f || min_bounds.x >= f32(uniforms.img_size.x) ||
       max_bounds.y <= 0.0f || min_bounds.y >= f32(uniforms.img_size.y) ||
       min_bounds.z <= -1.0f || max_bounds.z >= 1.0f {
        return;
    }

    let tile_bbox = helpers::get_tile_bbox(vec2f(center.xy), vec2f(extent.xy), uniforms.tile_bounds);
    let num_tiles_hit = (u32(tile_bbox.z) - u32(tile_bbox.x)) * (u32(tile_bbox.w) - u32(tile_bbox.y));
    if num_tiles_hit == 0u {
        return;
    }

    let sh_degree = uniforms.sh_degree;
    let num_coeffs = num_sh_coeffs(sh_degree);
    var base_id = u32(global_gid) * num_coeffs;

    var sh = ShCoeffs();
    sh.b0_c0 = read_coeffs(&base_id);

    if sh_degree >= 1 {
        sh.b1_c0 = read_coeffs(&base_id);
        sh.b1_c1 = read_coeffs(&base_id);
        sh.b1_c2 = read_coeffs(&base_id);

        if sh_degree >= 2 {
            sh.b2_c0 = read_coeffs(&base_id);
            sh.b2_c1 = read_coeffs(&base_id);
            sh.b2_c2 = read_coeffs(&base_id);
            sh.b2_c3 = read_coeffs(&base_id);
            sh.b2_c4 = read_coeffs(&base_id);

            if sh_degree >= 3 {
                sh.b3_c0 = read_coeffs(&base_id);
                sh.b3_c1 = read_coeffs(&base_id);
                sh.b3_c2 = read_coeffs(&base_id);
                sh.b3_c3 = read_coeffs(&base_id);
                sh.b3_c4 = read_coeffs(&base_id);
                sh.b3_c5 = read_coeffs(&base_id);
                sh.b3_c6 = read_coeffs(&base_id);

                if sh_degree >= 4 {
                    sh.b4_c0 = read_coeffs(&base_id);
                    sh.b4_c1 = read_coeffs(&base_id);
                    sh.b4_c2 = read_coeffs(&base_id);
                    sh.b4_c3 = read_coeffs(&base_id);
                    sh.b4_c4 = read_coeffs(&base_id);
                    sh.b4_c5 = read_coeffs(&base_id);
                    sh.b4_c6 = read_coeffs(&base_id);
                    sh.b4_c7 = read_coeffs(&base_id);
                    sh.b4_c8 = read_coeffs(&base_id);
                }
            }
        }
    }

    let viewdir = normalize(mean - uniforms.camera_position.xyz);
    let color = max(sh_coeffs_to_color(sh_degree, viewdir, sh) + vec3f(0.5), vec3f(0.0));

    let write_id = atomicAdd(&uniforms.num_visible, 1u);
    transformed[write_id] = helpers::create_transformed_splat(
        VPMT1,
        VPMT2,
        VPMT4,
        MT3,
        vec4f(color, opac)
    );
    splat_bounds[write_id] = helpers::create_splat_bounds(tile_bbox);
    atomicAdd(&uniforms.num_intersections, num_tiles_hit);
    splat_intersect_counts[write_id + 1u] = num_tiles_hit;

}
