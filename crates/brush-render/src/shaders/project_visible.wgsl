#import helpers;

struct IsectInfo {
    compact_gid: u32,
    tile_id: u32,
}

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;

@group(0) @binding(1) var<storage, read> means: array<helpers::PackedVec3>;
@group(0) @binding(2) var<storage, read> log_scales: array<helpers::PackedVec3>;
@group(0) @binding(3) var<storage, read> quats: array<vec4f>;
@group(0) @binding(4) var<storage, read> coeffs: array<helpers::PackedVec3>;
@group(0) @binding(5) var<storage, read> raw_opacities: array<f32>;
@group(0) @binding(6) var<storage, read> global_from_compact_gid: array<u32>;
@group(0) @binding(7) var<storage, read_write> projected: array<helpers::ProjectedSplat>;

struct SvCoeffs {
    s0_c: vec3f,
    s0_s: vec3f,

    s1_c: vec3f,
    s1_s: vec3f,

    s2_c: vec3f,
    s2_s: vec3f,

    s3_c: vec3f,
    s3_s: vec3f,

    s4_c: vec3f,
    s4_s: vec3f,

    s5_c: vec3f,
    s5_s: vec3f,

    s6_c: vec3f,
    s6_s: vec3f,

    s7_c: vec3f,
    s7_s: vec3f,
}

fn sv_coeffs_to_color(
    viewdir: vec3f,
    sv: SvCoeffs,
) -> vec3f {

    var weights_sum = 0.0f;
    var total_color = vec3f(0.0);

    // site 0
    let w0 = exp(dot(sv.s0_s, viewdir));
    weights_sum += w0;
    total_color += w0 * sv.s0_c;

    // site 1
    let w1 = exp(dot(sv.s1_s, viewdir));
    weights_sum += w1;
    total_color += w1 * sv.s1_c;

    // site 2
    let w2 = exp(dot(sv.s2_s, viewdir));
    weights_sum += w2;
    total_color += w2 * sv.s2_c;

    // site 3
    let w3 = exp(dot(sv.s3_s, viewdir));
    weights_sum += w3;
    total_color += w3 * sv.s3_c;

    // site 4
    let w4 = exp(dot(sv.s4_s, viewdir));
    weights_sum += w4;
    total_color += w4 * sv.s4_c;

    // site 5
    let w5 = exp(dot(sv.s5_s, viewdir));
    weights_sum += w5;
    total_color += w5 * sv.s5_c;

    // site 6
    let w6 = exp(dot(sv.s6_s, viewdir));
    weights_sum += w6;
    total_color += w6 * sv.s6_c;

    // site 7
    let w7 = exp(dot(sv.s7_s, viewdir));
    weights_sum += w7;
    total_color += w7 * sv.s7_c;

    return total_color / weights_sum;
}

fn read_coeffs(base_id: ptr<function, u32>) -> vec3f {
    let ret = helpers::as_vec(coeffs[*base_id]);
    *base_id += 1u;
    return ret;
}

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let compact_gid = gid.x;

    if compact_gid >= uniforms.num_visible {
        return;
    }

    let global_gid = global_from_compact_gid[compact_gid];

    // Project world space to camera space.
    let mean = helpers::as_vec(means[global_gid]);
    let scale = exp(helpers::as_vec(log_scales[global_gid]));

    // Safe to normalize, splats with length(quat) == 0 are invisible.
    let quat = normalize(quats[global_gid]);
    let opac = helpers::sigmoid(raw_opacities[global_gid]);

    let viewmat = uniforms.viewmat;
    let R = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let mean_c = R * mean + viewmat[3].xyz;

    let covar = helpers::calc_cov3d(scale, quat);
    let cov2d = helpers::calc_cov2d(covar, mean_c, uniforms.focal, uniforms.img_size, uniforms.pixel_center, viewmat);
    let conic = helpers::inverse(cov2d);

    // compute the projected mean
    let rz = 1.0 / mean_c.z;
    let mean2d = uniforms.focal * mean_c.xy * rz + uniforms.pixel_center;

    var base_id = u32(global_gid) * 16;

    var sv = SvCoeffs();
    sv.s0_c = read_coeffs(&base_id);
    sv.s0_s = read_coeffs(&base_id);
    sv.s1_c = read_coeffs(&base_id);
    sv.s1_s = read_coeffs(&base_id);
    sv.s2_c = read_coeffs(&base_id);
    sv.s2_s = read_coeffs(&base_id);
    sv.s3_c = read_coeffs(&base_id);
    sv.s3_s = read_coeffs(&base_id);
    sv.s4_c = read_coeffs(&base_id);
    sv.s4_s = read_coeffs(&base_id);
    sv.s5_c = read_coeffs(&base_id);
    sv.s5_s = read_coeffs(&base_id);
    sv.s6_c = read_coeffs(&base_id);
    sv.s6_s = read_coeffs(&base_id);
    sv.s7_c = read_coeffs(&base_id);
    sv.s7_s = read_coeffs(&base_id);

    // Write projected splat information.
    let viewdir = normalize(mean - uniforms.camera_position.xyz);
    let color = sv_coeffs_to_color(viewdir, sv);

    projected[compact_gid] = helpers::create_projected_splat(
        mean2d,
        vec3f(conic[0][0], conic[0][1], conic[1][1]),
        vec4f(color, opac)
    );
}
