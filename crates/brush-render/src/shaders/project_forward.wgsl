#define UNIFORM_WRITE

#import helpers;

// Unfiroms contains the splat count which we're writing to.
@group(0) @binding(0) var<storage, read_write> uniforms: helpers::RenderUniforms;

@group(0) @binding(1) var<storage, read> means: array<helpers::PackedVec3>;
@group(0) @binding(2) var<storage, read> quats: array<vec4f>;
@group(0) @binding(3) var<storage, read> log_scales: array<helpers::PackedVec3>;
@group(0) @binding(4) var<storage, read> raw_opacities: array<f32>;

@group(0) @binding(5) var<storage, read_write> global_from_compact_gid: array<u32>;
@group(0) @binding(6) var<storage, read_write> depths: array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let global_gid = global_id.x;

    if global_gid >= uniforms.total_splats {
        return;
    }

    const near = 0.2f;
    const far = 1000.0f;

    // early near/far culling based on view-space mean.
    let mean = helpers::as_vec(means[global_gid]);
    let viewmat = uniforms.viewmat;
    let depth = viewmat[0].z * mean.x + viewmat[1].z * mean.y + viewmat[2].z * mean.z + viewmat[3].z;
    if depth < near || depth > far {
        return;
    }

    let opac = helpers::sigmoid(raw_opacities[global_gid]);
    if opac < 1.0f / 255.0f {
        return;
    }

    var quat = quats[global_gid];
    let quat_norm_sqr = dot(quat, quat);
    if quat_norm_sqr < 1e-6f {
        return;
    }
    quat *= inverseSqrt(quat_norm_sqr);
    let rot = helpers::quat_to_mat(quat);

    let scale = exp(helpers::as_vec(log_scales[global_gid]));

    let u = rot[0] * scale.x;
    let v = rot[1] * scale.y;
    let w = rot[2] * scale.z;
    let T = mat4x4f(
        vec4f(u, 0.0f),
        vec4f(v, 0.0f),
        vec4f(w, 0.0f),
        vec4f(mean, 1.0f)
    );

    // compute VPMT transform
    let depth_range = far - near;
    let VP = mat4x4f(
        vec4f(uniforms.focal.x, 0.0f, 0.0f, 0.0f), // 1st col
        vec4f(0.0f, uniforms.focal.y, 0.0f, 0.0f), // 2nd col
        vec4f(uniforms.pixel_center.x, uniforms.pixel_center.y, (far + near) / depth_range, 1.0f), // 3rd col
        vec4f(0.0f, 0.0f, -2.0f * near * far / depth_range, 0.0f) // 4th col
    );
    let VPMT = VP * viewmat * T;
    let VPMT1 = vec4f(VPMT[0].x, VPMT[1].x, VPMT[2].x, VPMT[3].x);
    let VPMT2 = vec4f(VPMT[0].y, VPMT[1].y, VPMT[2].y, VPMT[3].y);
    let VPMT3 = vec4f(VPMT[0].z, VPMT[1].z, VPMT[2].z, VPMT[3].z);
    let VPMT4 = vec4f(VPMT[0].w, VPMT[1].w, VPMT[2].w, VPMT[3].w);

    // compute screen-space bounding box and cull if outside
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
    if (max_bounds.x <= 0.0f || min_bounds.x >= f32(uniforms.img_size.x) ||
        max_bounds.y <= 0.0f || min_bounds.y >= f32(uniforms.img_size.y) ||
        min_bounds.z <= -1.0f || max_bounds.z >= 1.0f) {
        return;
    }

    // Now write all the data to the buffers.
    let write_id = atomicAdd(&uniforms.num_visible, 1u);
    global_from_compact_gid[write_id] = global_gid;
    depths[write_id] = 1.0f;
}
