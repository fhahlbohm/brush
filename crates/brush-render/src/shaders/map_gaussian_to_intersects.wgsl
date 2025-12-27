#import helpers;

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> splat_bounds: array<helpers::SplatBounds>;
@group(0) @binding(2) var<storage, read> splat_cum_hit_counts: array<u32>;

@group(0) @binding(3) var<storage, read_write> tile_id_from_isect: array<u32>;
@group(0) @binding(4) var<storage, read_write> compact_gid_from_isect: array<u32>;
@group(0) @binding(5) var<storage, read_write> num_intersections: array<u32>;

const WG_SIZE: u32 = 256u;

@compute
@workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) num_wgs: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let compact_gid = helpers::get_global_id(wid, num_wgs, lid, WG_SIZE);

    if compact_gid == 0u {
        num_intersections[0] = splat_cum_hit_counts[uniforms.num_visible];
    }

    if compact_gid >= uniforms.num_visible {
        return;
    }

    let tile_bbox = splat_bounds[compact_gid];
    let tile_bounds_x = uniforms.tile_bounds.x;
    var offset = splat_cum_hit_counts[compact_gid];
    for (var ty = tile_bbox.min_y; ty < tile_bbox.max_y; ty++) {
        let row_start_id = ty * tile_bounds_x;
        for (var tx = tile_bbox.min_x; tx < tile_bbox.max_x; tx++) {
            let tile_id = tx + row_start_id;
            tile_id_from_isect[offset] = tile_id;
            compact_gid_from_isect[offset] = compact_gid;
            offset += 1u;
        }
    }
}
