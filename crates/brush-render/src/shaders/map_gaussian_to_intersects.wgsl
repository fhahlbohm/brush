#import helpers;

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> splat_bounds: array<helpers::SplatBounds>;

#ifdef PREPASS
    @group(0) @binding(2) var<storage, read_write> splat_intersect_counts: array<u32>;
#else
    @group(0) @binding(2) var<storage, read> splat_cum_hit_counts: array<u32>;
    @group(0) @binding(3) var<storage, read_write> tile_id_from_isect: array<u32>;
    @group(0) @binding(4) var<storage, read_write> compact_gid_from_isect: array<u32>;
    @group(0) @binding(5) var<storage, read_write> num_intersections: array<u32>;
#endif

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let compact_gid = gid.x;

#ifndef PREPASS
    if gid.x == 0 {
        num_intersections[0] = splat_cum_hit_counts[uniforms.num_visible];
    }
#endif

    if compact_gid >= uniforms.num_visible {
        return;
    }

    let tile_bbox = splat_bounds[compact_gid];
    let num_tiles_hit = (u32(tile_bbox.max_x) - u32(tile_bbox.min_x)) * (u32(tile_bbox.max_y) - u32(tile_bbox.min_y));

#ifdef PREPASS
    splat_intersect_counts[compact_gid + 1u] = num_tiles_hit;
#endif

#ifndef PREPASS
    var offset = splat_cum_hit_counts[compact_gid];

    for (var tx = tile_bbox.min_x; tx < tile_bbox.max_x; tx++) {
        for (var ty = tile_bbox.min_y; ty < tile_bbox.max_y; ty++) {
            let tile_id = tx + ty * uniforms.tile_bounds.x;
            tile_id_from_isect[offset] = tile_id;
            compact_gid_from_isect[offset] = compact_gid;
            offset += 1u;
        }
    }
#endif
}
