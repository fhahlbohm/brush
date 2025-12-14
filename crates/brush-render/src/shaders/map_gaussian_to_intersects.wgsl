#import helpers;

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> projected: array<helpers::ProjectedSplat>;

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

    let projected = projected[compact_gid];
    let center = vec2f(projected.center_x, projected.center_y);
    let extent = vec2f(projected.extent_x, projected.extent_y);

    let tile_bbox = helpers::get_tile_bbox(center, extent, uniforms.tile_bounds);
    let tile_bbox_min = tile_bbox.xy;
    let tile_bbox_max = tile_bbox.zw;

    var num_tiles_hit = 0u;

    #ifndef PREPASS
        let base_isect_id = splat_cum_hit_counts[compact_gid];
    #endif

    // Nb: It's really really important here the two dispatches
    // of this kernel arrive at the exact same num_tiles_hit count. Otherwise
    // we might not be writing some intersection data.
    // This is a bit scary given potential optimizations that might happen depending
    // on which version is being ran.
    for (var tx = tile_bbox_min.x; tx < tile_bbox_max.x; tx++) {
        for (var ty = tile_bbox_min.y; ty < tile_bbox_max.y; ty++) {
            let tile_id = tx + ty * uniforms.tile_bounds.x;

        #ifndef PREPASS
            let isect_id = base_isect_id + num_tiles_hit;
            // Nb: isect_id MIGHT be out of bounds here for degenerate cases.
            // These kernels should be launched with bounds checking, so that these
            // writes are ignored. This will skip these intersections.
            tile_id_from_isect[isect_id] = tile_id;
            compact_gid_from_isect[isect_id] = compact_gid;
        #endif

            num_tiles_hit += 1u;
        }
    }

    #ifdef PREPASS
        splat_intersect_counts[compact_gid + 1u] = num_tiles_hit;
    #endif
}
