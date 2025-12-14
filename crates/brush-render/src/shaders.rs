use brush_wgsl::wgsl_kernel;

// Define kernels using proc macro

#[wgsl_kernel(source = "src/shaders/project.wgsl")]
pub struct Project;

#[wgsl_kernel(source = "src/shaders/map_gaussian_to_intersects.wgsl")]
pub struct MapGaussiansToIntersect {
    pub prepass: bool,
}

#[wgsl_kernel(source = "src/shaders/rasterize.wgsl")]
pub struct Rasterize {
    pub webgpu: bool,
}

// Re-export helper types and constants from the kernel modules that use them
pub mod helpers {
    // Types used by multiple shaders - available from project
    pub use super::project::PackedVec3;
    pub use super::project::SplatBounds;
    pub use super::project::TransformedSplat;
    pub use super::project::RenderUniforms;

    // Constants are now associated with the kernel structs
    pub const TILE_SIZE: u32 = super::Rasterize::TILE_SIZE;
    pub const TILE_WIDTH: u32 = super::Rasterize::TILE_WIDTH;
}

// Re-export module-specific constants
pub const SH_C0: f32 = Project::SH_C0;
