use std::process::exit;

use wgsl_bindgen::{GlamWgslTypeMap, WgslBindgenOptionBuilder, WgslTypeSerializeStrategy};

fn main() {
    let bindgen = WgslBindgenOptionBuilder::default()
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .emit_rerun_if_change(true)
        .short_constructor(0)
        .type_map(GlamWgslTypeMap)
        .workspace_root("shaders")
        .add_entry_point("shaders/project_forward.wgsl")
        .output("src/splat_render/gen/bindings.rs")
        .build()
        .unwrap();
    match bindgen.generate() {
        Ok(_) => println!("Sucesfully updated wgsl bindings."),
        Err(e) => {
            println!("cargo:error={e}");
            exit(1);
        }
    }
}