use crate::{
    burn_texture::BurnTexture, dataset_readers, gaussian_splats::Splats, orbit_controls::OrbitControls, scene::{SceneBatcher, SceneLoader}, splat_import, train::{LrConfig, SplatTrainer, TrainConfig}
};
use brush_kernel::BurnBack;
use brush_render::camera::Camera;
use burn::{backend::Autodiff, module::AutodiffModule};
use burn_wgpu::RuntimeOptions;
use tracing::info_span;
use egui::{pos2, Color32, Rect};
use glam::{Mat4, Quat, Vec2, Vec3};
use std::time::{self, Duration};

type BurnDiffBack = Autodiff<BurnBack>;

pub struct Viewer {
    camera: Camera,
    dataloader: Option<SceneLoader<BurnDiffBack>>,
    splats: Option<Splats<BurnDiffBack>>,
    trainer: Option<SplatTrainer<BurnDiffBack>>,
    train_active: bool,

    rec: Option<rerun::RecordingStream>,
    reference_cameras: Option<Vec<Camera>>,
    backbuffer: Option<BurnTexture>,
    controls: OrbitControls,
    device: <BurnDiffBack as burn::prelude::Backend>::Device,
    start_transform: Mat4,
    last_render_time: time::Instant,
}

impl Viewer {
    pub fn new(cc: &eframe::CreationContext) -> Self {
        let state = cc.wgpu_render_state.as_ref().unwrap();

        // Run the burn backend on the egui WGPU device.
        let device = burn::backend::wgpu::init_existing_device(
            0,
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
            // Burn atm has pretty suboptimal memory management leading to weird behaviour. This effectively disables
            // it, with a GC run every couple seconds. Seems good enough for now.
            // (nb: it doesn't mean burn will never re-use memory, just that it hangs on to
            // GPU allocations for longer).
            RuntimeOptions {
                dealloc_strategy: burn_compute::memory_management::simple::DeallocStrategy::PeriodTime {
                    period: Duration::from_secs(5),
                    state: time::Instant::now(),
                },
                tasks_max: 128,
                ..Default::default()
            },
        );

        Viewer {
            camera: Camera::new(Vec3::ZERO, Quat::IDENTITY, 0.5, 0.5),
            splats: None,
            trainer: None,
            dataloader: None,
            reference_cameras: None,
            backbuffer: None,
            rec: None,
            train_active: true,
            controls: OrbitControls::new(15.0),
            device,
            start_transform: glam::Mat4::IDENTITY,
            last_render_time: time::Instant::now(),
        }
    }

    pub fn load_splats(&mut self, path: &str, reference_view: Option<&str>) {
        self.trainer = None;
        self.reference_cameras =
            reference_view.and_then(|s| dataset_readers::read_viewpoint_data(s).ok());

        if let Some(refs) = self.reference_cameras.as_ref() {
            self.camera = refs[0].clone();
        }

        self.splats = splat_import::load_splat_from_ply(path, &self.device).ok();
    }

    pub fn set_splats(&mut self, splats: Splats<BurnDiffBack>) {
        self.splats = Some(splats);
    }

    pub fn start_training(&mut self, path: &str) {
        self.rec = rerun::RecordingStreamBuilder::new("visualize training")
        .spawn().ok();

        <BurnDiffBack as burn::prelude::Backend>::seed(42);
    
        let config = TrainConfig::new(
            LrConfig::new().with_max_lr(2e-5).with_min_lr(5e-6),
            LrConfig::new().with_max_lr(4e-2).with_min_lr(2e-2),
            LrConfig::new().with_max_lr(1e-2).with_min_lr(6e-3),
            path.to_owned(),
        );

        let splats = Splats::<BurnDiffBack>::init_random(5000, 2.0, &self.device);

        println!("Loading dataset {:?}, {}", std::env::current_dir(), config.scene_path);
        let scene = dataset_readers::read_scene(&config.scene_path, "transforms_train.json", None);

        #[cfg(feature = "rerun")]
        scene.visualize(self.rec.as_ref().unwrap()).unwrap();

        // TODO: Unify reference views & training scene.
        self.trainer = Some(SplatTrainer::new(splats.num_splats(), &config, &splats));
        self.splats = Some(splats);
        let batcher_train = SceneBatcher::<BurnDiffBack>::new(self.device.clone());
        self.dataloader = Some(SceneLoader::new(scene, batcher_train, 2)); 
    }
}

impl eframe::App for Viewer {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui_extras::install_image_loaders(ctx);

        if let Some(cameras) = self.reference_cameras.as_ref() {
            egui::SidePanel::new(egui::panel::Side::Left, "cam panel").show(ctx, |ui| {
                for (i, camera) in cameras.iter().enumerate() {
                    if ui.button(format!("Camera {i}")).clicked() {
                        self.camera = camera.clone();
                        self.controls.position = self.camera.position;
                        self.controls.rotation = self.camera.rotation;
                    }
                }
            });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Brush splat viewer");
            ui.label("load pretrained models.");
            ui.horizontal(|ui| {
                for r in ["bonsai", "stump", "counter", "garden", "truck"] {
                    if ui.button(r.to_string()).clicked() {
                        self.load_splats(
                            &format!("./brush_data/pretrained/{r}/point_cloud/iteration_30000/point_cloud.ply"),
                            Some(&format!("./brush_data/pretrained/{r}/cameras.json")),
                        );
                    }
                }
            });

            ui.label("Train a new model.");
            ui.horizontal(|ui| {
                for r in ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"] {
                    if ui.button(r.to_string()).clicked() {
                        self.start_training(&format!("./brush_data/nerf_synthetic/{r}/"))
                    }
                }
            });

            let now = time::Instant::now();
            let ms = (now - self.last_render_time).as_secs_f64() * 1000.0;
            let fps = 1000.0 / ms;
            self.last_render_time = now;

            ui.label(format!("FPS: {fps:.0} {ms:.0} ms/frame"));

            if let Some(train) = self.trainer.as_mut() {    
                ui.label(format!("Training step {}", train.iter));
                ui.checkbox(&mut self.train_active, "Train");

                // TODO: On non wasm this really should be threaded.
                let get_span = info_span!("Get batch").entered();
                let batch = self.dataloader.as_mut().unwrap().next();
                drop(get_span);

                if self.train_active {
                    let splats = self.splats.take().unwrap();
                    
                    self.splats = Some(
                        train
                            .step(
                                batch,
                                splats,
                                self.rec.as_ref().unwrap(),
                            )
                            .unwrap(),
                    );
                }
            }

            let size = ui.available_size();

            // Round to 64 pixels for buffer alignment.
            let size = glam::uvec2(((size.x as u32).div_ceil(64) * 64).max(32), (size.y as u32).max(32));

            if let Some(splats) = &self.splats {
                let (rect, response) = ui.allocate_exact_size(
                    egui::Vec2::new(size.x as f32, size.y as f32),
                    egui::Sense::drag(),
                );

                let mouse_delta = glam::vec2(response.drag_delta().x, response.drag_delta().y);

                let (pan, rotate) = if response.dragged_by(egui::PointerButton::Primary) {
                    (Vec2::ZERO, mouse_delta)
                } else if response.dragged_by(egui::PointerButton::Secondary) {
                    (mouse_delta, Vec2::ZERO)
                } else {
                    (Vec2::ZERO, Vec2::ZERO)
                };

                let scrolled = ui.input(|r| r.smooth_scroll_delta).y;
    
                // TODO: Controls can be pretty borked.
                self.controls.pan_orbit_camera(
                    pan * 0.5,
                    rotate * 0.5,
                    scrolled * 0.02,
                    glam::vec2(rect.size().x, rect.size().y),
                );

                let controls_transform = glam::Mat4::from_rotation_translation(
                    self.controls.rotation,
                    self.controls.position,
                );

                let total_transform = self.start_transform * controls_transform;
                let (_, rot, pos) = total_transform.to_scale_rotation_translation();
                self.camera.position = pos;
                self.camera.rotation = rot;

                // TODO: For reference cameras just need to match fov?
                self.camera.fovx = 0.5;
                self.camera.fovy = 0.5 * (size.y as f32) / (size.x as f32);
    
                let (img, _) =
                    splats.clone().valid().render(&self.camera, size, glam::vec3(0.0, 0.0, 0.0), true);


                let back = self.backbuffer.get_or_insert_with(|| BurnTexture::new(img.clone(), frame));
                back.update_texture(img, frame);
                
                ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
                ui.painter().image(
                    back.id,
                    rect,
                    Rect {
                        min: pos2(0.0, 0.0),
                        max: pos2(1.0, 1.0),
                    },
                    Color32::WHITE,
                );
            
                ctx.request_repaint();
            }
        });
    }
  
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 1.0]
    }
}