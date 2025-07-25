use crate::parsed_gaussian::ParsedGaussian;
use brush_render::gaussian_splats::Splats;
use burn::prelude::Backend;
use glam::{Quat, Vec3};
use ply_rs::{
    ply::{self, Ply, PropertyDef, PropertyType, ScalarType},
    writer::Writer,
};

async fn read_splat_data<B: Backend>(splats: Splats<B>) -> Vec<ParsedGaussian<false>> {
    let means = splats
        .means
        .val()
        .into_data_async()
        .await
        .into_vec()
        .expect("Unreachable");
    let log_scales = splats
        .log_scales
        .val()
        .into_data_async()
        .await
        .into_vec()
        .expect("Unreachable");
    let rotations = splats
        .rotation
        .val()
        .into_data_async()
        .await
        .into_vec()
        .expect("Unreachable");
    let opacities = splats
        .raw_opacity
        .val()
        .into_data_async()
        .await
        .into_vec()
        .expect("Unreachable");
    let sh_coeffs = splats
        .sh_coeffs
        .val()
        .permute([0, 2, 1]) // Permute to inria format ([n, channel, coeffs]).
        .into_data_async()
        .await
        .into_vec()
        .expect("Unreachable");

    let sh_coeffs_num = splats.sh_coeffs.dims()[1];

    (0..splats.num_splats())
        .filter_map(|i| {
            let i = i as usize;
            // Read SH data from [coeffs, channel] format to
            let sh_start = i * sh_coeffs_num * 3;
            let sh_end = (i + 1) * sh_coeffs_num * 3;

            let splat_sh = &sh_coeffs[sh_start..sh_end];

            let [sh_red, sh_green, sh_blue] = [
                &splat_sh[0..sh_coeffs_num],
                &splat_sh[sh_coeffs_num..sh_coeffs_num * 2],
                &splat_sh[sh_coeffs_num * 2..sh_coeffs_num * 3],
            ];

            let sh_dc = glam::vec3(sh_red[0], sh_green[0], sh_blue[0]);
            let sh_coeffs_rest = [&sh_red[1..], &sh_green[1..], &sh_blue[1..]].concat();

            let splat = ParsedGaussian {
                mean: Vec3::new(means[i * 3], means[i * 3 + 1], means[i * 3 + 2]),
                log_scale: Vec3::new(
                    log_scales[i * 3],
                    log_scales[i * 3 + 1],
                    log_scales[i * 3 + 2],
                ),
                opacity: opacities[i],
                rotation: Quat::from_xyzw(
                    rotations[i * 4 + 1],
                    rotations[i * 4 + 2],
                    rotations[i * 4 + 3],
                    rotations[i * 4],
                ),
                sh_dc,
                sh_coeffs_rest,
            };

            splat.is_finite().then_some(splat)
        })
        .collect()
}

pub async fn splat_to_ply<B: Backend>(splats: Splats<B>) -> std::io::Result<Vec<u8>> {
    let splats = splats.with_normed_rotations();

    let data = read_splat_data(splats.clone()).await;

    let property_names = vec![
        "x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2",
        "rot_3", "f_dc_0", "f_dc_1", "f_dc_2",
    ];

    let mut properties: Vec<PropertyDef> = property_names
        .into_iter()
        .map(|name| PropertyDef::new(name, PropertyType::Scalar(ScalarType::Float)))
        .collect();

    let sh_coeffs_rest = (splats.sh_coeffs.dims()[1] - 1) * 3;

    for i in 0..sh_coeffs_rest {
        properties.push(PropertyDef::new(
            &format!("f_rest_{i}"),
            PropertyType::Scalar(ScalarType::Float),
        ));
    }

    let mut ply: Ply<ParsedGaussian<false>> = Ply::new();

    // Create PLY header
    let mut vertex = ply::ElementDef::new("vertex");
    vertex.properties = properties;
    ply.header.elements.push(vertex);
    ply.header.encoding = ply::Encoding::BinaryLittleEndian;
    ply.header.comments.push("Exported from Brush".to_owned());
    ply.header.comments.push("Vertical axis: y".to_owned());
    ply.payload.insert("vertex".to_owned(), data);

    let mut buf = vec![];
    let writer = Writer::<ParsedGaussian<false>>::new();
    writer.write_ply(&mut buf, &mut ply)?;
    Ok(buf)
}
