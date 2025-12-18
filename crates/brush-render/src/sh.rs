
use glam::Vec3;

pub const fn sh_coeffs_for_degree(degree: u32) -> u32 {
    (degree + 1).pow(2)
}

pub fn sh_degree_from_coeffs(coeffs_per_channel: u32) -> u32 {
    match coeffs_per_channel {
        1 => 0,
        4 => 1,
        9 => 2,
        16 => 3,
        25 => 4,
        _ => panic!("Invalid nr. of sh bases {coeffs_per_channel}"),
    }
}

pub fn rgb_to_sh(rgb: Vec3) -> Vec3 {
    rgb
}
