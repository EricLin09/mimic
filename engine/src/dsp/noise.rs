//! Noise keyed to absolute time: the same frame / sample always gets the same value, so two
//! renders that overlap in time feed the synthesizer identical randomness there.

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Standard normal value for (`stream`, `index`) via Box-Muller.
pub fn gaussian(stream: u64, index: u64) -> f32 {
    let h1 = splitmix64(index ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03));
    let h2 = splitmix64(h1);
    let scale = 1.0 / (1u64 << 53) as f64;
    let u1 = ((h1 >> 11) as f64 + 0.5) * scale;
    let u2 = (h2 >> 11) as f64 * scale;
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

pub const LATENT_STREAM: u64 = 1;
pub const SOURCE_STREAM: u64 = 2;
