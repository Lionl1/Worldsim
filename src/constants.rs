//! Fundamental (CODATA 2018 / SI 2019) and astronomical (IAU 2015) constants.

pub const G: f64 = 6.674_30e-11; // m^3 kg^-1 s^-2
pub const C: f64 = 299_792_458.0; // m/s, exact
pub const H_PLANCK: f64 = 6.626_070_15e-34; // J·s, exact
pub const K_B: f64 = 1.380_649e-23; // J/K, exact
pub const N_A: f64 = 6.022_140_76e23; // 1/mol, exact
pub const R_GAS: f64 = K_B * N_A; // J/(mol·K)
pub const SIGMA_SB: f64 = 5.670_374_419e-8; // W/(m^2·K^4)
pub const E_CHARGE: f64 = 1.602_176_634e-19; // C, exact
pub const M_E: f64 = 9.109_383_701_5e-31; // kg
pub const M_P: f64 = 1.672_621_923_69e-27; // kg
pub const M_ALPHA: f64 = 6.644_657_335_7e-27; // kg
pub const MU_0: f64 = 1.256_637_062_12e-6; // N/A^2

pub const AU: f64 = 1.495_978_707e11; // m, exact
pub const L_SUN: f64 = 3.828e26; // W
pub const DAY: f64 = 86_400.0;
pub const YEAR: f64 = 365.25 * DAY;
pub const MEV: f64 = 1.0e6 * E_CHARGE;

/// Molar mass (kg/mol) and c_p (J/(mol·K)) of an ideal gas at ~300 K.
pub fn gas(name: &str) -> Option<(f64, f64)> {
    Some(match name {
        "N2" => (0.028_013_4, 29.12),
        "O2" => (0.031_998_8, 29.38),
        "Ar" => (0.039_948, 20.79),
        "CO2" => (0.044_009_5, 37.12),
        "H2O" => (0.018_015_28, 33.58),
        "CO" => (0.028_010_1, 29.14),
        "Ne" => (0.020_179_7, 20.79),
        "He" => (0.004_002_602, 20.79),
        "H2" => (0.002_015_88, 28.84),
        "CH4" => (0.016_042_5, 35.69),
        _ => return None,
    })
}
