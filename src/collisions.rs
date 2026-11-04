//! Collisions: merging with momentum conservation, body disruption (Stewart and Leinhardt), crater scaling (Collins et al.), Roche limit.
//! These are scaling laws, not a strength calculation for a specific body.

use crate::constants::G;
use crate::vec3::Vec3;
use std::f64::consts::PI;

/// Perfectly inelastic merger: mass, velocity and energy lost to heat.
pub fn merge(m1: f64, v1: Vec3, m2: f64, v2: Vec3) -> (f64, Vec3, f64) {
    let m = m1 + m2;
    let v = (v1 * m1 + v2 * m2) * (1.0 / m);
    let ke = |mm: f64, u: Vec3| 0.5 * mm * u.dot(u);
    (m, v, ke(m1, v1) + ke(m2, v2) - ke(m, v))
}

/// Specific energy of catastrophic disruption Q*_RD, J/kg: half the mass disperses. Gravity-regime laws.
/// `c_star` ≈ 1.9 for uniform bodies after Stewart and Leinhardt (2009).
pub fn disruption_energy(total_mass: f64, density: f64, c_star: f64) -> f64 {
    let r = (3.0 * total_mass / (4.0 * PI * density)).cbrt();
    c_star * 4.0 / 5.0 * PI * density * G * r * r
}

/// Mass fraction in the largest remnant (universal law); Q is the specific collision energy.
pub fn largest_remnant_fraction(q: f64, q_star: f64) -> f64 {
    let x = q / q_star;
    if x < 1.8 { (1.0 - 0.5 * x).clamp(0.0, 1.0) } else { (0.1 * (x / 1.8).powf(-1.5)).clamp(0.0, 1.0) }
}

/// Specific collision energy Q_R, J/kg: kinetic energy in the center-of-mass frame per unit mass.
pub fn impact_energy(m1: f64, m2: f64, v: f64) -> f64 { 0.5 * (m1 * m2 / (m1 + m2)) * v * v / (m1 + m2) }

/// Collision outcome by energy: merger, partial disruption or catastrophe.
pub fn outcome(q: f64, q_star: f64) -> &'static str {
    let x = q / q_star;
    if x < 0.1 { "merger" } else if x < 1.0 { "partial disruption" } else if x < 1.8 { "catastrophic disruption" } else { "supercatastrophic disruption" }
}

/// Crater diameter, m: estimate after Collins, Melosh and Marcus (2005). Transient crater, simple ×1.25, complex with their correction.
pub fn crater_diameter(rho_imp: f64, rho_tgt: f64, d_imp: f64, v: f64, g: f64, angle_deg: f64) -> (f64, f64) {
    let tc = 1.161 * (rho_imp / rho_tgt).cbrt() * d_imp.powf(0.78) * v.powf(0.44) * g.powf(-0.22) * angle_deg.to_radians().sin().cbrt();
    let simple = 1.25 * tc;
    let dc = 3200.0 * 9.81 / g; // the simple-to-complex transition scales as 1/g
    let fin = if simple < dc { simple } else { 1.17 * tc.powf(1.13) / dc.powf(0.13) };
    (tc, fin)
}

/// Roche limit for a fluid satellite, m.
pub fn roche_limit(primary_radius: f64, rho_primary: f64, rho_satellite: f64) -> f64 { 2.44 * primary_radius * (rho_primary / rho_satellite).cbrt() }
