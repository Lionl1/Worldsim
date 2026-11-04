//! Tectonics: oceanic lithosphere cooling and seafloor depth, heat flow, mantle convection.
//! These are scaling laws, not a calculation of plate motion.

use std::f64::consts::PI;

/// Lithosphere parameters (Turcotte and Schubert).
#[derive(Clone, Copy)]
pub struct Lithosphere { pub k: f64, pub kappa: f64, pub dt: f64, pub alpha: f64, pub rho_m: f64, pub rho_w: f64 }

impl Lithosphere {
    pub fn earth() -> Lithosphere { Lithosphere { k: 3.3, kappa: 1e-6, dt: 1300.0, alpha: 3e-5, rho_m: 3300.0, rho_w: 1000.0 } }
    /// Surface heat flow, W/m²; age is the age, s.
    pub fn heat_flow(&self, age: f64) -> f64 { self.k * self.dt / (PI * self.kappa * age).sqrt() }
    /// Seafloor subsidence relative to the ridge, m: thermal contraction and isostasy.
    pub fn subsidence(&self, age: f64) -> f64 { 2.0 * self.rho_m * self.alpha * self.dt / (self.rho_m - self.rho_w) * (self.kappa * age / PI).sqrt() }
    /// Temperature at depth z at the given age, numerically: explicit heat conduction scheme, surface 0, depth ΔT.
    pub fn temperature_numeric(&self, z: f64, age: f64) -> f64 {
        let (l, n) = (8.0 * (self.kappa * age).sqrt(), 200usize);
        let dz = l / n as f64;
        let dt = 0.25 * dz * dz / self.kappa;
        let steps = (age / dt).ceil() as usize;
        let dt = age / steps as f64;
        let mut t = vec![self.dt; n + 1];
        t[0] = 0.0;
        for _ in 0..steps {
            let old = t.clone();
            for i in 1..n { t[i] = old[i] + self.kappa * dt / (dz * dz) * (old[i + 1] - 2.0 * old[i] + old[i - 1]); }
        }
        let f = z / dz;
        let i = (f as usize).min(n - 1);
        t[i] + (t[i + 1] - t[i]) * (f - i as f64)
    }
    /// Analytic: T = ΔT·erf(z / 2√(κt)).
    pub fn temperature_exact(&self, z: f64, age: f64) -> f64 { self.dt * erf(z / (2.0 * (self.kappa * age).sqrt())) }
}

/// Error function (Abramowitz and Stegun, series of sufficient accuracy).
pub fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let e = t * (-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))).exp();
    if x >= 0.0 { 1.0 - e } else { e - 1.0 }
}

/// Rayleigh number of mantle convection.
pub fn rayleigh(rho: f64, g: f64, alpha: f64, dt: f64, d: f64, kappa: f64, viscosity: f64) -> f64 { rho * g * alpha * dt * d.powi(3) / (kappa * viscosity) }

/// Convection speed from boundary layer theory: v ≈ 0.354·(κ/d)·Ra^(2/3). Order of plate speeds.
pub fn convection_speed(kappa: f64, d: f64, ra: f64) -> f64 { 0.354 * kappa / d * ra.powf(2.0 / 3.0) }
