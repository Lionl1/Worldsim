//! Quantum mechanics: Schrödinger equation, tunneling, hydrogen atom, white dwarf degeneracy.
//! Wave calculations are in atomic units (ħ = m = 1). This is a nonrelativistic single-particle model.

use crate::constants::{E_CHARGE, G, H_PLANCK, K_B, M_E, M_P, C};
use std::f64::consts::PI;

pub const HBAR: f64 = H_PLANCK / (2.0 * PI);
pub const M_SUN: f64 = 1.988_47e30;
const EPS0: f64 = 8.854_187_8128e-12;

/// Fine-structure constant.
pub fn alpha() -> f64 { E_CHARGE * E_CHARGE / (4.0 * PI * EPS0 * HBAR * C) }

/// Hydrogen energy level, Bohr model with reduced mass, eV.
pub fn hydrogen_level_ev(n: f64) -> f64 {
    let mu = M_E * M_P / (M_E + M_P);
    -0.5 * mu * (alpha() * C).powi(2) / (n * n) / E_CHARGE
}

/// Transition wavelength n2 → n1, nm (in vacuum).
pub fn hydrogen_line_nm(n1: f64, n2: f64) -> f64 {
    let de = (hydrogen_level_ev(n2) - hydrogen_level_ev(n1)) * E_CHARGE;
    H_PLANCK * C / de * 1e9
}

/// Eigenvalues of a tridiagonal matrix by bisection with a Sturm sequence: the k-th from the bottom.
fn sturm_eigenvalue(diag: &[f64], off: &[f64], k: usize) -> f64 {
    let count = |x: f64| {
        let (mut c, mut d) = (0, 1.0);
        for i in 0..diag.len() {
            let o = if i == 0 { 0.0 } else { off[i - 1] * off[i - 1] };
            d = diag[i] - x - o / if d == 0.0 { 1e-300 } else { d };
            if d < 0.0 { c += 1; }
        }
        c
    };
    let span = off.iter().fold(0.0f64, |a, b| a.max(b.abs()));
    let mut lo = diag.iter().cloned().fold(f64::MAX, f64::min) - 2.0 * span;
    let mut hi = diag.iter().cloned().fold(f64::MIN, f64::max) + 2.0 * span;
    for _ in 0..200 {
        let m = 0.5 * (lo + hi);
        if count(m) > k { hi = m } else { lo = m }
    }
    0.5 * (lo + hi)
}

/// k-th level in the potential V(x) on the grid [-l, l]; 4th-order differences are not needed, the step is small.
pub fn bound_level_1d(v: impl Fn(f64) -> f64, l: f64, n: usize, k: usize) -> f64 {
    let h = 2.0 * l / (n + 1) as f64;
    let diag: Vec<f64> = (1..=n).map(|i| 1.0 / (h * h) + v(-l + i as f64 * h)).collect();
    let off = vec![-0.5 / (h * h); n - 1];
    sturm_eigenvalue(&diag, &off, k)
}

/// Hydrogen s-state levels, atomic units (theory: −1/(2n²)). Uniform grid, u(r) = r·R(r).
pub fn hydrogen_s_level(k: usize, rmax: f64, n: usize) -> f64 {
    let h = rmax / (n + 1) as f64;
    let diag: Vec<f64> = (1..=n).map(|i| 1.0 / (h * h) - 1.0 / (i as f64 * h)).collect();
    let off = vec![-0.5 / (h * h); n - 1];
    sturm_eigenvalue(&diag, &off, k)
}

/// Wave packet on an interval: implicit Crank–Nicolson scheme. It is unitary, so the norm is conserved exactly.
#[derive(Clone)]
pub struct Wave {
    pub x: Vec<f64>, pub re: Vec<f64>, pub im: Vec<f64>, pub v: Vec<f64>, pub dx: f64, pub dt: f64, pub t: f64,
}

impl Wave {
    pub fn gaussian(xmin: f64, xmax: f64, n: usize, dt: f64, x0: f64, sigma: f64, k0: f64, v: impl Fn(f64) -> f64) -> Wave {
        let dx = (xmax - xmin) / (n - 1) as f64;
        let x: Vec<f64> = (0..n).map(|i| xmin + i as f64 * dx).collect();
        let a = (2.0 * PI * sigma * sigma).powf(-0.25);
        let env = |q: f64| a * (-(q - x0) * (q - x0) / (4.0 * sigma * sigma)).exp();
        Wave { re: x.iter().map(|q| env(*q) * (k0 * q).cos()).collect(), im: x.iter().map(|q| env(*q) * (k0 * q).sin()).collect(), v: x.iter().map(|q| v(*q)).collect(), x, dx, dt, t: 0.0 }
    }
    /// Step: (1 + iHdt/2)ψ⁺ = (1 − iHdt/2)ψ, Thomas algorithm for a complex tridiagonal matrix; at the edges ψ = 0.
    pub fn step(&mut self) {
        let n = self.x.len();
        let r = self.dt / (4.0 * self.dx * self.dx); // kin: -½∂² => coefficient i·dt/2·(−½/dx²)
        // b_i = 1 + i(dt/2)(1/dx² + V_i), a = c = −i r
        let (mut rhs_re, mut rhs_im) = (vec![0.0; n], vec![0.0; n]);
        for i in 1..n - 1 {
            let h = 2.0 * r + 0.5 * self.dt * self.v[i];
            let (lr, li) = (self.re[i - 1] + self.re[i + 1], self.im[i - 1] + self.im[i + 1]);
            // ψ − i(dt/2)Hψ, where Hψ·dt/2 = hψ − r(ψ₋ + ψ₊)
            rhs_re[i] = self.re[i] + h * self.im[i] - r * li;
            rhs_im[i] = self.im[i] - h * self.re[i] + r * lr;
        }
        let (mut cp_re, mut cp_im, mut dp_re, mut dp_im) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 1..n - 1 {
            let h = 2.0 * r + 0.5 * self.dt * self.v[i];
            // m = b − a·c'_{i−1}, a = −ir, b = 1 + ih
            let (cr, ci) = (cp_re[i - 1], cp_im[i - 1]);
            let (ac_re, ac_im) = (r * ci, -r * cr); // (−ir)(cr + i ci)
            let (mr, mi) = (1.0 - ac_re, h - ac_im);
            let den = mr * mr + mi * mi;
            // c' = c/m, c = −ir
            cp_re[i] = (-r * mi) / den;
            cp_im[i] = (-r * mr) / den;
            // d' = (d − a d'_{i−1})/m
            let (dr, di) = (dp_re[i - 1], dp_im[i - 1]);
            let (ad_re, ad_im) = (r * di, -r * dr);
            let (nr, ni) = (rhs_re[i] - ad_re, rhs_im[i] - ad_im);
            dp_re[i] = (nr * mr + ni * mi) / den;
            dp_im[i] = (ni * mr - nr * mi) / den;
        }
        let (mut nr, mut ni) = (vec![0.0; n], vec![0.0; n]);
        for i in (1..n - 1).rev() {
            let (xr, xi) = (nr[i + 1], ni[i + 1]);
            nr[i] = dp_re[i] - (cp_re[i] * xr - cp_im[i] * xi);
            ni[i] = dp_im[i] - (cp_re[i] * xi + cp_im[i] * xr);
        }
        self.re = nr;
        self.im = ni;
        self.t += self.dt;
    }
    pub fn density(&self) -> Vec<f64> { self.re.iter().zip(&self.im).map(|(a, b)| a * a + b * b).collect() }
    pub fn norm(&self) -> f64 { self.density().iter().sum::<f64>() * self.dx }
    pub fn mean_x(&self) -> f64 { self.density().iter().zip(&self.x).map(|(p, x)| p * x).sum::<f64>() * self.dx / self.norm() }
    pub fn sigma_x(&self) -> f64 { let m = self.mean_x(); (self.density().iter().zip(&self.x).map(|(p, x)| p * (x - m) * (x - m)).sum::<f64>() * self.dx / self.norm()).sqrt() }
    /// Δp through the wave function derivative.
    pub fn sigma_p(&self) -> f64 {
        let n = self.x.len();
        let (mut p1, mut p2) = (0.0, 0.0);
        for i in 1..n - 1 {
            let (dr, di) = ((self.re[i + 1] - self.re[i - 1]) / (2.0 * self.dx), (self.im[i + 1] - self.im[i - 1]) / (2.0 * self.dx));
            p1 += (self.re[i] * di - self.im[i] * dr) * self.dx;
            p2 += (dr * dr + di * di) * self.dx;
        }
        let norm = self.norm();
        (p2 / norm - (p1 / norm).powi(2)).sqrt()
    }
    pub fn right_of(&self, x0: f64) -> f64 { self.density().iter().zip(&self.x).filter(|(_, x)| **x > x0).map(|(p, _)| *p).sum::<f64>() * self.dx }
}

/// Exact transmission probability of a rectangular barrier (atomic units).
pub fn barrier_transmission(e: f64, v0: f64, a: f64) -> f64 {
    if (e - v0).abs() < 1e-12 { return 1.0 / (1.0 + e * a * a / 2.0); }
    if e < v0 {
        let k = (2.0 * (v0 - e)).sqrt();
        1.0 / (1.0 + v0 * v0 * (k * a).sinh().powi(2) / (4.0 * e * (v0 - e)))
    } else {
        let k = (2.0 * (e - v0)).sqrt();
        1.0 / (1.0 + v0 * v0 * (k * a).sin().powi(2) / (4.0 * e * (e - v0)))
    }
}

/// Gamow energy and Gamow peak for a reaction: tunneling through the Coulomb barrier, without which stars would not burn.
pub fn gamow(z1: f64, z2: f64, m1: f64, m2: f64, temp: f64) -> (f64, f64) {
    let mu = m1 * m2 / (m1 + m2);
    let eg = 2.0 * mu * C * C * (PI * alpha() * z1 * z2).powi(2);
    let kt = K_B * temp;
    (eg / E_CHARGE / 1e3, (eg * kt * kt / 4.0).powf(1.0 / 3.0) / E_CHARGE / 1e3)
}

/// Lane–Emden solution: first zero ξ₁ and −ξ²θ'(ξ₁).
pub fn lane_emden(n: f64) -> (f64, f64) {
    let f = |x: f64, y: [f64; 2]| -> [f64; 2] { [y[1], -y[0].max(0.0).powf(n) - 2.0 / x * y[1]] };
    let (mut x, mut y) = (1e-4, [1.0 - 1e-8 / 6.0, -1e-4 / 3.0]);
    let h = 1e-3;
    loop {
        let k1 = f(x, y);
        let k2 = f(x + h / 2.0, [y[0] + h / 2.0 * k1[0], y[1] + h / 2.0 * k1[1]]);
        let k3 = f(x + h / 2.0, [y[0] + h / 2.0 * k2[0], y[1] + h / 2.0 * k2[1]]);
        let k4 = f(x + h, [y[0] + h * k3[0], y[1] + h * k3[1]]);
        let ny = [y[0] + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]), y[1] + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])];
        if ny[0] <= 0.0 {
            let t = y[0] / (y[0] - ny[0]);
            let (xz, dz) = (x + t * h, y[1] + t * (ny[1] - y[1]));
            return (xz, -xz * xz * dz);
        }
        y = ny;
        x += h;
        if x > 100.0 { return (f64::NAN, f64::NAN); }
    }
}

/// Chandrasekhar limit, solar masses: degenerate electron pressure (ultrarelativistic limit, n = 3).
pub fn chandrasekhar_mass(mu_e: f64) -> f64 {
    let (_, w) = lane_emden(3.0);
    w * (3.0 * PI).sqrt() / 2.0 * (HBAR * C / G).powf(1.5) / (mu_e * M_P).powi(2) / M_SUN
}
