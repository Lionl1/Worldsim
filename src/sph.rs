//! Smoothed particle hydrodynamics: compressible gas as particles, with optional self-gravity.
//!
//! Cubic spline kernel, adaptive smoothing length, Monaghan artificial viscosity and a symmetric form that conserves
//! momentum and energy. Adiabatic gas: no cooling, chemistry or star formation, and a direct gravity sum,
//! so it suits thousands of particles on one machine, not cosmological runs.

use crate::vec3::Vec3;
use std::collections::HashMap;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Sph {
    pub dim: usize, pub gamma: f64, pub t: f64,
    pub x: Vec<Vec3>, pub v: Vec<Vec3>, pub u: Vec<f64>, pub m: Vec<f64>, pub h: Vec<f64>, pub rho: Vec<f64>,
    /// Softening of self-gravity with G = 1; None means no gravity.
    pub gravity: Option<f64>,
}

fn kernel_norm(dim: usize) -> f64 { match dim { 1 => 2.0 / 3.0, 2 => 10.0 / (7.0 * PI), _ => 1.0 / PI } }

fn w(q: f64, h: f64, dim: usize) -> f64 {
    let s = if q < 1.0 { 1.0 - 1.5 * q * q + 0.75 * q * q * q } else if q < 2.0 { 0.25 * (2.0 - q).powi(3) } else { 0.0 };
    kernel_norm(dim) * s / h.powi(dim as i32)
}

/// dW/dr
fn dw(q: f64, h: f64, dim: usize) -> f64 {
    let s = if q < 1.0 { -3.0 * q + 2.25 * q * q } else if q < 2.0 { -0.75 * (2.0 - q).powi(2) } else { 0.0 };
    kernel_norm(dim) * s / h.powi(dim as i32 + 1)
}

type Cell = (i64, i64, i64);

impl Sph {
    fn grid(&self, size: f64) -> HashMap<Cell, Vec<u32>> {
        let mut g: HashMap<Cell, Vec<u32>> = HashMap::new();
        for (i, p) in self.x.iter().enumerate() { g.entry(Self::cell(*p, size)).or_default().push(i as u32); }
        g
    }
    fn cell(p: Vec3, size: f64) -> Cell { ((p.x / size).floor() as i64, (p.y / size).floor() as i64, (p.z / size).floor() as i64) }
    fn near(&self, g: &HashMap<Cell, Vec<u32>>, p: Vec3, size: f64, mut f: impl FnMut(usize)) {
        let (cx, cy, cz) = Self::cell(p, size);
        let (ry, rz) = (if self.dim >= 2 { 1 } else { 0 }, if self.dim >= 3 { 1 } else { 0 });
        for a in cx - 1..=cx + 1 { for b in cy - ry..=cy + ry { for c in cz - rz..=cz + rz {
            if let Some(list) = g.get(&(a, b, c)) { for j in list { f(*j as usize); } }
        }}}
    }

    /// Density by kernel summation, with the smoothing length following it: h = 1.2·(m/ρ)^(1/dim).
    pub fn density(&mut self) {
        let d = self.dim as f64;
        for _ in 0..3 {
            let size = 2.0 * self.h.iter().cloned().fold(0.0, f64::max);
            let g = self.grid(size);
            let mut rho = vec![0.0; self.x.len()];
            for i in 0..self.x.len() {
                let (mut sum, xi, hi) = (0.0, self.x[i], self.h[i]);
                self.near(&g, xi, size, |j| {
                    let q = (xi - self.x[j]).norm() / hi;
                    if q < 2.0 { sum += self.m[j] * w(q, hi, self.dim); }
                });
                rho[i] = sum;
            }
            for i in 0..self.x.len() { self.h[i] = 1.2 * (self.m[i] / rho[i]).powf(1.0 / d); }
            self.rho = rho;
        }
    }

    fn pressure(&self, i: usize) -> f64 { (self.gamma - 1.0) * self.rho[i] * self.u[i].max(0.0) }
    fn sound(&self, i: usize) -> f64 { (self.gamma * (self.gamma - 1.0) * self.u[i].max(0.0)).sqrt() }

    /// Accelerations and heating. The pair form is symmetric, so momentum and total energy are conserved by construction.
    fn rates(&mut self) -> (Vec<Vec3>, Vec<f64>) {
        self.density();
        let n = self.x.len();
        let (mut acc, mut heat) = (vec![Vec3::ZERO; n], vec![0.0; n]);
        let size = 2.0 * self.h.iter().cloned().fold(0.0, f64::max);
        let g = self.grid(size);
        for i in 0..n {
            let (xi, vi, hi, pi, ci, ri) = (self.x[i], self.v[i], self.h[i], self.pressure(i), self.sound(i), self.rho[i]);
            let (mut a, mut e) = (Vec3::ZERO, 0.0);
            self.near(&g, xi, size, |j| {
                if j == i { return; }
                let d = xi - self.x[j];
                let r = d.norm();
                let hj = self.h[j];
                if r == 0.0 || (r >= 2.0 * hi && r >= 2.0 * hj) { return; }
                let grad = 0.5 * (dw(r / hi, hi, self.dim) + dw(r / hj, hj, self.dim)) / r;
                let dv = vi - self.v[j];
                let vr = dv.dot(d);
                let visc = if vr < 0.0 {
                    let hb = 0.5 * (hi + hj);
                    let mu = hb * vr / (r * r + 0.01 * hb * hb);
                    (-1.0 * 0.5 * (ci + self.sound(j)) * mu + 2.0 * mu * mu) / (0.5 * (ri + self.rho[j]))
                } else { 0.0 };
                let term = pi / (ri * ri) + self.pressure(j) / (self.rho[j] * self.rho[j]) + visc;
                a -= d * (self.m[j] * term * grad);
                e += 0.5 * self.m[j] * term * grad * vr;
            });
            acc[i] = a;
            heat[i] = e;
        }
        if let Some(eps) = self.gravity {
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = self.x[j] - self.x[i];
                    let s = (d.norm2() + eps * eps).powf(-1.5);
                    acc[i] += d * (self.m[j] * s);
                    acc[j] -= d * (self.m[i] * s);
                }
            }
        }
        (acc, heat)
    }

    /// Kick–drift–kick step; the step follows the sound speed and the fastest acceleration.
    pub fn step(&mut self) -> f64 {
        let (a0, e0) = self.rates();
        let n = self.x.len();
        let mut dt = f64::MAX;
        for i in 0..n {
            dt = dt.min(0.25 * self.h[i] / (self.sound(i) + self.v[i].norm().max(1e-12)));
            let an = a0[i].norm();
            if an > 0.0 { dt = dt.min(0.25 * (self.h[i] / an).sqrt()); }
        }
        for i in 0..n {
            self.v[i] += a0[i] * (0.5 * dt);
            self.u[i] = (self.u[i] + 0.5 * dt * e0[i]).max(0.0);
            self.x[i] += self.v[i] * dt;
        }
        let (a1, e1) = self.rates();
        for i in 0..n {
            self.v[i] += a1[i] * (0.5 * dt);
            self.u[i] = (self.u[i] + 0.5 * dt * e1[i]).max(0.0);
        }
        self.t += dt;
        dt
    }

    pub fn run_to(&mut self, t: f64) { while self.t < t { self.step(); } }

    pub fn momentum(&self) -> Vec3 { self.v.iter().zip(&self.m).fold(Vec3::ZERO, |s, (v, m)| s + *v * *m) }
    pub fn kinetic(&self) -> f64 { self.v.iter().zip(&self.m).map(|(v, m)| 0.5 * m * v.norm2()).sum() }
    pub fn thermal(&self) -> f64 { self.u.iter().zip(&self.m).map(|(u, m)| u * m).sum() }
    pub fn potential(&self) -> f64 {
        let Some(eps) = self.gravity else { return 0.0 };
        let mut e = 0.0;
        for i in 0..self.x.len() { for j in (i + 1)..self.x.len() { e -= self.m[i] * self.m[j] / ((self.x[i] - self.x[j]).norm2() + eps * eps).sqrt(); } }
        e
    }
    pub fn energy(&self) -> f64 { self.kinetic() + self.thermal() + self.potential() }

    /// The Sod tube as particles: the standard left and right states, in a tube long enough that the ends do not matter.
    pub fn sod(n_left: usize) -> Sph {
        let dx = 1.0 / n_left as f64;
        let (mut x, mut m) = (vec![], vec![]);
        let mass = dx;
        let mut p = -0.5 + 0.5 * dx;
        while p < 0.5 { x.push(p); p += dx; }
        let mut p = 0.5 + 4.0 * dx;
        while p < 1.5 { x.push(p); p += 8.0 * dx; }
        m.resize(x.len(), mass);
        let u: Vec<f64> = x.iter().map(|p| if *p < 0.5 { 1.0 / (0.4 * 1.0) } else { 0.1 / (0.4 * 0.125) }).collect();
        let h: Vec<f64> = x.iter().map(|p| if *p < 0.5 { 1.2 * dx } else { 9.6 * dx }).collect();
        let n = x.len();
        Sph { dim: 1, gamma: 1.4, t: 0.0, x: x.into_iter().map(|p| Vec3::new(p, 0.0, 0.0)).collect(), v: vec![Vec3::ZERO; n], u, m, h, rho: vec![1.0; n], gravity: None }
    }

    /// Evrard collapse: cold gas with ρ ∝ 1/r, G = M = R = 1, u = 0.05. Standard test of gas with gravity.
    pub fn evrard(n: usize, seed: u64) -> Sph {
        let mut rng = crate::rng::Rng::new(seed);
        let x: Vec<Vec3> = (0..n).map(|_| {
            let r = rng.f64().sqrt();
            let z = rng.uniform(-1.0, 1.0);
            let ph = rng.uniform(0.0, 2.0 * PI);
            let s = (1.0 - z * z).sqrt();
            Vec3::new(r * s * ph.cos(), r * s * ph.sin(), r * z)
        }).collect();
        let h0 = 0.25 * (1.0 / n as f64).powf(1.0 / 3.0) * 4.0;
        Sph { dim: 3, gamma: 5.0 / 3.0, t: 0.0, x, v: vec![Vec3::ZERO; n], u: vec![0.05; n], m: vec![1.0 / n as f64; n], h: vec![h0; n], rho: vec![1.0; n], gravity: Some(0.02) }
    }
}
