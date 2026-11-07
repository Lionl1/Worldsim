//! Smoothed particle hydrodynamics: compressible gas as particles, with optional self-gravity.
//!
//! Cubic spline kernel, adaptive smoothing length, Monaghan artificial viscosity and a symmetric form that conserves
//! momentum and energy. Adiabatic gas: no cooling, chemistry or star formation, and a direct gravity sum,
//! so it suits thousands of particles on one machine, not cosmological runs.

use crate::vec3::Vec3;
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

/// Particles by cell, in index order. Cells are visited in the same order on every run, so sums do not depend on threads.
struct Grid { size: f64, origin: [i64; 3], dims: [usize; 3], start: Vec<u32>, items: Vec<u32> }

impl Grid {
    fn cell(p: Vec3, size: f64) -> [i64; 3] { [(p.x / size).floor() as i64, (p.y / size).floor() as i64, (p.z / size).floor() as i64] }
    fn new(x: &[Vec3], size: f64) -> Grid {
        let (mut lo, mut hi) = ([i64::MAX; 3], [i64::MIN; 3]);
        for p in x { let c = Self::cell(*p, size); for k in 0..3 { lo[k] = lo[k].min(c[k]); hi[k] = hi[k].max(c[k]); } }
        let dims = [(hi[0] - lo[0] + 1) as usize, (hi[1] - lo[1] + 1) as usize, (hi[2] - lo[2] + 1) as usize];
        let index = |c: [i64; 3]| ((c[0] - lo[0]) as usize * dims[1] + (c[1] - lo[1]) as usize) * dims[2] + (c[2] - lo[2]) as usize;
        let mut start = vec![0u32; dims[0] * dims[1] * dims[2] + 1];
        for p in x { start[index(Self::cell(*p, size)) + 1] += 1; }
        for k in 1..start.len() { start[k] += start[k - 1]; }
        let mut fill = start.clone();
        let mut items = vec![0u32; x.len()];
        for (i, p) in x.iter().enumerate() { let c = index(Self::cell(*p, size)); items[fill[c] as usize] = i as u32; fill[c] += 1; }
        Grid { size, origin: lo, dims, start, items }
    }
    /// Calls `f` with every particle in the cell of `p` and around it.
    #[inline]
    fn near(&self, p: Vec3, dim: usize, mut f: impl FnMut(usize)) {
        let c = Self::cell(p, self.size);
        let (ry, rz) = (if dim >= 2 { 1 } else { 0 }, if dim >= 3 { 1 } else { 0 });
        for a in c[0] - 1..=c[0] + 1 {
            let i = a - self.origin[0];
            if i < 0 || i >= self.dims[0] as i64 { continue; }
            for b in c[1] - ry..=c[1] + ry {
                let j = b - self.origin[1];
                if j < 0 || j >= self.dims[1] as i64 { continue; }
                for d in c[2] - rz..=c[2] + rz {
                    let k = d - self.origin[2];
                    if k < 0 || k >= self.dims[2] as i64 { continue; }
                    let at = (i as usize * self.dims[1] + j as usize) * self.dims[2] + k as usize;
                    for n in &self.items[self.start[at] as usize..self.start[at + 1] as usize] { f(*n as usize); }
                }
            }
        }
    }
}

/// Fills `out` in parallel, one value per particle. Each value depends only on its index.
fn par_fill<T: Send>(out: &mut [T], f: impl Fn(usize) -> T + Sync) {
    let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
    let per = out.len().div_ceil(threads).max(1);
    let f = &f;
    std::thread::scope(|sc| {
        for (c, block) in out.chunks_mut(per).enumerate() {
            sc.spawn(move || { for (k, o) in block.iter_mut().enumerate() { *o = f(c * per + k); } });
        }
    });
}

impl Sph {
    /// Density by kernel summation, with the smoothing length following it: h = 1.2·(m/ρ)^(1/dim).
    pub fn density(&mut self) {
        let d = self.dim as f64;
        for _ in 0..3 {
            let size = 2.0 * self.h.iter().cloned().fold(0.0, f64::max);
            let g = Grid::new(&self.x, size);
            let mut rho = vec![0.0; self.x.len()];
            let me = &*self;
            par_fill(&mut rho, |i| {
                let (mut sum, xi, hi) = (0.0, me.x[i], me.h[i]);
                g.near(xi, me.dim, |j| {
                    let q = (xi - me.x[j]).norm() / hi;
                    if q < 2.0 { sum += me.m[j] * w(q, hi, me.dim); }
                });
                sum
            });
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
        let size = 2.0 * self.h.iter().cloned().fold(0.0, f64::max);
        let g = Grid::new(&self.x, size);
        let me = &*self;
        let mut rates = vec![(Vec3::ZERO, 0.0); n];
        par_fill(&mut rates, |i| {
            let (xi, vi, hi, pi, ci, ri) = (me.x[i], me.v[i], me.h[i], me.pressure(i), me.sound(i), me.rho[i]);
            let (mut a, mut e) = (Vec3::ZERO, 0.0);
            g.near(xi, me.dim, |j| {
                if j == i { return; }
                let d = xi - me.x[j];
                let r = d.norm();
                let hj = me.h[j];
                if r == 0.0 || (r >= 2.0 * hi && r >= 2.0 * hj) { return; }
                let grad = 0.5 * (dw(r / hi, hi, me.dim) + dw(r / hj, hj, me.dim)) / r;
                let dv = vi - me.v[j];
                let vr = dv.dot(d);
                let visc = if vr < 0.0 {
                    let hb = 0.5 * (hi + hj);
                    let mu = hb * vr / (r * r + 0.01 * hb * hb);
                    (-1.0 * 0.5 * (ci + me.sound(j)) * mu + 2.0 * mu * mu) / (0.5 * (ri + me.rho[j]))
                } else { 0.0 };
                let term = pi / (ri * ri) + me.pressure(j) / (me.rho[j] * me.rho[j]) + visc;
                a -= d * (me.m[j] * term * grad);
                e += 0.5 * me.m[j] * term * grad * vr;
            });
            (a, e)
        });
        let (mut acc, heat): (Vec<Vec3>, Vec<f64>) = rates.into_iter().unzip();
        if let Some(eps) = self.gravity {
            // each particle sums over the others in index order, the same order the pair loop would add them in
            let mut pull = vec![Vec3::ZERO; n];
            let me = &*self;
            par_fill(&mut pull, |i| {
                let mut a = acc[i];
                for j in 0..n {
                    if j == i { continue; }
                    // d runs from the lower index to the higher one, as in the pair form
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    let d = me.x[hi] - me.x[lo];
                    let s = (d.norm2() + eps * eps).powf(-1.5);
                    if i < j { a += d * (me.m[j] * s); } else { a -= d * (me.m[j] * s); }
                }
                a
            });
            acc = pull;
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
