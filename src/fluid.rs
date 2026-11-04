//! Fluid dynamics: 1D Euler and shallow water equations with the finite volume method (HLL).
//! These are reference problems for checking the solver; there is no 3D turbulence or global circulation here.

pub const GAMMA: f64 = 1.4;

/// Riemann problem (Sod): numerical solution from the conservation law, n cells, time t.
pub struct Euler1D { pub dx: f64, pub rho: Vec<f64>, pub mom: Vec<f64>, pub en: Vec<f64>, pub t: f64 }

impl Euler1D {
    pub fn sod(n: usize) -> Euler1D {
        let dx = 1.0 / n as f64;
        let (mut rho, mom, mut en) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let left = (i as f64 + 0.5) * dx < 0.5;
            rho[i] = if left { 1.0 } else { 0.125 };
            en[i] = (if left { 1.0 } else { 0.1 }) / (GAMMA - 1.0);
        }
        Euler1D { dx, rho, mom, en, t: 0.0 }
    }
    fn flux(r: f64, m: f64, e: f64) -> ([f64; 3], f64, f64) {
        let u = m / r;
        let p = (GAMMA - 1.0) * (e - 0.5 * m * u);
        ([m, m * u + p, u * (e + p)], u, (GAMMA * p / r).sqrt())
    }
    pub fn step(&mut self, cfl: f64) -> f64 {
        let n = self.rho.len();
        let mut smax = 1e-12f64;
        for i in 0..n { let (_, u, c) = Self::flux(self.rho[i], self.mom[i], self.en[i]); smax = smax.max(u.abs() + c); }
        let dt = cfl * self.dx / smax;
        let mut f = vec![[0.0; 3]; n + 1];
        for i in 1..n {
            let (fl, ul, cl) = Self::flux(self.rho[i - 1], self.mom[i - 1], self.en[i - 1]);
            let (fr, ur, cr) = Self::flux(self.rho[i], self.mom[i], self.en[i]);
            let (sl, sr) = ((ul - cl).min(ur - cr), (ul + cl).max(ur + cr));
            let ql = [self.rho[i - 1], self.mom[i - 1], self.en[i - 1]];
            let qr = [self.rho[i], self.mom[i], self.en[i]];
            for k in 0..3 {
                f[i][k] = if sl >= 0.0 { fl[k] } else if sr <= 0.0 { fr[k] } else { (sr * fl[k] - sl * fr[k] + sl * sr * (qr[k] - ql[k])) / (sr - sl) };
            }
        }
        // walls: impermeable, the momentum flux is the wall pressure
        let wall = |r: f64, m: f64, e: f64| (GAMMA - 1.0) * (e - 0.5 * m * m / r);
        f[0][1] = wall(self.rho[0], self.mom[0], self.en[0]);
        f[n][1] = wall(self.rho[n - 1], self.mom[n - 1], self.en[n - 1]);
        let k = dt / self.dx;
        for i in 0..n {
            self.rho[i] -= k * (f[i + 1][0] - f[i][0]);
            self.mom[i] -= k * (f[i + 1][1] - f[i][1]);
            self.en[i] -= k * (f[i + 1][2] - f[i][2]);
        }
        self.t += dt;
        dt
    }
    pub fn run_to(&mut self, t: f64) { while self.t < t { self.step(0.4); } }
    pub fn pressure(&self) -> Vec<f64> { (0..self.rho.len()).map(|i| (GAMMA - 1.0) * (self.en[i] - 0.5 * self.mom[i] * self.mom[i] / self.rho[i])).collect() }
    pub fn velocity(&self) -> Vec<f64> { (0..self.rho.len()).map(|i| self.mom[i] / self.rho[i]).collect() }
}

/// Exact solution of the Sod problem at point x, time t: (ρ, u, p).
pub fn sod_exact(x: f64, t: f64) -> (f64, f64, f64) {
    let g = GAMMA;
    let (rl, pl, rr, pr) = (1.0, 1.0, 0.125, 0.1);
    let (cl, cr) = ((g * pl / rl).sqrt(), (g * pr / rr).sqrt());
    let f = |p: f64| -> f64 {
        let fl = 2.0 * cl / (g - 1.0) * ((p / pl).powf((g - 1.0) / (2.0 * g)) - 1.0);
        let (ar, br) = (2.0 / ((g + 1.0) * rr), (g - 1.0) / (g + 1.0) * pr);
        fl + (p - pr) * (ar / (p + br)).sqrt()
    };
    let (mut lo, mut hi) = (1e-6, 1.0);
    for _ in 0..100 { let m = 0.5 * (lo + hi); if f(m) > 0.0 { hi = m } else { lo = m } }
    let ps = 0.5 * (lo + hi);
    let fl = 2.0 * cl / (g - 1.0) * ((ps / pl).powf((g - 1.0) / (2.0 * g)) - 1.0);
    let us = -fl;
    let rsl = rl * (ps / pl).powf(1.0 / g);
    let csl = (g * ps / rsl).sqrt();
    let rsr = rr * ((ps / pr + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * ps / pr + 1.0));
    let ss = cr * ((g + 1.0) / (2.0 * g) * ps / pr + (g - 1.0) / (2.0 * g)).sqrt();
    let s = (x - 0.5) / t;
    if s < -cl { (rl, 0.0, pl) }
    else if s < us - csl {
        let u = 2.0 / (g + 1.0) * (cl + s);
        let c = cl - (g - 1.0) / 2.0 * u;
        let r = rl * (c / cl).powf(2.0 / (g - 1.0));
        (r, u, pl * (c / cl).powf(2.0 * g / (g - 1.0)))
    }
    else if s < us { (rsl, us, ps) }
    else if s < ss { (rsr, us, ps) }
    else { (rr, 0.0, pr) }
}

/// Shallow water: dam break over a dry bed. Exact solution: Ritter (1892).
pub struct Shallow1D { pub dx: f64, pub g: f64, pub h: Vec<f64>, pub q: Vec<f64>, pub t: f64 }

impl Shallow1D {
    pub fn dam(n: usize, l: f64, g: f64, h0: f64) -> Shallow1D {
        let dx = l / n as f64;
        Shallow1D { dx, g, h: (0..n).map(|i| if (i as f64 + 0.5) * dx < l / 2.0 { h0 } else { 1e-9 }).collect(), q: vec![0.0; n], t: 0.0 }
    }
    pub fn step(&mut self, cfl: f64) {
        let n = self.h.len();
        let mut smax = 1e-9f64;
        for i in 0..n { smax = smax.max((self.q[i] / self.h[i]).abs() + (self.g * self.h[i]).sqrt()); }
        let dt = cfl * self.dx / smax;
        let flux = |h: f64, q: f64, g: f64| [q, q * q / h + 0.5 * g * h * h];
        let mut f = vec![[0.0; 2]; n + 1];
        for i in 1..n {
            let (hl, ql, hr, qr) = (self.h[i - 1], self.q[i - 1], self.h[i], self.q[i]);
            let (ul, ur, cl, cr) = (ql / hl, qr / hr, (self.g * hl).sqrt(), (self.g * hr).sqrt());
            let (sl, sr) = ((ul - cl).min(ur - cr), (ul + cl).max(ur + cr));
            let (fl, fr) = (flux(hl, ql, self.g), flux(hr, qr, self.g));
            let (ql2, qr2) = ([hl, ql], [hr, qr]);
            for k in 0..2 { f[i][k] = if sl >= 0.0 { fl[k] } else if sr <= 0.0 { fr[k] } else { (sr * fl[k] - sl * fr[k] + sl * sr * (qr2[k] - ql2[k])) / (sr - sl) }; }
        }
        f[0][1] = 0.5 * self.g * self.h[0] * self.h[0];
        f[n][1] = 0.5 * self.g * self.h[n - 1] * self.h[n - 1];
        let k = dt / self.dx;
        for i in 0..n {
            self.h[i] = (self.h[i] - k * (f[i + 1][0] - f[i][0])).max(1e-9);
            self.q[i] -= k * (f[i + 1][1] - f[i][1]);
        }
        self.t += dt;
    }
    pub fn run_to(&mut self, t: f64) { while self.t < t { self.step(0.4); } }
}

/// Exact Ritter depth at point x from the dam (x0), time t.
pub fn ritter_depth(x: f64, x0: f64, t: f64, g: f64, h0: f64) -> f64 {
    let c0 = (g * h0).sqrt();
    let xi = (x - x0) / t;
    if xi <= -c0 { h0 } else if xi >= 2.0 * c0 { 0.0 } else { (2.0 * c0 - xi).powi(2) / (9.0 * g) }
}
