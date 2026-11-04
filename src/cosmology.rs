//! Cosmology ΛCDM.
//!  * Friedmann equation H(a) = H₀ √(Ω_r a⁻⁴ + Ω_m a⁻³ + Ω_k a⁻² + Ω_Λ): age, distances, redshift.
//!  * Linear perturbation growth D(a) (Heath integral for matter + Λ).
//!  * Power spectrum: P(k) ∝ kⁿˢ T²(k), BBKS transfer function with Sugiyama shape parameter, normalization σ₈.
//!  * Zeldovich initial conditions, particle-mesh simulation (cloud-in-cell + Poisson solve via FFT).
//! PM units: length in Mpc/h (comoving), time in 1/H₀, φ: ∇²φ = (3/2) Ω_m δ / a.

use crate::rng::Rng;
use std::f64::consts::PI;

pub const C_KM_S: f64 = 299_792.458;
pub const GYR_S: f64 = 3.155_76e16;
pub const MPC_KM: f64 = 3.085_677_581e19;

#[derive(Clone, Copy, Debug)]
pub struct Cosmology {
    pub h: f64,        // H₀ / 100 km/s/Mpc
    pub omega_m: f64,
    pub omega_b: f64,
    pub omega_l: f64,
    pub omega_r: f64,
    pub n_s: f64,
    pub sigma8: f64,
}

pub const PRESETS: &[&str] = &["planck2018", "eds", "open", "lambda_heavy"];

impl Cosmology {
    /// Planck 2018 (TT,TE,EE+lowE+lensing+BAO).
    pub fn planck2018() -> Self {
        Cosmology { h: 0.6766, omega_m: 0.3111, omega_b: 0.0490, omega_l: 0.6889 - 9.2e-5, omega_r: 9.2e-5, n_s: 0.9665, sigma8: 0.8102 }
    }
    pub fn preset(name: &str) -> Result<Self, String> {
        let p = Cosmology::planck2018();
        Ok(match name {
            "planck2018" => p,
            "eds" => Cosmology { omega_m: 1.0, omega_l: 0.0, omega_r: 0.0, ..p },             // Einstein–de Sitter
            "open" => Cosmology { omega_m: 0.3, omega_l: 0.0, omega_r: 0.0, ..p },            // open without Λ
            "lambda_heavy" => Cosmology { omega_m: 0.15, omega_l: 0.85, omega_r: 0.0, ..p },
            _ => return Err(format!("unknown cosmology '{name}', available: {PRESETS:?}")),
        })
    }
    pub fn omega_k(&self) -> f64 { 1.0 - self.omega_m - self.omega_l - self.omega_r }
    pub fn h0_si(&self) -> f64 { 100.0 * self.h / MPC_KM } // 1/s

    /// E(a) = H(a)/H₀
    pub fn e(&self, a: f64) -> f64 {
        (self.omega_r / a.powi(4) + self.omega_m / a.powi(3) + self.omega_k() / (a * a) + self.omega_l).sqrt()
    }

    fn simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
        let n = n + n % 2;
        let h = (b - a) / n as f64;
        let mut s = f(a) + f(b);
        for i in 1..n { s += f(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 }; }
        s * h / 3.0
    }

    /// Cosmic time t(a) in units of 1/H₀ (substitution a = u² removes the singularity at zero).
    pub fn time_h0(&self, a: f64) -> f64 {
        Self::simpson(|u: f64| { if u == 0.0 { return 0.0; } let a = u * u; 2.0 * u / (a * self.e(a)) }, 0.0, a.sqrt(), 4000)
    }
    pub fn age_gyr(&self, a: f64) -> f64 { self.time_h0(a) / self.h0_si() / GYR_S }

    /// Comoving distance to z, Mpc.
    pub fn comoving_distance(&self, z: f64) -> f64 {
        C_KM_S / (100.0 * self.h) * Self::simpson(|zz| 1.0 / self.e(1.0 / (1.0 + zz)), 0.0, z, 2000)
    }
    pub fn transverse_distance(&self, z: f64) -> f64 {
        let dc = self.comoving_distance(z);
        let ok = self.omega_k();
        let dh = C_KM_S / (100.0 * self.h);
        if ok.abs() < 1e-8 { dc } else if ok > 0.0 { dh / ok.sqrt() * (ok.sqrt() * dc / dh).sinh() } else { dh / (-ok).sqrt() * ((-ok).sqrt() * dc / dh).sin() }
    }
    pub fn luminosity_distance(&self, z: f64) -> f64 { (1.0 + z) * self.transverse_distance(z) }
    pub fn angular_diameter_distance(&self, z: f64) -> f64 { self.transverse_distance(z) / (1.0 + z) }

    /// Linear growth factor D(a), normalized D(1)=1 (without radiation).
    pub fn growth(&self, a: f64) -> f64 {
        let g = |a: f64| -> f64 {
            let e = |x: f64| (self.omega_m / x.powi(3) + self.omega_k() / (x * x) + self.omega_l).sqrt();
            2.5 * self.omega_m * e(a) * Self::simpson(|x| if x == 0.0 { 0.0 } else { 1.0 / (x * e(x)).powi(3) }, 1e-6, a, 4000)
        };
        g(a) / g(1.0)
    }
    /// f = d ln D / d ln a
    pub fn growth_rate(&self, a: f64) -> f64 {
        let h = 1e-4;
        (self.growth(a * (1.0 + h)).ln() - self.growth(a * (1.0 - h)).ln()) / ((1.0 + h).ln() - (1.0 - h).ln())
    }

    /// BBKS transfer function, k in h/Mpc.
    pub fn transfer(&self, k: f64) -> f64 {
        let gamma = self.omega_m * self.h * (-self.omega_b * (1.0 + (2.0 * self.h).sqrt() / self.omega_m)).exp();
        let q = k / gamma;
        if q < 1e-9 { return 1.0; }
        (1.0 + 2.34 * q).ln() / (2.34 * q) * (1.0 + 3.89 * q + (16.1 * q).powi(2) + (5.46 * q).powi(3) + (6.71 * q).powi(4)).powf(-0.25)
    }

    fn p_unnorm(&self, k: f64) -> f64 { k.powf(self.n_s) * self.transfer(k).powi(2) }

    fn sigma_r_unnorm(&self, r: f64) -> f64 {
        let w = |x: f64| if x < 1e-3 { 1.0 - x * x / 10.0 } else { 3.0 * (x.sin() - x * x.cos()) / x.powi(3) };
        // ∫ P(k) W²(kR) k² dk / 2π², integral over ln k
        let f = |lk: f64| { let k = lk.exp(); self.p_unnorm(k) * w(k * r).powi(2) * k.powi(3) / (2.0 * PI * PI) };
        Self::simpson(f, (1e-5f64).ln(), (1e3f64).ln(), 20_000).sqrt()
    }

    /// Linear power spectrum at a=1, (Mpc/h)³, normalized to σ₈.
    pub fn power(&self, k: f64) -> f64 {
        let amp = (self.sigma8 / self.sigma_r_unnorm(8.0)).powi(2);
        amp * self.p_unnorm(k)
    }
    pub fn power_amplitude(&self) -> f64 { (self.sigma8 / self.sigma_r_unnorm(8.0)).powi(2) }
    pub fn sigma_r(&self, r: f64) -> f64 { self.power_amplitude().sqrt() * self.sigma_r_unnorm(r) }
}

// ------------------------------------------------------------------ FFT
#[derive(Clone, Copy, Debug, Default)]
pub struct Cx { pub re: f64, pub im: f64 }

fn fft1(buf: &mut [Cx], inverse: bool) {
    let n = buf.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 { j ^= bit; bit >>= 1; }
        j |= bit;
        if i < j { buf.swap(i, j); }
    }
    let mut len = 2;
    while len <= n {
        let ang = 2.0 * PI / len as f64 * if inverse { 1.0 } else { -1.0 };
        let (wr, wi) = (ang.cos(), ang.sin());
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0, 0.0);
            for k in 0..len / 2 {
                let a = buf[start + k];
                let b = buf[start + k + len / 2];
                let (br, bi) = (b.re * cr - b.im * ci, b.re * ci + b.im * cr);
                buf[start + k] = Cx { re: a.re + br, im: a.im + bi };
                buf[start + k + len / 2] = Cx { re: a.re - br, im: a.im - bi };
                let t = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = t;
            }
        }
        len <<= 1;
    }
    if inverse { for x in buf.iter_mut() { x.re /= n as f64; x.im /= n as f64; } }
}

/// 3D FFT on an n³ grid (n is a power of two), index (i·n + j)·n + k.
pub fn fft3(data: &mut [Cx], n: usize, inverse: bool) {
    let mut line = vec![Cx::default(); n];
    for axis in 0..3 {
        for a in 0..n {
            for b in 0..n {
                let idx = |c: usize| match axis { 0 => (c * n + a) * n + b, 1 => (a * n + c) * n + b, _ => (a * n + b) * n + c };
                for c in 0..n { line[c] = data[idx(c)]; }
                fft1(&mut line, inverse);
                for c in 0..n { data[idx(c)] = line[c]; }
            }
        }
    }
}

fn kfreq(i: usize, n: usize, box_size: f64) -> f64 {
    let ii = if i <= n / 2 { i as f64 } else { i as f64 - n as f64 };
    2.0 * PI * ii / box_size
}

// ------------------------------------------------------------------ PM
pub struct PmSim {
    pub cosmo: Cosmology,
    pub n: usize,        // particles per side = grid size
    pub box_size: f64,   // Mpc/h
    pub a: f64,
    pub pos: Vec<[f64; 3]>,
    pub mom: Vec<[f64; 3]>, // u = a² dx/dt (Mpc/h · H₀)
    pub density: Vec<f64>,  // δ+1 on the grid after the last step
}

impl PmSim {
    pub fn new(cosmo: Cosmology, n: usize, box_size: f64, a_init: f64, seed: u64) -> PmSim {
        assert!(n.is_power_of_two());
        let mut rng = Rng::new(seed);
        let n3 = n * n * n;
        // Gaussian field δ_k with |δ_k|² = P(k)·N⁶/V (discrete normalization), reality comes from an FFT of real noise
        let mut field: Vec<Cx> = (0..n3).map(|_| Cx { re: rng.normal(), im: 0.0 }).collect();
        fft3(&mut field, n, false);
        let vol = box_size.powi(3);
        let amp = cosmo.power_amplitude();
        let mut psi: [Vec<Cx>; 3] = [vec![Cx::default(); n3], vec![Cx::default(); n3], vec![Cx::default(); n3]];
        for i in 0..n { for j in 0..n { for k in 0..n {
            let idx = (i * n + j) * n + k;
            let (kx, ky, kz) = (kfreq(i, n, box_size), kfreq(j, n, box_size), kfreq(k, n, box_size));
            let k2 = kx * kx + ky * ky + kz * kz;
            if k2 == 0.0 { continue; }
            let kk = k2.sqrt();
            let s = (amp * cosmo.p_unnorm(kk) / vol).sqrt(); // white noise has ⟨|w_k|²⟩ = N³
            let d = Cx { re: field[idx].re * s, im: field[idx].im * s };
            // ψ_k = i k / k² · δ_k   (∇·ψ = −δ)
            for (c, kc) in [kx, ky, kz].iter().enumerate() {
                psi[c][idx] = Cx { re: -d.im * kc / k2, im: d.re * kc / k2 };
            }
        }}}
        for c in 0..3 { fft3(&mut psi[c], n, true); }
        // discrete normalization: ⟨|δ̃_k|²⟩ should be N⁶P/V, and white noise has ⟨|w_k|²⟩ = N³ — factor N^{3/2}
        let scale = (n3 as f64).sqrt();
        let d = cosmo.growth(a_init);
        let f = cosmo.growth_rate(a_init);
        let e = cosmo.e(a_init);
        let cell = box_size / n as f64;
        let mut pos = Vec::with_capacity(n3);
        let mut mom = Vec::with_capacity(n3);
        for i in 0..n { for j in 0..n { for k in 0..n {
            let idx = (i * n + j) * n + k;
            let q = [i as f64 * cell, j as f64 * cell, k as f64 * cell];
            let mut p = [0.0; 3];
            let mut u = [0.0; 3];
            for c in 0..3 {
                let s = psi[c][idx].re * scale;
                p[c] = (q[c] + d * s).rem_euclid(box_size);
                u[c] = a_init * a_init * d * f * e * s; // u = a² Ḋ ψ, Ḋ = D f H
            }
            pos.push(p);
            mom.push(u);
        }}}
        PmSim { cosmo, n, box_size, a: a_init, pos, mom, density: vec![1.0; n3] }
    }

    fn deposit(&mut self) {
        let n = self.n;
        let cell = self.box_size / n as f64;
        let mut rho = vec![0.0; n * n * n];
        for p in &self.pos {
            let (x, y, z) = (p[0] / cell, p[1] / cell, p[2] / cell);
            let (i, j, k) = (x.floor(), y.floor(), z.floor());
            let (dx, dy, dz) = (x - i, y - j, z - k);
            let (i, j, k) = (i as isize, j as isize, k as isize);
            for (oi, wx) in [(0, 1.0 - dx), (1, dx)] { for (oj, wy) in [(0, 1.0 - dy), (1, dy)] { for (ok, wz) in [(0, 1.0 - dz), (1, dz)] {
                let ii = (i + oi).rem_euclid(n as isize) as usize;
                let jj = (j + oj).rem_euclid(n as isize) as usize;
                let kk = (k + ok).rem_euclid(n as isize) as usize;
                rho[(ii * n + jj) * n + kk] += wx * wy * wz;
            }}}
        }
        // mean number of particles per cell = 1
        self.density = rho;
    }

    /// Acceleration −∇φ on the grid (3 components).
    fn forces(&mut self) -> [Vec<f64>; 3] {
        self.deposit();
        let n = self.n;
        let n3 = n * n * n;
        let mut d: Vec<Cx> = self.density.iter().map(|&r| Cx { re: r - 1.0, im: 0.0 }).collect();
        fft3(&mut d, n, false);
        let pref = 1.5 * self.cosmo.omega_m / self.a;
        let cell = self.box_size / n as f64;
        for i in 0..n { for j in 0..n { for k in 0..n {
            let idx = (i * n + j) * n + k;
            // discrete Laplacian (consistent with the finite-difference gradient)
            let s = |m: usize| { let x = PI * m as f64 / n as f64; (2.0 * x.sin() / cell).powi(2) };
            let k2 = s(i) + s(j) + s(k);
            if k2 == 0.0 { d[idx] = Cx::default(); continue; }
            let g = -pref / k2;
            d[idx] = Cx { re: d[idx].re * g, im: d[idx].im * g };
        }}}
        fft3(&mut d, n, true);
        let phi: Vec<f64> = d.iter().map(|c| c.re).collect();
        let mut acc = [vec![0.0; n3], vec![0.0; n3], vec![0.0; n3]];
        let w = |i: isize| (i.rem_euclid(n as isize)) as usize;
        for i in 0..n as isize { for j in 0..n as isize { for k in 0..n as isize {
            let idx = (w(i) * n + w(j)) * n + w(k);
            acc[0][idx] = -(phi[(w(i + 1) * n + w(j)) * n + w(k)] - phi[(w(i - 1) * n + w(j)) * n + w(k)]) / (2.0 * cell);
            acc[1][idx] = -(phi[(w(i) * n + w(j + 1)) * n + w(k)] - phi[(w(i) * n + w(j - 1)) * n + w(k)]) / (2.0 * cell);
            acc[2][idx] = -(phi[(w(i) * n + w(j)) * n + w(k + 1)] - phi[(w(i) * n + w(j)) * n + w(k - 1)]) / (2.0 * cell);
        }}}
        acc
    }

    fn kick(&mut self, da: f64, a_force: f64) {
        let acc = self.forces();
        let n = self.n;
        let cell = self.box_size / n as f64;
        let fac = da / (a_force * self.cosmo.e(a_force)); // du/da = −∇φ / (a E)
        for (p, u) in self.pos.iter().zip(self.mom.iter_mut()) {
            let (x, y, z) = (p[0] / cell, p[1] / cell, p[2] / cell);
            let (i, j, k) = (x.floor(), y.floor(), z.floor());
            let (dx, dy, dz) = (x - i, y - j, z - k);
            let (i, j, k) = (i as isize, j as isize, k as isize);
            let mut g = [0.0; 3];
            for (oi, wx) in [(0, 1.0 - dx), (1, dx)] { for (oj, wy) in [(0, 1.0 - dy), (1, dy)] { for (ok, wz) in [(0, 1.0 - dz), (1, dz)] {
                let ii = (i + oi).rem_euclid(n as isize) as usize;
                let jj = (j + oj).rem_euclid(n as isize) as usize;
                let kk = (k + ok).rem_euclid(n as isize) as usize;
                let idx = (ii * n + jj) * n + kk;
                for c in 0..3 { g[c] += wx * wy * wz * acc[c][idx]; }
            }}}
            for c in 0..3 { u[c] += g[c] * fac; }
        }
    }

    fn drift(&mut self, a0: f64, a1: f64) {
        // dx/da = u / (a³ E): integral over a at constant u
        let ds = Cosmology::simpson(|a| 1.0 / (a.powi(3) * self.cosmo.e(a)), a0, a1, 16);
        let l = self.box_size;
        for (p, u) in self.pos.iter_mut().zip(self.mom.iter()) {
            for c in 0..3 { p[c] = (p[c] + u[c] * ds).rem_euclid(l); }
        }
    }

    /// KDK step in scale factor.
    pub fn step(&mut self, da: f64) {
        let a0 = self.a;
        let am = a0 + 0.5 * da;
        self.kick(0.5 * da, a0.max(1e-4) + 0.25 * da);
        self.drift(a0, a0 + da);
        self.a = a0 + da;
        self.kick(0.5 * da, am + 0.25 * da);
        let _ = am;
    }

    /// RMS density contrast on the grid.
    pub fn delta_rms(&mut self) -> f64 {
        self.deposit();
        let n3 = self.density.len() as f64;
        (self.density.iter().map(|r| (r - 1.0).powi(2)).sum::<f64>() / n3).sqrt()
    }

    /// Measured power spectrum: (k, P(k)) in bins, with CIC (no deconvolution, large scales only).
    pub fn measure_power(&mut self, bins: usize) -> Vec<(f64, f64)> {
        self.deposit();
        let n = self.n;
        let mut d: Vec<Cx> = self.density.iter().map(|&r| Cx { re: r - 1.0, im: 0.0 }).collect();
        fft3(&mut d, n, false);
        let kf = 2.0 * PI / self.box_size;
        let kn = PI * n as f64 / self.box_size;
        let mut sum = vec![0.0; bins];
        let mut cnt = vec![0usize; bins];
        let vol = self.box_size.powi(3);
        let n3 = (n * n * n) as f64;
        for i in 0..n { for j in 0..n { for k in 0..n {
            let kk = (kfreq(i, n, self.box_size).powi(2) + kfreq(j, n, self.box_size).powi(2) + kfreq(k, n, self.box_size).powi(2)).sqrt();
            if kk == 0.0 || kk >= kn { continue; }
            let b = ((kk / kf).ln() / (kn / kf).ln() * bins as f64) as usize;
            if b >= bins { continue; }
            let c = d[(i * n + j) * n + k];
            sum[b] += (c.re * c.re + c.im * c.im) * vol / (n3 * n3);
            cnt[b] += 1;
        }}}
        (0..bins).filter(|&b| cnt[b] > 0).map(|b| {
            let k = kf * ((b as f64 + 0.5) / bins as f64 * (kn / kf).ln()).exp();
            (k, sum[b] / cnt[b] as f64)
        }).collect()
    }

    /// Total momentum (conserved with symmetric CIC + central differences).
    pub fn total_momentum(&self) -> f64 {
        let mut s = [0.0; 3];
        for u in &self.mom { for c in 0..3 { s[c] += u[c]; } }
        (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt()
    }
    pub fn momentum_scale(&self) -> f64 { self.mom.iter().map(|u| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt()).sum() }
}
