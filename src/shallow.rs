//! Winds on a sphere: the shallow water equations by the spectral transform method (vorticity and divergence, Gaussian grid).
//!
//! One layer, so this is an intermediate model and not a general circulation model: no vertical structure, no moisture
//! physics, no radiation scheme. It is checked against the exact steady flow of Williamson et al. (1992), test case 2.

use crate::climate::ClimateResult;
use crate::constants::R_GAS;
use crate::world::World;
use std::f64::consts::PI;

type C = (f64, f64);
fn cmul(a: C, b: C) -> C { (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0) }

fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut x, mut w) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n { let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64; p0 = p1; p1 = p2; }
            dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() < 1e-15 { break; }
        }
        x[n - 1 - i] = z;
        w[n - 1 - i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
    (x, w)
}

/// Spectral state: vorticity, divergence, geopotential.
#[derive(Clone)]
pub struct State { pub z: Vec<C>, pub d: Vec<C>, pub p: Vec<C> }

pub struct Sphere {
    pub trunc: usize, pub nlat: usize, pub nlon: usize, pub radius: f64, pub omega: f64,
    pub mu: Vec<f64>, weight: Vec<f64>, plm: Vec<Vec<f64>>, hlm: Vec<Vec<f64>>, modes: Vec<(usize, usize)>, cos_t: Vec<Vec<f64>>, sin_t: Vec<Vec<f64>>,
    /// Coriolis parameter on the grid, can be tilted for tests. [lat][lon]
    pub coriolis: Vec<Vec<f64>>,
    /// Relaxation: (equilibrium geopotential on the grid in spectral form, radiative time, drag time).
    pub forcing: Option<(Vec<C>, f64, f64)>,
    /// Damping time of the shortest scales, s; 0 means none.
    pub hyper: f64,
}

impl Sphere {
    pub fn new(trunc: usize, radius: f64, omega: f64, tilt: f64) -> Sphere {
        let nlon = (3 * trunc + 1).next_power_of_two().max(4);
        let nlat = nlon / 2;
        let (mu, weight) = gauss_legendre(nlat);
        let eps = |m: usize, n: usize| -> f64 { let (m, n) = (m as f64, n as f64); ((n * n - m * m) / (4.0 * n * n - 1.0)).sqrt() };
        let (mut modes, mut first) = (vec![], vec![]);
        for m in 0..=trunc { first.push(modes.len()); for n in m..=trunc { modes.push((m, n)); } }
        let (mut plm, mut hlm) = (vec![], vec![]);
        for &x in &mu {
            let (mut pj, mut hj) = (vec![0.0; modes.len()], vec![0.0; modes.len()]);
            let s = (1.0 - x * x).sqrt();
            let mut pmm = 1.0 / 2f64.sqrt();
            for m in 0..=trunc {
                if m > 0 { pmm *= ((2 * m + 1) as f64 / (2 * m) as f64).sqrt() * s; }
                // P up to trunc + 1: the derivative needs one more
                let mut col = vec![0.0; trunc + 3];
                col[m] = pmm;
                col[m + 1] = ((2 * m + 3) as f64).sqrt() * x * pmm;
                for n in (m + 2)..=(trunc + 1) { col[n] = (x * col[n - 1] - eps(m, n - 1) * col[n - 2]) / eps(m, n); }
                for n in m..=trunc {
                    let i = first[m] + n - m;
                    pj[i] = col[n];
                    let below = if n > m { (n + 1) as f64 * eps(m, n) * col[n - 1] } else { 0.0 };
                    hj[i] = below - n as f64 * eps(m, n + 1) * col[n + 1];
                }
            }
            plm.push(pj);
            hlm.push(hj);
        }
        let (mut cos_t, mut sin_t) = (vec![], vec![]);
        for m in 0..=trunc {
            cos_t.push((0..nlon).map(|i| (2.0 * PI * (m * i) as f64 / nlon as f64).cos()).collect());
            sin_t.push((0..nlon).map(|i| (2.0 * PI * (m * i) as f64 / nlon as f64).sin()).collect());
        }
        let coriolis = mu.iter().map(|&x| (0..nlon).map(|i| {
            let l = 2.0 * PI * i as f64 / nlon as f64;
            2.0 * omega * (-l.cos() * (1.0 - x * x).sqrt() * tilt.sin() + x * tilt.cos())
        }).collect()).collect();
        Sphere { trunc, nlat, nlon, radius, omega, mu, weight, plm, hlm, modes, cos_t, sin_t, coriolis, forcing: None, hyper: 0.0 }
    }

    /// Longitude of grid column i, radians.
    pub fn lon(&self, i: usize) -> f64 { 2.0 * PI * i as f64 / self.nlon as f64 }

    /// Fourier coefficients of a grid field by latitude, m = 0..=trunc.
    fn fourier(&self, f: &[Vec<f64>]) -> Vec<Vec<C>> {
        f.iter().map(|row| (0..=self.trunc).map(|m| {
            let (mut re, mut im) = (0.0, 0.0);
            for i in 0..self.nlon { re += row[i] * self.cos_t[m][i]; im -= row[i] * self.sin_t[m][i]; }
            (re / self.nlon as f64, im / self.nlon as f64)
        }).collect()).collect()
    }
    fn synthesis(&self, f: &[Vec<C>]) -> Vec<Vec<f64>> {
        f.iter().map(|row| (0..self.nlon).map(|i| {
            let mut v = row[0].0;
            for m in 1..=self.trunc { v += 2.0 * (row[m].0 * self.cos_t[m][i] - row[m].1 * self.sin_t[m][i]); }
            v
        }).collect()).collect()
    }

    /// Grid to spectral.
    pub fn analysis(&self, f: &[Vec<f64>]) -> Vec<C> {
        let four = self.fourier(f);
        self.modes.iter().enumerate().map(|(k, &(m, _))| {
            let (mut re, mut im) = (0.0, 0.0);
            for j in 0..self.nlat { re += self.weight[j] * four[j][m].0 * self.plm[j][k]; im += self.weight[j] * four[j][m].1 * self.plm[j][k]; }
            (re, im)
        }).collect()
    }
    /// Spectral to grid.
    pub fn grid(&self, s: &[C]) -> Vec<Vec<f64>> {
        let four: Vec<Vec<C>> = (0..self.nlat).map(|j| {
            let mut row = vec![(0.0, 0.0); self.trunc + 1];
            for (k, &(m, _)) in self.modes.iter().enumerate() { row[m].0 += s[k].0 * self.plm[j][k]; row[m].1 += s[k].1 * self.plm[j][k]; }
            row
        }).collect();
        self.synthesis(&four)
    }

    /// U = u·cosφ and V = v·cosφ from vorticity and divergence.
    pub fn winds(&self, z: &[C], d: &[C]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let a = self.radius;
        let inv = |s: &[C]| -> Vec<C> { s.iter().zip(&self.modes).map(|(x, &(_, n))| if n == 0 { (0.0, 0.0) } else { let f = -a * a / (n * (n + 1)) as f64; (x.0 * f, x.1 * f) }).collect() };
        let (psi, chi) = (inv(z), inv(d));
        let (mut fu, mut fv) = (vec![vec![(0.0, 0.0); self.trunc + 1]; self.nlat], vec![vec![(0.0, 0.0); self.trunc + 1]; self.nlat]);
        for j in 0..self.nlat {
            for (k, &(m, _)) in self.modes.iter().enumerate() {
                let (p, h, im) = (self.plm[j][k], self.hlm[j][k], (0.0, m as f64));
                let (dchi, dpsi) = (cmul(im, chi[k]), cmul(im, psi[k]));
                fu[j][m].0 += (dchi.0 * p - psi[k].0 * h) / a;
                fu[j][m].1 += (dchi.1 * p - psi[k].1 * h) / a;
                fv[j][m].0 += (dpsi.0 * p + chi[k].0 * h) / a;
                fv[j][m].1 += (dpsi.1 * p + chi[k].1 * h) / a;
            }
        }
        (self.synthesis(&fu), self.synthesis(&fv))
    }

    /// Divergence and curl of the vector field (A, B)/cosφ, in spectral form.
    pub fn div_curl(&self, a_grid: &[Vec<f64>], b_grid: &[Vec<f64>]) -> (Vec<C>, Vec<C>) {
        let (fa, fb) = (self.fourier(a_grid), self.fourier(b_grid));
        let (mut div, mut curl) = (vec![(0.0, 0.0); self.modes.len()], vec![(0.0, 0.0); self.modes.len()]);
        for j in 0..self.nlat {
            let w = self.weight[j] / (self.radius * (1.0 - self.mu[j] * self.mu[j]));
            for (k, &(m, _)) in self.modes.iter().enumerate() {
                let (p, h, im) = (self.plm[j][k], self.hlm[j][k], (0.0, m as f64));
                let (ia, ib) = (cmul(im, fa[j][m]), cmul(im, fb[j][m]));
                div[k].0 += w * (ia.0 * p - fb[j][m].0 * h);
                div[k].1 += w * (ia.1 * p - fb[j][m].1 * h);
                curl[k].0 += w * (ib.0 * p + fa[j][m].0 * h);
                curl[k].1 += w * (ib.1 * p + fa[j][m].1 * h);
            }
        }
        (div, curl)
    }

    fn tendency(&self, s: &State) -> State {
        let (u, v) = self.winds(&s.z, &s.d);
        let (zg, pg) = (self.grid(&s.z), self.grid(&s.p));
        let (nl, nl2) = (self.nlat, self.nlon);
        let (mut a1, mut b1, mut a2, mut b2, mut en) = (vec![vec![0.0; nl2]; nl], vec![vec![0.0; nl2]; nl], vec![vec![0.0; nl2]; nl], vec![vec![0.0; nl2]; nl], vec![vec![0.0; nl2]; nl]);
        for j in 0..nl {
            let c2 = 1.0 - self.mu[j] * self.mu[j];
            for i in 0..nl2 {
                let q = zg[j][i] + self.coriolis[j][i];
                a1[j][i] = q * u[j][i];
                b1[j][i] = q * v[j][i];
                a2[j][i] = pg[j][i] * u[j][i];
                b2[j][i] = pg[j][i] * v[j][i];
                en[j][i] = pg[j][i] + 0.5 * (u[j][i] * u[j][i] + v[j][i] * v[j][i]) / c2;
            }
        }
        let (div1, curl1) = self.div_curl(&a1, &b1);
        let (div2, _) = self.div_curl(&a2, &b2);
        let esp = self.analysis(&en);
        let a2i = 1.0 / (self.radius * self.radius);
        let mut out = State { z: vec![(0.0, 0.0); self.modes.len()], d: vec![(0.0, 0.0); self.modes.len()], p: vec![(0.0, 0.0); self.modes.len()] };
        for (k, &(_, n)) in self.modes.iter().enumerate() {
            let lap = (n * (n + 1)) as f64 * a2i;
            out.z[k] = (-div1[k].0, -div1[k].1);
            out.d[k] = (curl1[k].0 + lap * esp[k].0, curl1[k].1 + lap * esp[k].1);
            out.p[k] = (-div2[k].0, -div2[k].1);
            if let Some((eq, rad, drag)) = &self.forcing {
                out.p[k].0 += (eq[k].0 - s.p[k].0) / rad;
                out.p[k].1 += (eq[k].1 - s.p[k].1) / rad;
                out.z[k].0 -= s.z[k].0 / drag;
                out.z[k].1 -= s.z[k].1 / drag;
                out.d[k].0 -= s.d[k].0 / drag;
                out.d[k].1 -= s.d[k].1 / drag;
            }
        }
        out
    }

    /// Classical Runge–Kutta step, then damping of the shortest scales.
    pub fn step(&self, s: &State, dt: f64) -> State {
        let add = |a: &[C], b: &[C], h: f64| -> Vec<C> { a.iter().zip(b).map(|(x, y)| (x.0 + h * y.0, x.1 + h * y.1)).collect() };
        let shift = |h: f64, r: &State| State { z: add(&s.z, &r.z, h), d: add(&s.d, &r.d, h), p: add(&s.p, &r.p, h) };
        let k1 = self.tendency(s);
        let k2 = self.tendency(&shift(dt / 2.0, &k1));
        let k3 = self.tendency(&shift(dt / 2.0, &k2));
        let k4 = self.tendency(&shift(dt, &k3));
        let mix = |a: &[C], x1: &[C], x2: &[C], x3: &[C], x4: &[C]| -> Vec<C> {
            (0..a.len()).map(|i| (a[i].0 + dt / 6.0 * (x1[i].0 + 2.0 * x2[i].0 + 2.0 * x3[i].0 + x4[i].0), a[i].1 + dt / 6.0 * (x1[i].1 + 2.0 * x2[i].1 + 2.0 * x3[i].1 + x4[i].1))).collect()
        };
        let mut out = State { z: mix(&s.z, &k1.z, &k2.z, &k3.z, &k4.z), d: mix(&s.d, &k1.d, &k2.d, &k3.d, &k4.d), p: mix(&s.p, &k1.p, &k2.p, &k3.p, &k4.p) };
        if self.hyper > 0.0 {
            let top = (self.trunc * (self.trunc + 1)) as f64;
            for (k, &(_, n)) in self.modes.iter().enumerate() {
                let f = (-dt / self.hyper * ((n * (n + 1)) as f64 / top).powi(2)).exp();
                for field in [&mut out.z, &mut out.d, &mut out.p] { field[k] = (field[k].0 * f, field[k].1 * f); }
            }
        }
        out
    }

    /// Total mass in geopotential units: the mean over the sphere.
    pub fn mean_geopotential(&self, s: &State) -> f64 { s.p[0].0 / 2f64.sqrt() }
}

/// Williamson test case 2: a global steady geostrophic flow over a sphere tilted by α. Returns the relative L2 error in depth after `days`.
pub fn williamson2(trunc: usize, alpha: f64, days: f64, dt: f64) -> f64 {
    let (a, omega, g) = (6.37122e6, 7.292e-5, 9.80616);
    let u0 = 2.0 * PI * a / (12.0 * 86400.0);
    let gh0 = 2.94e4;
    let sw = Sphere::new(trunc, a, omega, alpha);
    let (mut ug, mut vg, mut pg) = (vec![vec![0.0; sw.nlon]; sw.nlat], vec![vec![0.0; sw.nlon]; sw.nlat], vec![vec![0.0; sw.nlon]; sw.nlat]);
    for j in 0..sw.nlat {
        let (x, c) = (sw.mu[j], (1.0 - sw.mu[j] * sw.mu[j]).sqrt());
        for i in 0..sw.nlon {
            let l = sw.lon(i);
            let u = u0 * (c * alpha.cos() + x * l.cos() * alpha.sin());
            let v = -u0 * l.sin() * alpha.sin();
            ug[j][i] = u * c;
            vg[j][i] = v * c;
            let tilt = -l.cos() * c * alpha.sin() + x * alpha.cos();
            pg[j][i] = gh0 - (a * omega * u0 + u0 * u0 / 2.0) * tilt * tilt;
        }
    }
    let (d, z) = sw.div_curl(&ug, &vg);
    let mut s = State { z, d, p: sw.analysis(&pg) };
    let exact = pg;
    let steps = (days * 86400.0 / dt).ceil() as usize;
    for _ in 0..steps { s = sw.step(&s, days * 86400.0 / steps as f64); }
    let now = sw.grid(&s.p);
    let (mut top, mut bot) = (0.0, 0.0);
    for j in 0..sw.nlat { for i in 0..sw.nlon {
        top += sw.weight[j] * (now[j][i] - exact[j][i]).powi(2) / (g * g);
        bot += sw.weight[j] * exact[j][i].powi(2) / (g * g);
    }}
    (top / bot).sqrt()
}

/// Gravity wave of degree n on a non-rotating sphere: the depth pattern oscillates with ω² = n(n+1)Φ/a². Returns (measured / theory) period.
pub fn gravity_wave_period(n: usize) -> f64 {
    let (a, phi0) = (6.37e6, 1e5);
    let sw = Sphere::new(21, a, 0.0, 0.0);
    let omega = ((n * (n + 1)) as f64 * phi0).sqrt() / a;
    let period = 2.0 * PI / omega;
    let modes = sw.modes.len();
    let k = sw.modes.iter().position(|m| *m == (0, n)).unwrap();
    let mut s = State { z: vec![(0.0, 0.0); modes], d: vec![(0.0, 0.0); modes], p: vec![(0.0, 0.0); modes] };
    s.p[0].0 = phi0 * 2f64.sqrt();
    s.p[k].0 = phi0 * 1e-4;
    // from the crest, the amplitude first crosses zero after a quarter period
    let dt = period / 400.0;
    let (mut t, mut prev) = (0.0, s.p[k].0);
    loop {
        s = sw.step(&s, dt);
        t += dt;
        let now = s.p[k].0;
        if now <= 0.0 { return (t - dt + dt * prev / (prev - now)) * 4.0 / period; }
        prev = now;
        if t > 2.0 * period { return f64::NAN; }
    }
}

#[derive(Clone, Debug)]
pub struct WindResult { pub lat: Vec<f64>, pub u: Vec<f64>, pub v: Vec<f64>, pub ascent: Vec<f64>, pub cloud: Vec<f64>, pub balance_error: f64, pub peak_u: f64, pub peak_lat: f64, pub days: f64 }

/// Zonal winds and diagnostic cloud cover for a world, driven by its annual-mean temperature from the energy balance model.
///
/// The temperature contrast sets an equilibrium geopotential at the upper troposphere (thickness up to 0.25 of surface pressure); the
/// layer relaxes to it in 15 days, with 30 days of drag. Cloud cover is diagnosed from ascent: a rule of thumb, not a cloud scheme,
/// and it does not feed back on temperature.
pub fn world_winds(world: &World, climate: &ClimateResult, trunc: usize, days: f64) -> Result<WindResult, String> {
    let atm = world.atmosphere.as_ref().ok_or("this world has no atmosphere, so there are no winds")?;
    let rs = R_GAS / atm.molar_mass;
    let omega = world.omega().abs();
    let mut sw = Sphere::new(trunc, world.physics.radius, omega, 0.0);
    let n_in = climate.lat.len();
    let mean_t = climate.global_mean;
    let thick = 4f64.ln();
    let temp_at = |lat: f64| -> f64 {
        let x = ((lat + 90.0) / 180.0 * n_in as f64 - 0.5).clamp(0.0, (n_in - 1) as f64);
        let (i, f) = (x.floor() as usize, x.fract());
        climate.annual_mean[i] * (1.0 - f) + climate.annual_mean[(i + 1).min(n_in - 1)] * f
    };
    let eq: Vec<Vec<f64>> = sw.mu.iter().map(|x| vec![rs * mean_t + rs * thick * (temp_at(x.asin().to_degrees()) - mean_t); sw.nlon]).collect();
    let spec_eq = sw.analysis(&eq);
    sw.forcing = Some((spec_eq.clone(), 15.0 * 86400.0, 30.0 * 86400.0));
    sw.hyper = 86400.0;
    let n = sw.modes.len();
    let mut s = State { z: vec![(0.0, 0.0); n], d: vec![(0.0, 0.0); n], p: spec_eq };
    let c = (rs * mean_t).sqrt();
    let dt = (0.8 * world.physics.radius / (c * trunc as f64)).min(3600.0);
    let steps = (days * 86400.0 / dt).ceil() as usize;
    for _ in 0..steps {
        s = sw.step(&s, days * 86400.0 / steps as f64);
        if s.p.iter().any(|x| !x.0.is_finite()) { return Err("winds: the solution diverged".into()); }
    }
    let (ug, vg) = sw.winds(&s.z, &s.d);
    let (dg, pg) = (sw.grid(&s.d), sw.grid(&s.p));
    let zonal = |f: &[Vec<f64>], j: usize| f[j].iter().sum::<f64>() / sw.nlon as f64;
    let (mut lat, mut u, mut v, mut ascent, mut phi) = (vec![], vec![], vec![], vec![], vec![]);
    for j in 0..sw.nlat {
        let cs = (1.0 - sw.mu[j] * sw.mu[j]).sqrt();
        lat.push(sw.mu[j].asin().to_degrees());
        u.push(zonal(&ug, j) / cs);
        v.push(zonal(&vg, j) / cs);
        // the layer is the upper troposphere: air diverging there has risen from below
        ascent.push(zonal(&dg, j));
        phi.push(zonal(&pg, j));
    }
    // gradient wind balance of the mean flow, away from the poles and the equator where it degenerates
    let (mut top, mut bot) = (0.0_f64, 0.0_f64);
    for j in 1..sw.nlat - 1 {
        if lat[j].abs() < 15.0 || lat[j].abs() > 75.0 { continue; }
        let f = 2.0 * omega * sw.mu[j];
        let dphi = (phi[j + 1] - phi[j - 1]) / ((lat[j + 1] - lat[j - 1]).to_radians() * world.physics.radius);
        let press = -dphi;
        let coriolis = f * u[j] + u[j] * u[j] * lat[j].to_radians().tan() / world.physics.radius;
        top = top.max((coriolis - press).abs());
        bot = bot.max(press.abs());
    }
    let spread = ascent.iter().map(|x| x * x).sum::<f64>().sqrt() / (ascent.len() as f64).sqrt();
    let cloud: Vec<f64> = ascent.iter().map(|a| (0.4 + 0.3 * a / spread.max(1e-30)).clamp(0.0, 1.0)).collect();
    let k = (0..u.len()).max_by(|a, b| u[*a].abs().total_cmp(&u[*b].abs())).unwrap_or(0);
    Ok(WindResult { peak_u: u[k], peak_lat: lat[k], lat, u, v, ascent, cloud, balance_error: top / bot.max(1e-30), days })
}
