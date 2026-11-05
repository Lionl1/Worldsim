//! N-body problem: law of universal gravitation + optional 1PN GR correction (Schwarzschild)
//! from the central body. Integrators: symplectic leapfrog (2nd order) and Yoshida (4th),
//! which preserve phase volume and show no secular energy drift.

use crate::constants::C;
use crate::orbit::elements_to_state;
use crate::presets;
use crate::vec3::Vec3;

#[derive(Clone, Debug)]
pub struct Body {
    pub name: String,
    pub gm: f64,
    pub r: Vec3,
    pub v: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Integrator {
    Leapfrog,
    Yoshida4,
    /// Gauss–Radau 15th order with an adaptive step (Rein & Spiegel 2015): `step(dt)` takes as many internal steps as it needs.
    Ias15,
    /// Wisdom–Holman in democratic heliocentric coordinates, body 0 is the star. Second order and fixed step, no close encounters and no GR.
    WisdomHolman,
}

#[derive(Clone, Debug)]
pub struct NBody {
    pub bodies: Vec<Body>,
    pub t: f64,
    /// Index of the body used for the 1PN GR correction (usually the Sun).
    pub relativity_center: Option<usize>,
    pub integrator: Integrator,
    ias_dt: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Invariants {
    pub energy: f64,           // G·E  (in GM units, to avoid losing precision on G)
    pub momentum: Vec3,        // G·P
    pub angular_momentum: Vec3,// G·L
    pub center_of_mass: Vec3,
}

impl NBody {
    pub fn new(bodies: Vec<Body>) -> Self {
        NBody { bodies, t: 0.0, relativity_center: None, integrator: Integrator::Yoshida4, ias_dt: 0.0 }
    }

    /// Solar system (or part of it) from presets using elements J2000.
    pub fn from_presets(names: &[&str]) -> Result<Self, String> {
        let mut presets_v = Vec::new();
        for n in names { presets_v.push(presets::physics(n)?); }
        let idx = |n: &str| presets_v.iter().position(|p| p.name == n);
        let mut bodies: Vec<Body> = presets_v.iter().map(|p| Body { name: p.name.clone(), gm: p.gm, r: Vec3::ZERO, v: Vec3::ZERO }).collect();
        // first bodies around the root, then moons (hierarchy levels)
        for pass in 0..3 {
            for (k, p) in presets_v.iter().enumerate() {
                let Some(o) = &p.orbit else { continue };
                let parent = match idx(&o.parent) {
                    Some(j) => j,
                    None => continue, // the parent is not in the system: the body stays at the origin
                };
                let depth = depth_of(&presets_v, k);
                if depth != pass + 1 { continue; }
                // moons of the parent present in the system also add mass to the barycenter
                let moons_gm: f64 = presets_v.iter().filter(|q| q.orbit.as_ref().map(|oo| oo.parent == p.name).unwrap_or(false)).map(|q| q.gm).sum();
                let mu = presets_v[parent].gm + p.gm + moons_gm;
                let (r, v) = elements_to_state(mu, o.a, o.e, o.i, o.node, o.long_peri - o.node, o.mean_long - o.long_peri);
                bodies[k].r = bodies[parent].r + r;
                bodies[k].v = bodies[parent].v + v;
                let _ = moons_gm;
            }
            // moons at this level: shift the parent so the barycenter of the pair stays in place
            for (k, p) in presets_v.iter().enumerate() {
                let Some(o) = &p.orbit else { continue };
                let Some(parent) = idx(&o.parent) else { continue };
                if depth_of(&presets_v, k) != pass + 1 || presets_v[parent].orbit.is_none() { continue; }
                let gm_p = presets_v[parent].gm;
                let gm_s = p.gm;
                let rel_r = bodies[k].r - bodies[parent].r;
                let rel_v = bodies[k].v - bodies[parent].v;
                let f = gm_s / (gm_p + gm_s);
                bodies[parent].r -= rel_r * f;
                bodies[parent].v -= rel_v * f;
                bodies[k].r -= rel_r * f;
                bodies[k].v -= rel_v * f;
            }
        }
        let mut s = NBody::new(bodies);
        s.to_barycentric();
        if let Some(sun) = idx("sun") { s.relativity_center = None; let _ = sun; }
        Ok(s)
    }

    pub fn index(&self, name: &str) -> Option<usize> { self.bodies.iter().position(|b| b.name == name) }

    pub fn enable_relativity(&mut self, center: &str) -> Result<(), String> {
        self.relativity_center = Some(self.index(center).ok_or(format!("no body {center}"))?);
        Ok(())
    }

    /// Convert to the center-of-mass frame (total momentum = 0).
    pub fn to_barycentric(&mut self) {
        let m: f64 = self.bodies.iter().map(|b| b.gm).sum();
        let rc = self.bodies.iter().fold(Vec3::ZERO, |a, b| a + b.r * b.gm) / m;
        let vc = self.bodies.iter().fold(Vec3::ZERO, |a, b| a + b.v * b.gm) / m;
        for b in &mut self.bodies { b.r -= rc; b.v -= vc; }
    }

    fn accelerations(&self, acc: &mut [Vec3]) {
        let n = self.bodies.len();
        for a in acc.iter_mut() { *a = Vec3::ZERO; }
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.bodies[j].r - self.bodies[i].r;
                let r2 = d.norm2();
                let inv_r3 = 1.0 / (r2 * r2.sqrt());
                acc[i] += d * (self.bodies[j].gm * inv_r3);
                acc[j] -= d * (self.bodies[i].gm * inv_r3);
            }
        }
        if let Some(c) = self.relativity_center {
            let (gm, rc, vc) = (self.bodies[c].gm, self.bodies[c].r, self.bodies[c].v);
            for i in 0..n {
                if i == c { continue; }
                let r = self.bodies[i].r - rc;
                let v = self.bodies[i].v - vc;
                let rn = r.norm();
                // a_1PN = GM/(c² r³) [ (4GM/r − v²) r + 4 (r·v) v ]
                acc[i] += (r * (4.0 * gm / rn - v.norm2()) + v * (4.0 * r.dot(v))) * (gm / (C * C * rn.powi(3)));
            }
        }
    }

    fn drift(&mut self, h: f64) { for b in &mut self.bodies { b.r += b.v * h; } }
    fn kick(&mut self, h: f64, acc: &mut [Vec3]) {
        self.accelerations(acc);
        for (b, a) in self.bodies.iter_mut().zip(acc.iter()) { b.v += *a * h; }
    }

    pub fn step(&mut self, dt: f64) {
        let mut acc = vec![Vec3::ZERO; self.bodies.len()];
        match self.integrator {
            Integrator::Leapfrog => {
                self.drift(0.5 * dt);
                self.kick(dt, &mut acc);
                self.drift(0.5 * dt);
            }
            Integrator::Yoshida4 => {
                let cbrt2 = 2f64.powf(1.0 / 3.0);
                let w1 = 1.0 / (2.0 - cbrt2);
                let w0 = -cbrt2 * w1;
                let c = [w1 / 2.0, (w0 + w1) / 2.0, (w0 + w1) / 2.0, w1 / 2.0];
                let d = [w1, w0, w1];
                for k in 0..3 {
                    self.drift(c[k] * dt);
                    self.kick(d[k] * dt, &mut acc);
                }
                self.drift(c[3] * dt);
            }
            Integrator::Ias15 => { self.ias15_advance(dt); return; }
            Integrator::WisdomHolman => self.wisdom_holman(dt),
        }
        self.t += dt;
    }

    /// Forgets the adaptive step, for example after the set of bodies changed.
    pub fn reset_step(&mut self) { self.ias_dt = 0.0; }

    fn ias15_advance(&mut self, dt: f64) {
        let end = self.t + dt;
        if self.ias_dt == 0.0 { self.ias_dt = dt / 16.0; }
        while (end - self.t).abs() > 1e-13 * end.abs().max(dt.abs()) {
            let left = end - self.t;
            let wish = self.ias_dt;
            let (done, next) = self.ias15_step(if wish.abs() >= left.abs() { left } else { wish });
            self.t += done;
            // a shortened last step says nothing about the size the orbit allows
            self.ias_dt = if done == wish || next.abs() < wish.abs() { next } else { wish };
        }
    }

    /// One accepted step. Returns (the step taken, the next step to try).
    fn ias15_step(&mut self, mut h: f64) -> (f64, f64) {
        let n = self.bodies.len();
        let (x0, v0): (Vec<Vec3>, Vec<Vec3>) = (self.bodies.iter().map(|b| b.r).collect(), self.bodies.iter().map(|b| b.v).collect());
        let mut a0 = vec![Vec3::ZERO; n];
        let mut a = vec![Vec3::ZERO; n];
        self.accelerations(&mut a0);
        let c = ias15_coefficients();
        loop {
            let mut g = [vec![Vec3::ZERO; n], vec![Vec3::ZERO; n], vec![Vec3::ZERO; n], vec![Vec3::ZERO; n], vec![Vec3::ZERO; n], vec![Vec3::ZERO; n], vec![Vec3::ZERO; n]];
            let mut b = g.clone();
            let mut last_b6 = vec![Vec3::ZERO; n];
            for _ in 0..16 {
                for j in 1..8 {
                    let tau = IAS15_NODES[j];
                    for i in 0..n {
                        let (mut px, mut pv) = (a0[i] * 0.5, a0[i]);
                        let mut p = tau;
                        for (k, bk) in b.iter().enumerate() {
                            px += bk[i] * (p / ((k + 2) * (k + 3)) as f64);
                            pv += bk[i] * (p / (k + 2) as f64);
                            p *= tau;
                        }
                        self.bodies[i].r = x0[i] + v0[i] * (tau * h) + px * (h * h * tau * tau);
                        self.bodies[i].v = v0[i] + pv * (h * tau);
                    }
                    self.accelerations(&mut a);
                    for i in 0..n {
                        // Newton divided differences on the Gauss–Radau nodes
                        let mut d = (a[i] - a0[i]) / tau;
                        for l in 0..(j - 1) { d = (d - g[l][i]) / (tau - IAS15_NODES[l + 1]); }
                        g[j - 1][i] = d;
                    }
                    for k in 0..7 { for i in 0..n { b[k][i] = (k..7).fold(Vec3::ZERO, |s, m| s + g[m][i] * c[m][k]); } }
                }
                let top = a.iter().map(|x| x.norm()).fold(0.0, f64::max).max(1e-300);
                let change = b[6].iter().zip(&last_b6).map(|(x, y)| (*x - *y).norm()).fold(0.0, f64::max);
                last_b6.clone_from(&b[6]);
                if change / top < 1e-16 { break; }
            }
            let top = a.iter().map(|x| x.norm()).fold(0.0, f64::max).max(1e-300);
            let phi = b[6].iter().map(|x| x.norm()).fold(0.0, f64::max) / top;
            let wish = if phi > 0.0 { h * (1e-9 / phi).powf(1.0 / 7.0) } else { h * 4.0 };
            if wish.abs() < 0.25 * h.abs() {
                for i in 0..n { self.bodies[i].r = x0[i]; self.bodies[i].v = v0[i]; }
                h = wish;
                continue;
            }
            for i in 0..n {
                let (mut px, mut pv) = (a0[i] * 0.5, a0[i]);
                for (k, bk) in b.iter().enumerate() {
                    px += bk[i] * (1.0 / ((k + 2) * (k + 3)) as f64);
                    pv += bk[i] * (1.0 / (k + 2) as f64);
                }
                self.bodies[i].r = x0[i] + v0[i] * h + px * (h * h);
                self.bodies[i].v = v0[i] + pv * h;
            }
            let limit = 4.0 * h.abs();
            return (h, wish.clamp(-limit, limit));
        }
    }

    fn wisdom_holman(&mut self, dt: f64) {
        let n = self.bodies.len();
        let total: f64 = self.bodies.iter().map(|b| b.gm).sum();
        let (gm0, r_com, v_com) = (self.bodies[0].gm, self.bodies.iter().fold(Vec3::ZERO, |s, b| s + b.r * b.gm) / total, self.bodies.iter().fold(Vec3::ZERO, |s, b| s + b.v * b.gm) / total);
        let r0 = self.bodies[0].r;
        // Q: position relative to the star; V: velocity relative to the center of mass
        let mut q: Vec<Vec3> = self.bodies.iter().map(|b| b.r - r0).collect();
        let mut v: Vec<Vec3> = self.bodies.iter().map(|b| b.v - v_com).collect();
        let jump = |q: &mut [Vec3], v: &[Vec3], h: f64, bodies: &[Body]| {
            let p = (1..n).fold(Vec3::ZERO, |s, i| s + v[i] * bodies[i].gm) / gm0;
            for x in q.iter_mut().skip(1) { *x += p * h; }
        };
        let kick = |q: &[Vec3], v: &mut [Vec3], h: f64, bodies: &[Body]| {
            for i in 1..n {
                for j in (i + 1)..n {
                    let d = q[j] - q[i];
                    let inv = 1.0 / (d.norm2() * d.norm());
                    v[i] += d * (bodies[j].gm * inv * h);
                    v[j] -= d * (bodies[i].gm * inv * h);
                }
            }
        };
        jump(&mut q, &v, dt / 2.0, &self.bodies);
        kick(&q, &mut v, dt / 2.0, &self.bodies);
        for i in 1..n { let (a, b) = kepler_step(gm0, q[i], v[i], dt); q[i] = a; v[i] = b; }
        kick(&q, &mut v, dt / 2.0, &self.bodies);
        jump(&mut q, &v, dt / 2.0, &self.bodies);
        let com = r_com + v_com * dt;
        let shift = (1..n).fold(Vec3::ZERO, |s, i| s + q[i] * self.bodies[i].gm) / total;
        let star_v = (1..n).fold(Vec3::ZERO, |s, i| s + v[i] * self.bodies[i].gm) / gm0;
        self.bodies[0].r = com - shift;
        self.bodies[0].v = v_com - star_v;
        for i in 1..n {
            self.bodies[i].r = self.bodies[0].r + q[i];
            self.bodies[i].v = v[i] + v_com;
        }
    }

    pub fn invariants(&self) -> Invariants {
        let mut e = 0.0;
        let mut p = Vec3::ZERO;
        let mut l = Vec3::ZERO;
        let mut cm = Vec3::ZERO;
        let mut m = 0.0;
        for (i, b) in self.bodies.iter().enumerate() {
            e += 0.5 * b.gm * b.v.norm2();
            p += b.v * b.gm;
            l += b.r.cross(b.v) * b.gm;
            cm += b.r * b.gm;
            m += b.gm;
            for c in &self.bodies[i + 1..] {
                e -= b.gm * c.gm / (c.r - b.r).norm();
            }
        }
        Invariants { energy: e, momentum: p, angular_momentum: l, center_of_mass: cm / m }
    }

    /// Characteristic momentum scale for relative errors.
    pub fn momentum_scale(&self) -> f64 { self.bodies.iter().map(|b| b.gm * b.v.norm()).sum::<f64>().max(1e-300) }
}

fn depth_of(p: &[presets::PhysicsPreset], k: usize) -> usize {
    let mut d = 0;
    let mut cur = k;
    while let Some(o) = &p[cur].orbit {
        match p.iter().position(|q| q.name == o.parent) {
            Some(j) => { d += 1; cur = j; }
            None => break,
        }
        if d > 10 { break; }
    }
    d
}

pub fn integrator_by_name(name: &str) -> Result<Integrator, String> {
    match name {
        "yoshida" => Ok(Integrator::Yoshida4),
        "leapfrog" => Ok(Integrator::Leapfrog),
        "ias15" => Ok(Integrator::Ias15),
        "wh" => Ok(Integrator::WisdomHolman),
        _ => Err(format!("unknown integrator '{name}': yoshida, leapfrog, ias15, wh")),
    }
}

/// Gauss–Radau nodes on [0, 1], the first is the start of the step.
const IAS15_NODES: [f64; 8] = [0.0, 0.056_262_560_536_922_146, 0.180_240_691_736_892_36, 0.352_624_717_113_169_64, 0.547_153_626_330_555_4, 0.734_210_177_215_410_5, 0.885_320_946_839_095_8, 0.977_520_613_561_287_5];

/// `c[m][k]`: the coefficient of τᵏ in (τ − h₁)…(τ − h_m), which turns divided differences into polynomial coefficients.
fn ias15_coefficients() -> [[f64; 7]; 7] {
    let mut c = [[0.0; 7]; 7];
    c[0][0] = 1.0;
    for m in 1..7 {
        for k in 0..=m {
            c[m][k] = (if k > 0 { c[m - 1][k - 1] } else { 0.0 }) - IAS15_NODES[m] * c[m - 1][k];
        }
    }
    c
}

/// Exact two-body motion over time `dt`, universal variables. Works for any orbit type.
pub fn kepler_step(mu: f64, r0: Vec3, v0: Vec3, dt: f64) -> (Vec3, Vec3) {
    let (r0n, vr0) = (r0.norm(), r0.dot(v0) / r0.norm());
    let alpha = 2.0 / r0n - v0.norm2() / mu;
    let sq = mu.sqrt();
    let stumpff = |z: f64| -> (f64, f64) {
        if z > 1e-8 { let s = z.sqrt(); ((1.0 - s.cos()) / z, (s - s.sin()) / (s * z)) }
        else if z < -1e-8 { let s = (-z).sqrt(); ((s.cosh() - 1.0) / -z, (s.sinh() - s) / (s * -z)) }
        else { (0.5 - z / 24.0, 1.0 / 6.0 - z / 120.0) }
    };
    let mut x = sq * alpha.abs() * dt;
    for _ in 0..60 {
        let z = alpha * x * x;
        let (c, s) = stumpff(z);
        let f = r0n * vr0 / sq * x * x * c + (1.0 - alpha * r0n) * x * x * x * s + r0n * x - sq * dt;
        let df = r0n * vr0 / sq * x * (1.0 - z * s) + (1.0 - alpha * r0n) * x * x * c + r0n;
        let dx = f / df;
        x -= dx;
        if dx.abs() <= 1e-14 * x.abs().max(1.0) { break; }
    }
    let z = alpha * x * x;
    let (c, s) = stumpff(z);
    let f = 1.0 - x * x / r0n * c;
    let g = dt - x * x * x / sq * s;
    let r = r0 * f + v0 * g;
    let rn = r.norm();
    let fd = sq / (rn * r0n) * (z * s - 1.0) * x;
    let gd = 1.0 - x * x / rn * c;
    (r, r0 * fd + v0 * gd)
}
