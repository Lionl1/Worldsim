//! Задача N тел: закон всемирного тяготения + опционально 1PN-поправка ОТО (Шварцшильд)
//! от центрального тела. Интеграторы: симплектические leapfrog (2-й порядок) и Йошида (4-й),
//! которые сохраняют фазовый объём и не дают векового дрейфа энергии.

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
pub enum Integrator { Leapfrog, Yoshida4 }

#[derive(Clone, Debug)]
pub struct NBody {
    pub bodies: Vec<Body>,
    pub t: f64,
    /// Индекс тела, от которого считается 1PN-поправка ОТО (обычно Солнце).
    pub relativity_center: Option<usize>,
    pub integrator: Integrator,
}

#[derive(Clone, Copy, Debug)]
pub struct Invariants {
    pub energy: f64,           // G·E  (в единицах GM, чтобы не терять точность на G)
    pub momentum: Vec3,        // G·P
    pub angular_momentum: Vec3,// G·L
    pub center_of_mass: Vec3,
}

impl NBody {
    pub fn new(bodies: Vec<Body>) -> Self {
        NBody { bodies, t: 0.0, relativity_center: None, integrator: Integrator::Yoshida4 }
    }

    /// Солнечная система (или её часть) из пресетов по элементам J2000.
    pub fn from_presets(names: &[&str]) -> Result<Self, String> {
        let mut presets_v = Vec::new();
        for n in names { presets_v.push(presets::physics(n)?); }
        let idx = |n: &str| presets_v.iter().position(|p| p.name == n);
        let mut bodies: Vec<Body> = presets_v.iter().map(|p| Body { name: p.name.clone(), gm: p.gm, r: Vec3::ZERO, v: Vec3::ZERO }).collect();
        // сначала тела вокруг корня, потом спутники (уровни иерархии)
        for pass in 0..3 {
            for (k, p) in presets_v.iter().enumerate() {
                let Some(o) = &p.orbit else { continue };
                let parent = match idx(&o.parent) {
                    Some(j) => j,
                    None => continue, // родителя нет в системе — тело остаётся в начале координат
                };
                let depth = depth_of(&presets_v, k);
                if depth != pass + 1 { continue; }
                // спутники родителя, присутствующие в системе, тоже вносят массу в барицентр
                let moons_gm: f64 = presets_v.iter().filter(|q| q.orbit.as_ref().map(|oo| oo.parent == p.name).unwrap_or(false)).map(|q| q.gm).sum();
                let mu = presets_v[parent].gm + p.gm + moons_gm;
                let (r, v) = elements_to_state(mu, o.a, o.e, o.i, o.node, o.long_peri - o.node, o.mean_long - o.long_peri);
                bodies[k].r = bodies[parent].r + r;
                bodies[k].v = bodies[parent].v + v;
                let _ = moons_gm;
            }
            // спутники этого уровня: сдвигаем родителя так, чтобы барицентр пары остался на месте
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
        self.relativity_center = Some(self.index(center).ok_or(format!("нет тела {center}"))?);
        Ok(())
    }

    /// Перевод в систему центра масс (полный импульс = 0).
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
        }
        self.t += dt;
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

    /// Характерный масштаб импульса для относительных ошибок.
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
