//! Генератор планетных систем.
//! Звезда — из начальной функции масс Крупы (или пресет), планеты — с разносом в 10–25
//! взаимных радиусах Хилла (критерий динамической устойчивости, Chambers et al. 1996),
//! радиусы — из соотношения масса–радиус (Chen & Kipping 2017), температура — из баланса излучения.
//! Каждую планету можно превратить в полноценный `World` и запустить на ней климат/баллистику.

use crate::constants::{AU, G};
use crate::nbody::{Body, NBody};
use crate::orbit::elements_to_state;
use crate::presets::{self, PhysicsPreset};
use crate::rng::Rng;
use crate::stars::{self, Star};
use crate::vec3::Vec3;
use crate::world::World;

pub const M_EARTH: f64 = 5.972_2e24;
pub const R_EARTH: f64 = 6.371e6;
pub const GM_EARTH: f64 = 3.986_004_418e14;

#[derive(Clone, Debug)]
pub struct Planet {
    pub name: String,
    pub mass: f64,   // M⊕
    pub radius: f64, // R⊕
    pub a: f64,      // а.е.
    pub e: f64,
    pub inc: f64,    // градусы
    pub node: f64,
    pub argp: f64,
    pub mean_anom: f64,
    pub kind: &'static str,
    pub t_eq: f64,
    pub in_hz: bool,
    pub albedo: f64,
}

#[derive(Clone, Debug)]
pub struct StarSystem {
    pub star: Star,
    pub planets: Vec<Planet>,
    pub hz: (f64, f64),
    pub seed: u64,
}

/// R(M) в земных единицах, непрерывная кусочно-степенная форма Chen & Kipping (2017).
pub fn mass_radius(m: f64) -> f64 {
    if m < 2.04 { m.powf(0.279) } else if m < 132.0 { 1.22 * (m / 2.04).powf(0.589) } else { 14.2 * (m / 132.0).powf(-0.044) }
}

pub fn classify(m: f64, t_eq: f64) -> &'static str {
    match m {
        m if m < 0.1 => "карликовая",
        m if m < 2.0 => if t_eq > 400.0 { "горячая каменная" } else { "каменная" },
        m if m < 10.0 => "суперземля / мини-нептун",
        m if m < 50.0 => if t_eq < 150.0 { "ледяной гигант" } else { "тёплый нептун" },
        _ => if t_eq > 1000.0 { "горячий юпитер" } else { "газовый гигант" },
    }
}

impl StarSystem {
    pub fn generate(seed: u64, star_name: Option<&str>) -> Result<StarSystem, String> {
        let mut rng = Rng::new(seed);
        let star = match star_name {
            Some(n) => stars::star(n)?,
            None => Star::main_sequence(stars::sample_kroupa(rng.f64(), rng.f64(), 0.1, 2.0)),
        };
        let hz = star.habitable_zone();
        let n = 2 + (rng.f64() * 7.0) as usize;
        let mut planets = Vec::new();
        let mut a = rng.log_uniform(0.02, 0.15) * star.mass.powf(0.5).max(0.3);
        let mut prev_m: Option<f64> = None;
        for k in 0..n {
            // ближе — мелкие, дальше — шанс на гиганта (линия льда ~2.7 а.е. · √L)
            let snow = 2.7 * star.luminosity.sqrt();
            // масса гиганта ограничена ~1% массы звезды
            let m_max = (3000.0 * star.mass).min(1000.0).max(20.0);
            let m = if a > snow && rng.f64() < 0.5 { rng.log_uniform(15.0, m_max) } else { rng.log_uniform(0.05, 12.0) };
            if let Some(pm) = prev_m {
                let delta = rng.uniform(10.0, 25.0);
                let kh = ((pm + m) * M_EARTH / (3.0 * star.mass * stars::M_SUN)).powf(1.0 / 3.0);
                // a₂ − a₁ = Δ · R_H, R_H ≈ kh·a₁ (взаимный радиус Хилла)
                a *= 1.0 + delta * kh;
            }
            prev_m = Some(m);
            let albedo: f64 = if m > 15.0 { 0.34 } else { 0.3 };
            let t_eq = star.teff * (star.radius * stars::R_SUN / (2.0 * a * AU)).sqrt() * (1.0 - albedo).powf(0.25);
            let ecc = (rng.normal() * 0.03).abs().min(0.3);
            planets.push(Planet {
                name: format!("{}", (b'b' + k as u8) as char), mass: m, radius: mass_radius(m), a, e: ecc,
                inc: (rng.normal() * 1.5).abs(), node: rng.uniform(0.0, 360.0), argp: rng.uniform(0.0, 360.0),
                mean_anom: rng.uniform(0.0, 360.0), kind: classify(m, t_eq), t_eq,
                in_hz: a >= hz.0 && a <= hz.1, albedo,
            });
            if a > 60.0 { break; }
        }
        Ok(StarSystem { star, planets, hz, seed })
    }

    /// Солнечная система как StarSystem (реальные данные, для сравнения с генератором).
    pub fn solar() -> StarSystem {
        let star = stars::star("sun").unwrap();
        let mut planets = Vec::new();
        for name in ["mercury", "venus", "earth", "mars", "jupiter"] {
            let p = presets::physics(name).unwrap();
            let o = p.orbit.clone().unwrap();
            let m = p.gm / GM_EARTH;
            let a = o.a / AU;
            let t_eq = star.teff * (stars::R_SUN / (2.0 * o.a)).sqrt() * (1.0 - p.bond_albedo).powf(0.25);
            planets.push(Planet { name: name.into(), mass: m, radius: p.radius / R_EARTH, a, e: o.e, inc: o.i, node: o.node,
                argp: o.long_peri - o.node, mean_anom: o.mean_long - o.long_peri, kind: classify(m, t_eq), t_eq,
                in_hz: false, albedo: p.bond_albedo });
        }
        let hz = star.habitable_zone();
        for p in &mut planets { p.in_hz = p.a >= hz.0 && p.a <= hz.1; }
        StarSystem { star, planets, hz, seed: 0 }
    }

    pub fn nbody(&self) -> NBody {
        let gm_star = self.star.gm();
        let mut bodies = vec![Body { name: "star".into(), gm: gm_star, r: Vec3::ZERO, v: Vec3::ZERO }];
        for p in &self.planets {
            let gm = p.mass * GM_EARTH;
            let (r, v) = elements_to_state(gm_star + gm, p.a * AU, p.e, p.inc, p.node, p.argp, p.mean_anom);
            bodies.push(Body { name: p.name.clone(), gm, r, v });
        }
        let mut s = NBody::new(bodies);
        s.to_barycentric();
        s
    }

    /// Превратить планету в мир: физика из массы/радиуса, атмосфера и грунт — по типу и температуре.
    pub fn world(&self, idx: usize) -> Result<World, String> {
        let p = &self.planets[idx];
        let gm = p.mass * GM_EARTH;
        let radius = p.radius * R_EARTH;
        // приливный захват: время ~ a⁶ — грубо считаем захваченными планеты ближе 0.1·√L а.е.
        let locked = p.a < 0.1 * self.star.luminosity.sqrt().max(0.3);
        let year = 2.0 * std::f64::consts::PI * ((p.a * AU).powi(3) / (self.star.gm() + gm)).sqrt();
        let rot = if locked { year } else { 86_400.0 * (0.5 + p.mass.sqrt()) };
        let solar_day = if locked { f64::INFINITY } else { 1.0 / (1.0 / rot - 1.0 / year) };
        let phys = PhysicsPreset {
            name: format!("{}-{}", self.star.name, p.name), gm, radius, rotation_period: rot,
            solar_day: if solar_day.is_finite() { solar_day.abs() } else { year * 1e3 },
            j2: 0.0, bond_albedo: p.albedo, obliquity: if locked { 0.0 } else { 23.0 }, dipole_moment: 0.0,
            moment_of_inertia: 0.33, has_surface: p.mass < 10.0, luminosity: 0.0, orbit: None,
        };
        let (atmo, surf) = if p.mass > 15.0 {
            (Some("jupiter"), "earth")
        } else if p.t_eq > 330.0 && p.mass > 0.5 {
            (Some("venus"), "venus")
        } else if p.in_hz && p.mass > 0.3 {
            (Some("earth"), "earth")
        } else if p.mass > 0.08 {
            (Some("mars"), "mars")
        } else {
            (None, "moon")
        };
        let ap = match atmo { Some(a) => presets::atmosphere(a)?, None => None };
        World::from_parts(phys, ap, presets::surface(surf)?, self.star.luminosity_w(), self.star.gm(), p.a * AU)
    }

    pub fn total_planet_mass_kg(&self) -> f64 { self.planets.iter().map(|p| p.mass * M_EARTH).sum() }
}

pub fn star_mass_kg(s: &Star) -> f64 { s.gm() / G }
