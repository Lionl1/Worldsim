//! Заряженная частица в дипольном магнитном поле планеты (радиационные пояса).
//! Законы: сила Лоренца, специальная теория относительности (γ), поле магнитного диполя.
//! Интегратор Бориса (релятивистский) сохраняет |p| в чистом магнитном поле точно.

use crate::constants::{C, E_CHARGE, MEV, MU_0, M_ALPHA, M_E, M_P};
use crate::vec3::Vec3;
use crate::world::World;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct Species { pub mass: f64, pub charge: f64 }

pub fn species(name: &str) -> Result<Species, String> {
    Ok(match name {
        "proton" | "p" => Species { mass: M_P, charge: E_CHARGE },
        "electron" | "e" => Species { mass: M_E, charge: -E_CHARGE },
        "alpha" => Species { mass: M_ALPHA, charge: 2.0 * E_CHARGE },
        _ => return Err(format!("неизвестная частица '{name}' (proton, electron, alpha)")),
    })
}

/// Поле диполя с моментом m = −M ẑ (как у Земли: северный магнитный полюс у южного географического).
pub fn dipole_field(moment: f64, r: Vec3) -> Vec3 {
    let m = Vec3::new(0.0, 0.0, -moment);
    let rn = r.norm();
    let rh = r / rn;
    (rh * (3.0 * m.dot(rh)) - m) * (MU_0 / (4.0 * PI * rn.powi(3)))
}

#[derive(Clone, Copy, Debug)]
pub struct ParticleLaunch {
    pub l_shell: f64,       // экваториальное расстояние в радиусах планеты
    pub energy_mev: f64,    // кинетическая энергия
    pub pitch_angle: f64,   // градусы, угол между v и B на экваторе
}

#[derive(Clone, Debug)]
pub struct ParticleResult {
    pub t: Vec<f64>,
    pub pos: Vec<Vec3>,
    pub max_abs_lat: f64,          // широта зеркальной точки, численно
    pub mirror_lat_theory: f64,    // из сохранения магнитного момента μ
    pub energy_drift: f64,         // |ΔE_k|/E_k
    pub drift_longitude: f64,      // градусы, на которые частица сместилась по долготе
    pub hit_surface: bool,
    pub trapped: bool,
}

/// Теоретическая зеркальная широта: sin²α_eq = cos⁶λ / sqrt(1 + 3 sin²λ).
pub fn mirror_latitude(pitch_deg: f64) -> f64 {
    let s2 = pitch_deg.to_radians().sin().powi(2);
    let f = |l: f64| l.cos().powi(6) / (1.0 + 3.0 * l.sin().powi(2)).sqrt() - s2;
    let (mut lo, mut hi) = (0.0, PI / 2.0);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 { lo = mid } else { hi = mid }
    }
    lo.to_degrees()
}

pub fn simulate(world: &World, sp: Species, l: ParticleLaunch, duration: f64, samples: usize) -> ParticleResult {
    let rr = world.physics.radius;
    let moment = world.physics.dipole_moment;
    let r0 = Vec3::new(l.l_shell * rr, 0.0, 0.0);
    let ek = l.energy_mev * MEV;
    let mc2 = sp.mass * C * C;
    let gamma = 1.0 + ek / mc2;
    let p = sp.mass * C * (gamma * gamma - 1.0).sqrt();
    let a = l.pitch_angle.to_radians();
    // на экваторе B ∥ z; перпендикулярная компонента — вдоль y
    let mut u = Vec3::new(0.0, a.sin(), a.cos()) * (p / sp.mass); // u = γv
    let mut r = r0;
    let mut t = 0.0;
    let mut out_t = Vec::new();
    let mut out_p = Vec::new();
    let mut max_lat: f64 = 0.0;
    let mut hit = false;
    let mut lon_unwrapped = 0.0;
    let mut prev_lon = 0.0;
    let sample_every = duration / samples.max(1) as f64;
    let mut next = 0.0;
    while t < duration {
        let b = dipole_field(moment, r);
        let bn = b.norm();
        let g = (1.0 + u.norm2() / (C * C)).sqrt();
        let gyro = if bn > 0.0 { 2.0 * PI * g * sp.mass / (sp.charge.abs() * bn) } else { duration / 1e4 };
        let dt = (gyro / 40.0).min(duration - t).max(1e-12);
        // Борис: поворот u в поле B
        let tv = b * (sp.charge * dt / (2.0 * g * sp.mass));
        let sv = tv * (2.0 / (1.0 + tv.norm2()));
        let up = u + u.cross(tv);
        u += up.cross(sv);
        let g1 = (1.0 + u.norm2() / (C * C)).sqrt();
        r += u * (dt / g1);
        t += dt;
        let lat = (r.z / r.norm()).asin().to_degrees();
        max_lat = max_lat.max(lat.abs());
        let lon = r.y.atan2(r.x);
        let mut dl = lon - prev_lon;
        if dl > PI { dl -= 2.0 * PI } else if dl < -PI { dl += 2.0 * PI }
        lon_unwrapped += dl;
        prev_lon = lon;
        if t >= next { out_t.push(t); out_p.push(r); next += sample_every; }
        if r.norm() < rr { hit = true; break; }
        if r.norm() > 50.0 * l.l_shell * rr { break; }
    }
    let g_end = (1.0 + u.norm2() / (C * C)).sqrt();
    let ek_end = (g_end - 1.0) * mc2;
    let trapped = !hit && r.norm() < 2.0 * l.l_shell * rr && moment > 0.0;
    ParticleResult {
        t: out_t, pos: out_p, max_abs_lat: max_lat, mirror_lat_theory: mirror_latitude(l.pitch_angle),
        energy_drift: ((ek_end - ek) / ek).abs(), drift_longitude: lon_unwrapped.to_degrees(), hit_surface: hit, trapped,
    }
}
