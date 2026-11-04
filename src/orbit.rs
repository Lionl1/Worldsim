//! Кеплеровы элементы <-> вектор состояния.

use crate::vec3::Vec3;
use std::f64::consts::PI;

pub fn solve_kepler(m: f64, e: f64) -> f64 {
    let m = m.rem_euclid(2.0 * PI);
    let mut ea = if e < 0.8 { m } else { PI };
    for _ in 0..100 {
        let f = ea - e * ea.sin() - m;
        let d = f / (1.0 - e * ea.cos());
        ea -= d;
        if d.abs() < 1e-15 { break; }
    }
    ea
}

/// Элементы (градусы для углов) -> (r, v) в системе отсчёта родителя.
pub fn elements_to_state(mu: f64, a: f64, e: f64, i_deg: f64, node_deg: f64, argp_deg: f64, mean_anom_deg: f64) -> (Vec3, Vec3) {
    let (i, om, w, m) = (i_deg.to_radians(), node_deg.to_radians(), argp_deg.to_radians(), mean_anom_deg.to_radians());
    let ea = solve_kepler(m, e);
    let (se, ce) = ea.sin_cos();
    let b = a * (1.0 - e * e).sqrt();
    let n = (mu / a.powi(3)).sqrt();
    let rdot = n * a / (1.0 - e * ce);
    let p = Vec3::new(a * (ce - e), b * se, 0.0);
    let v = Vec3::new(-a * se * rdot / a, b * ce * rdot / a, 0.0);
    let rot = |q: Vec3| -> Vec3 {
        let (sw, cw) = w.sin_cos();
        let (so, co) = om.sin_cos();
        let (si, ci) = i.sin_cos();
        let x1 = cw * q.x - sw * q.y;
        let y1 = sw * q.x + cw * q.y;
        let y2 = ci * y1;
        let z2 = si * y1;
        Vec3::new(co * x1 - so * y2, so * x1 + co * y2, z2)
    };
    (rot(p), rot(v))
}

#[derive(Clone, Copy, Debug)]
pub struct Elements {
    pub a: f64,
    pub e: f64,
    pub i: f64,
    pub periapsis_dir: Vec3, // единичный вектор Лапласа–Рунге–Ленца
}

/// Вектор состояния -> элементы (используется для проверки законов Кеплера и прецессии).
pub fn state_to_elements(mu: f64, r: Vec3, v: Vec3) -> Elements {
    let h = r.cross(v);
    let e_vec = v.cross(h) / mu - r.unit();
    let energy = 0.5 * v.norm2() - mu / r.norm();
    Elements { a: -mu / (2.0 * energy), e: e_vec.norm(), i: (h.z / h.norm()).acos(), periapsis_dir: e_vec.unit() }
}
