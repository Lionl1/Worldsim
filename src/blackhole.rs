//! Чёрные дыры и сильная гравитация: Шварцшильд и Керр, геодезические, излучение Хокинга, гравитационные волны.
//! Орбитальные формулы — в геометрических единицах (G = c = M = 1). Рендер — геодезические Шварцшильда с тонким диском.

use crate::constants::{C, G, H_PLANCK, K_B, YEAR};
use crate::quantum::M_SUN;
use std::f64::consts::PI;

const HBAR: f64 = H_PLANCK / (2.0 * PI);

/// Гравитационный радиус 2GM/c², м.
pub fn schwarzschild_radius(m_kg: f64) -> f64 { 2.0 * G * m_kg / (C * C) }

/// Эффективный потенциал времениподобной орбиты: V² = (1 − 2/r)(1 + L²/r²).
fn circular_l2(r: f64) -> f64 { r * r / (r - 3.0) }
/// d²V²/dr² на круговой орбите радиуса r: где становится нулём, там ISCO.
fn stability(r: f64) -> f64 {
    let l2 = circular_l2(r);
    let v2 = |x: f64| (1.0 - 2.0 / x) * (1.0 + l2 / (x * x));
    let h = r * 1e-3;
    (v2(r + h) - 2.0 * v2(r) + v2(r - h)) / (h * h)
}

/// ISCO Шварцшильда численно, из устойчивости орбиты (теория: 6).
pub fn isco_numeric() -> f64 {
    let (mut lo, mut hi) = (3.5, 30.0);
    for _ in 0..80 { let m = 0.5 * (lo + hi); if stability(m) > 0.0 { hi = m } else { lo = m } }
    0.5 * (lo + hi)
}

/// ISCO Керра, формула Бардина–Пресса–Теукольского; a ∈ (−1, 1), a > 0 — орбита по вращению.
pub fn kerr_isco(a: f64) -> f64 {
    let z1 = 1.0 + (1.0 - a * a).powf(1.0 / 3.0) * ((1.0 + a).powf(1.0 / 3.0) + (1.0 - a).powf(1.0 / 3.0));
    let z2 = (3.0 * a * a + z1 * z1).sqrt();
    3.0 + z2 - a.signum() * ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt()
}

/// Удельная энергия круговой орбиты Керра.
pub fn kerr_energy(r: f64, a: f64) -> f64 {
    let s = r.sqrt();
    (r * s - 2.0 * s + a) / (r.powf(0.75) * (r * s - 3.0 * s + 2.0 * a).sqrt())
}

/// ISCO Керра независимо: минимум энергии круговой орбиты. Сверка с замкнутой формулой.
pub fn kerr_isco_numeric(a: f64) -> f64 {
    let rh = 1.0 + (1.0 - a * a).sqrt();
    let rp = 2.0 * (1.0 + (2.0 / 3.0 * (-a).acos()).cos());
    let (mut lo, mut hi) = (rp.max(rh) + 1e-6, 12.0);
    let gr = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..200 {
        let (c, d) = (hi - gr * (hi - lo), lo + gr * (hi - lo));
        if kerr_energy(c, a) < kerr_energy(d, a) { hi = d } else { lo = c }
    }
    0.5 * (lo + hi)
}

/// Доля массы покоя, которая излучается при падении до ISCO.
pub fn radiative_efficiency(a: f64) -> f64 { 1.0 - kerr_energy(kerr_isco(a), a) }

/// Горизонт и прямая фотонная орбита Керра.
pub fn kerr_horizon(a: f64) -> f64 { 1.0 + (1.0 - a * a).sqrt() }
pub fn kerr_photon_orbit(a: f64) -> f64 { 2.0 * (1.0 + (2.0 / 3.0 * (-a).acos()).cos()) }

/// Шаг РК4 для уравнения фотона u'' = −u + 3u² (u = 1/r, φ — угол).
fn photon_step(u: f64, v: f64, h: f64) -> (f64, f64) {
    let a = |u: f64| -u + 3.0 * u * u;
    let (k1u, k1v) = (v, a(u));
    let (k2u, k2v) = (v + h / 2.0 * k1v, a(u + h / 2.0 * k1u));
    let (k3u, k3v) = (v + h / 2.0 * k2v, a(u + h / 2.0 * k2u));
    let (k4u, k4v) = (v + h * k3v, a(u + h * k3u));
    (u + h / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u), v + h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v))
}

/// Луч с прицельным параметром b из бесконечности: Some(полный поворот φ), None — захват.
fn photon_turn(b: f64, h: f64) -> Option<f64> {
    // старт далеко от дыры: первый интеграл (du/dφ)² = 1/b² − u² + 2u³ даёт точную скорость
    let u0 = 1e-4 / b;
    let (mut u, mut v, mut phi) = (u0, (1.0 / (b * b) - u0 * u0 + 2.0 * u0.powi(3)).sqrt(), u0 * b);
    for _ in 0..4_000_000 {
        let (nu, nv) = photon_step(u, v, h);
        if nu >= 0.5 { return None; } // горизонт
        if nu <= 0.0 && v < 0.0 {
            return Some(phi + h * u / (u - nu));
        }
        u = nu; v = nv; phi += h;
    }
    None
}

/// Отклонение луча, рад; None — луч захвачен.
pub fn deflection(b: f64) -> Option<f64> { photon_turn(b, 2e-4).map(|p| p - PI) }

/// Критический прицельный параметр (радиус тени), численно, в единицах M. Теория: 3√3.
pub fn shadow_radius_numeric() -> f64 {
    let (mut lo, mut hi) = (4.0, 7.0);
    for _ in 0..40 { let m = 0.5 * (lo + hi); if photon_turn(m, 1e-3).is_none() { lo = m } else { hi = m } }
    0.5 * (lo + hi)
}

/// Фотонная сфера численно: неустойчивая круговая орбита (теория: r = 3). Экстремум b²(r) = r³/(r − 2).
pub fn photon_sphere_numeric() -> f64 {
    let f = |r: f64| r * r * r / (r - 2.0);
    let (mut lo, mut hi) = (2.1, 10.0);
    let gr = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..200 { let (c, d) = (hi - gr * (hi - lo), lo + gr * (hi - lo)); if f(c) < f(d) { hi = d } else { lo = c } }
    0.5 * (lo + hi)
}

/// Прецессия перицентра за оборот, рад, численно по Бине: u'' = −u + 1/L² + 3u².
pub fn precession_per_orbit(a: f64, e: f64) -> f64 {
    let p = a * (1.0 - e * e);
    // в метрике Шварцшильда L² = p²/(p − 3 − e²)
    let l2 = p * p / (p - 3.0 - e * e);
    let acc = |u: f64| -u + 1.0 / l2 + 3.0 * u * u;
    let (mut u, mut v, mut phi) = ((1.0 + e) / p, 0.0f64, 0.0);
    let h = 1e-3;
    // до следующего перицентра: v меняет знак с + на −, а старт v = 0
    let mut was_pos = false;
    loop {
        let (k1u, k1v) = (v, acc(u));
        let (k2u, k2v) = (v + h / 2.0 * k1v, acc(u + h / 2.0 * k1u));
        let (k3u, k3v) = (v + h / 2.0 * k2v, acc(u + h / 2.0 * k2u));
        let (k4u, k4v) = (v + h * k3v, acc(u + h * k3u));
        let (nu, nv) = (u + h / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u), v + h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v));
        if was_pos && nv <= 0.0 && v > 0.0 { return phi + h * v / (v - nv) - 2.0 * PI; }
        if nv > 0.0 { was_pos = true; }
        u = nu; v = nv; phi += h;
        if phi > 100.0 { return f64::NAN; }
    }
}

/// Температура Хокинга, К.
pub fn hawking_temperature(m_kg: f64) -> f64 { HBAR * C.powi(3) / (8.0 * PI * G * m_kg * K_B) }
/// Энтропия Бекенштейна–Хокинга, Дж/К.
pub fn bh_entropy(m_kg: f64) -> f64 { 4.0 * PI * G * m_kg * m_kg * K_B / (HBAR * C) }
/// Время испарения, лет (только фотоны).
pub fn evaporation_years(m_kg: f64) -> f64 { 5120.0 * PI * G * G * m_kg.powi(3) / (HBAR * C.powi(4)) / YEAR }
/// Приливное ускорение на расстоянии 1 м от центра тела у горизонта, 1/с².
pub fn horizon_tide(m_kg: f64) -> f64 { let r = schwarzschild_radius(m_kg); 2.0 * G * m_kg / r.powi(3) }

/// Множитель Петерса для эллиптической орбиты.
pub fn peters_enhancement(e: f64) -> f64 { (1.0 + 73.0 / 24.0 * e * e + 37.0 / 96.0 * e.powi(4)) / (1.0 - e * e).powf(3.5) }

/// Скорость убывания орбитального периода двойной из-за гравитационных волн, с/с.
pub fn period_decay(m1: f64, m2: f64, pb: f64, e: f64) -> f64 {
    let tsun = G * M_SUN / C.powi(3);
    let (a, b) = (m1 / M_SUN, m2 / M_SUN);
    -192.0 * PI / 5.0 * (tsun * 2.0 * PI / pb).powf(5.0 / 3.0) * peters_enhancement(e) * a * b / (a + b).powf(1.0 / 3.0)
}

/// Время слияния круговой двойной, с: формула Петерса.
pub fn merger_time(m1: f64, m2: f64, a: f64) -> f64 { 5.0 * C.powi(5) * a.powi(4) / (256.0 * G.powi(3) * m1 * m2 * (m1 + m2)) }

/// То же численно: интегрируем da/dt = −(64/5) G³m₁m₂(m₁+m₂)/(c⁵a³).
pub fn merger_time_numeric(m1: f64, m2: f64, a0: f64) -> f64 {
    let k = 64.0 / 5.0 * G.powi(3) * m1 * m2 * (m1 + m2) / C.powi(5);
    // по переменной x = a⁴ процесс линеен, поэтому интегрируем по a c мелким шагом по ln a
    let n = 200_000;
    let mut t = 0.0;
    let dl = (a0 / (a0 * 1e-4)).ln() / n as f64;
    let mut a = a0;
    for _ in 0..n { let a1 = a * (-dl).exp(); let am = 0.5 * (a + a1); t += (a - a1) * am.powi(3) / k; a = a1; }
    t
}

/// Рендер: чёрная дыра с тонким диском и фоном из звёзд, луч трассируется по геодезической.
pub struct RenderOpts { pub w: usize, pub h: usize, pub incl_deg: f64, pub r_in: f64, pub r_out: f64, pub distance: f64, pub fov_deg: f64, pub doppler: bool }

fn hash3(a: f64, b: f64) -> f64 { let s = (a * 127.1 + b * 311.7).sin() * 43758.5453; s - s.floor() }

/// Цвет чёрного тела приближённо, по температуре в относительных единицах.
fn temp_color(t: f64) -> [f64; 3] {
    let k = (t * 5200.0).clamp(1200.0, 30000.0) / 100.0;
    let r = if k <= 66.0 { 1.0 } else { (1.292 * (k - 60.0).powf(-0.1332)).min(1.0) };
    let g = if k <= 66.0 { (0.39 * k.ln() - 0.63).clamp(0.0, 1.0) } else { (1.129 * (k - 60.0).powf(-0.0755)).min(1.0) };
    let b = if k >= 66.0 { 1.0 } else if k <= 19.0 { 0.0 } else { (0.543 * (k - 10.0).ln() - 1.196).clamp(0.0, 1.0) };
    [r, g, b]
}

pub fn render(o: &RenderOpts, threads: usize) -> Vec<u8> {
    let inc = o.incl_deg.to_radians();
    // камера в плоскости x–z, нормаль диска — ось z
    let cam = [inc.sin(), 0.0, inc.cos()];
    let right = [0.0, 1.0, 0.0];
    let up = [cam[1] * right[2] - cam[2] * right[1], cam[2] * right[0] - cam[0] * right[2], cam[0] * right[1] - cam[1] * right[0]];
    let tan = (o.fov_deg.to_radians() / 2.0).tan();
    let mut out = vec![0u8; o.w * o.h * 3];
    let threads = threads.max(1);
    let rows = o.h.div_ceil(threads);
    std::thread::scope(|sc| {
        for (ci, chunk) in out.chunks_mut(rows * o.w * 3).enumerate() {
            sc.spawn(move || {
                for (j, row) in chunk.chunks_mut(o.w * 3).enumerate() {
                    let y = ci * rows + j;
                    for i in 0..o.w {
                        let sx = (2.0 * (i as f64 + 0.5) / o.w as f64 - 1.0) * tan * o.w as f64 / o.h as f64;
                        let sy = (1.0 - 2.0 * (y as f64 + 0.5) / o.h as f64) * tan;
                        let c = trace(o, &cam, &right, &up, sx, sy);
                        // экспозиция: яркий край диска не выгорает в плоское белое пятно
                        for k in 0..3 { row[i * 3 + k] = ((1.0 - (-c[k]).exp()).powf(1.0 / 2.2) * 255.0) as u8; }
                    }
                }
            });
        }
    });
    out
}

fn trace(o: &RenderOpts, cam: &[f64; 3], right: &[f64; 3], up: &[f64; 3], sx: f64, sy: f64) -> [f64; 3] {
    // начальное направление луча от камеры: в плоскости (e1 = радиальный, e2 = поперечный)
    let t = (sx * sx + sy * sy).sqrt().max(1e-9);
    let e2 = [right[0] * sx / t + up[0] * sy / t, right[1] * sx / t + up[1] * sy / t, right[2] * sx / t + up[2] * sy / t];
    let r0 = o.distance;
    let (mut u, mut v) = (1.0 / r0, 1.0 / r0 / t); // du/dφ при выходе на бесконечность приближённо
    let mut phi = 0.0f64;
    let h = 0.012;
    let mut col = [0.0; 3];
    let mut alpha = 1.0;
    let dz = |p: f64| cam[2] * p.cos() + e2[2] * p.sin();
    let mut z_prev = dz(0.0);
    for _ in 0..6000 {
        let (nu, nv) = photon_step(u, v, h);
        let z = dz(phi + h);
        // пересечение плоскости диска
        if z_prev * z <= 0.0 && z_prev != z {
            let f = z_prev / (z_prev - z);
            let ru = u + (nu - u) * f;
            let r = 1.0 / ru.max(1e-9);
            if r > o.r_in && r < o.r_out {
                let p = phi + h * f;
                // азимут диска
                let (xh, yh) = (cam[0] * p.cos() + e2[0] * p.sin(), cam[1] * p.cos() + e2[1] * p.sin());
                // тонкий диск: T ∝ (ISCO-поправка) r^(−3/4); Доплер и красное смещение относительно наблюдателя
                let temp = (r.powf(-3.0) * (1.0 - (o.r_in / r).sqrt()).max(0.0)).powf(0.25) * 6.5;
                let om = r.powf(-1.5);
                // скорость диска на луч к наблюдателю
                let bphi = if o.doppler { (-yh * cam[0] + xh * cam[1]) * om * r } else { 0.0 };
                let g = (1.0 - 3.0 / r).max(0.01).sqrt() / (1.0 - bphi).max(0.05);
                let bri = (temp * g / 0.72).powi(4);
                let tc = temp_color(temp * g);
                // спиральные неоднородности и мягкий внешний край: диск не обрывается резкой линией
                let swirl = 0.55 + 0.45 * ((60.0 / r.sqrt() + 5.0 * yh.atan2(xh)).sin() * (23.0 / r.sqrt() - 3.0 * yh.atan2(xh)).cos());
                let edge = ((o.r_out - r) / (0.45 * o.r_out)).clamp(0.0, 1.0);
                let opacity = edge * edge * (3.0 - 2.0 * edge);
                for k in 0..3 { col[k] += alpha * tc[k] * bri * swirl * opacity * 0.75; }
                alpha *= 1.0 - 0.85 * opacity;
            }
        }
        z_prev = z;
        if nu >= 0.5 { return col; } // горизонт
        if nu <= 1e-4 && nv < 0.0 {
            // выход: фон из звёзд по итоговому направлению луча
            let p = phi + h;
            let dir = [-cam[0] * p.sin() + e2[0] * p.cos(), -cam[1] * p.sin() + e2[1] * p.cos(), -cam[2] * p.sin() + e2[2] * p.cos()];
            let (la, lo) = (dir[2].asin(), dir[1].atan2(dir[0]));
            // звезда — маленькая точка внутри ячейки неба, а не вся ячейка
            let (fa, fb) = (la * 90.0, lo * 90.0);
            let (ca, cb) = (fa.floor(), fb.floor());
            let hs = hash3(ca, cb);
            if hs > 0.965 {
                let (px, py) = (0.5 + 0.35 * (hash3(cb, ca) - 0.5), 0.5 + 0.35 * (hash3(ca + 7.0, cb - 3.0) - 0.5));
                // ячейки на широте сжаты по долготе: исправляем, чтобы звёзды были круглыми
                let d = ((fa - ca - px).powi(2) + ((fb - cb - py) * la.cos().abs().max(0.05)).powi(2)).sqrt();
                let b = (1.0 - d / 0.11).max(0.0) * (hs - 0.965) * 22.0 * alpha;
                for c in &mut col { *c += b; }
            }
            return col;
        }
        u = nu; v = nv; phi += h;
    }
    col
}
