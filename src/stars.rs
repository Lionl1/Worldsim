//! Звёзды: соотношения главной последовательности, закон Стефана–Больцмана,
//! спектр Планка → цвет (через функции сложения цветов CIE 1931), закон смещения Вина,
//! обитаемая зона (Kopparapu et al. 2014).

use crate::constants::{C, H_PLANCK, K_B, L_SUN};
use std::f64::consts::PI;

pub const R_SUN: f64 = 6.957e8;
pub const M_SUN: f64 = 1.988_47e30;
pub const GM_SUN: f64 = 1.327_124_400_18e20;
pub const T_SUN: f64 = 5772.0;

#[derive(Clone, Debug)]
pub struct Star {
    pub name: String,
    pub mass: f64,       // M☉
    pub radius: f64,     // R☉
    pub luminosity: f64, // L☉
    pub teff: f64,       // К
}

impl Star {
    /// Звезда главной последовательности из массы (эмпирические степенные законы).
    pub fn main_sequence(mass: f64) -> Star {
        let l = if mass < 0.43 { 0.23 * mass.powf(2.3) } else if mass < 2.0 { mass.powi(4) } else if mass < 55.0 { 1.4 * mass.powf(3.5) } else { 32_000.0 * mass };
        let r = if mass < 1.0 { mass.powf(0.8) } else { mass.powf(0.57) };
        // L = 4πR²σT⁴  ⇒  T = T☉ (L/R²)^¼
        let t = T_SUN * (l / (r * r)).powf(0.25);
        Star { name: format!("MS {mass:.2} M☉"), mass, radius: r, luminosity: l, teff: t }
    }

    pub fn gm(&self) -> f64 { GM_SUN * self.mass }
    pub fn luminosity_w(&self) -> f64 { self.luminosity * L_SUN }

    pub fn spectral_class(&self) -> &'static str {
        match self.teff {
            t if t >= 30_000.0 => "O",
            t if t >= 10_000.0 => "B",
            t if t >= 7_500.0 => "A",
            t if t >= 6_000.0 => "F",
            t if t >= 5_200.0 => "G",
            t if t >= 3_700.0 => "K",
            _ => "M",
        }
    }

    /// Время жизни на главной последовательности, лет: ~10¹⁰ · M/L.
    pub fn lifetime_years(&self) -> f64 { 1.0e10 * self.mass / self.luminosity }

    /// Обитаемая зона (внутренняя: runaway greenhouse; внешняя: maximum greenhouse), а.е.
    pub fn habitable_zone(&self) -> (f64, f64) {
        let t = (self.teff.clamp(2600.0, 7200.0)) - 5780.0;
        let seff = |c: [f64; 5]| c[0] + c[1] * t + c[2] * t * t + c[3] * t.powi(3) + c[4] * t.powi(4);
        let inner = seff([1.107, 1.332e-4, 1.58e-8, -8.308e-12, -1.931e-15]);
        let outer = seff([0.356, 6.171e-5, 1.698e-9, -3.198e-12, -5.575e-16]);
        ((self.luminosity / inner).sqrt(), (self.luminosity / outer).sqrt())
    }

    /// Цвет абсолютно чёрного тела при T_eff в sRGB (0..1).
    pub fn rgb(&self) -> [f64; 3] { blackbody_rgb(self.teff) }
}

pub const STAR_NAMES: &[&str] = &["sun", "proxima", "barnard", "trappist1", "alpha_cen_a", "sirius_a", "vega", "betelgeuse"];

/// Измеренные параметры известных звёзд (не из соотношений).
pub fn star(name: &str) -> Result<Star, String> {
    let s = |n: &str, m, r, l, t| Star { name: n.into(), mass: m, radius: r, luminosity: l, teff: t };
    Ok(match name {
        "sun" => s("Солнце", 1.0, 1.0, 1.0, 5772.0),
        "proxima" => s("Проксима Центавра", 0.122, 0.154, 0.00155, 3042.0),
        "barnard" => s("Звезда Барнарда", 0.144, 0.196, 0.0035, 3134.0),
        "trappist1" => s("TRAPPIST-1", 0.0898, 0.1192, 0.000553, 2566.0),
        "alpha_cen_a" => s("α Центавра A", 1.1, 1.22, 1.52, 5790.0),
        "sirius_a" => s("Сириус A", 2.063, 1.711, 25.4, 9940.0),
        "vega" => s("Вега", 2.135, 2.36, 40.12, 9602.0),
        "betelgeuse" => s("Бетельгейзе (сверхгигант)", 16.5, 764.0, 126_000.0, 3600.0),
        _ => {
            if let Some(m) = name.strip_prefix("ms:") {
                return Ok(Star::main_sequence(m.parse().map_err(|_| format!("ms:<масса>, получено {name}"))?));
            }
            return Err(format!("неизвестная звезда '{name}', доступны: {STAR_NAMES:?} или ms:<масса в M☉>"));
        }
    })
}

/// Спектральная плотность энергетической яркости B_λ(T), Вт/(м²·ср·м).
pub fn planck(lambda: f64, t: f64) -> f64 {
    2.0 * H_PLANCK * C * C / lambda.powi(5) / ((H_PLANCK * C / (lambda * K_B * t)).exp() - 1.0)
}

/// ∫ π B_λ dλ — должно совпасть с σT⁴ (проверка закона Стефана–Больцмана).
pub fn planck_integrated(t: f64) -> f64 {
    // интеграл по x = hc/(λkT): π·∫B dλ = 2πk⁴T⁴/(h³c²) ∫ x³/(eˣ−1) dx
    let n = 20_000;
    let (a, b) = (1e-6, 60.0);
    let h = (b - a) / n as f64;
    let f = |x: f64| x.powi(3) / (x.exp() - 1.0);
    let mut s = f(a) + f(b);
    for i in 1..n { s += f(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 }; }
    let integral = s * h / 3.0;
    2.0 * PI * K_B.powi(4) * t.powi(4) / (H_PLANCK.powi(3) * C * C) * integral
}

/// Длина волны максимума B_λ, найденная численно (для проверки закона Вина).
pub fn wien_peak(t: f64) -> f64 {
    let (mut lo, mut hi) = (1e-8, 1e-3);
    for _ in 0..200 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if planck(m1, t) < planck(m2, t) { lo = m1 } else { hi = m2 }
    }
    0.5 * (lo + hi)
}

fn lobe(x: f64, mu: f64, s1: f64, s2: f64) -> f64 {
    let s = if x < mu { s1 } else { s2 };
    (-0.5 * ((x - mu) / s).powi(2)).exp()
}

/// CIE 1931 2° (аналитическая аппроксимация Wyman, Sloan, Shirley 2013), λ в нм.
fn cie_xyz(nm: f64) -> [f64; 3] {
    [
        1.056 * lobe(nm, 599.8, 37.9, 31.0) + 0.362 * lobe(nm, 442.0, 16.0, 26.7) - 0.065 * lobe(nm, 501.1, 20.4, 26.2),
        0.821 * lobe(nm, 568.8, 46.9, 40.5) + 0.286 * lobe(nm, 530.9, 16.3, 31.1),
        1.217 * lobe(nm, 437.0, 11.8, 36.0) + 0.681 * lobe(nm, 459.0, 26.0, 13.8),
    ]
}

pub fn blackbody_rgb(t: f64) -> [f64; 3] {
    let mut xyz = [0.0; 3];
    let mut nm = 380.0;
    while nm <= 780.0 {
        let b = planck(nm * 1e-9, t);
        let c = cie_xyz(nm);
        for k in 0..3 { xyz[k] += b * c[k]; }
        nm += 2.0;
    }
    let lin = [
        3.2406 * xyz[0] - 1.5372 * xyz[1] - 0.4986 * xyz[2],
        -0.9689 * xyz[0] + 1.8758 * xyz[1] + 0.0415 * xyz[2],
        0.0557 * xyz[0] - 0.2040 * xyz[1] + 1.0570 * xyz[2],
    ];
    let mx = lin.iter().cloned().fold(f64::MIN, f64::max).max(1e-300);
    let mut out = [0.0; 3];
    for k in 0..3 {
        let v = (lin[k] / mx).max(0.0);
        out[k] = if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    }
    out
}

/// Начальная функция масс Крупы (2001): выборка массы в [m_lo, m_hi] M☉.
pub fn sample_kroupa(u: f64, v: f64, m_lo: f64, m_hi: f64) -> f64 {
    // кусочно-степенная: α=1.3 (0.08–0.5), α=2.3 (>0.5); выборка через выбор сегмента + обратную функцию
    let seg = |a: f64, lo: f64, hi: f64| -> f64 { if (a - 1.0).abs() < 1e-9 { (hi / lo).ln() } else { (hi.powf(1.0 - a) - lo.powf(1.0 - a)) / (1.0 - a) } };
    let lo1 = m_lo.max(0.08);
    let hi1 = m_hi.min(0.5);
    let w1 = if hi1 > lo1 { seg(1.3, lo1, hi1) } else { 0.0 };
    let lo2 = m_lo.max(0.5);
    let w2 = if m_hi > lo2 { 0.5 * seg(2.3, lo2, m_hi) } else { 0.0 }; // 0.5 = непрерывность в 0.5 M☉
    let inv = |a: f64, lo: f64, hi: f64, x: f64| -> f64 {
        let (p, q) = (lo.powf(1.0 - a), hi.powf(1.0 - a));
        (p + x * (q - p)).powf(1.0 / (1.0 - a))
    };
    if u * (w1 + w2) < w1 { inv(1.3, lo1, hi1, v) } else { inv(2.3, lo2, m_hi, v) }
}
