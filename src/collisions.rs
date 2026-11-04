//! Столкновения: слияние с сохранением импульса, разрушение тел (Стюарт и Лейнхардт), масштабирование кратеров (Коллинз и др.), предел Роша.
//! Это масштабные законы, а не расчёт прочности конкретного тела.

use crate::constants::G;
use crate::vec3::Vec3;
use std::f64::consts::PI;

/// Абсолютно неупругое слияние: масса, скорость и энергия, ушедшая в тепло.
pub fn merge(m1: f64, v1: Vec3, m2: f64, v2: Vec3) -> (f64, Vec3, f64) {
    let m = m1 + m2;
    let v = (v1 * m1 + v2 * m2) * (1.0 / m);
    let ke = |mm: f64, u: Vec3| 0.5 * mm * u.dot(u);
    (m, v, ke(m1, v1) + ke(m2, v2) - ke(m, v))
}

/// Удельная энергия катастрофического разрушения Q*_RD, Дж/кг: половина массы разлетается. Законы гравитационного режима.
/// `c_star` ≈ 1.9 для однородных тел по Стюарту и Лейнхардту (2009).
pub fn disruption_energy(total_mass: f64, density: f64, c_star: f64) -> f64 {
    let r = (3.0 * total_mass / (4.0 * PI * density)).cbrt();
    c_star * 4.0 / 5.0 * PI * density * G * r * r
}

/// Доля массы в крупнейшем остатке (универсальный закон); Q — удельная энергия столкновения.
pub fn largest_remnant_fraction(q: f64, q_star: f64) -> f64 {
    let x = q / q_star;
    if x < 1.8 { (1.0 - 0.5 * x).clamp(0.0, 1.0) } else { (0.1 * (x / 1.8).powf(-1.5)).clamp(0.0, 1.0) }
}

/// Удельная энергия столкновения Q_R, Дж/кг: кинетическая энергия в системе центра масс на единицу массы.
pub fn impact_energy(m1: f64, m2: f64, v: f64) -> f64 { 0.5 * (m1 * m2 / (m1 + m2)) * v * v / (m1 + m2) }

/// Исход столкновения по энергии: слияние, частичное разрушение или катастрофа.
pub fn outcome(q: f64, q_star: f64) -> &'static str {
    let x = q / q_star;
    if x < 0.1 { "слияние" } else if x < 1.0 { "частичное разрушение" } else if x < 1.8 { "катастрофическое разрушение" } else { "сверхкатастрофа" }
}

/// Диаметр кратера, м: оценка по Коллинзу, Мелошу и Марквику (2005). Переходный кратер, простой ×1.25, сложный — их поправка.
pub fn crater_diameter(rho_imp: f64, rho_tgt: f64, d_imp: f64, v: f64, g: f64, angle_deg: f64) -> (f64, f64) {
    let tc = 1.161 * (rho_imp / rho_tgt).cbrt() * d_imp.powf(0.78) * v.powf(0.44) * g.powf(-0.22) * angle_deg.to_radians().sin().cbrt();
    let simple = 1.25 * tc;
    let dc = 3200.0 * 9.81 / g; // переход простой–сложный масштабируется как 1/g
    let fin = if simple < dc { simple } else { 1.17 * tc.powf(1.13) / dc.powf(0.13) };
    (tc, fin)
}

/// Предел Роша для жидкого спутника, м.
pub fn roche_limit(primary_radius: f64, rho_primary: f64, rho_satellite: f64) -> f64 { 2.44 * primary_radius * (rho_primary / rho_satellite).cbrt() }
