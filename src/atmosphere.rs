//! Атмосфера в гидростатическом равновесии: dP/dz = -ρ g(z), ρ = P M / (R T),
//! g(z) = GM/(R+z)^2 (закон тяготения), T(z) — кусочно-линейный профиль.
//! Таблица строится численным интегрированием для любого тела, на которое «поставлена» атмосфера.

use crate::constants::{gas, R_GAS};
use crate::presets::AtmospherePreset;

#[derive(Clone, Debug)]
pub struct Atmosphere {
    pub preset: AtmospherePreset,
    pub molar_mass: f64, // кг/моль
    pub cp_molar: f64,   // Дж/(моль·К)
    pub gamma: f64,
    pub t0: f64,
    dz: f64,
    temp: Vec<f64>,
    ln_p: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct AirState {
    pub temperature: f64,
    pub pressure: f64,
    pub density: f64,
    pub sound_speed: f64,
}

impl Atmosphere {
    pub fn build(preset: &AtmospherePreset, gm: f64, radius: f64, t0: f64) -> Result<Self, String> {
        let mut m = 0.0;
        let mut cp = 0.0;
        let total: f64 = preset.composition.iter().map(|c| c.1).sum();
        for (g, x) in &preset.composition {
            let (mm, c) = gas(g).ok_or(format!("неизвестный газ {g}"))?;
            m += x / total * mm;
            cp += x / total * c;
        }
        let gamma = cp / (cp - R_GAS); // Майер: c_p - c_v = R
        let dz = 50.0;
        let n = (preset.top / dz).ceil() as usize + 1;
        let tz = |z: f64| -> f64 {
            let mut t = t0;
            let layers = &preset.layers;
            for (k, &(base, lapse)) in layers.iter().enumerate() {
                let next = layers.get(k + 1).map(|l| l.0).unwrap_or(f64::INFINITY);
                if z <= base { break; }
                t += lapse * (z.min(next) - base);
            }
            t.max(20.0)
        };
        let g = |z: f64| gm / (radius + z).powi(2);
        let mut temp = Vec::with_capacity(n);
        let mut ln_p = Vec::with_capacity(n);
        let mut lp = preset.surface_pressure.ln();
        for i in 0..n {
            let z = i as f64 * dz;
            if i > 0 {
                // правило Симпсона для d lnP/dz = -M g / (R T)
                let f = |z: f64| -m * g(z) / (R_GAS * tz(z));
                lp += dz / 6.0 * (f(z - dz) + 4.0 * f(z - dz / 2.0) + f(z));
            }
            temp.push(tz(z));
            ln_p.push(lp);
        }
        Ok(Atmosphere { preset: preset.clone(), molar_mass: m, cp_molar: cp, gamma, t0, dz, temp, ln_p })
    }

    pub fn top(&self) -> f64 { (self.temp.len() - 1) as f64 * self.dz }

    pub fn at(&self, z: f64) -> AirState {
        if z > self.top() {
            let t = *self.temp.last().unwrap();
            return AirState { temperature: t, pressure: 0.0, density: 0.0, sound_speed: self.sound(t) };
        }
        let zz = z.max(0.0);
        let f = zz / self.dz;
        let i = (f.floor() as usize).min(self.temp.len() - 2);
        let w = f - i as f64;
        let t = self.temp[i] * (1.0 - w) + self.temp[i + 1] * w;
        let p = (self.ln_p[i] * (1.0 - w) + self.ln_p[i + 1] * w).exp();
        AirState { temperature: t, pressure: p, density: p * self.molar_mass / (R_GAS * t), sound_speed: self.sound(t) }
    }

    fn sound(&self, t: f64) -> f64 { (self.gamma * R_GAS * t / self.molar_mass).sqrt() }

    /// Шкала высот у поверхности H = RT/(Mg).
    pub fn scale_height(&self, g: f64) -> f64 { R_GAS * self.t0 / (self.molar_mass * g) }

    /// Масса столба атмосферы на м^2 (интеграл ρ dz) — для проверки P0 = ∫ρ g dz.
    pub fn column_weight(&self, gm: f64, radius: f64) -> f64 {
        let mut s = 0.0;
        for i in 0..self.temp.len() - 1 {
            let z = (i as f64 + 0.5) * self.dz;
            let a = self.at(z);
            s += a.density * gm / (radius + z).powi(2) * self.dz;
        }
        s
    }
}
