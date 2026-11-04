//! Суточный ход температуры поверхности.
//! Законы: закон обратных квадратов для инсоляции, геометрия освещения (зенитный угол),
//! Стефан–Больцман для излучения, закон Фурье для теплопроводности грунта,
//! первое начало термодинамики (метод конечных объёмов сохраняет энергию дискретно),
//! парниковое противоизлучение атмосферы из серой модели Эддингтона.
//! Схема: неявная (обратный Эйлер) + Ньютон по нелинейному члену T⁴ — безусловно устойчива.

use crate::constants::SIGMA_SB;
use crate::world::World;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct ThermalOptions {
    pub latitude: f64,     // градусы
    pub declination: f64,  // склонение Солнца, градусы (сезон)
    pub steps_per_day: usize,
    pub spinup_days: usize,
    pub cells: usize,
}

impl Default for ThermalOptions {
    fn default() -> Self { ThermalOptions { latitude: 0.0, declination: 0.0, steps_per_day: 720, spinup_days: 15, cells: 80 } }
}

#[derive(Clone, Debug)]
pub struct ThermalResult {
    pub local_time_h: Vec<f64>, // местное солнечное время, часы (0..24)
    pub surface_t: Vec<f64>,
    pub t_min: f64,
    pub t_max: f64,
    pub t_mean: f64,
    pub skin_depth: f64,
    /// |ΔH − ∫(поглощено − излучено)dt| / ∫поглощено dt за последние сутки
    pub energy_residual: f64,
    pub depth_profile: Vec<(f64, f64)>,
}

fn solve_tridiag(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for i in 1..n {
        let m = b[i] - a[i] * cp[i - 1];
        cp[i] = if i < n - 1 { c[i] / m } else { 0.0 };
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() { x[i] = dp[i] - cp[i] * x[i + 1]; }
    x
}

pub fn simulate(world: &World, o: ThermalOptions) -> ThermalResult {
    let s = &world.surface;
    let (rho, cp, k, eps) = (s.density, s.heat_capacity, s.conductivity(), s.emissivity);
    let period = world.physics.solar_day.abs();
    let kappa = k / (rho * cp);
    let skin = (kappa * period / PI).sqrt();
    let n = o.cells;
    let dz = 10.0 * skin / n as f64;
    let dt = period / o.steps_per_day as f64;
    let s0 = world.insolation() * (1.0 - world.physics.bond_albedo);
    let (phi, dec) = (o.latitude.to_radians(), o.declination.to_radians());
    let t_eq = world.equilibrium_temperature();
    let f_down = SIGMA_SB * t_eq.powi(4) * 0.75 * world.greenhouse_tau(); // противоизлучение
    let q_geo = s.geothermal_flux;
    let absorbed = |t: f64| -> f64 {
        let hour = 2.0 * PI * t / period - PI; // t=0 — полночь
        let cz = phi.sin() * dec.sin() + phi.cos() * dec.cos() * hour.cos();
        s0 * cz.max(0.0)
    };
    // начальное условие: радиационное равновесие со среднесуточным потоком
    let mean_abs: f64 = (0..o.steps_per_day).map(|i| absorbed(i as f64 * dt)).sum::<f64>() / o.steps_per_day as f64;
    let t_init = ((mean_abs + f_down + q_geo) / (eps * SIGMA_SB)).powf(0.25).max(30.0);
    let mut temp = vec![t_init; n];
    let cap = rho * cp * dz; // теплоёмкость ячейки на м^2
    let g = k / dz;
    let total_steps = o.steps_per_day * (o.spinup_days + 1);
    let mut out_t = Vec::new();
    let mut out_s = Vec::new();
    let mut flux_in = 0.0;
    let mut flux_net = 0.0;
    let mut h_start = 0.0;
    for step in 0..total_steps {
        let t_new = (step + 1) as f64 * dt;
        let f_in = absorbed(t_new) + f_down;
        let mut guess = temp.clone();
        for _ in 0..8 {
            let ts = guess[0];
            let rad = eps * SIGMA_SB * ts.powi(4);
            let drad = 4.0 * eps * SIGMA_SB * ts.powi(3);
            let mut a = vec![0.0; n];
            let mut b = vec![0.0; n];
            let mut c = vec![0.0; n];
            let mut d = vec![0.0; n];
            for i in 0..n {
                b[i] = cap / dt;
                d[i] = cap / dt * temp[i];
                if i > 0 { a[i] = -g; b[i] += g; }
                if i < n - 1 { c[i] = -g; b[i] += g; }
            }
            b[0] += drad;
            d[0] += f_in - rad + drad * ts;
            d[n - 1] += q_geo;
            let new = solve_tridiag(&a, &b, &c, &d);
            let delta = (new[0] - guess[0]).abs();
            guess = new;
            if delta < 1e-10 { break; }
        }
        let last_day = step >= o.steps_per_day * o.spinup_days;
        if step == o.steps_per_day * o.spinup_days {
            h_start = temp.iter().sum::<f64>() * cap;
        }
        if last_day {
            flux_in += absorbed(t_new) * dt;
            flux_net += (f_in + q_geo - eps * SIGMA_SB * guess[0].powi(4)) * dt;
            out_t.push(((t_new / period) * 24.0) % 24.0);
            out_s.push(guess[0]);
        }
        temp = guess;
    }
    let h_end = temp.iter().sum::<f64>() * cap;
    let residual = ((h_end - h_start) - flux_net).abs() / flux_in.max(1e-30);
    let tmin = out_s.iter().cloned().fold(f64::INFINITY, f64::min);
    let tmax = out_s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let tmean = out_s.iter().sum::<f64>() / out_s.len() as f64;
    let depth_profile = temp.iter().enumerate().map(|(i, t)| ((i as f64 + 0.5) * dz, *t)).collect();
    ThermalResult { local_time_h: out_t, surface_t: out_s, t_min: tmin, t_max: tmax, t_mean: tmean, skin_depth: skin, energy_residual: residual, depth_profile }
}
