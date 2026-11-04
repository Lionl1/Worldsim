//! Climate: seasonal latitudinal energy balance model (Budyko–Sellers–North class),
//! generalized to any world worldsim.
//!
//!   C ∂T/∂t = Q(φ,t)·(1 − α(T)) − OLR(T) + ∂/∂x [ D (1 − x²) ∂T/∂x ],   x = sin φ
//!
//!  * Q — daily mean insolation from orbital mechanics (eccentricity, axial tilt, perihelion),
//!  * OLR = σT⁴/(1 + ¾τ) — the same gray Eddington atmosphere as in `World`,
//!  * α(T) — ice-albedo feedback (water ice at 263 K or CO₂ frost at 148 K),
//!  * D — meridional heat transport, scaled with the atmospheric column mass (P/g),
//!  * C — heat capacity: mixed ocean layer + atmosphere + soil.
//! The implicit finite-volume scheme conserves energy discretely (this is checked).

use crate::constants::SIGMA_SB;
use crate::orbit::solve_kepler;
use crate::world::World;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct ClimateOptions {
    pub bands: usize,
    pub steps_per_year: usize,
    pub years: usize,
    pub ocean_fraction: Option<f64>,
    pub solar_scale: f64,     // stellar luminosity factor
    pub tau_scale: f64,       // greenhouse depth factor (roughly “more CO₂”)
    pub obliquity: Option<f64>,
    pub eccentricity: Option<f64>,
    pub perihelion: Option<f64>, // longitude of perihelion from the vernal equinox, degrees
    pub initial_t: Option<f64>,  // K; None means the radiative equilibrium of the world. A cold start gives a “snowball Earth”
    pub samples_per_year: usize,
}

impl Default for ClimateOptions {
    fn default() -> Self {
        ClimateOptions { bands: 90, steps_per_year: 360, years: 30, ocean_fraction: None, solar_scale: 1.0, tau_scale: 1.0,
            obliquity: None, eccentricity: None, perihelion: None, initial_t: None, samples_per_year: 48 }
    }
}

#[derive(Clone, Debug)]
pub struct ClimateResult {
    pub lat: Vec<f64>,              // band centers, degrees
    pub time_frac: Vec<f64>,        // fraction of a year
    pub temp: Vec<Vec<f64>>,        // [time][latitude], K
    pub annual_mean: Vec<f64>,      // by latitude
    pub global_mean: f64,
    pub ice_edge: f64,              // ice edge latitude (annual mean), degrees; 90 means no ice
    pub energy_residual: f64,       // |ΔH − ∫(ASR−OLR)| / ∫ASR over the last year
    pub toa_imbalance: f64,         // ⟨ASR − OLR⟩, W/m² over the last year
    pub d: f64,
    pub heat_capacity: f64,
    pub ice_temperature: f64,
}

/// Daily mean insolation at latitude φ and solar longitude λs (radians), S at mean distance.
pub fn daily_insolation(s: f64, phi: f64, lambda_s: f64, obliq: f64, ecc: f64, peri: f64) -> f64 {
    let dec = (obliq.sin() * lambda_s.sin()).asin();
    let rho2 = ((1.0 + ecc * (lambda_s - peri).cos()) / (1.0 - ecc * ecc)).powi(2);
    let c = -phi.tan() * dec.tan();
    let h0 = if c >= 1.0 { 0.0 } else if c <= -1.0 { PI } else { c.acos() };
    (s / PI * rho2 * (h0 * phi.sin() * dec.sin() + phi.cos() * dec.cos() * h0.sin())).max(0.0)
}

fn solve_tridiag(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for i in 1..n {
        let m = b[i] - a[i] * cp[i - 1];
        cp[i] = c[i] / m;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() { x[i] = dp[i] - cp[i] * x[i + 1]; }
    x
}

pub fn simulate(w: &World, o: &ClimateOptions) -> ClimateResult {
    let n = o.bands;
    let dx = 2.0 / n as f64;
    let xs: Vec<f64> = (0..n).map(|j| -1.0 + (j as f64 + 0.5) * dx).collect();
    let phis: Vec<f64> = xs.iter().map(|x| x.asin()).collect();
    let s = w.insolation() * o.solar_scale;
    let tau = w.greenhouse_tau() * o.tau_scale;
    let olr_fac = SIGMA_SB / (1.0 + 0.75 * tau);
    let is_earth_air = w.atmosphere.as_ref().map(|a| a.preset.composition.iter().any(|(g, x)| g == "N2" && *x > 0.5)).unwrap_or(false);
    let is_co2 = w.atmosphere.as_ref().map(|a| a.preset.composition.iter().any(|(g, x)| g == "CO2" && *x > 0.5)).unwrap_or(false);
    let ocean = o.ocean_fraction.unwrap_or(if w.physics.name == "earth" { 0.7 } else { 0.0 });
    let has_water = ocean > 0.0 || is_earth_air;
    let ice_t = if has_water { 263.0 } else if is_co2 { 148.0 } else { 0.0 };
    let a_warm = if w.physics.name == "earth" { 0.29 } else { w.physics.bond_albedo };
    let a_ice: f64 = 0.62;
    // zenith effect: reflection is higher with a low Sun (∝ P₂(x), the area mean does not change)
    let zen = if w.atmosphere.is_some() { 0.12 } else { 0.0 };
    let albedo = |t: f64, x: f64| {
        let aw = a_warm + zen * 0.5 * (3.0 * x * x - 1.0);
        if ice_t > 0.0 { aw + (a_ice.max(aw) - aw) * 0.5 * (1.0 - ((t - ice_t) / 2.0).tanh()) } else { aw }
    };
    // heat capacity
    let g = w.g_surface();
    let year = orbital_period(w);
    let (p0, cp_air) = w.atmosphere.as_ref().map(|a| (a.preset.surface_pressure, a.cp_molar / a.molar_mass)).unwrap_or((0.0, 0.0));
    let c_air = cp_air * p0 / g;
    let c_ocean = 4.18e6 * 50.0;
    let c_ground = w.surface.thermal_inertia * (year / PI).sqrt() * 0.1; // effective seasonal soil depth
    let heat_c = ocean * c_ocean + (1.0 - ocean) * c_ground + c_air;
    // transport: Earth D=0.55 W/(m²K), scale ∝ column mass
    let col = p0 / g;
    let d = if p0 > 0.0 { (0.40 * col / (101_325.0 / 9.81)).clamp(1e-4, 50.0) } else { 0.0 };
    let obl = o.obliquity.unwrap_or(w.physics.obliquity).to_radians();
    let obl = if obl > PI / 2.0 { PI - obl } else { obl };
    let (ecc, peri) = match &w.physics.orbit {
        Some(orb) if orb.parent == "sun" => (orb.e, (orb.long_peri + 180.0).to_radians()),
        _ => (0.0167, 282.9f64.to_radians()),
    };
    let ecc = o.eccentricity.unwrap_or(ecc);
    let peri = o.perihelion.map(|p| p.to_radians()).unwrap_or(peri);
    let dt = year / o.steps_per_year as f64;
    // conductances on faces
    let k_face: Vec<f64> = (0..=n).map(|f| { let xf = -1.0 + f as f64 * dx; d * (1.0 - xf * xf) / (dx * dx) }).collect();
    let mut t = vec![o.initial_t.unwrap_or_else(|| w.mean_surface_temperature() * (o.solar_scale).powf(0.25)); n];
    let total = o.steps_per_year * o.years;
    let mut out_t = Vec::new();
    let mut out_time = Vec::new();
    let mut annual = vec![0.0; n];
    let (mut asr_int, mut net_int) = (0.0, 0.0);
    let mut h_start = 0.0;
    let sample_every = (o.steps_per_year / o.samples_per_year.max(1)).max(1);
    for step in 0..total {
        let mean_anom = 2.0 * PI * (step as f64 + 0.5) / o.steps_per_year as f64;
        // mean anomaly → true → solar longitude; M=0 at perihelion
        let ea = solve_kepler(mean_anom, ecc);
        let nu = 2.0 * (((1.0 + ecc) / (1.0 - ecc)).sqrt() * (ea / 2.0).tan()).atan();
        let lambda_s = nu + peri;
        let last_year = step >= o.steps_per_year * (o.years - 1);
        if step == o.steps_per_year * (o.years - 1) { h_start = heat_c * t.iter().sum::<f64>() / n as f64; }
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut c = vec![0.0; n];
        let mut rhs = vec![0.0; n];
        let mut asr = vec![0.0; n];
        for j in 0..n {
            let q = daily_insolation(s, phis[j], lambda_s, obl, ecc, peri);
            asr[j] = q * (1.0 - albedo(t[j], xs[j]));
            let olr = olr_fac * t[j].powi(4);
            let dolr = 4.0 * olr_fac * t[j].powi(3);
            b[j] = heat_c / dt + dolr + k_face[j] + k_face[j + 1];
            a[j] = -k_face[j];
            c[j] = -k_face[j + 1];
            rhs[j] = heat_c / dt * t[j] + asr[j] - olr + dolr * t[j];
        }
        let tn = solve_tridiag(&a, &b, &c, &rhs);
        if last_year {
            for j in 0..n {
                let olr_lin = olr_fac * t[j].powi(4) + 4.0 * olr_fac * t[j].powi(3) * (tn[j] - t[j]);
                asr_int += asr[j] * dt / n as f64;
                net_int += (asr[j] - olr_lin) * dt / n as f64;
                annual[j] += tn[j] / o.steps_per_year as f64;
            }
            let k = step - o.steps_per_year * (o.years - 1);
            if k % sample_every == 0 {
                out_t.push(tn.clone());
                out_time.push(k as f64 / o.steps_per_year as f64);
            }
        }
        t = tn;
    }
    let h_end = heat_c * t.iter().sum::<f64>() / n as f64;
    let global = annual.iter().sum::<f64>() / n as f64;
    // ice edge: the lowest latitude (by magnitude) with annual mean T < T_ice
    let ice_edge = if ice_t > 0.0 {
        phis.iter().zip(&annual).filter(|(_, &tt)| tt < ice_t).map(|(p, _)| p.abs().to_degrees()).fold(90.0, f64::min)
    } else { 90.0 };
    ClimateResult {
        lat: phis.iter().map(|p| p.to_degrees()).collect(), time_frac: out_time, temp: out_t, annual_mean: annual, global_mean: global,
        ice_edge, energy_residual: ((h_end - h_start) - net_int).abs() / asr_int.max(1e-30),
        toa_imbalance: net_int / year, d, heat_capacity: heat_c, ice_temperature: ice_t,
    }
}

/// Orbital period around the star (Kepler third law; for a moon, the period of its parent).
pub fn orbital_period(w: &World) -> f64 { 2.0 * PI * (w.star_distance.powi(3) / w.star_gm).sqrt() }
