//! Celestial body presets. Three independent layers that can be mixed:
//!   * `PhysicsPreset`    — GM, radius, rotation, J2, albedo, magnetic dipole, orbit
//!   * `AtmospherePreset` — pressure, composition, temperature profile, greenhouse optical depth
//!   * `SurfacePreset`    — crust chemistry (oxides), soil thermophysics
//!
//! Sources: NASA Planetary Fact Sheets, JPL (Standish, elements J2000), US Std Atmosphere 1976,
//! Rudnick & Gao 2014, Taylor & McLennan 2009, Apollo/Diviner/Curiosity. Mean values.

use crate::constants::{AU, DAY};

#[derive(Clone, Debug)]
pub struct Orbit {
    pub parent: String,
    pub a: f64,         // m
    pub e: f64,
    pub i: f64,         // degrees
    pub node: f64,      // Ω, degrees
    pub long_peri: f64, // ϖ = Ω + ω, degrees
    pub mean_long: f64, // L = ϖ + M, degrees (J2000)
}

#[derive(Clone, Debug)]
pub struct PhysicsPreset {
    pub name: String,
    pub gm: f64,              // m^3/s^2
    pub radius: f64,          // m
    pub rotation_period: f64, // sidereal, s (<0 — retrograde)
    pub solar_day: f64,       // synodic day, s
    pub j2: f64,
    pub bond_albedo: f64,
    pub obliquity: f64,       // degrees
    pub dipole_moment: f64,   // A·m^2
    pub moment_of_inertia: f64,
    pub has_surface: bool,
    pub luminosity: f64,      // W (for stars)
    pub orbit: Option<Orbit>,
}

#[derive(Clone, Debug)]
pub struct AtmospherePreset {
    pub name: String,
    pub surface_pressure: f64,         // Pa
    pub composition: Vec<(String, f64)>, // mole fractions
    pub layers: Vec<(f64, f64)>,       // (base altitude m, dT/dz K/m)
    pub reference_temperature: f64,    // K
    pub greenhouse_tau: f64,           // IR optical depth (gray Eddington model)
    pub reference_g: f64,              // g home body: τ scales ∝ column mass P/g
    pub fixed_temperature: bool,
    pub top: f64,                      // top of the grid, m
}

#[derive(Clone, Debug)]
pub struct SurfacePreset {
    pub name: String,
    pub oxides_wt: Vec<(String, f64)>, // mass.%
    pub density: f64,                  // kg/m^3
    pub heat_capacity: f64,            // J/(kg·K)
    pub thermal_inertia: f64,          // sqrt(kρc)
    pub emissivity: f64,
    pub geothermal_flux: f64,          // W/m^2
    pub description: String,
}

impl SurfacePreset {
    pub fn conductivity(&self) -> f64 {
        self.thermal_inertia.powi(2) / (self.density * self.heat_capacity)
    }
}

fn d(x: f64) -> f64 { x * DAY }
fn s(x: &str) -> String { x.to_string() }
fn comp(v: &[(&str, f64)]) -> Vec<(String, f64)> { v.iter().map(|(k, x)| (k.to_string(), *x)).collect() }

pub const PHYSICS_NAMES: &[&str] = &["sun", "mercury", "venus", "earth", "moon", "mars", "jupiter"];
pub const ATMOSPHERE_NAMES: &[&str] = &["none", "earth", "mars", "venus", "jupiter"];
pub const SURFACE_NAMES: &[&str] = &["earth", "moon", "moon_mare", "mars", "venus", "mercury"];

fn orbit(parent: &str, a: f64, e: f64, i: f64, node: f64, lp: f64, ml: f64) -> Option<Orbit> {
    Some(Orbit { parent: s(parent), a, e, i, node, long_peri: lp, mean_long: ml })
}

pub fn physics(name: &str) -> Result<PhysicsPreset, String> {
    let p = |name: &str, gm, radius, rot, sday, j2, alb, obl, dip, moi, surf, orb| PhysicsPreset {
        name: s(name), gm, radius, rotation_period: rot, solar_day: sday, j2, bond_albedo: alb,
        obliquity: obl, dipole_moment: dip, moment_of_inertia: moi, has_surface: surf,
        luminosity: 0.0, orbit: orb,
    };
    Ok(match name {
        "sun" => PhysicsPreset { luminosity: 3.828e26, ..p("sun", 1.327_124_400_18e20, 695_700e3, d(25.38), d(25.38), 2.2e-7, 0.0, 7.25, 0.0, 0.070, false, None) },
        "mercury" => p("mercury", 2.203_186_8e13, 2_439.4e3, d(58.6462), d(175.9421), 5.03e-5, 0.088, 0.034, 4.0e19, 0.346, true,
                       orbit("sun", 0.387_099_27 * AU, 0.205_635_93, 7.004_979_02, 48.330_765_93, 77.457_796_28, 252.250_323_50)),
        "venus" => p("venus", 3.248_585_92e14, 6_051.8e3, -d(243.0226), d(116.75), 4.458e-6, 0.76, 177.36, 0.0, 0.33, true,
                     orbit("sun", 0.723_335_66 * AU, 0.006_776_72, 3.394_676_05, 76.679_842_55, 131.602_467_18, 181.979_099_50)),
        "earth" => p("earth", 3.986_004_418e14, 6_371.0e3, 86_164.0905, DAY, 1.082_63e-3, 0.306, 23.44, 8.0e22, 0.3307, true,
                     orbit("sun", 1.000_002_61 * AU, 0.016_711_23, -0.000_015_31, 0.0, 102.937_681_93, 100.464_571_66)),
        "moon" => p("moon", 4.904_869_5e12, 1_737.4e3, d(27.321_661), d(29.530_589), 2.034e-4, 0.11, 1.54, 0.0, 0.3929, true,
                    orbit("earth", 384_399e3, 0.0549, 5.145, 125.08, 208.31, 218.32)),
        "mars" => p("mars", 4.282_837e13, 3_389.5e3, 88_642.66, 88_775.244, 1.960_45e-3, 0.25, 25.19, 0.0, 0.3644, true,
                    orbit("sun", 1.523_710_34 * AU, 0.093_394_10, 1.849_691_42, 49.559_538_91, -23.943_629_59, -4.553_432_05)),
        "jupiter" => p("jupiter", 1.266_865_34e17, 69_911e3, 35_730.0, 35_733.0, 1.4736e-2, 0.343, 3.13, 1.56e27, 0.254, false,
                       orbit("sun", 5.202_887_00 * AU, 0.048_386_24, 1.304_396_95, 100.473_909_09, 14.728_479_83, 34.396_440_51)),
        _ => return Err(format!("unknown physics preset '{name}', available: {PHYSICS_NAMES:?}")),
    })
}

pub fn atmosphere(name: &str) -> Result<Option<AtmospherePreset>, String> {
    let a = |name: &str, p0, c: &[(&str, f64)], layers: &[(f64, f64)], t0, tau, g, top| AtmospherePreset {
        name: s(name), surface_pressure: p0, composition: comp(c), layers: layers.to_vec(),
        reference_temperature: t0, greenhouse_tau: tau, reference_g: g, fixed_temperature: false, top,
    };
    Ok(Some(match name {
        "none" | "vacuum" | "moon" | "mercury" => return Ok(None),
        "earth" => a("earth", 101_325.0, &[("N2", 0.78084), ("O2", 0.20946), ("Ar", 0.00934), ("CO2", 0.00042)],
                     &[(0.0, -0.0065), (11e3, 0.0), (20e3, 0.001), (32e3, 0.0028), (47e3, 0.0), (51e3, -0.0028), (71e3, -0.002), (84.852e3, 0.0)],
                     288.15, 0.875, 9.81, 150e3),
        "mars" => a("mars", 636.0, &[("CO2", 0.951), ("N2", 0.0259), ("Ar", 0.0194), ("O2", 0.00161), ("CO", 0.0006)],
                    &[(0.0, -0.0025), (40e3, -0.001), (80e3, 0.0)], 210.0, 0.02, 3.73, 150e3),
        "venus" => a("venus", 9.2e6, &[("CO2", 0.965), ("N2", 0.035)],
                     &[(0.0, -0.0077), (60e3, -0.003), (90e3, 0.0)], 737.0, 140.0, 8.87, 150e3),
        "jupiter" => AtmospherePreset { fixed_temperature: true,
                     ..a("jupiter", 1.0e5, &[("H2", 0.862), ("He", 0.136), ("CH4", 0.002)], &[(0.0, -0.0018), (30e3, 0.0)], 165.0, 0.0, 24.79, 300e3) },
        _ => return Err(format!("unknown atmosphere preset '{name}', available: {ATMOSPHERE_NAMES:?}")),
    }))
}

pub fn surface(name: &str) -> Result<SurfacePreset, String> {
    let sp = |name: &str, ox: &[(&str, f64)], rho, cp, ti, eps, q, desc: &str| SurfacePreset {
        name: s(name), oxides_wt: comp(ox), density: rho, heat_capacity: cp, thermal_inertia: ti,
        emissivity: eps, geothermal_flux: q, description: s(desc),
    };
    Ok(match name {
        "earth" => sp("earth", &[("SiO2", 60.6), ("TiO2", 0.7), ("Al2O3", 15.9), ("FeO", 6.7), ("MnO", 0.1), ("MgO", 4.7), ("CaO", 6.4), ("Na2O", 3.1), ("K2O", 1.8), ("P2O5", 0.1)],
                      1800.0, 1000.0, 1200.0, 0.95, 0.087, "Average continental crust (Rudnick & Gao 2014), dry soil"),
        "moon" => sp("moon", &[("SiO2", 45.0), ("TiO2", 0.6), ("Al2O3", 24.6), ("FeO", 5.9), ("MgO", 7.5), ("CaO", 15.9), ("Na2O", 0.5)],
                     1500.0, 600.0, 55.0, 0.95, 0.016, "Lunar highland regolith (Apollo 16), thermophysics after Diviner"),
        "moon_mare" => sp("moon_mare", &[("SiO2", 45.4), ("TiO2", 3.9), ("Al2O3", 14.9), ("FeO", 14.1), ("MgO", 9.2), ("CaO", 11.8), ("Na2O", 0.6)],
                          1500.0, 600.0, 55.0, 0.95, 0.021, "Lunar mare regolith (basalts)"),
        "mars" => sp("mars", &[("SiO2", 49.3), ("TiO2", 0.98), ("Al2O3", 10.5), ("FeO", 18.2), ("MnO", 0.36), ("MgO", 9.06), ("CaO", 6.92), ("Na2O", 2.97), ("K2O", 0.45), ("P2O5", 0.9)],
                     1600.0, 800.0, 250.0, 0.95, 0.019, "Average Martian crust (Taylor & McLennan 2009), thermal inertia after TES"),
        "venus" => sp("venus", &[("SiO2", 45.1), ("TiO2", 1.59), ("Al2O3", 15.8), ("FeO", 9.3), ("MgO", 11.4), ("CaO", 7.1), ("K2O", 0.1), ("Na2O", 2.0)],
                      2800.0, 1000.0, 1500.0, 0.9, 0.05, "Venus basalts (Venera 14, Vega-2)"),
        "mercury" => sp("mercury", &[("SiO2", 49.0), ("MgO", 18.0), ("Al2O3", 13.0), ("CaO", 7.0), ("FeO", 1.5), ("Na2O", 5.0), ("TiO2", 0.8), ("SO3", 5.0)],
                        1500.0, 600.0, 60.0, 0.95, 0.01, "Mercury surface (MESSENGER XRS, rough)"),
        _ => return Err(format!("unknown surface preset '{name}', available: {SURFACE_NAMES:?}")),
    })
}

// ---------------------------------------------------------------------------
// Composition: oxides -> elements (stoichiometry, mass conservation)

fn atomic_mass(el: &str) -> f64 {
    match el {
        "O" => 15.999, "Si" => 28.085, "Ti" => 47.867, "Al" => 26.982, "Fe" => 55.845, "Mn" => 54.938,
        "Mg" => 24.305, "Ca" => 40.078, "Na" => 22.990, "K" => 39.098, "P" => 30.974, "Cr" => 51.996,
        "S" => 32.06, _ => f64::NAN,
    }
}

/// (element, atoms of the element, atoms O)
pub fn oxide_formula(ox: &str) -> Option<(&'static str, f64, f64)> {
    Some(match ox {
        "SiO2" => ("Si", 1.0, 2.0), "TiO2" => ("Ti", 1.0, 2.0), "Al2O3" => ("Al", 2.0, 3.0),
        "Cr2O3" => ("Cr", 2.0, 3.0), "FeO" => ("Fe", 1.0, 1.0), "MnO" => ("Mn", 1.0, 1.0),
        "MgO" => ("Mg", 1.0, 1.0), "CaO" => ("Ca", 1.0, 1.0), "Na2O" => ("Na", 2.0, 1.0),
        "K2O" => ("K", 2.0, 1.0), "P2O5" => ("P", 2.0, 5.0), "SO3" => ("S", 1.0, 3.0),
        _ => return None,
    })
}

pub fn normalized(v: &[(String, f64)]) -> Vec<(String, f64)> {
    let total: f64 = v.iter().map(|x| x.1).sum();
    v.iter().map(|(k, x)| (k.clone(), x / total)).collect()
}

/// Elemental composition of the crust, mass fractions (sum = 1).
pub fn oxides_to_elements(oxides: &[(String, f64)]) -> Result<Vec<(String, f64)>, String> {
    let mut out: Vec<(String, f64)> = Vec::new();
    let mut add = |el: &str, m: f64| match out.iter_mut().find(|e| e.0 == el) {
        Some(e) => e.1 += m,
        None => out.push((el.to_string(), m)),
    };
    for (ox, w) in normalized(oxides) {
        let (el, n_el, n_o) = oxide_formula(&ox).ok_or(format!("unknown oxide {ox}"))?;
        let m_el = n_el * atomic_mass(el);
        let m_o = n_o * atomic_mass("O");
        add(el, w * m_el / (m_el + m_o));
        add("O", w * m_o / (m_el + m_o));
    }
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok(out)
}
