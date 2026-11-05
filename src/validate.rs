//! Check suite: each compares the simulation with an analytic solution,
//! a conservation law or an observed value. `worldsim validate` prints a table.

use crate::ballistics::{self, Launch, Options, Projectile};
use crate::constants::{C, DAY, YEAR};
use crate::magneto::{self, ParticleLaunch};
use crate::nbody::{Body, NBody};
use crate::orbit::{elements_to_state, state_to_elements};
use crate::presets;
use crate::thermal::{self, ThermalOptions};
use crate::vec3::Vec3;
use crate::world::World;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Check {
    pub law: &'static str,
    pub test: String,
    pub value: String,
    pub expected: String,
    pub pass: bool,
}

fn rel(a: f64, b: f64) -> f64 { ((a - b) / b).abs() }

fn chk(law: &'static str, test: &str, value: f64, expected: f64, tol_rel: f64, unit: &str) -> Check {
    Check { law, test: test.into(), value: format!("{value:.6} {unit}"), expected: format!("{expected:.6} {unit} ±{:.2}%", tol_rel * 100.0), pass: rel(value, expected) <= tol_rel }
}
fn chk_max(law: &'static str, test: &str, value: f64, max: f64) -> Check {
    Check { law, test: test.into(), value: format!("{value:.3e}"), expected: format!("< {max:.0e}"), pass: value.abs() < max }
}

pub fn conservation_solar_system(years: f64) -> Vec<Check> {
    let mut s = NBody::from_presets(&["sun", "mercury", "venus", "earth", "moon", "mars", "jupiter"]).unwrap();
    let i0 = s.invariants();
    let dt = 0.05 * DAY;
    let steps = (years * YEAR / dt) as usize;
    let (mut de, mut dl): (f64, f64) = (0.0, 0.0);
    for _ in 0..steps {
        s.step(dt);
        let i = s.invariants();
        de = de.max(rel(i.energy, i0.energy));
        dl = dl.max((i.angular_momentum - i0.angular_momentum).norm() / i0.angular_momentum.norm());
    }
    let i = s.invariants();
    let dp = (i.momentum - i0.momentum).norm() / s.momentum_scale();
    // The Moon must stay in Earth orbit
    let (e, m) = (s.index("earth").unwrap(), s.index("moon").unwrap());
    let d_em = (s.bodies[m].r - s.bodies[e].r).norm();
    vec![
        chk_max("Energy conservation", &format!("Solar system, {years} yr, max |ΔE/E|"), de, 1e-9),
        chk_max("Momentum conservation", "|ΔP| / Σ|p_i|", dp, 1e-12),
        chk_max("Angular momentum conservation", "max |ΔL|/|L|", dl, 1e-12),
        Check { law: "Gravity (hierarchy)", test: "The Moon stayed in Earth orbit".into(), value: format!("{:.0} km", d_em / 1e3), expected: "356 000–407 000 km".into(), pass: (3.5e8..4.1e8).contains(&d_em) },
    ]
}

pub fn kepler() -> Vec<Check> {
    let earth = presets::physics("earth").unwrap();
    let a = earth.radius + 400e3;
    let mu = earth.gm;
    let (r, v) = elements_to_state(mu, a, 0.1, 51.6, 30.0, 40.0, 0.0);
    let mut s = NBody::new(vec![
        Body { name: "earth".into(), gm: mu, r: Vec3::ZERO, v: Vec3::ZERO },
        Body { name: "sat".into(), gm: 1e-6, r, v },
    ]);
    let t_theory = 2.0 * PI * (a.powi(3) / mu).sqrt();
    let n = 2000;
    for _ in 0..n { s.step(t_theory / n as f64); }
    let back = (s.bodies[1].r - s.bodies[0].r - r).norm();
    vec![
        Check { law: "3rd Kepler law", test: "orbit e=0.1, h=400 km: return after T=2π√(a³/μ)".into(), value: format!("{back:.2e} m"), expected: "< 1 m".into(), pass: back < 1.0 },
    ]
}

/// Mercury perihelion precession from the 1PN GR correction (≈ 43″ per century).
pub fn mercury_precession() -> Vec<Check> {
    let run = |gr: bool| -> f64 {
        let mut s = NBody::from_presets(&["sun", "mercury"]).unwrap();
        if gr { s.enable_relativity("sun").unwrap(); }
        let p0 = state_to_elements(s.bodies[0].gm + s.bodies[1].gm, s.bodies[1].r - s.bodies[0].r, s.bodies[1].v - s.bodies[0].v).periapsis_dir;
        // a whole number of revolutions: short-period 1PN terms cancel
        let mu = s.bodies[0].gm + s.bodies[1].gm;
        let a = state_to_elements(mu, s.bodies[1].r - s.bodies[0].r, s.bodies[1].v - s.bodies[0].v).a;
        let period = 2.0 * PI * (a.powi(3) / mu).sqrt();
        let orbits = 10usize;
        let per = 8000usize;
        let dt = period / per as f64;
        for _ in 0..orbits * per { s.step(dt); }
        let years = orbits as f64 * period / YEAR;
        let el = state_to_elements(s.bodies[0].gm + s.bodies[1].gm, s.bodies[1].r - s.bodies[0].r, s.bodies[1].v - s.bodies[0].v);
        let ang = p0.cross(el.periapsis_dir).z.atan2(p0.dot(el.periapsis_dir));
        ang / years * 100.0 // rad/century
    };
    let d = (run(true) - run(false)).to_degrees() * 3600.0;
    let m = presets::physics("mercury").unwrap();
    let sun = presets::physics("sun").unwrap();
    let o = m.orbit.unwrap();
    let per_orbit = 6.0 * PI * sun.gm / (C * C * o.a * (1.0 - o.e * o.e));
    let period = 2.0 * PI * (o.a.powi(3) / sun.gm).sqrt();
    let theory = per_orbit * (100.0 * YEAR / period) * 180.0 / PI * 3600.0;
    vec![chk("GR (1PN)", "Mercury perihelion precession, ″/century", d, theory, 0.02, "″")]
}

pub fn atmosphere() -> Vec<Check> {
    let e = World::preset("earth").unwrap();
    let ea = e.atmosphere.as_ref().unwrap();
    let m = World::preset("mars").unwrap();
    let ma = m.atmosphere.as_ref().unwrap();
    let mut v = vec![
        chk("Hydrostatics", "Earth: P0 = ∫ρg dz", ea.column_weight(e.physics.gm, e.physics.radius), 101_325.0, 0.005, "Pa"),
        chk("Ideal gas equation of state", "Earth: ρ at the surface", ea.at(0.0).density, 1.225, 0.005, "kg/m³"),
        chk("Hydrostatics + profile T", "Earth: P at 11 km (US Std 1976)", ea.at(11_000.0).pressure, 22_632.0, 0.01, "Pa"),
        chk("Speed of sound √(γRT/M)", "Earth, 0 m", ea.at(0.0).sound_speed, 340.29, 0.005, "m/s"),
        chk("Hydrostatics", "Mars: scale height H=RT/Mg", ma.scale_height(m.g_surface()) / 1e3, 11.1, 0.05, "km"),
        chk("Hydrostatics", "Mars: P0 = ∫ρg dz", ma.column_weight(m.physics.gm, m.physics.radius), 636.0, 0.005, "Pa"),
    ];
    let comp = presets::oxides_to_elements(&presets::surface("earth").unwrap().oxides_wt).unwrap();
    let sum: f64 = comp.iter().map(|x| x.1).sum();
    let o = comp.iter().find(|x| x.0 == "O").unwrap().1;
    v.push(chk_max("Mass conservation (stoichiometry)", "Σ crust elements − 1", sum - 1.0, 1e-12));
    v.push(chk("Stoichiometry", "O fraction in Earth crust", o * 100.0, 45.5, 0.04, "%"));
    v
}

pub fn temperatures() -> Vec<Check> {
    let e = World::preset("earth").unwrap();
    let m = World::preset("mars").unwrap();
    let mut v = vec![
        chk("Stefan–Boltzmann", "Earth: T_eq", e.equilibrium_temperature(), 254.0, 0.01, "K"),
        chk("Greenhouse effect (Eddington)", "Earth: mean surface T", e.mean_surface_temperature(), 288.0, 0.01, "K"),
        chk("Stefan–Boltzmann", "Mars: T_eq", m.equilibrium_temperature(), 210.0, 0.02, "K"),
        chk("Greenhouse effect (Eddington)", "Venus: mean surface T", World::preset("venus").unwrap().mean_surface_temperature(), 737.0, 0.02, "K"),
    ];
    let moon = World::preset("moon").unwrap();
    let r = thermal::simulate(&moon, ThermalOptions { spinup_days: 6, ..Default::default() });
    v.push(chk("Fourier + Stefan–Boltzmann", "Moon, equator: T max (Diviner ≈ 390 K)", r.t_max, 390.0, 0.04, "K"));
    v.push(chk("Fourier + Stefan–Boltzmann", "Moon, equator: T min (Diviner ≈ 95 K)", r.t_min, 95.0, 0.15, "K"));
    v.push(chk_max("1st law of thermodynamics", "Moon: daily energy balance residual", r.energy_residual, 1e-6));
    let rm = thermal::simulate(&m, ThermalOptions { spinup_days: 10, ..Default::default() });
    v.push(Check { law: "Fourier + Stefan–Boltzmann", test: "Mars, equator: diurnal cycle (Curiosity ~ 190–280 K)".into(),
        value: format!("{:.0}–{:.0} K", rm.t_min, rm.t_max), expected: "min 170–215, max 260–300".into(),
        pass: (170.0..215.0).contains(&rm.t_min) && (260.0..300.0).contains(&rm.t_max) });
    v.push(chk_max("1st law of thermodynamics", "Mars: energy balance residual", rm.energy_residual, 1e-6));
    v
}

pub fn ballistics_checks() -> Vec<Check> {
    let mut v = Vec::new();
    // 1) vacuum, no rotation: range v²/g
    let moon = World::preset("moon").unwrap();
    let o = Options { rotating: false, j2: false, drag: false, buoyancy: false, sample_dt: 0.05, ..Default::default() };
    let mut moon_static = moon.clone();
    moon_static.physics.rotation_period = 1e30;
    let r = ballistics::simulate(&moon_static, Projectile::sphere(0.05, 2000.0),
        Launch { lat: 0.0, lon: 0.0, altitude: 0.0, speed: 10.0, elevation: 45.0, azimuth: 90.0 }, o);
    v.push(chk("2nd Newton law (vacuum)", "Moon: throw range 10 m/s at 45° = v²/g", r.downrange, 100.0 / moon.g_surface(), 1e-4, "m"));
    // 2) Coriolis: eastward deflection when falling 100 m at the equator
    let earth = World::preset("earth").unwrap();
    let h = 100.0;
    let o2 = Options { j2: false, drag: false, buoyancy: false, sample_dt: 0.01, ..Default::default() };
    let r = ballistics::simulate(&earth, Projectile::sphere(0.05, 7800.0),
        Launch { lat: 0.0, lon: 0.0, altitude: h, speed: 0.0, elevation: -90.0, azimuth: 0.0 }, o2);
    let last = r.samples.last().unwrap();
    let g = earth.g_surface() - earth.omega().powi(2) * earth.physics.radius;
    let t = (2.0 * h / g).sqrt();
    let theory = earth.omega() * g * t.powi(3) / 3.0;
    let east = last.lon.to_radians().sin() * earth.physics.radius;
    v.push(chk("Coriolis force", "Earth: eastward deflection falling from 100 m", east * 1000.0, theory * 1000.0, 0.02, "mm"));
    // 3) energy balance with drag, J2 and rotation
    let r = ballistics::simulate(&earth, Projectile::sphere(0.1, 1000.0),
        Launch { lat: 55.75, lon: 37.6, altitude: 0.0, speed: 600.0, elevation: 50.0, azimuth: 30.0 }, Options::default());
    v.push(chk_max("1st law (drag work → heat)", "Earth: |E−E0−W_nc| / scale", r.energy_error, 1e-8));
    v.push(Check { law: "Medium drag", test: "range in air is less than in vacuum".into(),
        value: format!("{:.1} km", r.downrange / 1e3), expected: format!("< {:.1} km", 600f64.powi(2) / 9.81 / 1e3), pass: r.downrange < 600f64.powi(2) / 9.81 && r.dissipated_energy > 0.0 });
    // 4) Archimedes: a helium balloon rises
    let balloon = Projectile { mass: 0.5, area: PI, cd: 0.47, volume: 4.0 / 3.0 * PI };
    let r = ballistics::simulate(&earth, balloon, Launch { lat: 0.0, lon: 0.0, altitude: 1.0, speed: 0.0, elevation: 90.0, azimuth: 0.0 },
        Options { max_time: 120.0, ..Default::default() });
    v.push(Check { law: "Archimedes principle", test: "balloon 0.5 kg, V=4.19 m³ rises in Earth atmosphere".into(),
        value: format!("+{:.0} m in 120 s", r.max_altitude), expected: "> 100 m".into(), pass: r.max_altitude > 100.0 });
    v
}

pub fn magnetosphere() -> Vec<Check> {
    let earth = World::preset("earth").unwrap();
    let sp = magneto::species("proton").unwrap();
    let r = magneto::simulate(&earth, sp, ParticleLaunch { l_shell: 3.0, energy_mev: 1.0, pitch_angle: 40.0 }, 20.0, 100);
    vec![
        chk_max("Lorentz force + SR (Boris)", "proton 1 MeV, L=3: |ΔE_k|/E_k", r.energy_drift, 1e-10),
        Check { law: "Adiabatic invariant μ", test: "mirror latitude, α_eq=40°".into(),
            value: format!("{:.2}°", r.max_abs_lat), expected: format!("{:.2}° ±1.5°", r.mirror_lat_theory),
            pass: (r.max_abs_lat - r.mirror_lat_theory).abs() < 1.5 },
        Check { law: "Gradient-curvature drift", test: "proton drifts west".into(),
            value: format!("{:.2}°", r.drift_longitude), expected: "< 0°".into(), pass: r.drift_longitude < 0.0 && r.trapped },
    ]
}

pub fn stars_checks() -> Vec<Check> {
    use crate::stars;
    let sun = stars::star("sun").unwrap();
    let hz = sun.habitable_zone();
    let ms = stars::Star::main_sequence(1.0);
    vec![
        chk("Stefan–Boltzmann from the Planck spectrum", "π∫B_λdλ / σT⁴ at 5772 K", stars::planck_integrated(5772.0) / (crate::constants::SIGMA_SB * 5772f64.powi(4)), 1.0, 1e-6, ""),
        chk("Wien displacement law", "λ_max·T numerically from Planck", stars::wien_peak(5772.0) * 5772.0 * 1e3, 2.897_771_955, 1e-4, "mm·K"),
        chk("Mass–luminosity + Stefan–Boltzmann", "T_eff of a 1 M☉ main sequence star", ms.teff, 5772.0, 0.001, "K"),
        chk("Habitable zone (Kopparapu 2014)", "inner boundary at the Sun", hz.0, 0.95, 0.01, "AU"),
        chk("Habitable zone (Kopparapu 2014)", "outer boundary at the Sun", hz.1, 1.67, 0.01, "AU"),
        Check { law: "Black-body radiation → color", test: "Proxima is red, Sirius is bluish".into(),
            value: format!("R/B: {:.2} / {:.2}", stars::star("proxima").unwrap().rgb()[0] / stars::star("proxima").unwrap().rgb()[2], stars::star("sirius_a").unwrap().rgb()[0] / stars::star("sirius_a").unwrap().rgb()[2]),
            expected: "> 1.5 / < 1".into(),
            pass: { let p = stars::star("proxima").unwrap().rgb(); let s = stars::star("sirius_a").unwrap().rgb(); p[0] / p[2] > 1.5 && s[0] / s[2] < 1.0 } },
    ]
}

pub fn starsystem_checks() -> Vec<Check> {
    use crate::starsystem::StarSystem;
    let mut worst: f64 = 0.0;
    let mut stable = 0;
    let n = 12;
    for seed in 0..n {
        let sys = StarSystem::generate(seed, None).unwrap();
        let mut nb = sys.nbody();
        let i0 = nb.invariants();
        let p_in = sys.planets[0].a.powf(1.5) / sys.star.mass.sqrt() * YEAR;
        let dt = p_in / 200.0;
        let a0: Vec<f64> = nb.bodies.iter().skip(1).map(|b| (b.r - nb.bodies[0].r).norm()).collect();
        for _ in 0..4000 { nb.step(dt); }
        let i1 = nb.invariants();
        worst = worst.max(rel(i1.energy, i0.energy));
        let ok = nb.bodies.iter().skip(1).zip(&a0).all(|(b, r0)| { let r = (b.r - nb.bodies[0].r).norm(); r < 3.0 * r0 && r > r0 / 3.0 });
        if ok { stable += 1; }
    }
    vec![
        chk_max("Energy conservation", "12 generated systems, 20 orbits of the inner planet", worst, 1e-6),
        Check { law: "Hill stability", test: "spacing 10–25 R_H: planets stay in orbit".into(), value: format!("{stable}/{n}"), expected: format!("{n}/{n}"), pass: stable == n },
    ]
}

pub fn galaxy_checks() -> Vec<Check> {
    use crate::galaxy;
    let mut g = galaxy::preset("plummer", 3000, 1).unwrap();
    let (k, w) = g.energy();
    let acc = g.tree_acc();
    let mut errs: Vec<f64> = (0..150).map(|i| { let d = g.direct_acc(i * 20); (acc[i * 20] - d).norm() / d.norm() }).collect();
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let e0 = k + w;
    let p_scale: f64 = (0..g.len()).map(|i| g.mass[i] * g.vel[i].norm()).sum();
    let tc = galaxy::crossing_time(1.0, 1.0);
    for _ in 0..100 { g.step(tc / 100.0); }
    let (k1, w1) = g.energy();
    let mw = galaxy::preset("milky_way", 20000, 2).unwrap();
    let rc = mw.rotation_curve(20.0, 10);
    let v8 = rc.iter().find(|(r, _)| *r > 7.0).map(|x| x.1).unwrap_or(0.0);
    vec![
        chk("Virial theorem", "Plummer sphere: 2K/|W|", 2.0 * k / w.abs(), 1.0, 0.03, ""),
        chk_max("Barnes–Hut (θ=0.6) vs direct sum", "median relative force error", errs[errs.len() / 2], 0.01),
        chk_max("Energy conservation", "Plummer, 1 crossing time, |ΔE/E|", ((k1 + w1) - e0) / e0.abs(), 1e-3),
        chk_max("Momentum conservation", "|ΔP| / Σm|v|", g.momentum().norm() / p_scale, 1e-3),
        chk("Rotation curve (dark halo)", "Milky Way model: v_c at 8 kpc", v8, 220.0, 0.1, "km/s"),
    ]
}

pub fn cosmology_checks() -> Vec<Check> {
    use crate::cosmology::{Cosmology, PmSim, GYR_S};
    let c = Cosmology::planck2018();
    let eds = Cosmology::preset("eds").unwrap();
    let a0 = 0.02;
    let a1 = 0.1;
    let mut pm = PmSim::new(c, 32, 128.0, a0, 3);
    let p0 = pm.measure_power(8);
    let ps = pm.momentum_scale();
    for _ in 0..40 { pm.step((a1 - a0) / 40.0); }
    let p1 = pm.measure_power(8);
    let ratio: f64 = p0.iter().zip(&p1).take(3).map(|(x, y)| y.1 / x.1).sum::<f64>() / 3.0;
    vec![
        chk("Friedmann equation (Planck 2018)", "age of the Universe", c.age_gyr(1.0), 13.787, 0.002, "Gyr"),
        chk("Friedmann equation (analytic)", "Einstein–de Sitter: t₀ = 2/(3H₀)", eds.age_gyr(1.0), 2.0 / 3.0 / eds.h0_si() / GYR_S, 1e-4, "Gyr"),
        chk("Linear growth", "Einstein–de Sitter: D(a) = a", eds.growth(0.5), 0.5, 1e-4, ""),
        chk("Linear growth", "ΛCDM: f(z=0) ≈ Ω_m^0.55", c.growth_rate(1.0), c.omega_m.powf(0.55), 0.01, ""),
        chk("Spectrum normalization", "σ₈ of P(k)", c.sigma_r(8.0), 0.8102, 1e-6, ""),
        chk("Luminosity distance", "d_L(z=1), Planck 2018 (≈6.78 Gpc)", c.luminosity_distance(1.0) / 1e3, 6.78, 0.01, "Gpc"),
        chk("Gravitational instability (PM)", "P(k) growth on large scales a=0.02→0.1 vs D²", ratio, (c.growth(a1) / c.growth(a0)).powi(2), 0.08, ""),
        chk_max("Momentum conservation (PM, CIC)", "|ΣP| / Σ|p|", pm.total_momentum() / ps, 1e-10),
    ]
}

pub fn climate_checks() -> Vec<Check> {
    use crate::climate::{self, ClimateOptions};
    let e = World::preset("earth").unwrap();
    let r = climate::simulate(&e, &ClimateOptions::default());
    let eq = r.annual_mean[r.lat.len() / 2];
    let snow = climate::simulate(&e, &ClimateOptions { initial_t: Some(220.0), years: 60, ..Default::default() });
    let m = World::preset("mars").unwrap();
    let rm = climate::simulate(&m, &ClimateOptions::default());
    vec![
        chk("Energy balance + transport (EBM)", "Earth: annual global mean T", r.global_mean, 288.0, 0.01, "K"),
        chk("Energy balance + transport (EBM)", "Earth: equator, annual mean (≈299 K)", eq, 299.0, 0.02, "K"),
        chk("Ice-albedo feedback", "Earth: ice edge (observed ≈70°)", r.ice_edge, 70.0, 0.12, "°"),
        chk_max("1st law of thermodynamics", "Earth: annual energy residual", r.energy_residual, 1e-9),
        chk_max("Radiative equilibrium", "Earth: |⟨ASR−OLR⟩| at the top of the atmosphere", r.toa_imbalance, 0.05),
        Check { law: "Bistability (Budyko)", test: "cold start with the same parameters → “snowball Earth”".into(),
            value: format!("{:.0} K, ice to {:.0}°", snow.global_mean, snow.ice_edge), expected: "ice down to the equator".into(), pass: snow.ice_edge < 5.0 },
        Check { law: "CO₂ condensation (148 K)", test: "Mars: polar caps CO₂".into(),
            value: format!("edge {:.0}°, pole {:.0} K", rm.ice_edge, rm.annual_mean[rm.lat.len() - 1]), expected: "caps above 70°".into(), pass: rm.ice_edge > 70.0 && rm.ice_edge < 90.0 },
    ]
}

fn e_steps_for_seconds(e: &crate::env::Env, s: f64) -> usize { (s / e.control_dt()) as usize }

pub fn env_checks() -> Vec<Check> {
    use crate::env::{Env, EnvConfig, Terrain, G0};
    let mut out = Vec::new();
    // 1) Tsiolkovsky: full fuel burn in weightlessness and vacuum
    let mut cfg = EnvConfig::new("lander", "moon").unwrap();
    cfg.world.gravity_scale = 1e-12;
    cfg.noise = 0.0;
    let mut e = Env::new(cfg).unwrap();
    e.reset(1);
    let h0 = e.terrain.height(0.0, 0.0);
    e.set_state(Vec3::new(0.0, 0.0, h0 + 100.0), Vec3::ZERO);
    let m0 = e.mass();
    for _ in 0..4000 { e.step(&[1.0, 0.0, 0.0, 0.0]); }
    let dv = e.velocity().norm();
    let theory = e.isp() * G0 * (m0 / e.dry_mass()).ln();
    out.push(chk("Tsiolkovsky equation", "lander: Δv with a full fuel burn", dv, theory, 0.002, "m/s"));
    // 2) Drone: hover on Earth with thrust = weight; no lift on the Moon; ordinary rotors are not enough on Mars
    let hover = |planet: &str, thr: Option<f64>, rotor: f64| -> f64 {
        let mut c = EnvConfig::new("drone", planet).unwrap();
        c.noise = 0.0;
        c.rotor_scale = rotor;
        let mut e = Env::new(c).unwrap();
        e.reset(3);
        let h = e.terrain.height(0.0, 0.0);
        e.set_state(Vec3::new(0.0, 0.0, h + 30.0), Vec3::ZERO);
        let rho = e.air_density();
        let u = thr.unwrap_or(if rho > 0.0 { 0.4 * 1.225 / rho / rotor } else { 1.0 });
        for _ in 0..50 { e.step(&[u; 4]); }
        e.velocity().z
    };
    out.push(chk_max("Rotor momentum theory", "Earth: thrust = weight → vertical speed after 1 s, m/s", hover("earth", None, 1.0), 0.05));
    out.push(Check { law: "Lift ∝ ρ", test: "Moon (vacuum): drone at full thrust falls".into(),
        value: format!("v_z = {:.2} m/s", hover("moon", Some(1.0), 1.0)), expected: "< 0".into(), pass: hover("moon", Some(1.0), 1.0) < -1.0 });
    out.push(Check { law: "Lift ∝ ρ", test: "Mars: ordinary rotors do not lift, ×40 (like Ingenuity) do".into(),
        value: format!("{:.2} / {:+.2} m/s", hover("mars", Some(1.0), 1.0), hover("mars", Some(1.0), 40.0)), expected: "< 0 / > 0".into(),
        pass: hover("mars", Some(1.0), 1.0) < 0.0 && hover("mars", Some(1.0), 40.0) > 0.0 });
    // 3) Rover: Coulomb traction, climbing a slope tanθ < μ − C_rr, sliding at tanθ > μ
    let climb = |grade: f64| -> f64 {
        let mut c = EnvConfig::new("rover", "earth").unwrap();
        c.noise = 0.0;
        let mut e = Env::new(c).unwrap();
        e.reset(1);
        e.set_terrain(Terrain::plane(grade, 0.6));
        e.set_state(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO);
        for _ in 0..200 { e.step(&[1.0, 1.0]); }
        e.pose().0.x
    };
    let (x1, x2) = (climb(0.3), climb(0.75));
    out.push(Check { law: "Coulomb friction (wheel traction)", test: "μ=0.6: slope 17° climbs, 37° slides".into(),
        value: format!("{x1:+.1} m / {x2:+.1} m"), expected: "> 0 / < 0".into(), pass: x1 > 0.0 && x2 < 0.0 });
    // 4) Determinism
    let run = || { let mut e = Env::new(EnvConfig::new("lander", "mars").unwrap()).unwrap(); e.reset(42); let mut o = vec![]; for k in 0..200 { o = e.step(&[0.6, (k as f64 * 0.1).sin(), 0.0, 0.1]).0; } o };
    let (a, b) = (run(), run());
    out.push(chk_max("Reproducibility", "same seed and actions → same observations", a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max), 1e-12));
    // 5) Sensor noise
    let mut e = Env::new(EnvConfig::new("rover", "moon").unwrap()).unwrap();
    e.reset(5);
    let xs: Vec<f64> = (0..2000).map(|_| e.step(&[0.0, 0.0]).0[0]).collect();
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    let sd = (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt();
    out.push(chk("Sensor model", "GPS rover: noise standard deviation", sd, 0.5, 0.06, "m"));
    // Contact: on a plane the lander stands on its legs, so the reaction equals the weight and the accelerometer shows g
    let mut c = EnvConfig::new("lander", "mars").unwrap();
    c.noise = 0.0;
    let mut e = Env::new(c).unwrap();
    e.reset(1);
    e.set_terrain(Terrain::plane(0.0, 0.8));
    e.set_state(Vec3::new(0.0, 0.0, 1.7), Vec3::ZERO);
    let g_mars = e.world.g_at(0.0);
    let mut az = 0.0;
    for _ in 0..40 { let (o, _, t, _) = e.step(&[0.0, 0.0, 0.0, 0.0]); az = o[15]; if t { break; } }
    out.push(chk("Newton third law (legs)", "lander stands on Mars: accelerometer shows g", az, g_mars, 0.03, "m/s²"));
    // Tip-over: the stability margin depends on g, so the same maneuver is dangerous on the Moon and safe on Earth
    let turn = |planet: &str| -> bool {
        let mut c = EnvConfig::new("rover", planet).unwrap();
        c.noise = 0.0;
        let mut e = Env::new(c).unwrap();
        e.reset(1);
        e.set_terrain(Terrain::plane(0.0, 0.8));
        e.set_state(Vec3::ZERO, Vec3::ZERO);
        for _ in 0..30 { e.step(&[1.0, 1.0]); }
        for _ in 0..200 { let (_, _, t, _) = e.step(&[1.0, -1.0]); if t { return e.info.crashed; } }
        false
    };
    let (moon_tip, earth_tip) = (turn("moon"), turn("earth"));
    out.push(Check { law: "Static stability (v·ω / g)", test: "same turn: tips over on the Moon, not on Earth".into(),
        value: format!("{moon_tip} / {earth_tip}"), expected: "true / false".into(), pass: moon_tip && !earth_tip });
    // Gusts: stationary standard deviation of the Ornstein–Uhlenbeck process
    let mut c = EnvConfig::new("drone", "earth").unwrap();
    c.gust = 3.0;
    c.max_steps = usize::MAX;
    let mut e = Env::new(c).unwrap();
    e.reset(4);
    let h = e.terrain.height(0.0, 0.0);
    let mut sum = 0.0;
    let mut k = 0.0;
    for t in 0..60_000 {
        e.set_state(Vec3::new(0.0, 0.0, h + 100.0), Vec3::ZERO);
        e.step(&[0.4; 4]);
        if t > 1000 { let v = e.gust_velocity().x; sum += v * v; k += 1.0; }
    }
    out.push(chk("Ornstein–Uhlenbeck process (gusts)", "gust standard deviation equals the set value", (sum / k).sqrt(), 3.0, 0.1, "m/s"));
    // Actuators: first-order lag, within time τ the output reaches 1 − 1/e
    let mut c = EnvConfig::new("lander", "moon").unwrap();
    c.noise = 0.0;
    c.actuator_lag = 0.2;
    let mut e = Env::new(c).unwrap();
    e.reset(1);
    let h = e.terrain.height(0.0, 0.0);
    e.set_state(Vec3::new(0.0, 0.0, h + 400.0), Vec3::ZERO);
    for _ in 0..10 { e.step(&[1.0, 0.0, 0.0, 0.0]); } // 10 steps of 20 ms = τ
    out.push(chk("First-order lag (actuators)", "within time τ the command is executed to 1 − 1/e", e.applied_action()[0], 1.0 - (-1f64).exp(), 0.01, ""));
    // Rover motor: with no load the speed is limited by the no-load speed
    let mut c = EnvConfig::new("rover", "earth").unwrap();
    c.noise = 0.0;
    c.world.atmosphere = Some("none".into());
    let mut e = Env::new(c).unwrap();
    e.reset(1);
    e.set_terrain(Terrain::plane(0.0, 0.8));
    e.set_state(Vec3::ZERO, Vec3::ZERO);
    for _ in 0..300 { e.step(&[1.0, 1.0]); }
    out.push(Check { law: "Motor characteristic", test: "rover on flat ground: speed below no-load speed".into(), value: format!("{:.2} m/s", e.rover_speed()), expected: "1…2.5 m/s".into(), pass: e.rover_speed() > 1.0 && e.rover_speed() < 2.5 });
    // Branching: an environment copy runs exactly like the original, otherwise counterfactual data is wrong
    let mut a = Env::new(EnvConfig::new("lander", "earth").unwrap()).unwrap();
    a.reset(9);
    for _ in 0..50 { a.step(&[0.6, 0.0, 0.1, 0.0]); }
    let mut b = a.clone();
    let mut diff: f64 = 0.0;
    for _ in 0..100 {
        let (x, y) = (a.step(&[0.5, 0.1, 0.0, 0.0]).0, b.step(&[0.5, 0.1, 0.0, 0.0]).0);
        diff = x.iter().zip(&y).map(|(p, q)| (p - q).abs()).fold(diff, f64::max);
    }
    out.push(chk_max("Branching accuracy", "environment copy repeats the original exactly", diff, 1e-12));
    // 6) Solvability: the autopilot lands on the Moon
    let mut cfg = EnvConfig::new("lander", "moon").unwrap();
    cfg.randomize = true;
    let mut e = Env::new(cfg).unwrap();
    let mut ok = 0;
    for ep in 0..10 {
        e.reset(ep + 1);
        loop { let a = e.autopilot(); let (_, _, t, tr) = e.step(&a); if t || tr { break; } }
        if e.info.success { ok += 1; }
    }
    // Latency: a delayed command must give exactly the flight of the same commands shifted in time
    let flown = |set: &[(&str, f64)], shift: usize| {
        let mut c = EnvConfig::new("lander", "moon").unwrap();
        c.noise = 0.0;
        for (k, v) in set { c.set(k, *v).unwrap(); }
        let mut e = Env::new(c).unwrap();
        e.reset(2);
        for k in 0..200usize {
            let i = k.saturating_sub(shift);
            let a = if k < shift { [0.0; 4] } else { [0.4 + 0.3 * (i as f64 * 0.05).sin(), 0.1 * (i as f64 * 0.03).cos(), 0.0, 0.0] };
            e.step(&a);
        }
        e.state()
    };
    let gap = |a: Vec<f64>, b: Vec<f64>| a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let d = gap(flown(&[("cmd_delay", 5.0)], 0), flown(&[], 5));
    out.push(Check { law: "Command latency", test: "5 steps of latency = same commands shifted by 5 steps".into(), value: format!("{d:.2e}"), expected: "< 1e-12".into(), pass: d < 1e-12 });
    let mut c = EnvConfig::new("lander", "moon").unwrap();
    c.noise = 0.0;
    let dt_ms = Env::new(c.clone()).unwrap().control_dt() * 1e3;
    let d = gap(flown(&[("compute_ms", 3.5 * dt_ms)], 0), flown(&[], 3));
    out.push(Check { law: "Decision time", test: "3.5 control steps of computing = 3 steps of latency".into(), value: format!("{d:.2e}"), expected: "< 1e-12".into(), pass: d < 1e-12 });
    // Outage: the observation freezes and the last command stays active
    let mut c = EnvConfig::new("lander", "moon").unwrap();
    c.noise = 0.0;
    for (k, v) in [("link_period", 1.0), ("link_outage", 0.5), ("start_alt_min", 3000.0), ("start_alt_max", 3000.0)] { c.set(k, v).unwrap(); }
    let mut e = Env::new(c).unwrap();
    e.reset(1);
    let (mut frozen, mut total, mut prev) = (0usize, 0usize, e.last_obs.clone());
    for k in 0..e_steps_for_seconds(&e, 10.0) {
        let (o, _, _, _) = e.step(&[0.2 + 0.001 * k as f64 % 0.5, 0.0, 0.0, 0.0]);
        total += 1;
        if o == prev { frozen += 1; }
        prev = o;
    }
    let share = frozen as f64 / total as f64;
    out.push(chk("Link outage", "0.5 s outage in a 1 s cycle: share of frozen observations", share, 0.5, 0.05, ""));
    // Thermal: radiative equilibrium and adiabatic heating against closed forms
    let thermal = |set: &[(&str, f64)], seconds: f64| {
        let mut c = EnvConfig::new("rover", "moon").unwrap();
        for (k, v) in [("terrain_scale", 0.0), ("crater_scale", 0.0), ("temp_min", 1.0), ("temp_max", 3000.0), ("absorptance", 0.0)] { c.set(k, v).unwrap(); }
        for (k, v) in set { c.set(k, *v).unwrap(); }
        c.max_steps = usize::MAX;
        let mut e = Env::new(c).unwrap();
        e.reset(1);
        while e.t < seconds && !e.done { e.step(&[0.0, 0.0]); }
        (e.temperature(), e.t)
    };
    let (q, area, eps) = (200.0, 1.0, 0.85);
    let (t_eq, _) = thermal(&[("thermal_mass", 200.0), ("heat_load", q), ("radiator_area", area), ("emissivity", eps)], 3000.0);
    out.push(chk("Stefan–Boltzmann (radiator)", "200 W, 1 m², ε = 0.85: T_eq = (Q/εσA)^¼", t_eq, (q / (eps * crate::constants::SIGMA_SB * area)).powf(0.25), 1e-6, "K"));
    let (t_ad, t) = thermal(&[("thermal_mass", 5000.0), ("heat_load", 100.0), ("radiator_area", 0.0)], 100.0);
    out.push(chk("First law (heating)", "no radiator, 100 W, 5000 J/K: T = T₀ + Qt/C", t_ad, 293.0 + 100.0 * t / 5000.0, 1e-9, "K"));
    let mut c = EnvConfig::new("rover", "moon").unwrap();
    for (k, v) in [("thermal_mass", 100.0), ("heat_load", 1000.0), ("radiator_area", 0.0), ("absorptance", 0.0), ("terrain_scale", 0.0), ("crater_scale", 0.0)] { c.set(k, v).unwrap(); }
    let mut e = Env::new(c).unwrap();
    e.reset(1);
    while !e.step(&[0.0, 0.0]).2 && e.t < 60.0 {}
    out.push(Check { law: "Thermal limits", test: "100 J/K at 1 kW: reaches 323 K in 3 s and the mission ends".into(), value: format!("{:.2} s, {}", e.t, e.info.reason), expected: "3 s, overheated".into(), pass: (e.t - 3.0).abs() < 0.1 && e.info.reason == "overheated" });
    // Soil: closed forms of the Bekker model against numerical integration
    let soil = crate::terramech::Soil { kc: 1400.0, kphi: 820_000.0, n: 1.0, cohesion: 170.0, phi: 35f64.to_radians(), shear_k: 0.018, radius: 0.25, width: 0.2 };
    let w = soil.wheel(73.0);
    let k = soil.kc + soil.width * soil.kphi;
    let parts = 20_000;
    let (mut load, mut work) = (0.0, 0.0);
    for i in 0..parts {
        // parabolic rim: the depth falls from z at the bottom to 0 at the front of the contact
        let x = (i as f64 + 0.5) / parts as f64 * w.contact;
        load += k * w.sinkage * (1.0 - (x / w.contact).powi(2)) * w.contact / parts as f64;
        let z = (i as f64 + 0.5) / parts as f64 * w.sinkage;
        work += k * z * w.sinkage / parts as f64;
    }
    out.push(chk("Bekker pressure-sinkage", "vertical force over the contact = wheel load", load, 73.0, 1e-3, "N"));
    out.push(chk("Bekker compaction", "R_c = ∫ b·p dz over the sinkage", w.compaction, work, 1e-3, "N"));
    let mut mean = 0.0;
    for i in 0..parts {
        let x = (i as f64 + 0.5) / parts as f64 * w.contact;
        mean += (1.0 - (-0.2 * x / w.shear_k).exp()) / parts as f64;
    }
    out.push(chk("Janosi–Hanamoto shear", "slip 0.2: closed form = mean over the contact", w.shear_factor(0.2), mean, 1e-3, ""));
    let (h, s) = w.traction(0.5 * w.h_max * w.shear_factor(1.0));
    out.push(chk("Janosi–Hanamoto shear", "inverse: the slip found gives the demanded force", w.h_max * w.shear_factor(s), h, 1e-6, "N"));
    // Rover on soft soil: the limit slope follows from drawbar pull
    let m_rover: f64 = 180.0;
    let g_moon = World::preset("moon").unwrap().g_surface();
    let climbs = |deg: f64| {
        let mut c = EnvConfig::new("rover", "moon").unwrap();
        for (k, v) in [("terramechanics", 1.0), ("motor_force", 5000.0)] { c.set(k, v).unwrap(); }
        c.noise = 0.0;
        let mut e = Env::new(c).unwrap();
        e.reset(1);
        e.set_terrain(Terrain::plane(deg.to_radians().tan(), 0.6));
        e.set_state(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO);
        for _ in 0..200 { e.step(&[1.0, 1.0]); }
        e.pose().0.x
    };
    // limit angle: tanθ·W·cosθ = 4·(H(1) − R_c), the load itself falls with cosθ
    let limit = (0..9000).map(|i| i as f64 * 0.01).find(|deg: &f64| {
        let t = deg.to_radians();
        let w = soil.wheel(m_rover * g_moon * t.cos() / 4.0);
        m_rover * g_moon * t.sin() >= 4.0 * (w.h_max * w.shear_factor(1.0) - w.compaction)
    }).unwrap_or(90.0);
    let (below, above) = (climbs(limit - 2.0), climbs(limit + 2.0));
    out.push(Check { law: "Drawbar pull", test: format!("limit slope {limit:.1}°: climbs 2° below, stalls 2° above (4 s)"), value: format!("{below:.1} m / {above:.2} m"), expected: "> 2 m / < 0.2 m".into(), pass: below > 2.0 && above < 0.2 });
    // Power budget: a constant bus load draws exactly P·t, and used + left equals the initial charge
    let mut c = EnvConfig::new("lander", "moon").unwrap();
    c.set("bus_power", 500.0).unwrap();
    let mut e = Env::new(c).unwrap();
    e.reset(1);
    for _ in 0..100 { let a = e.autopilot(); e.step(&a); }
    out.push(chk("Energy balance (bus load)", "lander, 500 W: energy used = P·t", e.info.energy_used, 500.0 * e.t, 1e-9, "J"));
    let mut c = EnvConfig::new("rover", "moon").unwrap();
    for (k, v) in [("bus_power", 100.0), ("panel_area", 0.0), ("terrain_scale", 0.0), ("crater_scale", 0.0)] { c.set(k, v).unwrap(); }
    let mut e = Env::new(c).unwrap();
    e.reset(1);
    let full = e.battery_frac();
    for _ in 0..100 { e.step(&[0.0, 0.0]); }
    out.push(chk("Energy balance (bus load)", "rover at rest, no panel: 15 W electronics + 100 W bus", (full - e.battery_frac()) * 0.6 * 3.6e6, 115.0 * e.t, 1e-9, "J"));
    let mut c = EnvConfig::new("drone", "earth").unwrap();
    c.set("bus_power", 50.0).unwrap();
    c.set("battery_kj", 1.0).unwrap();
    let mut e = Env::new(c).unwrap();
    e.reset(1);
    while !e.step(&[0.6; 4]).2 && e.t < 60.0 {}
    out.push(chk("Energy balance (bus load)", "drone, battery runs out: energy used = initial charge", e.info.energy_used, 1000.0, 1e-9, "J"));
    let mut c = EnvConfig::new("lander", "moon").unwrap();
    for (k, v) in [("bus_power", 2000.0), ("lander_battery_kwh", 0.01), ("start_alt_min", 3000.0), ("start_alt_max", 3000.0), ("terrain_scale", 0.0), ("crater_scale", 0.0)] { c.set(k, v).unwrap(); }
    c.max_steps = 100_000;
    let mut e = Env::new(c).unwrap();
    e.reset(1);
    let mut empty_t = 0.0;
    loop {
        let a = e.autopilot();
        let (_, _, t, tr) = e.step(&a);
        if empty_t == 0.0 && e.info.energy_used >= 36_000.0 { empty_t = e.t; }
        if t || tr { break; }
    }
    out.push(Check { law: "Power budget", test: "lander, 36 kJ at 2 kW: battery lasts 18 s, then the vehicle crashes".into(), value: format!("{empty_t:.2} s, {}", e.info.reason), expected: "18 s, battery depleted, then crash".into(),
        pass: (empty_t - 18.0).abs() < 0.2 && e.info.reason == "battery depleted, then crash" });
    out.push(Check { law: "Task solvability", test: "PD autopilot: Moon landing, 10 episodes".into(), value: format!("{ok}/10"), expected: "≥ 7/10".into(), pass: ok >= 7 });
    out
}

pub fn quantum_checks() -> Vec<Check> {
    use crate::quantum as q;
    let mut v = vec![
        chk("Fine-structure constant", "1/α of e, ħ, c, ε₀", 1.0 / q::alpha(), 137.035999, 1e-6, ""),
        chk("Hydrogen spectrum", "Lyman-α, vacuum (NIST 121.567 nm)", q::hydrogen_line_nm(1.0, 2.0), 121.567, 1e-3, "nm"),
        chk("Hydrogen spectrum", "Balmer-α, vacuum (NIST 656.46 nm)", q::hydrogen_line_nm(2.0, 3.0), 656.46, 1e-3, "nm"),
        chk("Schrödinger equation", "hydrogen atom numerically: E₁ = −1/2 a.u.", q::hydrogen_s_level(0, 40.0, 4000), -0.5, 0.002, "a.u."),
        chk("Schrödinger equation", "hydrogen atom numerically: E₂ = −1/8 a.u.", q::hydrogen_s_level(1, 60.0, 6000), -0.125, 0.002, "a.u."),
        chk("Quantum oscillator", "E₀ = ½ħω numerically", q::bound_level_1d(|x| 0.5 * x * x, 8.0, 2000, 0), 0.5, 1e-3, "a.u."),
        chk("Quantum oscillator", "E₃ = 7/2 ħω numerically", q::bound_level_1d(|x| 0.5 * x * x, 8.0, 2000, 3), 3.5, 1e-3, "a.u."),
    ];
    // Packet: norm, uncertainty, spreading
    let mut w = q::Wave::gaussian(-60.0, 60.0, 2400, 0.01, 0.0, 1.0, 0.0, |_| 0.0);
    v.push(chk("Uncertainty relation", "Gaussian packet: Δx·Δp = ħ/2", w.sigma_x() * w.sigma_p(), 0.5, 0.005, "ħ"));
    let n0 = w.norm();
    for _ in 0..500 { w.step(); }
    v.push(chk_max("Probability conservation", "norm after 500 Crank–Nicolson steps", (w.norm() - n0).abs(), 1e-9));
    v.push(chk("Packet spreading", "σ(t) = σ₀√(1 + (t/2σ₀²)²), t = 5", w.sigma_x(), (1.0f64 + 2.5 * 2.5).sqrt(), 0.005, ""));
    // Tunneling: narrow-spectrum packet against the exact formula
    let (e0, v0, a) = (0.5f64, 0.8f64, 1.0f64);
    let k0 = (2.0 * e0).sqrt();
    let mut t = q::Wave::gaussian(-150.0, 150.0, 6000, 0.02, -60.0, 12.0, k0, move |x| if (0.0..a).contains(&x) { v0 } else { 0.0 });
    for _ in 0..6500 { t.step(); }
    v.push(chk("Tunnel effect", "barrier transmission at E < V₀", t.right_of(a), q::barrier_transmission(e0, v0, a), 0.03, ""));
    let (eg, peak) = q::gamow(1.0, 1.0, crate::constants::M_P, crate::constants::M_P, 1.57e7);
    v.push(chk("Tunneling in stars", "Gamow energy p–p", eg, 493.0, 0.01, "keV"));
    v.push(chk("Tunneling in stars", "Gamow peak at the center of the Sun", peak, 6.0, 0.05, "keV"));
    let (x1, w3) = q::lane_emden(3.0);
    v.push(chk("Lane–Emden equation", "n = 3: first zero ξ₁", x1, 6.89685, 1e-4, ""));
    v.push(chk("Lane–Emden equation", "n = 3: −ξ²θ'", w3, 2.01824, 1e-4, ""));
    v.push(chk("Chandrasekhar limit", "white dwarf, μₑ = 2", q::chandrasekhar_mass(2.0), 1.44, 0.015, "M☉"));
    v
}

pub fn blackhole_checks() -> Vec<Check> {
    use crate::blackhole as b;
    use crate::constants::G;
    use crate::quantum::M_SUN;
    let c = crate::constants::C;
    let sec = 180.0 / std::f64::consts::PI * 3600.0;
    let rg_sun = G * M_SUN / (c * c);
    let r_sun = 6.957e8;
    let mut v = vec![
        chk("Schwarzschild metric", "ISCO from orbital stability", b::isco_numeric(), 6.0, 1e-3, "M"),
        chk("Schwarzschild metric", "photon sphere", b::photon_sphere_numeric(), 3.0, 1e-3, "M"),
        chk("Black hole shadow", "critical impact parameter of rays", b::shadow_radius_numeric(), 27f64.sqrt(), 2e-3, "M"),
        chk("Light deflection", "geodesic at the edge of the Sun (Eddington 1.75″)", b::deflection(r_sun / rg_sun).unwrap_or(f64::NAN) * sec, 4.0 * rg_sun / r_sun * sec, 0.02, "″"),
        chk("Light deflection", "strong field, b = 20: 4/b + 15π/4b²", b::deflection(20.0).unwrap_or(f64::NAN), 4.0 / 20.0 + 15.0 * std::f64::consts::PI / 4.0 / 400.0 + 128.0 / 3.0 / 8000.0, 0.02, "rad"),
        chk("Orbital precession", "numerically against 6π/p at p = 1000", b::precession_per_orbit(1000.0 / (1.0 - 0.04), 0.2), 6.0 * std::f64::consts::PI / 1000.0, 0.01, "rad"),
        chk("Kerr metric", "ISCO a = 0 by minimum energy", b::kerr_isco_numeric(1e-9), 6.0, 1e-3, "M"),
        chk("Kerr metric", "ISCO a = 0.9: numerically against the formula", b::kerr_isco_numeric(0.9), b::kerr_isco(0.9), 1e-3, "M"),
        chk("Kerr metric", "ISCO a = 0.998 (theory 1.237)", b::kerr_isco(0.998), 1.237, 2e-3, "M"),
        chk("Accretion efficiency", "a = 0: 1 − √(8/9)", b::radiative_efficiency(1e-9), 1.0 - (8f64 / 9.0).sqrt(), 1e-3, ""),
        chk("Accretion efficiency", "a = 0.998 (theory ≈ 0.32)", b::radiative_efficiency(0.998), 0.321, 0.01, ""),
        chk("Hawking radiation", "T for 1 M☉ (6.17·10⁻⁸ K)", b::hawking_temperature(M_SUN) * 1e8, 6.17, 2e-3, "·10⁻⁸ K"),
    ];
    // the first law of black hole thermodynamics checks T and S independently of each other
    let (m, dm) = (10.0 * M_SUN, 1e-6 * M_SUN);
    let t_ds = b::hawking_temperature(m) * (b::bh_entropy(m + dm) - b::bh_entropy(m - dm)) / (2.0 * dm * c * c);
    v.push(chk("Black hole thermodynamics", "dE = T·dS", t_ds, 1.0, 1e-6, ""));
    v.push(chk("Hawking radiation", "evaporation time of 1 M☉ (2.1·10⁶⁷ years)", b::evaporation_years(M_SUN) / 1e67, 2.1, 0.02, "·10⁶⁷ yr"));
    // PSR B1913+16: observed decay rate agrees with GR
    v.push(chk("Gravitational waves", "PSR B1913+16: dP/dt (GR −2.40·10⁻¹²)", b::period_decay(1.4398 * M_SUN, 1.3886 * M_SUN, 27906.98, 0.6171) * 1e12, -2.402, 0.005, "·10⁻¹²"));
    let (m1, m2, a) = (30.0 * M_SUN, 30.0 * M_SUN, 1e9);
    v.push(chk("Gravitational waves", "merger time: numerically against Peters", b::merger_time_numeric(m1, m2, a), b::merger_time(m1, m2, a), 1e-3, "s"));
    // GPS: total clock divergence
    let (re, rs) = (6.371e6, 2.6571e7);
    let gm = 3.986004418e14;
    let grav = gm * (1.0 / re - 1.0 / rs) / (c * c);
    let vel = gm / rs / (2.0 * c * c);
    v.push(chk("Time dilation", "GPS clock per day (+38.6 μs)", (grav - vel) * 86400.0 * 1e6, 38.6, 0.02, "μs"));
    v
}

pub fn fluid_checks() -> Vec<Check> {
    use crate::fluid as f;
    let mut s = f::Euler1D::sod(800);
    let (m0, e0) = (s.rho.iter().sum::<f64>(), s.en.iter().sum::<f64>());
    s.run_to(0.2);
    let n = s.rho.len();
    let l1 = (0..n).map(|i| { let x = (i as f64 + 0.5) * s.dx; (s.rho[i] - f::sod_exact(x, s.t).0).abs() * s.dx }).sum::<f64>();
    let mut v = vec![
        chk_max("Sod problem", "L1 density error against the exact solution", l1, 0.01),
        chk_max("Mass conservation", "Euler equations, relative error", (s.rho.iter().sum::<f64>() - m0).abs() / m0, 1e-12),
        chk_max("Energy conservation", "Euler equations, relative error", (s.en.iter().sum::<f64>() - e0).abs() / e0, 1e-12),
    ];
    let (g, h0, l) = (9.81, 1.0, 100.0);
    let mut d = f::Shallow1D::dam(1000, l, g, h0);
    let v0: f64 = d.h.iter().sum();
    d.run_to(4.0);
    let err = (0..d.h.len()).map(|i| { let x = (i as f64 + 0.5) * d.dx; (d.h[i] - f::ritter_depth(x, l / 2.0, d.t, g, h0)).abs() * d.dx }).sum::<f64>() / (h0 * l / 2.0);
    v.push(chk_max("Dam break (Ritter)", "depth error against the exact solution", err, 0.02));
    v.push(chk_max("Volume conservation", "shallow water, relative error", (d.h.iter().sum::<f64>() - v0).abs() / v0, 1e-10));
    v
}

pub fn chemistry_checks() -> Vec<Check> {
    use crate::chemistry as c;
    let (kf, kr) = (2.0, 0.5);
    let (a, b) = c::reversible(1.0, 0.0, kf, kr, 0.7, 2000);
    let exact = kr / (kf + kr) + kf / (kf + kr) * (-(kf + kr) * 0.7f64).exp();
    let (ae, be) = c::reversible(1.0, 0.0, kf, kr, 50.0, 20000);
    let mut v = vec![
        chk("First-order kinetics", "A ⇌ B against the exact solution", a, exact, 1e-8, ""),
        chk_max("Mass conservation of species", "A + B = const", (a + b - 1.0).abs(), 1e-12),
        chk("Chemical equilibrium", "B/A → K = k₊/k₋", be / ae, kf / kr, 1e-6, ""),
        chk("Arrhenius law", "Eₐ = 50 kJ/mol: growth of k at 298 → 308 K", c::arrhenius(1.0, 5e4, 308.0) / c::arrhenius(1.0, 5e4, 298.0), 1.927, 0.01, "times"),
    ];
    let ch = c::Chapman::stratosphere();
    let mut y = [1e7, 0.21 * ch.m, 1e12];
    let atoms0 = y[0] + 2.0 * y[1] + 3.0 * y[2];
    for _ in 0..4000 { y = ch.implicit_step(y, 3600.0); }
    v.push(chk("Chapman cycle", "steady O₃/O₂ against the analytic value", y[2] / y[1], ch.steady_ratio(), 0.02, ""));
    v.push(chk_max("Atom conservation", "Chapman cycle, relative error", (y[0] + 2.0 * y[1] + 3.0 * y[2] - atoms0).abs() / atoms0, 1e-8));
    // RS-25: hydrogen-oxygen, ~3600 K, M ≈ 13.5 g/mol, γ ≈ 1.2, expansion ratio 69:1. Rated vacuum Isp 452 s
    let (g, tc, mol): (f64, f64, f64) = (1.2, 3600.0, 0.0135);
    let (mut lo, mut hi): (f64, f64) = (1e-6, 0.5);
    for _ in 0..80 { let m = (lo * hi).sqrt(); if c::area_ratio(g, m) > 69.0 { lo = m } else { hi = m } }
    let pr = (lo * hi).sqrt();
    let ve = c::exhaust_velocity(g, tc, mol, pr);
    let area_e = c::area_ratio(g, pr);
    // vacuum impulse adds the exit pressure term to the velocity; estimate through the characteristic quantity
    let cstar = (g * crate::constants::R_GAS / mol * tc).sqrt() / (g * (2.0 / (g + 1.0)).powf((g + 1.0) / (2.0 * (g - 1.0))));
    let isp_vac = (ve + pr * area_e * cstar) / crate::env::G0;
    v.push(chk("Ideal nozzle", "hydrogen-oxygen, vacuum (RS-25 452 s, losses not included)", isp_vac, 452.0, 0.1, "s"));
    v
}

pub fn collision_checks() -> Vec<Check> {
    use crate::collisions as k;
    let (m, u, de) = k::merge(3.0, Vec3::new(2.0, 0.0, 1.0), 1.0, Vec3::new(-6.0, 4.0, 1.0));
    let p0 = Vec3::new(2.0, 0.0, 1.0) * 3.0 + Vec3::new(-6.0, 4.0, 1.0);
    let mut v = vec![
        chk_max("Momentum conservation", "inelastic merger", (u * m - p0).norm(), 1e-12),
        Check { law: "Second law", test: "kinetic energy is only lost in a merger".into(), value: format!("{de:.2} J"), expected: "> 0".into(), pass: de > 0.0 },
    ];
    // Giant impact hypothesis: 0.1 M⊕ at about escape velocity does not disrupt Earth
    let (me, mt, vesc) = (5.972e24, 6.4e23, 1.12e4);
    let q = k::impact_energy(me, mt, vesc * 1.1);
    let qs = k::disruption_energy(me + mt, 5500.0, 1.9);
    v.push(Check { law: "Disruption law (Leinhardt–Stewart)", test: "Theia impact does not disrupt Earth".into(), value: k::outcome(q, qs).into(), expected: "not catastrophic".into(), pass: q < qs });
    // Barringer Crater (Meteor Crater): about 1.2 km; body about 50 m, iron, 12.8 km/s
    v.push(chk("Crater scaling", "Barringer Crater (1.19 km), order of magnitude", k::crater_diameter(7800.0, 2500.0, 50.0, 12800.0, 9.81, 45.0).1 / 1e3, 1.19, 0.35, "km"));
    // Chicxulub: about 180 km; body about 12 km, 20 km/s
    v.push(chk("Crater scaling", "Chicxulub (≈180 km), order of magnitude", k::crater_diameter(2700.0, 2600.0, 12000.0, 20000.0, 9.81, 45.0).1 / 1e3, 180.0, 0.35, "km"));
    v.push(chk("Roche limit", "Saturn and ice: the outer edge of the A ring is observed at 2.27 R", k::roche_limit(1.0, 687.0, 934.0), 2.27, 0.05, "R"));
    v
}

pub fn tectonics_checks() -> Vec<Check> {
    use crate::tectonics as t;
    let l = t::Lithosphere::earth();
    let myr = YEAR * 1e6;
    let mut v = vec![
        chk("Heat conduction", "plate cooling: numerically against erf", l.temperature_numeric(5e4, 80.0 * myr), l.temperature_exact(5e4, 80.0 * myr), 0.005, "K"),
        chk("Lithosphere cooling", "ocean depth: ~350 m·√(Myr) (Parsons–Sclater)", l.subsidence(myr), 350.0, 0.12, "m"),
        chk("Lithosphere cooling", "heat flow in an 80 Myr plate (observed ≈ 55 mW/m²)", l.heat_flow(80.0 * myr) * 1e3, 55.0, 0.2, "mW/m²"),
    ];
    let ra = t::rayleigh(4000.0, 9.8, 3e-5, 3000.0, 2.9e6, 1e-6, 1e22);
    v.push(Check { law: "Mantle convection", test: "Rayleigh number above critical (≈ 10³)".into(), value: format!("{ra:.2e}"), expected: "> 1e3".into(), pass: ra > 1e3 });
    let s = t::convection_speed(1e-6, 2.9e6, ra) * YEAR * 100.0;
    v.push(Check { law: "Mantle convection", test: "plate speed, order of magnitude (cm/yr)".into(), value: format!("{s:.1} cm/yr"), expected: "1…30 cm/yr".into(), pass: (1.0..=30.0).contains(&s) });
    v
}

pub fn all() -> Vec<Check> {
    let mut v = Vec::new();
    v.extend(quantum_checks());
    v.extend(blackhole_checks());
    v.extend(fluid_checks());
    v.extend(chemistry_checks());
    v.extend(collision_checks());
    v.extend(tectonics_checks());
    v.extend(kepler());
    v.extend(conservation_solar_system(1.0));
    v.extend(mercury_precession());
    v.extend(atmosphere());
    v.extend(temperatures());
    v.extend(ballistics_checks());
    v.extend(magnetosphere());
    v.extend(stars_checks());
    v.extend(starsystem_checks());
    v.extend(galaxy_checks());
    v.extend(cosmology_checks());
    v.extend(climate_checks());
    v.extend(env_checks());
    v
}
