//! Генерация датасетов. Каждый сэмпл проходит проверку закона сохранения;
//! непрошедшие помечаются `valid=0` (или отбрасываются в режиме strict).
//!
//! Выход (каталог):
//!   params.csv  — параметры и итоги каждого сэмпла
//!   series.csv  — временные ряды (sample_id, t, ...)
//!   series_noisy.csv — то же с шумом измерений (если --noise > 0)
//!   meta.json   — задача, миры, seed, константы, статистика валидации

use crate::ballistics::{self, Launch, Options, Projectile};
use crate::constants::{AU, DAY, G, YEAR};
use crate::magneto::{self, ParticleLaunch};
use crate::nbody::{Body, NBody};
use crate::orbit::elements_to_state;
use crate::rng::Rng;
use crate::thermal::{self, ThermalOptions};
use crate::vec3::Vec3;
use crate::world::{World, WorldSpec};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct DatasetConfig {
    pub task: String,
    pub n: usize,
    pub worlds: Vec<WorldSpec>,
    pub seed: u64,
    pub out_dir: String,
    pub noise: f64,
    pub strict: bool,
    pub points: usize, // точек на временной ряд
}

pub struct Stats { pub generated: usize, pub valid: usize, pub max_error: f64 }

struct Sink {
    params: BufWriter<File>,
    series: BufWriter<File>,
    noisy: Option<BufWriter<File>>,
    rng_noise: Rng,
    noise: f64,
    header_done: bool,
}

impl Sink {
    fn row(&mut self, id: usize, vals: &[f64]) -> std::io::Result<()> {
        write!(self.series, "{id}")?;
        for v in vals { write!(self.series, ",{v:.9e}")?; }
        writeln!(self.series)?;
        if let Some(w) = &mut self.noisy {
            write!(w, "{id},{:.9e}", vals[0])?;
            for v in &vals[1..] { write!(w, ",{:.9e}", v * (1.0 + self.noise * self.rng_noise.normal()))?; }
            writeln!(w)?;
        }
        Ok(())
    }
}

pub fn generate(cfg: &DatasetConfig) -> Result<Stats, String> {
    let io = |e: std::io::Error| e.to_string();
    fs::create_dir_all(&cfg.out_dir).map_err(io)?;
    let dir = Path::new(&cfg.out_dir);
    let worlds: Vec<World> = cfg.worlds.iter().map(World::build).collect::<Result<_, _>>()?;
    let (series_cols, param_cols) = columns(&cfg.task)?;
    let mut sink = Sink {
        params: BufWriter::new(File::create(dir.join("params.csv")).map_err(io)?),
        series: BufWriter::new(File::create(dir.join("series.csv")).map_err(io)?),
        noisy: if cfg.noise > 0.0 { Some(BufWriter::new(File::create(dir.join("series_noisy.csv")).map_err(io)?)) } else { None },
        rng_noise: Rng::new(cfg.seed ^ 0xA5A5),
        noise: cfg.noise,
        header_done: false,
    };
    writeln!(sink.params, "sample_id,world,{},validation_error,valid", param_cols.join(",")).map_err(io)?;
    writeln!(sink.series, "sample_id,{}", series_cols.join(",")).map_err(io)?;
    if let Some(w) = &mut sink.noisy { writeln!(w, "sample_id,{}", series_cols.join(",")).map_err(io)?; }
    sink.header_done = true;

    let mut rng = Rng::new(cfg.seed);
    let mut stats = Stats { generated: 0, valid: 0, max_error: 0.0 };
    let mut id = 0usize;
    let mut attempts = 0usize;
    while id < cfg.n && attempts < cfg.n * 20 {
        attempts += 1;
        let wi = (rng.f64() * worlds.len() as f64) as usize % worlds.len();
        let w = &worlds[wi];
        let (params, rows, err, tol) = match cfg.task.as_str() {
            "projectile" => sample_projectile(w, &mut rng, cfg.points),
            "orbit" => sample_orbit(w, &mut rng, cfg.points),
            "thermal" => sample_thermal(w, &mut rng, cfg.points),
            "particle" => {
                if w.physics.dipole_moment <= 0.0 { return Err(format!("у мира {} нет магнитного диполя (используйте earth, jupiter, mercury)", w.physics.name)); }
                sample_particle(w, &mut rng, cfg.points)
            }
            "system" => sample_system(&mut rng, cfg.points),
            _ => unreachable!(),
        };
        let valid = err.is_finite() && err < tol;
        stats.generated += 1;
        if !valid && cfg.strict { continue; }
        if valid { stats.valid += 1; }
        stats.max_error = stats.max_error.max(if err.is_finite() { err } else { f64::INFINITY });
        let wname = if cfg.task == "system" { "random_star_system".to_string() } else { w.name().replace(' ', ";") };
        write!(sink.params, "{id},{wname}").map_err(io)?;
        for p in &params { write!(sink.params, ",{p:.9e}").map_err(io)?; }
        writeln!(sink.params, ",{err:.3e},{}", valid as u8).map_err(io)?;
        for r in &rows { sink.row(id, r).map_err(io)?; }
        id += 1;
    }
    sink.params.flush().map_err(io)?;
    sink.series.flush().map_err(io)?;
    if let Some(w) = &mut sink.noisy { w.flush().map_err(io)?; }
    write_meta(cfg, &worlds, &stats, dir).map_err(io)?;
    Ok(stats)
}

fn columns(task: &str) -> Result<(Vec<&'static str>, Vec<&'static str>), String> {
    Ok(match task {
        "projectile" => (vec!["t", "altitude", "downrange", "lat", "lon", "speed", "mach"],
                         vec!["lat0", "speed0", "elevation", "azimuth", "radius", "density", "mass", "g", "rho0", "flight_time", "range", "max_alt", "impact_speed", "dissipated_J"]),
        "orbit" => (vec!["t", "x", "y", "z", "vx", "vy", "vz", "r"],
                    vec!["a", "e", "inc", "node", "argp", "mean_anom", "gm", "period"]),
        "thermal" => (vec!["local_time_h", "surface_T"],
                      vec!["lat", "declination", "thermal_inertia", "insolation", "albedo", "tau", "t_min", "t_max", "t_mean"]),
        "particle" => (vec!["t", "x", "y", "z", "lat"],
                       vec!["l_shell", "energy_mev", "pitch", "charge", "mass", "bounces", "mirror_lat_theory", "mirror_lat_sim", "drift_deg"]),
        "system" => (vec!["t", "body", "x", "y", "z", "vx", "vy", "vz"],
                     vec!["n_bodies", "star_gm", "total_planet_gm", "years"]),
        _ => return Err(format!("неизвестная задача '{task}': projectile, orbit, thermal, particle, system")),
    })
}

type Sample = (Vec<f64>, Vec<Vec<f64>>, f64, f64);

fn sample_projectile(w: &World, rng: &mut Rng, points: usize) -> Sample {
    let lat0 = rng.uniform(-80.0, 80.0);
    let speed = rng.log_uniform(5.0, 1500.0);
    let elev = rng.uniform(5.0, 85.0);
    let az = rng.uniform(0.0, 360.0);
    let radius = rng.log_uniform(0.01, 0.5);
    let dens = rng.uniform(300.0, 8000.0);
    let p = Projectile::sphere(radius, dens);
    // грубая оценка времени полёта для шага выборки
    let t_guess = (2.0 * speed * elev.to_radians().sin() / w.g_surface()).max(0.5);
    let opts = Options { sample_dt: t_guess / points as f64, max_time: t_guess * 3.0 + 10.0, ..Default::default() };
    let r = ballistics::simulate(w, p, Launch { lat: lat0, lon: 0.0, altitude: 0.0, speed, elevation: elev, azimuth: az }, opts);
    let rows = r.samples.iter().map(|s| vec![s.t, s.altitude, s.downrange, s.lat, s.lon, s.speed, if s.mach.is_finite() { s.mach } else { 0.0 }]).collect();
    let params = vec![lat0, speed, elev, az, radius, dens, p.mass, w.g_surface(), w.density(0.0), r.flight_time, r.downrange, r.max_altitude, r.impact_speed, r.dissipated_energy];
    (params, rows, r.energy_error, 1e-7)
}

fn sample_orbit(w: &World, rng: &mut Rng, points: usize) -> Sample {
    let rr = w.physics.radius;
    let gm = w.physics.gm;
    let peri = rr * rng.uniform(1.03, 3.0);
    let e = rng.uniform(0.0, 0.6);
    let a = peri / (1.0 - e);
    let (inc, node, argp, m0) = (rng.uniform(0.0, 180.0), rng.uniform(0.0, 360.0), rng.uniform(0.0, 360.0), rng.uniform(0.0, 360.0));
    let (r, v) = elements_to_state(gm, a, e, inc, node, argp, m0);
    let mut s = NBody::new(vec![
        Body { name: "planet".into(), gm, r: Vec3::ZERO, v: Vec3::ZERO },
        Body { name: "sat".into(), gm: gm * 1e-12, r, v },
    ]);
    let period = 2.0 * std::f64::consts::PI * (a.powi(3) / gm).sqrt();
    let total = 2.0 * period;
    let sub = 200usize; // шагов интегратора на точку
    let dt = total / (points * sub) as f64;
    let i0 = s.invariants();
    let mut rows = Vec::with_capacity(points + 1);
    let mut err: f64 = 0.0;
    for k in 0..=points {
        if k > 0 { for _ in 0..sub { s.step(dt); } }
        let rel_r = s.bodies[1].r - s.bodies[0].r;
        let rel_v = s.bodies[1].v - s.bodies[0].v;
        rows.push(vec![s.t, rel_r.x, rel_r.y, rel_r.z, rel_v.x, rel_v.y, rel_v.z, rel_r.norm()]);
        let i = s.invariants();
        err = err.max(((i.energy - i0.energy) / i0.energy).abs());
    }
    (vec![a, e, inc, node, argp, m0, gm, period], rows, err, 1e-8)
}

fn sample_thermal(w: &World, rng: &mut Rng, points: usize) -> Sample {
    let lat = rng.uniform(-70.0, 70.0);
    let obl = w.physics.obliquity.min(180.0 - w.physics.obliquity);
    let dec = rng.uniform(-obl, obl);
    let mut ww = w.clone();
    ww.surface.thermal_inertia *= rng.log_uniform(0.5, 2.0);
    let spd = (points * 4).max(240);
    let r = thermal::simulate(&ww, ThermalOptions { latitude: lat, declination: dec, steps_per_day: spd, spinup_days: 8, cells: 60 });
    let stride = (r.surface_t.len() / points).max(1);
    let rows = r.local_time_h.iter().zip(&r.surface_t).step_by(stride).map(|(t, s)| vec![*t, *s]).collect();
    let params = vec![lat, dec, ww.surface.thermal_inertia, ww.insolation(), ww.physics.bond_albedo, ww.greenhouse_tau(), r.t_min, r.t_max, r.t_mean];
    (params, rows, r.energy_residual, 1e-6)
}

fn sample_particle(w: &World, rng: &mut Rng, points: usize) -> Sample {
    let names = ["proton", "electron", "alpha"];
    let sp = magneto::species(names[(rng.f64() * 3.0) as usize % 3]).unwrap();
    let l = rng.uniform(2.0, 6.0);
    let en = rng.log_uniform(0.05, 20.0);
    let pitch = rng.uniform(15.0, 89.0);
    // длительность = 3 периода баунс-колебаний: T_b ≈ 4LR/v · (1.30 − 0.56 sin α)
    let gamma = 1.0 + en * crate::constants::MEV / (sp.mass * crate::constants::C.powi(2));
    let v = crate::constants::C * (1.0 - 1.0 / (gamma * gamma)).sqrt();
    let t_bounce = 4.0 * l * w.physics.radius / v * (1.30 - 0.56 * pitch.to_radians().sin());
    // бюджет ≤ ~2·10⁶ шагов Бориса (у Юпитера электроны вращаются в ~10⁷ раз быстрее, чем баунсят)
    let b_eq = magneto::dipole_field(w.physics.dipole_moment, crate::vec3::Vec3::new(l * w.physics.radius, 0.0, 0.0)).norm();
    let gyro_eq = 2.0 * std::f64::consts::PI * gamma * sp.mass / (sp.charge.abs() * b_eq);
    let budget = 2.0e6 * gyro_eq / 40.0 * pitch.to_radians().sin().powi(2);
    let duration = (3.0 * t_bounce).min(budget);
    let r = magneto::simulate(w, sp, ParticleLaunch { l_shell: l, energy_mev: en, pitch_angle: pitch }, duration, points);
    let rows = r.t.iter().zip(&r.pos).map(|(t, p)| vec![*t, p.x, p.y, p.z, (p.z / p.norm()).asin().to_degrees()]).collect();
    (vec![l, en, pitch, sp.charge, sp.mass, duration / t_bounce, r.mirror_lat_theory, r.max_abs_lat, r.drift_longitude], rows, r.energy_drift, 1e-9)
}

/// Случайная планетная система: звезда 0.5–1.5 M☉, 2–6 планет от массы Марса до Юпитера.
fn sample_system(rng: &mut Rng, points: usize) -> Sample {
    let star = 1.327_124_400_18e20 * rng.uniform(0.5, 1.5);
    let n = 2 + (rng.f64() * 5.0) as usize;
    let mut bodies = vec![Body { name: "star".into(), gm: star, r: Vec3::ZERO, v: Vec3::ZERO }];
    let mut a = AU * rng.uniform(0.3, 0.8);
    let mut total = 0.0;
    for k in 0..n {
        let gm = rng.log_uniform(4.28e13, 1.27e17);
        total += gm;
        let (r, v) = elements_to_state(star + gm, a, rng.uniform(0.0, 0.1), rng.uniform(0.0, 5.0), rng.uniform(0.0, 360.0), rng.uniform(0.0, 360.0), rng.uniform(0.0, 360.0));
        bodies.push(Body { name: format!("p{k}"), gm, r, v });
        a *= rng.uniform(1.4, 2.2); // разнос орбит ради устойчивости
    }
    let mut s = NBody::new(bodies);
    s.to_barycentric();
    let years = 10.0;
    let p_in = 2.0 * std::f64::consts::PI * ((AU * 0.3).powi(3) / star).sqrt();
    let dt = p_in / 400.0;
    let sub = ((years * YEAR / points as f64) / dt).ceil() as usize;
    let i0 = s.invariants();
    let mut rows = Vec::new();
    let mut err: f64 = 0.0;
    for k in 0..=points {
        if k > 0 { for _ in 0..sub { s.step(years * YEAR / points as f64 / sub as f64); } }
        for (j, b) in s.bodies.iter().enumerate() {
            rows.push(vec![s.t, j as f64, b.r.x, b.r.y, b.r.z, b.v.x, b.v.y, b.v.z]);
        }
        err = err.max(((s.invariants().energy - i0.energy) / i0.energy).abs());
    }
    let _ = (DAY, G);
    (vec![(n + 1) as f64, star, total, years], rows, err, 1e-8)
}

fn write_meta(cfg: &DatasetConfig, worlds: &[World], st: &Stats, dir: &Path) -> std::io::Result<()> {
    let mut f = File::create(dir.join("meta.json"))?;
    let ws: Vec<String> = worlds.iter().map(|w| format!(
        "    {{\"physics\": \"{}\", \"atmosphere\": \"{}\", \"surface\": \"{}\", \"g\": {:.6}, \"radius_m\": {:.1}, \"surface_pressure_pa\": {:.3}, \"mean_surface_T\": {:.2}}}",
        w.physics.name, w.atmosphere.as_ref().map(|a| a.preset.name.as_str()).unwrap_or("none"), w.surface.name,
        w.g_surface(), w.physics.radius, w.atmosphere.as_ref().map(|a| a.preset.surface_pressure).unwrap_or(0.0), w.mean_surface_temperature())).collect();
    writeln!(f, "{{\n  \"generator\": \"worldsim {}\",\n  \"task\": \"{}\",\n  \"seed\": {},\n  \"requested\": {},\n  \"generated\": {},\n  \"valid\": {},\n  \"max_validation_error\": {:e},\n  \"noise_rel_sigma\": {},\n  \"constants\": {{\"G\": {:e}, \"c\": {:e}, \"sigma_SB\": {:e}}},\n  \"worlds\": [\n{}\n  ]\n}}",
        env!("CARGO_PKG_VERSION"), cfg.task, cfg.seed, cfg.n, st.generated, st.valid, st.max_error, cfg.noise,
        G, crate::constants::C, crate::constants::SIGMA_SB, ws.join(",\n"))
}
