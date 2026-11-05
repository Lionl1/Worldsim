//! Mission Lab: from a mission description to an engineering answer.
//!
//! Input: JSON with the vehicle, dispersions and requirements. Output:
//! 1) requirements scorecard: met / not met / not enough data, with a confidence interval;
//! 2) main drivers: which dispersion hurts the mission most (one at a time, like OVAT in the analysis of EDL);
//! 3) value of technologies: what each improvement buys and what it costs in mass.
//!
//! The result does not depend on thread count. This is a model for early design decisions and training,
//! not a replacement for high-fidelity tools and testing.

use crate::collect::episode_seed;
use crate::env::{param_def, param_default, Env, EnvConfig, Task};
use crate::json::J;
use crate::rng::Rng;
use crate::vec3::Vec3;
use crate::world::{World, WorldSpec};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

const METRICS: &[&str] = &["success", "crashed", "touch_speed", "touch_tilt", "touch_offset", "fuel_used", "energy_used_kj", "duration_s", "peak_temp_k", "min_temp_k", "max_slip", "overruns"];
/// Dispersions with a special meaning; any other registry parameter can be dispersed relatively.
const SPECIAL_DISPERSIONS: &[&str] = &["gravity", "nav_noise", "gust", "actuator_lag", "fault"];
const RELATIVE_FIRST: &[&str] = &["dry_mass", "isp", "t_max", "torque_max", "rover_mass", "motor_force"];
const MAX_RUNS: usize = 20_000;

#[derive(Clone, Debug)]
pub struct Requirement { pub id: String, pub text: String, pub metric: String, pub percentile: Option<f64>, pub op: String, pub value: f64 }

#[derive(Clone, Debug, Default)]
pub struct Technology { pub name: String, pub set: Vec<(String, f64)>, pub set_dispersion: Vec<(String, f64)>, pub set_top: Vec<(String, f64)>, pub dry_mass_delta: f64 }

/// Calculation progress: a background job shows it and can cancel.
#[derive(Default)]
pub struct Progress { pub done: AtomicUsize, pub total: AtomicUsize, pub cancel: AtomicBool }

#[derive(Clone, Debug)]
pub struct MissionSpec {
    pub name: String, pub task: String, pub world: WorldSpec, pub runs: usize, pub seed: u64, pub rotor_scale: f64,
    pub vehicle: Vec<(String, f64)>, pub dispersions: Vec<(String, f64)>, pub requirements: Vec<Requirement>, pub technologies: Vec<Technology>,
    pub terrain: Option<std::sync::Arc<crate::env::Terrain>>,
    pub terrain_sha: Option<String>, // terrain file fingerprint: the mission description stores only the path
}

fn pairs(j: Option<&J>, ctx: &str) -> Result<Vec<(String, f64)>, String> {
    let Some(j) = j else { return Ok(vec![]) };
    j.as_obj().ok_or(format!("{ctx}: expected an object"))?.iter().map(|(k, v)| {
        let x = v.as_f64().ok_or(format!("{ctx}.{k}: expected a number"))?;
        if !x.is_finite() { return Err(format!("{ctx}.{k}: not a number")); }
        Ok((k.clone(), x))
    }).collect()
}

impl MissionSpec {
    /// Description where terrain may be in a file. For local calls only: the API does not read server files.
    pub fn parse_local(text: &str) -> Result<MissionSpec, String> {
        let j = J::parse(text)?;
        let mut spec = Self::parse_inner(text, true)?;
        if let Some(t) = j.get("terrain") {
            if let Some(path) = t.get("file").and_then(|x| x.as_str()) {
                let fr = t.get("friction").and_then(|x| x.as_f64()).unwrap_or(0.6);
                spec.terrain_sha = Some(crate::store::sha256(&std::fs::read(path).map_err(|e| format!("{path}: {e}"))?));
                spec.terrain = Some(std::sync::Arc::new(crate::dem::load(path, t.get("size").and_then(|x| x.as_f64()), fr)?));
            }
        }
        Ok(spec)
    }

    /// Description from an API client: reading server files by a path from the request is not allowed.
    pub fn parse(text: &str) -> Result<MissionSpec, String> { Self::parse_inner(text, false) }

    fn parse_inner(text: &str, allow_file: bool) -> Result<MissionSpec, String> {
        let j = J::parse(text)?;
        if !allow_file && j.get("terrain").and_then(|t| t.get("file")).is_some() {
            return Err("terrain.file: the API does not read server files, pass terrain.n, terrain.size and terrain.heights".into());
        }
        let st = |k: &str, d: &str| j.get(k).and_then(|x| x.as_str()).unwrap_or(d).to_string();
        let task = st("task", "lander");
        let mut check = EnvConfig::new(&task, "earth")?;
        let task_kind = check.task;
        let world = WorldSpec::parse(&st("world", "moon"));
        World::build(&world)?;
        let runs = j.get("runs").and_then(|x| x.as_f64()).unwrap_or(200.0);
        if !(1.0..=MAX_RUNS as f64).contains(&runs) { return Err(format!("runs: must be from 1 to {MAX_RUNS}")); }
        let rotor_scale = j.get("rotor_scale").and_then(|x| x.as_f64()).unwrap_or(1.0);
        if rotor_scale <= 0.0 { return Err("rotor_scale must be greater than zero".into()); }
        // parameters can be split into topical sections, for the engine this is one list
        let mut vehicle = pairs(j.get("vehicle"), "vehicle")?;
        for section in ["scenario", "environment", "sensors", "control", "criteria", "params"] { vehicle.extend(pairs(j.get(section), section)?); }
        for (k, v) in &vehicle { check.set(k, *v).map_err(|e| format!("mission parameters: {e}"))?; }
        let mut dispersions = pairs(j.get("dispersions"), "dispersions")?;
        let mut rotor_scale = rotor_scale;
        // these can be set both as a parameter and as a dispersion; stored one way, otherwise a parameter would silently override a technology
        vehicle.retain(|(k, v)| {
            let moved = match k.as_str() { "noise" => "nav_noise", "gust" => "gust", "actuator_lag" => "actuator_lag", "fault_prob" => "fault", _ => "" };
            if k == "rotor_scale" { rotor_scale = *v; return false; }
            if moved.is_empty() { return true; }
            if !dispersions.iter().any(|(n, _)| n == moved) { dispersions.push((moved.to_string(), *v)); }
            false
        });
        for (k, v) in &dispersions {
            if !SPECIAL_DISPERSIONS.contains(&k.as_str()) {
                param_def(k).ok_or(format!("dispersions.{k}: unknown parameter"))?;
                let nom = nominal(task_kind, k, &vehicle);
                if nom == 0.0 { return Err(format!("dispersions.{k}: the parameter has no nonzero value to compute a dispersion from")); }
                check.clone().set(k, nom).map_err(|e| format!("dispersions: {e}"))?;
            }
            if *v < 0.0 { return Err(format!("dispersions.{k}: cannot be negative")); }
        }
        let mut requirements = vec![];
        for (i, r) in j.get("requirements").and_then(|x| x.as_arr()).unwrap_or(&[]).iter().enumerate() {
            let metric = r.get("metric").and_then(|x| x.as_str()).ok_or(format!("requirements[{i}].metric: must be specified"))?.to_string();
            if !METRICS.contains(&metric.as_str()) { return Err(format!("requirements[{i}].metric = {metric}: available {METRICS:?}")); }
            let op = r.get("op").and_then(|x| x.as_str()).unwrap_or("<=").to_string();
            if op != "<=" && op != ">=" { return Err(format!("requirements[{i}].op: only <= or >=")); }
            let value = r.get("value").and_then(|x| x.as_f64()).ok_or(format!("requirements[{i}].value: a number is required"))?;
            let percentile = r.get("percentile").and_then(|x| x.as_f64());
            if let Some(p) = percentile { if !(0.0..=100.0).contains(&p) { return Err(format!("requirements[{i}].percentile: from 0 to 100")); } }
            // a requirement on a model that is off would pass silently
            let on = |k: &str| vehicle.iter().find(|(n, _)| n == k).map(|x| x.1).unwrap_or(0.0) > 0.0;
            if (metric == "peak_temp_k" || metric == "min_temp_k") && !on("thermal_mass") { return Err(format!("requirements[{i}].metric = {metric}: set thermal_mass, the thermal model is off")); }
            if metric == "max_slip" && !on("terramechanics") { return Err(format!("requirements[{i}].metric = max_slip: set terramechanics = 1, the soil model is off")); }
            let binary = metric == "success" || metric == "crashed";
            if !binary && percentile.is_none() { return Err(format!("requirements[{i}]: for {metric} specify percentile")); }
            requirements.push(Requirement { id: r.get("id").and_then(|x| x.as_str()).map(|s| s.to_string()).unwrap_or(format!("REQ-{:02}", i + 1)),
                text: r.get("text").and_then(|x| x.as_str()).unwrap_or(&metric).to_string(), metric, percentile, op, value });
        }
        if requirements.is_empty() { return Err("requirements: at least one requirement is required".into()); }
        let mut technologies = vec![];
        for (i, t) in j.get("technologies").and_then(|x| x.as_arr()).unwrap_or(&[]).iter().enumerate() {
            let mut tech = Technology {
                name: t.get("name").and_then(|x| x.as_str()).map(|s| s.to_string()).unwrap_or(format!("Technology {}", i + 1)),
                set: pairs(t.get("set"), "technologies.set")?, set_dispersion: pairs(t.get("set_dispersion"), "technologies.set_dispersion")?,
                set_top: pairs(t.get("set_top"), "technologies.set_top")?, dry_mass_delta: t.get("dry_mass_delta").and_then(|x| x.as_f64()).unwrap_or(0.0) };
            // the same normalization as for mission parameters
            for k in ["noise", "gust", "actuator_lag", "fault_prob", "rotor_scale"] {
                if let Some(p) = tech.set.iter().position(|(n, _)| n == k) {
                    let (_, v) = tech.set.remove(p);
                    match k {
                        "rotor_scale" => tech.set_top.push((k.to_string(), v)),
                        "noise" => tech.set_dispersion.push(("nav_noise".to_string(), v)),
                        "fault_prob" => tech.set_dispersion.push(("fault".to_string(), v)),
                        _ => tech.set_dispersion.push((k.to_string(), v)),
                    }
                }
            }
            let mut c = check.clone();
            for (k, v) in &tech.set { c.set(k, *v).map_err(|e| format!("technologies[{i}].set: {e}"))?; }
            if tech.dry_mass_delta != 0.0 {
                if task_kind != Task::Lander { return Err(format!("technologies[{i}].dry_mass_delta: dry mass delta exists only for the lander; on other vehicles set the mass through set")); }
                let dm = tech.set.iter().find(|(n, _)| n == "dry_mass").map(|x| x.1).unwrap_or(nominal(task_kind, "dry_mass", &vehicle)) + tech.dry_mass_delta;
                c.set("dry_mass", dm).map_err(|e| format!("technologies[{i}]: {e}"))?;
            }
            for (k, _) in &tech.set_dispersion { if !SPECIAL_DISPERSIONS.contains(&k.as_str()) && param_def(k).is_none() { return Err(format!("technologies[{i}].set_dispersion.{k}: unknown parameter")); } }
            for (k, _) in &tech.set_top { if k != "rotor_scale" { return Err(format!("technologies[{i}].set_top.{k}: only available rotor_scale")); } }
            technologies.push(tech);
        }
        Ok(MissionSpec { name: st("name", "mission"), task, world, runs: runs as usize, seed: j.get("seed").and_then(|x| x.as_f64()).unwrap_or(1.0) as u64,
            rotor_scale, vehicle, dispersions, requirements, technologies, terrain: parse_inline_terrain(&j)?, terrain_sha: None })
    }
}

/// Terrain directly in the mission description: `{"size": 400, "n": 65, "heights": [...]}`. File `{"file": ...}` allowed only locally.
fn parse_inline_terrain(j: &J) -> Result<Option<std::sync::Arc<crate::env::Terrain>>, String> {
    let Some(t) = j.get("terrain") else { return Ok(None) };
    if t.get("file").is_some() { return Ok(None); }
    let n = t.get("n").and_then(|x| x.as_f64()).ok_or("terrain.n: grid side is required")? as usize;
    let size = t.get("size").and_then(|x| x.as_f64()).ok_or("terrain.size: map size is required, m")?;
    // null means a data hole; any other non-numeric value is an error, not a silently filled point
    let h: Vec<f64> = t.get("heights").and_then(|x| x.as_arr()).ok_or("terrain.heights: height array is required")?.iter().map(|v| match v { J::Null => Ok(f64::NAN), _ => v.as_f64().ok_or("terrain.heights: only numbers and null".to_string()) }).collect::<Result<_, _>>()?;
    Ok(Some(std::sync::Arc::new(crate::dem::terrain_from_grid(h, n, size, t.get("friction").and_then(|x| x.as_f64()).unwrap_or(0.6))?)))
}

/// Parameter value: set in the mission or the default; dispersion is computed from it.
fn nominal(task: Task, k: &str, spec: &[(String, f64)]) -> f64 {
    spec.iter().find(|(n, _)| n == k).map(|(_, v)| *v).or(param_default(task, k)).unwrap_or(1.0)
}

#[derive(Clone, Debug, Default)]
struct RunResult { success: f64, crashed: f64, touch_speed: f64, touch_tilt: f64, touch_offset: f64, fuel_used: f64, energy_kj: f64, duration: f64, peak_temp: f64, min_temp: f64, max_slip: f64, overruns: f64, reason: String }

/// Reason without numbers, so similar failures are counted together.
fn failure_mode(reason: &str) -> String {
    let head = reason.split(':').next().unwrap_or(reason);
    let cut = head.find(|c: char| c.is_ascii_digit()).map(|i| &head[..i]).unwrap_or(head);
    let t = cut.trim().trim_end_matches(" at").trim();
    if t.is_empty() { "no reason".into() } else { t.to_string() }
}

/// Failures by type, most frequent first. For landing, specifies what exactly was violated at touchdown.
fn failure_modes(rs: &[RunResult]) -> J {
    let mut m: Vec<(String, usize)> = vec![];
    for r in rs.iter().filter(|r| r.success < 0.5) {
        let k = if r.reason.starts_with("crash") {
            if r.touch_speed >= 2.5 { "crash: high touchdown speed".to_string() } else if r.touch_tilt >= 15.0 { "crash: high tilt".to_string() } else { "crash: steep slope under the legs".to_string() }
        } else { failure_mode(&r.reason) };
        match m.iter_mut().find(|x| x.0 == k) { Some(x) => x.1 += 1, None => m.push((k, 1)) }
    }
    m.sort_by(|a, b| b.1.cmp(&a.1));
    J::Arr(m.into_iter().map(|(k, n)| J::obj().set("mode", k).set("count", n).set("share", n as f64 / rs.len().max(1) as f64)).collect())
}

impl RunResult {
    fn metric(&self, m: &str) -> f64 {
        match m { "success" => self.success, "crashed" => self.crashed, "touch_speed" => self.touch_speed, "touch_tilt" => self.touch_tilt, "touch_offset" => self.touch_offset,
            "fuel_used" => self.fuel_used, "energy_used_kj" => self.energy_kj, "peak_temp_k" => self.peak_temp, "min_temp_k" => self.min_temp, "max_slip" => self.max_slip, "overruns" => self.overruns, _ => self.duration }
    }
}

fn disp(d: &[(String, f64)], k: &str) -> f64 { d.iter().find(|(n, _)| n == k).map(|(_, v)| *v).unwrap_or(0.0) }

/// Environment of a single flight. Dispersions: relative ones are normal, truncated at ±3σ.
fn flight_env(spec: &MissionSpec, vehicle: &[(String, f64)], d: &[(String, f64)], rotor_scale: f64, seed: u64) -> Result<Env, String> {
    let mut r = Rng::new(seed ^ 0xD15B);
    let mut cfg = EnvConfig::new(&spec.task, "earth")?;
    cfg.world = spec.world.clone();
    let mut n3 = || r.normal().clamp(-3.0, 3.0);
    cfg.world.gravity_scale *= (1.0 + disp(d, "gravity") * n3()).max(0.05);
    cfg.rotor_scale = rotor_scale;
    cfg.noise = disp(d, "nav_noise");
    cfg.gust = disp(d, "gust");
    cfg.actuator_lag = disp(d, "actuator_lag");
    cfg.fault_prob = disp(d, "fault").min(1.0);
    let task = cfg.task;
    let mut veh = vehicle.to_vec();
    let extras = d.iter().map(|x| x.0.as_str()).filter(|k| !RELATIVE_FIRST.contains(k) && !SPECIAL_DISPERSIONS.contains(k));
    for k in RELATIVE_FIRST.iter().copied().chain(extras) {
        let s = disp(d, k);
        if s > 0.0 {
            let def = param_def(k).ok_or(format!("unknown parameter {k}"))?;
            let v = (nominal(task, k, vehicle) * (1.0 + s * n3()).max(0.05)).clamp(def.min, def.max);
            veh.retain(|(n, _)| n != k);
            veh.push((k.to_string(), v));
        }
    }
    for (k, v) in &veh { cfg.set(k, *v)?; }
    cfg.custom_terrain = spec.terrain.clone();
    let mut env = Env::new(cfg)?;
    env.reset(seed);
    Ok(env)
}

fn result_of(env: &Env) -> RunResult {
    let i = &env.info;
    RunResult { success: i.success as u8 as f64, crashed: i.crashed as u8 as f64, touch_speed: i.touch_speed, touch_tilt: i.touch_tilt, touch_offset: i.touch_offset,
        fuel_used: i.fuel_used, energy_kj: i.energy_used / 1e3, duration: env.t, peak_temp: i.peak_temp, min_temp: i.min_temp, max_slip: i.max_slip, overruns: i.overruns as f64, reason: i.reason.clone() }
}

/// One flight with the built-in autopilot.
fn fly(spec: &MissionSpec, vehicle: &[(String, f64)], d: &[(String, f64)], rotor_scale: f64, seed: u64) -> Result<RunResult, String> {
    let mut env = flight_env(spec, vehicle, d, rotor_scale, seed)?;
    // navigation estimate: noisy GPS and velocity smoothed by a filter; ideal sensors give zero error
    let (mut pe, mut ve) = (Vec3::ZERO, Vec3::ZERO);
    let flying = env.cfg.task != Task::Rover;
    let sign = if env.cfg.task == Task::Lander { 1.0 } else { -1.0 };
    loop {
        let a = env.autopilot_nav(pe, ve);
        let (o, _, t, c) = env.step(&a);
        if t || c { break; }
        let rel = env.true_rel_pos();
        if flying {
            let v = env.velocity();
            pe = pe * 0.9 + Vec3::new(o[0] - rel.x, o[1] - rel.y, o[2] - rel.z) * (0.1 * sign);
            ve = ve * 0.8 + Vec3::new(o[3] - v.x, o[4] - v.y, o[5] - v.z) * 0.2;
        } else {
            // rover: noisy position to the goal in the plane
            pe = pe * 0.9 + Vec3::new(o[0] - rel.x, o[1] - rel.y, 0.0) * (0.1 * sign);
        }
    }
    Ok(result_of(&env))
}

/// Same flights and requirements, but controlled from outside: Python calls `step` and passes actions of all flights at once.
pub struct Runner { pub spec: MissionSpec, pub envs: Vec<Env>, pub done: Vec<bool>, threads: usize, baseline: Option<Vec<RunResult>>,
    plan: Vec<Variant>, current: usize, outcomes: Vec<Option<Vec<RunResult>>>, unfinished: usize }

/// One flight series of the analysis: the mission itself, no dispersions, one driver or one technology.
#[derive(Clone)]
struct Variant { kind: &'static str, name: String, sigma: f64, dry_mass_delta: f64, vehicle: Vec<(String, f64)>, dispersions: Vec<(String, f64)>, rotor: f64, runs: usize }

/// Same order and same flight counts as in `run_with`, so the answers can be compared directly.
fn analysis_plan(spec: &MissionSpec) -> Result<Vec<Variant>, String> {
    let driver_runs = spec.runs.min(300);
    let v = |kind, name: &str, sigma, d: Vec<(String, f64)>, runs| Variant { kind, name: name.to_string(), sigma, dry_mass_delta: 0.0, vehicle: spec.vehicle.clone(), dispersions: d, rotor: spec.rotor_scale, runs };
    let mut plan = vec![v("base", "mission", 0.0, spec.dispersions.clone(), spec.runs), v("clean", "no dispersions", 0.0, vec![], driver_runs)];
    for (k, s) in spec.dispersions.iter().filter(|d| d.1 != 0.0) { plan.push(v("driver", k, *s, vec![(k.clone(), *s)], driver_runs)); }
    for t in &spec.technologies {
        let (vehicle, dispersions, rotor) = tech_setup(spec, t)?;
        plan.push(Variant { kind: "technology", name: t.name.clone(), sigma: 0.0, dry_mass_delta: t.dry_mass_delta, vehicle, dispersions, rotor, runs: spec.runs });
    }
    Ok(plan)
}

/// What a technology changes: vehicle parameters, dispersions and rotor size.
fn tech_setup(spec: &MissionSpec, t: &Technology) -> Result<(Vec<(String, f64)>, Vec<(String, f64)>, f64), String> {
    let mut veh = spec.vehicle.clone();
    for (k, v) in &t.set { veh.retain(|(n, _)| n != k); veh.push((k.clone(), *v)); }
    if t.dry_mass_delta != 0.0 {
        let dm = t.set.iter().find(|(n, _)| n == "dry_mass").map(|x| x.1).unwrap_or(nominal(EnvConfig::new(&spec.task, "earth")?.task, "dry_mass", &veh)) + t.dry_mass_delta;
        veh.retain(|(n, _)| n != "dry_mass");
        veh.push(("dry_mass".into(), dm));
    }
    let mut d = spec.dispersions.clone();
    for (k, v) in &t.set_dispersion { d.retain(|(n, _)| n != k); d.push((k.clone(), *v)); }
    Ok((veh, d, t.set_top.iter().find(|(n, _)| n == "rotor_scale").map(|x| x.1).unwrap_or(spec.rotor_scale)))
}

/// Everything after the report head: scorecard, failure modes, drivers and technologies. Shared by the autopilot and a custom controller.
fn analysis(head: J, spec: &MissionSpec, base: &[RunResult], clean: &[RunResult], drivers: &[(&Variant, &[RunResult])], techs: &[(&Variant, &[RunResult])]) -> J {
    let base_s = success_rate(base);
    let clean_s = success_rate(clean);
    let mut dr: Vec<(String, f64, f64, f64)> = drivers.iter().map(|(v, rs)| { let s = success_rate(rs); (v.name.clone(), v.sigma, s, s - clean_s) }).collect();
    dr.sort_by(|a, b| a.3.total_cmp(&b.3));
    let mut tr = vec![];
    for (v, rs) in techs {
        let s = success_rate(rs);
        // paired comparison: under the same conditions, what was saved and what was lost
        let saved = rs.iter().zip(base).filter(|(a, b)| a.success > 0.5 && b.success < 0.5).count();
        let lost = rs.iter().zip(base).filter(|(a, b)| a.success < 0.5 && b.success > 0.5).count();
        let se = ((saved + lost) as f64).sqrt() / spec.runs as f64;
        tr.push((v.name.clone(), s, s - base_s, saved, lost, 1.96 * se, v.dry_mass_delta, margin_score(spec, rs), scorecard(spec, rs)));
    }
    tr.sort_by(|a, b| b.2.total_cmp(&a.2));
    let k = base.iter().map(|r| r.success).sum::<f64>();
    let (sl, sh) = wilson(k, spec.runs as f64);
    head.set("success", base_s).set("success_ci", vec![sl, sh]).set("requirements_met", margin_score(spec, base)).set("requirements_total", spec.requirements.len())
        .set("scorecard", scorecard(spec, base))
        .set("failure_modes", failure_modes(base))
        .set("baseline_no_dispersions", clean_s).set("driver_runs", clean.len())
        .set("drivers", J::Arr(dr.into_iter().map(|(k, v, s, ds)| J::obj().set("name", k).set("sigma", v).set("success", s).set("delta", ds)).collect()))
        .set("technologies", J::Arr(tr.into_iter().map(|(n, s, ds, sv, ls, ci, dm, met, sc)| J::obj().set("name", n).set("success", s).set("delta", ds)
            .set("delta_ci", ci).set("saved", sv).set("lost", ls).set("dry_mass_delta", dm).set("requirements_met", met).set("scorecard", sc)
            .set("delta_per_kg", if dm > 0.0 { ds / dm } else { f64::NAN })).collect()))
        .set("histograms", J::obj()
            .set("touch_speed", base.iter().filter(|r| r.touch_speed > 0.0).map(|r| r.touch_speed).collect::<Vec<_>>())
            .set("touch_offset", base.iter().filter(|r| r.touch_offset > 0.0).map(|r| r.touch_offset).collect::<Vec<_>>())
            .set("fuel_used", base.iter().map(|r| r.fuel_used).collect::<Vec<_>>()))
        .set("limits", "Simplified physics for early design decisions: does not replace high-fidelity tools, testing and qualification.")
}

/// Each flight has its own terrain map; keeping more than this in memory makes no sense.
pub const MAX_RUNNER_FLIGHTS: usize = 5000;

impl Runner {
    pub fn new(spec: MissionSpec, threads: usize) -> Result<Runner, String> {
        if spec.runs > MAX_RUNNER_FLIGHTS { return Err(format!("runs = {}: for a custom controller at most {MAX_RUNNER_FLIGHTS} flights", spec.runs)); }
        let envs = (0..spec.runs).map(|i| flight_env(&spec, &spec.vehicle, &spec.dispersions, spec.rotor_scale, episode_seed(spec.seed, i as u64))).collect::<Result<Vec<_>, _>>()?;
        let plan = analysis_plan(&spec)?;
        Ok(Runner { done: vec![false; envs.len()], envs, outcomes: vec![None; plan.len()], plan, current: 0, unfinished: 0, spec, threads: threads.max(1), baseline: None })
    }
    pub fn reset(&mut self) -> Vec<Vec<f64>> {
        for (i, e) in self.envs.iter_mut().enumerate() { e.reset(episode_seed(self.spec.seed, i as u64)); self.done[i] = false; }
        self.envs.iter().map(|e| e.last_obs.clone()).collect()
    }
    /// Step for unfinished flights; returns how many are still flying.
    pub fn step(&mut self, actions: &[Vec<f64>]) -> usize {
        let per = self.envs.len().div_ceil(self.threads).max(1);
        let left = AtomicUsize::new(0);
        // flights are independent, so the result does not depend on thread count
        std::thread::scope(|sc| {
            for ((envs, done), acts) in self.envs.chunks_mut(per).zip(self.done.chunks_mut(per)).zip(actions.chunks(per)) {
                let left = &left;
                sc.spawn(move || {
                    for ((e, d), a) in envs.iter_mut().zip(done.iter_mut()).zip(acts) {
                        if *d { continue; }
                        let (_, _, t, c) = e.step(a);
                        if t || c { *d = true; } else { left.fetch_add(1, Ordering::Relaxed); }
                    }
                });
            }
        });
        left.into_inner()
    }
    /// The flight series of an analysis, in the order they must be flown.
    pub fn analysis_plan(&self) -> J {
        J::Arr(self.plan.iter().map(|v| J::obj().set("kind", v.kind).set("name", v.name.clone()).set("runs", v.runs)).collect())
    }
    /// Rebuilds the flights for one series; 0 is the mission itself.
    pub fn select(&mut self, i: usize) -> Result<(), String> {
        let v = self.plan.get(i).ok_or(format!("no analysis series {i}"))?.clone();
        self.envs = (0..v.runs).map(|k| flight_env(&self.spec, &v.vehicle, &v.dispersions, v.rotor, episode_seed(self.spec.seed, k as u64))).collect::<Result<Vec<_>, _>>()?;
        self.done = vec![false; v.runs];
        self.current = i;
        Ok(())
    }
    /// Remembers the outcome of the flown series.
    pub fn record(&mut self) {
        self.outcomes[self.current] = Some(self.envs.iter().map(result_of).collect());
        self.unfinished += self.done.iter().filter(|d| !**d).count();
    }
    /// Same answer as `run`, but for the controller that flew every series.
    pub fn analysis_report(&mut self, name: &str) -> Result<J, String> {
        if self.outcomes.iter().any(|o| o.is_none()) { return Err("not all analysis series were flown".into()); }
        self.autopilot_baseline()?;
        let (base, auto) = (self.outcomes[0].as_ref().unwrap(), self.baseline.as_ref().unwrap());
        let of = |kind: &str| self.plan.iter().zip(&self.outcomes).filter(|(v, _)| v.kind == kind).map(|(v, o)| (v, o.as_ref().unwrap().as_slice())).collect::<Vec<_>>();
        let saved = base.iter().zip(auto).filter(|(a, b)| a.success > 0.5 && b.success < 0.5).count();
        let lost = base.iter().zip(auto).filter(|(a, b)| a.success < 0.5 && b.success > 0.5).count();
        let head = J::obj().set("controller", name).set("name", self.spec.name.clone()).set("task", self.spec.task.clone()).set("world", World::build(&self.spec.world)?.name()).set("runs", self.spec.runs).set("seed", self.spec.seed)
            .set("terrain_sha256", self.spec.terrain_sha.clone().map(J::from).unwrap_or(J::Null)).set("generator", format!("worldsim {}", env!("CARGO_PKG_VERSION")))
            .set("unfinished", self.unfinished).set("autopilot_success", success_rate(auto)).set("delta_vs_autopilot", success_rate(base) - success_rate(auto))
            .set("delta_ci", 1.96 * ((saved + lost) as f64).sqrt() / self.spec.runs as f64).set("saved", saved).set("lost", lost);
        let r = analysis(head, &self.spec, base, of("clean")[0].1, &of("driver"), &of("technology"));
        let md = markdown(&r);
        Ok(r.set("markdown", md))
    }
    /// The autopilot under the same conditions is computed once: any number of controllers can be compared.
    fn autopilot_baseline(&mut self) -> Result<(), String> {
        if self.baseline.is_none() {
            self.baseline = Some(campaign(&self.spec, &self.spec.vehicle, &self.spec.dispersions, self.spec.rotor_scale, self.spec.runs, self.threads, &Progress::default())?);
        }
        Ok(())
    }
    /// Report and paired comparison with the autopilot under the same conditions.
    pub fn report(&mut self, name: &str) -> Result<J, String> {
        if self.current != 0 { self.select(0)?; }
        let rs: Vec<RunResult> = self.envs.iter().map(result_of).collect();
        self.autopilot_baseline()?;
        let base = self.baseline.as_ref().unwrap();
        let s = success_rate(&rs);
        let k: f64 = rs.iter().map(|r| r.success).sum();
        let (sl, sh) = wilson(k, rs.len() as f64);
        let saved = rs.iter().zip(base).filter(|(a, b)| a.success > 0.5 && b.success < 0.5).count();
        let lost = rs.iter().zip(base).filter(|(a, b)| a.success < 0.5 && b.success > 0.5).count();
        let unfinished = self.done.iter().filter(|d| !**d).count();
        Ok(J::obj().set("controller", name).set("name", self.spec.name.clone()).set("task", self.spec.task.clone()).set("runs", self.spec.runs).set("seed", self.spec.seed)
            .set("success", s).set("success_ci", vec![sl, sh]).set("requirements_met", margin_score(&self.spec, &rs)).set("requirements_total", self.spec.requirements.len())
            .set("scorecard", scorecard(&self.spec, &rs)).set("failure_modes", failure_modes(&rs)).set("unfinished", unfinished)
            .set("autopilot_success", success_rate(base)).set("delta_vs_autopilot", s - success_rate(base)).set("delta_ci", 1.96 * ((saved + lost) as f64).sqrt() / self.spec.runs as f64).set("saved", saved).set("lost", lost)
            .set("generator", format!("worldsim {}", env!("CARGO_PKG_VERSION"))))
    }
}

/// A series of flights: the run number sets the seed, so options are compared under the same conditions.
fn campaign(spec: &MissionSpec, vehicle: &[(String, f64)], d: &[(String, f64)], rotor: f64, runs: usize, threads: usize, progress: &Progress) -> Result<Vec<RunResult>, String> {
    let results = Mutex::new(vec![RunResult::default(); runs]);
    let (next, err) = (AtomicUsize::new(0), Mutex::new(None));
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= runs { return; }
                if progress.cancel.load(Ordering::Relaxed) { *err.lock().unwrap() = Some("calculation cancelled".to_string()); next.store(runs, Ordering::Relaxed); return; }
                progress.done.fetch_add(1, Ordering::Relaxed);
                match fly(spec, vehicle, d, rotor, episode_seed(spec.seed, i as u64)) {
                    Ok(r) => results.lock().unwrap()[i] = r,
                    Err(e) => { *err.lock().unwrap() = Some(e); next.store(runs, Ordering::Relaxed); }
                }
            });
        }
    });
    if let Some(e) = err.into_inner().unwrap() { return Err(e); }
    Ok(results.into_inner().unwrap())
}

/// Wilson 95% interval for a success rate.
pub fn wilson(k: f64, n: f64) -> (f64, f64) {
    let z = 1.96;
    let p = k / n;
    let den = 1.0 + z * z / n;
    let c = (p + z * z / (2.0 * n)) / den;
    let h = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / den;
    ((c - h).max(0.0), (c + h).min(1.0))
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return f64::NAN; }
    let x = p / 100.0 * (sorted.len() - 1) as f64;
    let (lo, hi) = (x.floor() as usize, x.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (x - lo as f64)
}

/// Requirement value and 95% interval. Touchdown metrics count only where touchdown happened.
fn evaluate(q: &Requirement, rs: &[RunResult]) -> (f64, f64, f64) {
    if q.percentile.is_none() {
        let k: f64 = rs.iter().map(|r| r.metric(&q.metric)).sum();
        let (l, h) = wilson(k, rs.len() as f64);
        return (k / rs.len() as f64, l, h);
    }
    let touch = q.metric.starts_with("touch_");
    let mut v: Vec<f64> = rs.iter().filter(|r| !touch || r.touch_speed > 0.0 || r.metric(&q.metric) > 0.0).map(|r| r.metric(&q.metric)).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    let p = q.percentile.unwrap();
    if v.is_empty() { return (f64::NAN, f64::NAN, f64::NAN); }
    // order statistics: a percentile interval with no distribution assumption
    let n = v.len() as f64;
    let f = p / 100.0;
    let se = (n * f * (1.0 - f)).sqrt();
    let idx = |x: f64| (x.round().max(0.0) as usize).min(v.len() - 1);
    (percentile(&v, p), v[idx(n * f - 1.96 * se - 1.0)], v[idx(n * f + 1.96 * se)])
}

/// met / not met / not enough data (threshold inside the interval).
fn verdict(q: &Requirement, val: f64, lo: f64, hi: f64) -> &'static str {
    if !val.is_finite() { return "no data"; }
    let (sure_ok, sure_bad) = if q.op == "<=" { (hi <= q.value, lo > q.value) } else { (lo >= q.value, hi < q.value) };
    if sure_ok { "met" } else if sure_bad { "not met" } else { "not enough data" }
}

fn scorecard(spec: &MissionSpec, rs: &[RunResult]) -> J {
    J::Arr(spec.requirements.iter().map(|q| {
        let (v, l, h) = evaluate(q, rs);
        J::obj().set("id", q.id.clone()).set("text", q.text.clone()).set("metric", q.metric.clone()).set("op", q.op.clone()).set("target", q.value)
            .set("percentile", q.percentile.map(J::from).unwrap_or(J::Null)).set("value", v).set("ci_low", l).set("ci_high", h).set("verdict", verdict(q, v, l, h))
    }).collect())
}

fn success_rate(rs: &[RunResult]) -> f64 { rs.iter().map(|r| r.success).sum::<f64>() / rs.len().max(1) as f64 }

/// How many requirements are met: a score for comparing options.
fn margin_score(spec: &MissionSpec, rs: &[RunResult]) -> usize {
    spec.requirements.iter().filter(|q| { let (v, l, h) = evaluate(q, rs); verdict(q, v, l, h) == "met" }).count()
}

pub fn run(spec: &MissionSpec, threads: usize) -> Result<J, String> { run_with(spec, threads, &Progress::default()) }

pub fn run_with(spec: &MissionSpec, threads: usize, progress: &Progress) -> Result<J, String> {
    let t0 = std::time::Instant::now();
    let active = spec.dispersions.iter().filter(|d| d.1 != 0.0).count();
    progress.total.store(spec.runs + spec.runs.min(300) * (1 + active) + spec.runs * spec.technologies.len(), Ordering::Relaxed);
    let base = campaign(spec, &spec.vehicle, &spec.dispersions, spec.rotor_scale, spec.runs, threads, progress)?;
    // Main drivers: each dispersion alone, the rest off; plus no dispersions at all.
    // Value of technologies: same conditions and seeds, only one thing changed.
    let plan = analysis_plan(spec)?;
    let mut flown = vec![];
    for v in plan.iter().skip(1) { flown.push((v, campaign(spec, &v.vehicle, &v.dispersions, v.rotor, v.runs, threads, progress)?)); }
    let of = |kind: &str| flown.iter().filter(|(v, _)| v.kind == kind).map(|(v, rs)| (*v, rs.as_slice())).collect::<Vec<_>>();
    let head = J::obj().set("name", spec.name.clone()).set("task", spec.task.clone()).set("world", World::build(&spec.world)?.name()).set("runs", spec.runs).set("seed", spec.seed)
        .set("terrain_sha256", spec.terrain_sha.clone().map(J::from).unwrap_or(J::Null))
        .set("generator", format!("worldsim {}", env!("CARGO_PKG_VERSION")));
    Ok(analysis(head, spec, &base, of("clean")[0].1, &of("driver"), &of("technology")).set("seconds", t0.elapsed().as_secs_f64()))
}

fn pct(x: f64) -> String { if x.is_finite() { format!("{:.1}%", 100.0 * x) } else { "—".into() } }
fn num(x: f64) -> String { if x.is_finite() { format!("{x:.2}") } else { "—".into() } }

/// Report for people: can be attached to a project review.
pub fn markdown(r: &J) -> String {
    let st = |j: &J, k: &str| j.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    let n = |j: &J, k: &str| j.get(k).and_then(|x| x.as_f64()).unwrap_or(f64::NAN);
    let mut s = format!("# {}\n\n{} · {} · runs: {} · seed {}\n\n", st(r, "name"), st(r, "task"), st(r, "world"), n(r, "runs"), n(r, "seed"));
    if r.get("controller").is_some() {
        s += &format!("Controller: {}. Built-in autopilot under the same conditions: {} ({:+.1} pp. for the controller).\n\n", st(r, "controller"), pct(n(r, "autopilot_success")), 100.0 * n(r, "delta_vs_autopilot"));
    }
    let ci = r.get("success_ci").and_then(|x| x.as_arr()).unwrap_or(&[]);
    s += &format!("**Successful flights: {}** (95%: {}–{}). Requirements met: {} of {}.\n\n## Requirements\n\n| ID | Requirement | Target | Result | 95% | Verdict |\n|---|---|---|---|---|---|\n",
        pct(n(r, "success")), pct(ci.first().and_then(|x| x.as_f64()).unwrap_or(f64::NAN)), pct(ci.get(1).and_then(|x| x.as_f64()).unwrap_or(f64::NAN)), n(r, "requirements_met"), n(r, "requirements_total"));
    for q in r.get("scorecard").and_then(|x| x.as_arr()).unwrap_or(&[]) {
        let binary = q.get("percentile").map(|p| matches!(p, J::Null)).unwrap_or(true);
        let f = |x: f64| if binary { pct(x) } else { num(x) };
        s += &format!("| {} | {} | {} {} | {} | {}–{} | {} |\n", st(q, "id"), st(q, "text"), st(q, "op"), f(n(q, "target")), f(n(q, "value")), f(n(q, "ci_low")), f(n(q, "ci_high")), st(q, "verdict"));
    }
    let fm = r.get("failure_modes").and_then(|x| x.as_arr()).unwrap_or(&[]);
    if !fm.is_empty() {
        s += "\n## How exactly it fails\n\n| Failure | Runs | Share |\n|---|---|---|\n";
        for f in fm { s += &format!("| {} | {} | {} |\n", st(f, "mode"), n(f, "count"), pct(n(f, "share"))); }
    }
    s += &format!("\n## Main drivers\n\nSuccess without dispersions {}. Each driver enabled separately, with {} runs.\n\n| Driver | σ | Success | Change |\n|---|---|---|---|\n", pct(n(r, "baseline_no_dispersions")), n(r, "driver_runs"));
    for d in r.get("drivers").and_then(|x| x.as_arr()).unwrap_or(&[]) {
        s += &format!("| {} | {} | {} | {:+.1} pp. |\n", st(d, "name"), n(d, "sigma"), pct(n(d, "success")), 100.0 * n(d, "delta"));
    }
    let techs = r.get("technologies").and_then(|x| x.as_arr()).unwrap_or(&[]);
    if !techs.is_empty() {
        s += "\n## Value of technologies\n\nPaired comparison: same flight conditions, only the technology changed. Significant means the interval excludes zero.\n\n| Technology | Success | Change | Saved / lost | Mass, kg | Requirements | Significant |\n|---|---|---|---|---|---|---|\n";
        for t in techs {
            s += &format!("| {} | {} | {:+.1} ± {:.1} pp. | {} / {} | {:+} | {} of {} | {} |\n", st(t, "name"), pct(n(t, "success")), 100.0 * n(t, "delta"), 100.0 * n(t, "delta_ci"), n(t, "saved"), n(t, "lost"),
                n(t, "dry_mass_delta"), n(t, "requirements_met"), n(r, "requirements_total"), if n(t, "delta").abs() > n(t, "delta_ci") { "yes" } else { "no" });
        }
    }
    s += &format!("\n*{}*\n", st(r, "limits"));
    s
}
