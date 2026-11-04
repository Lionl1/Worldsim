//! Сбор переходов для обучения world model: `worldsim collect`.
//!
//! Эпизоды считаются параллельно, но результат не зависит от числа потоков: у каждого эпизода свой seed,
//! а запись идёт строго по порядку. Формат — `.npy` (читается numpy без зависимостей) + `meta.json`.
//!
//! Файлы: `obs`, `next_obs`, `act`, `state`, `next_state` (T×…), `reward`, `terminated`, `truncated`, `episode`, `step`,
//! `params` (эпизод × параметр), `episodes.csv`, необязательно `terrain` и `rgb` + `rgb_index`.

use crate::env::{Env, EnvConfig, PARAM_NAMES};
use crate::json::J;
use crate::rng::Rng;
use crate::world::{World, WorldSpec};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

/// Чем выбирать действия. Автопилот один дал бы только «хорошие» траектории, случайные — только падения.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Policy { Iid, Smooth, Hold, Auto, AutoNoise }

impl Policy {
    pub fn name(self) -> &'static str {
        match self { Policy::Iid => "iid", Policy::Smooth => "smooth", Policy::Hold => "hold", Policy::Auto => "auto", Policy::AutoNoise => "auto_noise" }
    }
    pub fn parse_list(s: &str) -> Result<Vec<Policy>, String> {
        s.split(',').map(|p| match p.trim() {
            "iid" => Ok(Policy::Iid), "smooth" => Ok(Policy::Smooth), "hold" => Ok(Policy::Hold), "auto" => Ok(Policy::Auto), "auto_noise" => Ok(Policy::AutoNoise),
            x => Err(format!("неизвестная политика '{x}': iid, smooth, hold, auto, auto_noise")),
        }).collect()
    }
}

/// Состояние политики внутри эпизода.
pub struct Actor { pub policy: Policy, rng: Rng, cur: Vec<f64>, hold_left: usize }

impl Actor {
    pub fn new(policy: Policy, seed: u64, env: &Env) -> Actor {
        let (lo, hi) = env.act_bounds();
        Actor { policy, rng: Rng::new(seed), cur: lo.iter().zip(&hi).map(|(l, h)| (l + h) / 2.0).collect(), hold_left: 0 }
    }
    pub fn act(&mut self, env: &Env) -> Vec<f64> {
        let (lo, hi) = env.act_bounds();
        let r = &mut self.rng;
        match self.policy {
            Policy::Iid => lo.iter().zip(&hi).map(|(l, h)| r.uniform(*l, *h)).collect(),
            // медленный случайный процесс: приводы двигаются плавно, как у настоящего оператора
            Policy::Smooth => {
                for i in 0..lo.len() {
                    let (m, s) = ((lo[i] + hi[i]) / 2.0, (hi[i] - lo[i]) / 2.0);
                    self.cur[i] = (m + 0.97 * (self.cur[i] - m) + 0.24 * s * r.normal()).clamp(lo[i], hi[i]);
                }
                self.cur.clone()
            }
            // команда держится 10–100 шагов: видно отклик на ступеньку
            Policy::Hold => {
                if self.hold_left == 0 {
                    self.cur = lo.iter().zip(&hi).map(|(l, h)| r.uniform(*l, *h)).collect();
                    self.hold_left = 10 + (r.next_u64() % 91) as usize;
                }
                self.hold_left -= 1;
                self.cur.clone()
            }
            Policy::Auto => env.autopilot(),
            // автопилот с помехами: доходит до контакта с поверхностью, но не идеален
            Policy::AutoNoise => {
                let a = env.autopilot();
                let burst = r.f64() < 0.3;
                a.iter().enumerate().map(|(i, x)| {
                    let k = if burst { 0.25 } else { 0.05 };
                    (x + k * (hi[i] - lo[i]) * r.normal()).clamp(lo[i], hi[i])
                }).collect()
            }
        }
    }
}

#[derive(Clone)]
pub struct CollectConfig {
    pub task: String,
    pub worlds: Vec<WorldSpec>,
    pub policies: Vec<Policy>,
    pub episodes: usize,
    pub seed: u64,
    pub noise: f64,
    pub randomize: bool,
    pub frame_skip: Option<usize>,
    pub max_steps: Option<usize>,
    pub rotor_scale: f64,
    pub gust: f64,
    pub lag: f64,
    pub fault: f64,
    pub threads: usize,
    pub out_dir: String,
    pub terrain_n: usize,
    pub camera_n: usize,
    pub camera_every: usize,
    pub params: String,         // параметры реестра: dry_mass=900,gust=2
    pub counterfactuals: usize, // точек ветвления на эпизод: от одного состояния несколько действий с истинным ответом
}

/// Ветвлений в точке и шагов в ветви. Ветви длиннее не нужны: при падении эпизод всё равно обрывается.
pub const CF_BRANCHES: usize = 4;
pub const CF_HORIZON: usize = 25;

impl CollectConfig {
    pub fn new(task: &str, worlds: &[&str], episodes: usize, out_dir: &str) -> CollectConfig {
        CollectConfig { task: task.into(), worlds: worlds.iter().map(|w| WorldSpec::parse(w)).collect(),
            policies: vec![Policy::Smooth, Policy::Hold, Policy::AutoNoise], episodes, seed: 42, noise: 1.0, randomize: true,
            frame_skip: None, max_steps: None, rotor_scale: 1.0, gust: 0.0, lag: 0.0, fault: 0.0, threads: 1, out_dir: out_dir.into(),
            terrain_n: 0, camera_n: 0, camera_every: 1, counterfactuals: 0, params: String::new() }
    }

    fn env_config(&self, world: usize) -> Result<EnvConfig, String> {
        let mut c = EnvConfig::new(&self.task, "earth")?;
        c.world = self.worlds[world].clone();
        c.randomize = self.randomize;
        c.noise = self.noise;
        c.rotor_scale = self.rotor_scale;
        c.gust = self.gust;
        c.actuator_lag = self.lag;
        c.fault_prob = self.fault;
        c.set_list(&self.params)?;
        if let Some(k) = self.frame_skip { c.frame_skip = k; }
        if let Some(m) = self.max_steps { c.max_steps = m; }
        Ok(c)
    }

    fn validate(&self) -> Result<(), String> {
        if self.episodes == 0 { return Err("--episodes: нужен хотя бы один эпизод".into()); }
        if self.worlds.is_empty() { return Err("--worlds: нужен хотя бы один мир".into()); }
        if self.policies.is_empty() { return Err("--policies: нужна хотя бы одна политика".into()); }
        if self.threads == 0 { return Err("--threads: нужен хотя бы один поток".into()); }
        if self.frame_skip == Some(0) { return Err("--frame-skip: нужен хотя бы один шаг физики".into()); }
        if self.max_steps == Some(0) { return Err("--max-steps: нужен хотя бы один шаг".into()); }
        if self.camera_every == 0 { return Err("--camera-every: нужно целое ≥ 1".into()); }
        if self.terrain_n == 1 || self.camera_n > 1024 { return Err("--terrain: нужно ≥ 2; --camera: не больше 1024".into()); }
        for (name, x, lo, hi) in [("noise", self.noise, 0.0, 100.0), ("gust", self.gust, 0.0, 100.0), ("lag", self.lag, 0.0, 10.0), ("fault", self.fault, 0.0, 1.0)] {
            if !x.is_finite() || x < lo || x > hi { return Err(format!("--{name} = {x} вне диапазона [{lo}, {hi}]")); }
        }
        if !self.rotor_scale.is_finite() || self.rotor_scale <= 0.0 { return Err("--rotor-scale должен быть больше нуля".into()); }
        for w in &self.worlds { World::build(w)?; }
        self.env_config(0).map(|_| ())
    }
}

pub struct CollectStats { pub episodes: usize, pub transitions: usize, pub successes: usize, pub non_finite: usize, pub seconds: f64 }

/// Смешивание seed и номера эпизода (SplitMix64): соседние эпизоды не коррелируют.
pub fn episode_seed(seed: u64, ep: u64) -> u64 {
    let mut z = seed.wrapping_add(ep.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) >> 12 // без старших бит: seed переживает путь через f64 в JSON
}

/// Запись `.npy` потоком: длина заранее неизвестна, поэтому заголовок переписывается в конце.
struct Npy { f: BufWriter<File>, descr: &'static str, tail: Vec<usize>, rows: usize }

const NPY_HEADER: usize = 128;

impl Npy {
    fn create(path: &Path, descr: &'static str, tail: Vec<usize>) -> Result<Npy, String> {
        let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut n = Npy { f: BufWriter::with_capacity(1 << 20, file), descr, tail, rows: 0 };
        n.header().map_err(|e| e.to_string())?;
        Ok(n)
    }
    fn header(&mut self) -> std::io::Result<()> {
        let mut shape = self.rows.to_string();
        for t in &self.tail { shape.push_str(&format!(", {t}")); }
        if self.tail.is_empty() { shape.push(','); }
        let mut d = format!("{{'descr': '{}', 'fortran_order': False, 'shape': ({shape}), }}", self.descr);
        while d.len() < NPY_HEADER - 10 - 1 { d.push(' '); }
        d.push('\n');
        self.f.write_all(b"\x93NUMPY\x01\x00")?;
        self.f.write_all(&(d.len() as u16).to_le_bytes())?;
        self.f.write_all(d.as_bytes())
    }
    fn finish(mut self) -> Result<(), String> {
        let io = |e: std::io::Error| e.to_string();
        self.f.flush().map_err(io)?;
        self.f.seek(SeekFrom::Start(0)).map_err(io)?;
        self.header().map_err(io)?;
        self.f.flush().map_err(io)
    }
}

fn write_f32(n: &mut Npy, v: &[f32], rows: usize) -> Result<(), String> {
    // на little-endian это просто байты массива; так запись не тратит время на преобразование
    let mut buf = Vec::with_capacity(v.len() * 4);
    for x in v { buf.extend_from_slice(&x.to_le_bytes()); }
    n.f.write_all(&buf).map_err(|e| e.to_string())?;
    n.rows += rows;
    Ok(())
}

fn write_u8(n: &mut Npy, v: &[u8], rows: usize) -> Result<(), String> {
    n.f.write_all(v).map_err(|e| e.to_string())?;
    n.rows += rows;
    Ok(())
}

fn write_u32(n: &mut Npy, v: &[u32], rows: usize) -> Result<(), String> {
    let mut buf = Vec::with_capacity(v.len() * 4);
    for x in v { buf.extend_from_slice(&x.to_le_bytes()); }
    n.f.write_all(&buf).map_err(|e| e.to_string())?;
    n.rows += rows;
    Ok(())
}

#[derive(Default)]
struct Episode {
    idx: usize, seed: u64, world: usize, policy: &'static str,
    obs: Vec<f32>, next_obs: Vec<f32>, act: Vec<f32>, state: Vec<f32>, next_state: Vec<f32>,
    reward: Vec<f32>, terminated: Vec<u8>, truncated: Vec<u8>, params: Vec<f32>, terrain: Vec<f32>,
    rgb: Vec<u8>, rgb_steps: Vec<u32>,
    cf_obs: Vec<f32>, cf_act: Vec<f32>, cf_next: Vec<f32>, cf_alive: Vec<u8>, cf_params: Vec<f32>, cf_points: usize,
    steps: usize, ret: f64, success: bool, crashed: bool, reason: String, non_finite: usize,
}

fn push(dst: &mut Vec<f32>, v: &[f64], bad: &mut usize) {
    for x in v {
        if !x.is_finite() { *bad += 1; }
        dst.push(*x as f32);
    }
}

fn run_episode(cfg: &CollectConfig, idx: usize) -> Result<Episode, String> {
    let world = idx % cfg.worlds.len();
    let policy = cfg.policies[(idx / cfg.worlds.len()) % cfg.policies.len()];
    let seed = episode_seed(cfg.seed, idx as u64);
    let mut env = Env::new(cfg.env_config(world)?)?;
    let mut obs = env.reset(seed);
    let mut actor = Actor::new(policy, seed ^ 0xAC70, &env);
    let mut ep = Episode { idx, seed, world, policy: policy.name(), ..Default::default() };
    let mut bad = 0;
    push(&mut ep.params, &env.params(), &mut bad);
    if cfg.terrain_n >= 2 {
        let (n, size) = (cfg.terrain_n, env.terrain.size);
        for j in 0..n { for i in 0..n {
            let x = -size / 2.0 + size * i as f64 / (n - 1) as f64;
            let y = -size / 2.0 + size * j as f64 / (n - 1) as f64;
            ep.terrain.push(env.terrain.height(x, y) as f32);
        }}
    }
    let mut state = env.state();
    // точки ветвления заранее: один и тот же seed даёт те же точки при любом числе потоков
    let mut cf_rng = Rng::new(seed ^ 0xCF);
    let cf_every = if cfg.counterfactuals > 0 { (env.cfg.max_steps / cfg.counterfactuals / 8).max(10) } else { 0 };
    let mut cf_left = cfg.counterfactuals;
    let (act_lo, act_hi) = env.act_bounds();
    loop {
        if cf_left > 0 && ep.steps > 0 && ep.steps % cf_every == 0 {
            // копия среды: то же состояние и ГПСЧ; каждая ветвь — своя постоянная команда
            cf_left -= 1;
            ep.cf_points += 1;
            push(&mut ep.cf_obs, &obs, &mut bad);
            push(&mut ep.cf_params, &env.params(), &mut bad);
            for _ in 0..CF_BRANCHES {
                let a: Vec<f64> = act_lo.iter().zip(&act_hi).map(|(l, h)| cf_rng.uniform(*l, *h)).collect();
                push(&mut ep.cf_act, &a, &mut bad);
                let mut fork = env.clone();
                let mut last = obs.clone();
                let mut finished = false;
                for _ in 0..CF_HORIZON {
                    if !finished {
                        let (o, _, t, c) = fork.step(&a);
                        last = o;
                        finished = t || c;
                    }
                    push(&mut ep.cf_next, &last, &mut bad);
                    ep.cf_alive.push(!finished as u8);
                }
            }
        }
        if cfg.camera_n > 0 && ep.steps % cfg.camera_every == 0 {
            ep.rgb.extend(env.rgb_camera(cfg.camera_n, cfg.camera_n, 70.0));
            ep.rgb_steps.push(ep.steps as u32);
        }
        let a = actor.act(&env);
        let (next, r, term, trunc) = env.step(&a);
        let next_state = env.state();
        push(&mut ep.obs, &obs, &mut bad);
        push(&mut ep.next_obs, &next, &mut bad);
        push(&mut ep.act, &a, &mut bad);
        push(&mut ep.state, &state, &mut bad);
        push(&mut ep.next_state, &next_state, &mut bad);
        if !r.is_finite() { bad += 1; }
        ep.reward.push(r as f32);
        ep.terminated.push(term as u8);
        ep.truncated.push(trunc as u8);
        ep.steps += 1;
        obs = next;
        state = next_state;
        if term || trunc { break; }
    }
    ep.ret = env.ep_return;
    ep.success = env.info.success;
    ep.crashed = env.info.crashed;
    ep.reason = env.info.reason.clone();
    ep.non_finite = bad;
    Ok(ep)
}

pub fn run(cfg: &CollectConfig) -> Result<CollectStats, String> {
    cfg.validate()?;
    let t0 = std::time::Instant::now();
    let dir = Path::new(&cfg.out_dir);
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let probe = Env::new(cfg.env_config(0)?)?;
    let (nobs, nact, nstate, npar) = (probe.obs_names().len(), probe.act_names().len(), probe.state_names().len(), PARAM_NAMES.len());
    let file = |n: &str| dir.join(n);
    let mut fobs = Npy::create(&file("obs.npy"), "<f4", vec![nobs])?;
    let mut fnext = Npy::create(&file("next_obs.npy"), "<f4", vec![nobs])?;
    let mut fact = Npy::create(&file("act.npy"), "<f4", vec![nact])?;
    let mut fstate = Npy::create(&file("state.npy"), "<f4", vec![nstate])?;
    let mut fnstate = Npy::create(&file("next_state.npy"), "<f4", vec![nstate])?;
    let mut frew = Npy::create(&file("reward.npy"), "<f4", vec![])?;
    let mut fterm = Npy::create(&file("terminated.npy"), "|u1", vec![])?;
    let mut ftrunc = Npy::create(&file("truncated.npy"), "|u1", vec![])?;
    let mut fep = Npy::create(&file("episode.npy"), "<u4", vec![])?;
    let mut fstep = Npy::create(&file("step.npy"), "<u4", vec![])?;
    let mut fpar = Npy::create(&file("params.npy"), "<f4", vec![npar])?;
    let mut fterr = if cfg.terrain_n >= 2 { Some(Npy::create(&file("terrain.npy"), "<f4", vec![cfg.terrain_n, cfg.terrain_n])?) } else { None };
    let mut fcf = if cfg.counterfactuals > 0 { Some((
        Npy::create(&file("cf_obs.npy"), "<f4", vec![nobs])?,
        Npy::create(&file("cf_act.npy"), "<f4", vec![CF_BRANCHES, nact])?,
        Npy::create(&file("cf_next.npy"), "<f4", vec![CF_BRANCHES, CF_HORIZON, nobs])?,
        Npy::create(&file("cf_alive.npy"), "|u1", vec![CF_BRANCHES, CF_HORIZON])?,
        Npy::create(&file("cf_params.npy"), "<f4", vec![npar])?)) } else { None };
    let mut frgb = if cfg.camera_n > 0 { Some((Npy::create(&file("rgb.npy"), "|u1", vec![cfg.camera_n, cfg.camera_n, 3])?, Npy::create(&file("rgb_index.npy"), "<u4", vec![])?)) } else { None };
    let mut csv = BufWriter::new(File::create(file("episodes.csv")).map_err(|e| e.to_string())?);
    writeln!(csv, "episode,seed,world,policy,steps,return,success,crashed,reason").map_err(|e| e.to_string())?;

    let names: Vec<String> = cfg.worlds.iter().map(|w| World::build(w).map(|b| b.name())).collect::<Result<_, _>>()?;
    let mut stats = CollectStats { episodes: 0, transitions: 0, successes: 0, non_finite: 0, seconds: 0.0 };
    let next = AtomicUsize::new(0);
    let (tx, rx) = mpsc::sync_channel::<Result<Episode, String>>(cfg.threads * 4);
    let mut total: u64 = 0;
    let mut error = None;
    std::thread::scope(|sc| {
        for _ in 0..cfg.threads {
            let (tx, next) = (tx.clone(), &next);
            sc.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= cfg.episodes { return; }
                if tx.send(run_episode(cfg, i)).is_err() { return; }
            });
        }
        drop(tx);
        // воркеры заканчивают эпизоды в разном порядке — пишем строго по номерам
        let mut pending: BTreeMap<usize, Episode> = BTreeMap::new();
        let mut want = 0;
        for res in rx.iter() {
            let ep = match res { Ok(e) => e, Err(e) => { error = Some(e); next.store(cfg.episodes, Ordering::Relaxed); break; } };
            pending.insert(ep.idx, ep);
            while let Some(ep) = pending.remove(&want) {
                let r = (|| -> Result<(), String> {
                    write_f32(&mut fobs, &ep.obs, ep.steps)?;
                    write_f32(&mut fnext, &ep.next_obs, ep.steps)?;
                    write_f32(&mut fact, &ep.act, ep.steps)?;
                    write_f32(&mut fstate, &ep.state, ep.steps)?;
                    write_f32(&mut fnstate, &ep.next_state, ep.steps)?;
                    write_f32(&mut frew, &ep.reward, ep.steps)?;
                    write_u8(&mut fterm, &ep.terminated, ep.steps)?;
                    write_u8(&mut ftrunc, &ep.truncated, ep.steps)?;
                    write_u32(&mut fep, &vec![ep.idx as u32; ep.steps], ep.steps)?;
                    write_u32(&mut fstep, &(0..ep.steps as u32).collect::<Vec<_>>(), ep.steps)?;
                    write_f32(&mut fpar, &ep.params, 1)?;
                    if let Some(f) = &mut fterr { write_f32(f, &ep.terrain, 1)?; }
                    if let Some((o, a, n, l, p)) = &mut fcf {
                        write_f32(o, &ep.cf_obs, ep.cf_points)?;
                        write_f32(a, &ep.cf_act, ep.cf_points)?;
                        write_f32(n, &ep.cf_next, ep.cf_points)?;
                        write_u8(l, &ep.cf_alive, ep.cf_points)?;
                        write_f32(p, &ep.cf_params, ep.cf_points)?;
                    }
                    if let Some((f, g)) = &mut frgb {
                        write_u8(f, &ep.rgb, ep.rgb_steps.len())?;
                        let idx: Vec<u32> = ep.rgb_steps.iter().map(|s| total as u32 + s).collect();
                        write_u32(g, &idx, idx.len())?;
                    }
                    writeln!(csv, "{},{},{},{},{},{:.4},{},{},{}", ep.idx, ep.seed, names[ep.world], ep.policy, ep.steps, ep.ret, ep.success as u8, ep.crashed as u8, ep.reason.replace(',', ";")).map_err(|e| e.to_string())
                })();
                if let Err(e) = r { error = Some(e); next.store(cfg.episodes, Ordering::Relaxed); }
                total += ep.steps as u64;
                stats.episodes += 1;
                stats.transitions += ep.steps;
                stats.successes += ep.success as usize;
                stats.non_finite += ep.non_finite;
                want += 1;
            }
        }
    });
    if let Some(e) = error { return Err(e); }
    for f in [fobs, fnext, fact, fstate, fnstate, frew, fterm, ftrunc, fep, fstep, fpar] { f.finish()?; }
    if let Some(f) = fterr { f.finish()?; }
    if let Some((o, a, n, l, p)) = fcf { o.finish()?; a.finish()?; n.finish()?; l.finish()?; p.finish()?; }
    if let Some((f, g)) = frgb { f.finish()?; g.finish()?; }
    csv.flush().map_err(|e| e.to_string())?;
    stats.seconds = t0.elapsed().as_secs_f64();

    let strs = |v: Vec<String>| J::from(v);
    let (lo, hi) = probe.act_bounds();
    let meta = J::obj()
        .set("generator", format!("worldsim {}", env!("CARGO_PKG_VERSION"))).set("task", cfg.task.clone())
        .set("obs_names", strs(probe.obs_names())).set("act_names", strs(probe.act_names()))
        .set("state_names", strs(probe.state_names().iter().map(|s| s.to_string()).collect()))
        .set("param_names", strs(PARAM_NAMES.iter().map(|s| s.to_string()).collect()))
        .set("act_low", lo).set("act_high", hi).set("control_dt", probe.control_dt())
        .set("worlds", strs(cfg.worlds.iter().map(|w| format!("{}:{}:{}", w.physics, w.atmosphere.clone().unwrap_or("auto".into()), w.surface.clone().unwrap_or("auto".into()))).collect()))
        .set("world_names", strs(names))
        .set("policies", strs(cfg.policies.iter().map(|p| p.name().to_string()).collect()))
        .set("episodes", stats.episodes).set("transitions", stats.transitions).set("successes", stats.successes)
        .set("seed", cfg.seed).set("noise", cfg.noise).set("randomize", cfg.randomize)
        .set("frame_skip", probe.cfg.frame_skip).set("max_steps", probe.cfg.max_steps)
        .set("rotor_scale", cfg.rotor_scale).set("gust", cfg.gust).set("actuator_lag", cfg.lag).set("fault_prob", cfg.fault)
        .set("terrain_n", cfg.terrain_n).set("camera_n", cfg.camera_n).set("camera_every", cfg.camera_every)
        .set("params", cfg.params.clone()).set("counterfactuals", cfg.counterfactuals).set("cf_branches", CF_BRANCHES).set("cf_horizon", CF_HORIZON);
    fs::write(file("meta.json"), meta.render()).map_err(|e| e.to_string())?;
    Ok(stats)
}
