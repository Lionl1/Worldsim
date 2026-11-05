//! Vector environment for Python via ctypes (`python/worldsim/vec.py`): no HTTP and no dependencies.
//! All functions are unsafe under the C contract: pointers from `ws_vec_new`, buffers of the right size (see vec.py).
//! All arrays are f32 by rows; finished environments start a new episode on their own (as in Gymnasium VectorEnv).

use crate::collect::episode_seed;
use crate::env::{Env, EnvConfig, PARAM_NAMES};
use crate::json::J;
use crate::world::WorldSpec;
use std::cell::RefCell;
use std::ffi::{c_char, CStr};

pub struct VecEnv { envs: Vec<Env>, seed: u64, episodes: Vec<u64>, threads: usize, nobs: usize, nact: usize, nstate: usize }

thread_local! { static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) }; }

fn set_error(e: String) { LAST_ERROR.with(|l| *l.borrow_mut() = e); }

fn write_str(s: &str, buf: *mut u8, cap: usize) -> usize {
    if !buf.is_null() && cap > 0 {
        let n = s.len().min(cap);
        // The caller passes a buffer of length cap
        unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), buf, n); }
    }
    s.len()
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_last_error(buf: *mut u8, cap: usize) -> usize { LAST_ERROR.with(|l| write_str(&l.borrow(), buf, cap)) }

#[allow(clippy::too_many_arguments)]
fn build(task: &str, worlds: &str, n: usize, seed: u64, noise: f64, randomize: bool, frame_skip: usize, max_steps: usize, rotor_scale: f64, gust: f64, lag: f64, fault: f64, threads: usize, params: &str) -> Result<VecEnv, String> {
    if n == 0 { return Err("at least one environment is required".into()); }
    let specs: Vec<WorldSpec> = worlds.split(',').map(|w| WorldSpec::parse(w.trim())).collect();
    let mut cfg = EnvConfig::new(task, "earth")?;
    cfg.world = specs[0].clone();
    cfg.worlds = if specs.len() > 1 { specs } else { vec![] };
    cfg.noise = noise;
    cfg.randomize = randomize;
    cfg.rotor_scale = rotor_scale;
    cfg.gust = gust;
    cfg.actuator_lag = lag;
    cfg.fault_prob = fault;
    if frame_skip > 0 { cfg.frame_skip = frame_skip; }
    if max_steps > 0 { cfg.max_steps = max_steps; }
    cfg.set_list(params)?;
    let proto = Env::new(cfg)?;
    let (nobs, nact, nstate) = (proto.obs_names().len(), proto.act_names().len(), proto.state_names().len());
    Ok(VecEnv { envs: vec![proto; n], seed, episodes: vec![0; n], threads: threads.max(1), nobs, nact, nstate })
}

/// Returns a pointer or null; the reason comes from `ws_last_error`.
#[no_mangle]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_vec_new(task: *const c_char, worlds: *const c_char, n: usize, seed: u64, noise: f64, randomize: bool, frame_skip: usize, max_steps: usize, rotor_scale: f64, gust: f64, lag: f64, fault: f64, threads: usize, params: *const c_char) -> *mut VecEnv {
    let text = |p: *const c_char| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    match build(&text(task), &text(worlds), n, seed, noise, randomize, frame_skip, max_steps, rotor_scale, gust, lag, fault, threads, &text(params)) {
        Ok(v) => Box::into_raw(Box::new(v)),
        Err(e) => { set_error(e); std::ptr::null_mut() }
    }
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_vec_free(p: *mut VecEnv) { if !p.is_null() { drop(unsafe { Box::from_raw(p) }); } }

/// Environment description in JSON: sizes, names, action bounds.
#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_vec_spec(p: *mut VecEnv, buf: *mut u8, cap: usize) -> usize {
    let v = unsafe { &*p };
    let e = &v.envs[0];
    let (lo, hi) = e.act_bounds();
    let s = J::obj().set("obs_names", e.obs_names()).set("act_names", e.act_names())
        .set("state_names", e.state_names().iter().map(|x| x.to_string()).collect::<Vec<_>>())
        .set("param_names", PARAM_NAMES.iter().map(|x| x.to_string()).collect::<Vec<_>>())
        .set("act_low", lo).set("act_high", hi).set("control_dt", e.control_dt()).set("n", v.envs.len()).render();
    write_str(&s, buf, cap)
}

fn copy(dst: &mut [f32], src: &[f64]) { for (d, s) in dst.iter_mut().zip(src) { *d = *s as f32; } }

/// Run `f` for each environment, in parallel chunks. The result does not depend on thread count: environments share no state.
fn each<T: Send>(envs: Vec<T>, threads: usize, f: impl Fn(T) + Sync) {
    if threads <= 1 || envs.len() < 2 { envs.into_iter().for_each(f); return; }
    let per = envs.len().div_ceil(threads);
    let mut it = envs.into_iter();
    std::thread::scope(|sc| {
        loop {
            let chunk: Vec<T> = it.by_ref().take(per).collect();
            if chunk.is_empty() { break; }
            let f = &f;
            sc.spawn(move || chunk.into_iter().for_each(f));
        }
    });
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_vec_reset(p: *mut VecEnv, obs: *mut f32) {
    let v = unsafe { &mut *p };
    let n = v.envs.len();
    let obs = unsafe { std::slice::from_raw_parts_mut(obs, n * v.nobs) };
    let seed = v.seed;
    let items: Vec<_> = v.envs.iter_mut().zip(v.episodes.iter_mut()).zip(obs.chunks_mut(v.nobs)).enumerate().collect();
    each(items, v.threads, |(i, ((e, ep), o))| {
        *ep = 0;
        copy(o, &e.reset(episode_seed(seed, (i as u64) << 32)));
    });
}

/// `final_obs` — observation at the end of the episode: for finished environments `obs` already belongs to the new episode.
#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_vec_step(p: *mut VecEnv, act: *const f32, obs: *mut f32, reward: *mut f32, terminated: *mut u8, truncated: *mut u8, final_obs: *mut f32) {
    let v = unsafe { &mut *p };
    let (n, no, na) = (v.envs.len(), v.nobs, v.nact);
    let (act, obs, reward, terminated, truncated, final_obs) = unsafe { (
        std::slice::from_raw_parts(act, n * na), std::slice::from_raw_parts_mut(obs, n * no), std::slice::from_raw_parts_mut(reward, n),
        std::slice::from_raw_parts_mut(terminated, n), std::slice::from_raw_parts_mut(truncated, n), std::slice::from_raw_parts_mut(final_obs, n * no)) };
    let seed = v.seed;
    let items: Vec<_> = v.envs.iter_mut().zip(v.episodes.iter_mut()).zip(act.chunks(na)).zip(obs.chunks_mut(no)).zip(reward.iter_mut())
        .zip(terminated.iter_mut()).zip(truncated.iter_mut()).zip(final_obs.chunks_mut(no)).enumerate().collect();
    each(items, v.threads, |(i, (((((((e, ep), a), o), r), te), tr), fo))| {
        let action: Vec<f64> = a.iter().map(|x| *x as f64).collect();
        let (next, rew, t, c) = e.step(&action);
        *r = rew as f32;
        *te = t as u8;
        *tr = c as u8;
        copy(fo, &next);
        if t || c {
            *ep += 1;
            copy(o, &e.reset(episode_seed(seed, ((i as u64) << 32) | *ep)));
        } else {
            copy(o, &next);
        }
    });
}

macro_rules! exporter {
    ($name:ident, $width:ident, $get:expr) => {
        #[no_mangle]
        #[allow(clippy::missing_safety_doc)]
        pub unsafe extern "C" fn $name(p: *mut VecEnv, out: *mut f32) {
            let v = unsafe { &mut *p };
            let w = $width(v);
            let out = unsafe { std::slice::from_raw_parts_mut(out, v.envs.len() * w) };
            for (e, o) in v.envs.iter().zip(out.chunks_mut(w)) { copy(o, &$get(e)); }
        }
    };
}
fn w_state(v: &VecEnv) -> usize { v.nstate }
fn w_params(_: &VecEnv) -> usize { PARAM_NAMES.len() }
fn w_act(v: &VecEnv) -> usize { v.nact }
exporter!(ws_vec_state, w_state, |e: &Env| e.state());
exporter!(ws_vec_params, w_params, |e: &Env| e.params());
exporter!(ws_vec_autopilot, w_act, |e: &Env| e.autopilot());

// ---------------------------------------------------------------- custom controller in Mission Lab

pub struct MissionHandle { runner: crate::mission::Runner }

/// Mission description in JSON; terrain from a file is allowed: this is a local call from user code.
#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_mission_new(spec: *const c_char, threads: usize) -> *mut MissionHandle {
    let text = unsafe { CStr::from_ptr(spec) }.to_string_lossy().into_owned();
    match crate::mission::MissionSpec::parse_local(&text).and_then(|s| crate::mission::Runner::new(s, threads.max(1))) {
        Ok(runner) => Box::into_raw(Box::new(MissionHandle { runner })),
        Err(e) => { set_error(e); std::ptr::null_mut() }
    }
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_mission_free(p: *mut MissionHandle) { if !p.is_null() { drop(unsafe { Box::from_raw(p) }); } }

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_mission_spec(p: *mut MissionHandle, buf: *mut u8, cap: usize) -> usize {
    let h = unsafe { &*p };
    let e = &h.runner.envs[0];
    let (lo, hi) = e.act_bounds();
    write_str(&J::obj().set("runs", h.runner.envs.len()).set("obs_names", e.obs_names()).set("act_names", e.act_names()).set("act_low", lo).set("act_high", hi).set("control_dt", e.control_dt()).set("param_names", PARAM_NAMES.iter().map(|s| s.to_string()).collect::<Vec<_>>()).render(), buf, cap)
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_mission_reset(p: *mut MissionHandle, obs: *mut f32) {
    let h = unsafe { &mut *p };
    let no = h.runner.envs[0].obs_names().len();
    let all = h.runner.reset();
    let out = unsafe { std::slice::from_raw_parts_mut(obs, all.len() * no) };
    for (o, v) in out.chunks_mut(no).zip(&all) { copy(o, v); }
}

/// Returns the number of unfinished flights. For finished ones `done` = 1, the observation stays at its last value.
#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_mission_step(p: *mut MissionHandle, act: *const f32, obs: *mut f32, done: *mut u8) -> usize {
    let h = unsafe { &mut *p };
    let n = h.runner.envs.len();
    let (no, na) = (h.runner.envs[0].obs_names().len(), h.runner.envs[0].act_names().len());
    let act = unsafe { std::slice::from_raw_parts(act, n * na) };
    let actions: Vec<Vec<f64>> = act.chunks(na).map(|c| c.iter().map(|x| *x as f64).collect()).collect();
    let left = h.runner.step(&actions);
    let (obs, done) = unsafe { (std::slice::from_raw_parts_mut(obs, n * no), std::slice::from_raw_parts_mut(done, n)) };
    for i in 0..n {
        copy(&mut obs[i * no..(i + 1) * no], &h.runner.envs[i].last_obs);
        done[i] = h.runner.done[i] as u8;
    }
    left
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_mission_report(p: *mut MissionHandle, buf: *mut u8, cap: usize, name: *const c_char) -> usize {
    let h = unsafe { &mut *p };
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
    let text = match h.runner.report(&name) { Ok(r) => r.render(), Err(e) => J::obj().set("error", e).render() };
    write_str(&text, buf, cap)
}

/// Hidden parameters of the current flights, by rows (see `param_names` in the spec).
#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_mission_params(p: *mut MissionHandle, out: *mut f32) {
    let h = unsafe { &*p };
    let w = PARAM_NAMES.len();
    let out = unsafe { std::slice::from_raw_parts_mut(out, h.runner.envs.len() * w) };
    for (e, o) in h.runner.envs.iter().zip(out.chunks_mut(w)) { copy(o, &e.params()); }
}

/// The flight series of a full analysis, JSON.
#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_mission_plan(p: *mut MissionHandle, buf: *mut u8, cap: usize) -> usize {
    write_str(&unsafe { &*p }.runner.analysis_plan().render(), buf, cap)
}

/// Rebuilds the flights for series `i`. Returns the number of flights, or 0 on error.
#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_mission_select(p: *mut MissionHandle, i: usize) -> usize {
    let h = unsafe { &mut *p };
    match h.runner.select(i) { Ok(()) => h.runner.envs.len(), Err(e) => { set_error(e); 0 } }
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_mission_record(p: *mut MissionHandle) { unsafe { &mut *p }.runner.record(); }

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_mission_analysis(p: *mut MissionHandle, buf: *mut u8, cap: usize, name: *const c_char) -> usize {
    let h = unsafe { &mut *p };
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
    let text = match h.runner.analysis_report(&name) { Ok(r) => r.render(), Err(e) => J::obj().set("error", e).render() };
    write_str(&text, buf, cap)
}

// ---------------------------------------------------------------- replaying recorded commands (calibration)

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_replay_names(task: *const c_char, buf: *mut u8, cap: usize) -> usize {
    let task = unsafe { CStr::from_ptr(task) }.to_string_lossy().into_owned();
    let names = EnvConfig::new(&task, "earth").and_then(Env::new).map(|e| e.state_names().iter().map(|x| x.to_string()).collect::<Vec<_>>()).unwrap_or_default();
    write_str(&J::from(names).render(), buf, cap)
}

/// Initial position and velocity are explicit, terrain comes from the parameters. Returns the step count, or −1 on error.
#[no_mangle]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe extern "C" fn ws_replay(task: *const c_char, world: *const c_char, params: *const c_char, init: *const f32, steps: usize, act_dim: usize, actions: *const f32, states: *mut f32) -> i64 {
    let text = |p: *const c_char| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    let run = || -> Result<i64, String> {
        let mut cfg = EnvConfig::new(&text(task), "earth")?;
        cfg.world = WorldSpec::parse(&text(world));
        cfg.noise = 0.0;
        cfg.max_steps = usize::MAX;
        cfg.set_list(&text(params))?;
        let mut e = Env::new(cfg)?;
        let na = e.act_names().len();
        if na != act_dim { return Err(format!("the recording has {act_dim} commands, but the vehicle takes {na}")); }
        e.reset(1);
        let x0 = unsafe { std::slice::from_raw_parts(init, 6) };
        e.set_state(crate::vec3::Vec3::new(x0[0] as f64, x0[1] as f64, x0[2] as f64), crate::vec3::Vec3::new(x0[3] as f64, x0[4] as f64, x0[5] as f64));
        let ns = e.state_names().len();
        let act = unsafe { std::slice::from_raw_parts(actions, steps * na) };
        let out = unsafe { std::slice::from_raw_parts_mut(states, steps * ns) };
        for t in 0..steps {
            let a: Vec<f64> = act[t * na..(t + 1) * na].iter().map(|x| *x as f64).collect();
            let (_, _, te, tr) = e.step(&a);
            copy(&mut out[t * ns..(t + 1) * ns], &e.state());
            if te || tr { return Ok(t as i64 + 1); }
        }
        Ok(steps as i64)
    };
    match run() { Ok(n) => n, Err(m) => { set_error(m); -1 } }
}
