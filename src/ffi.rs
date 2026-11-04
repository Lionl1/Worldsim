//! Векторная среда для Python через ctypes (`python/worldsim/vec.py`): без HTTP и без зависимостей.
//! Все массивы — f32 по строкам; завершившиеся среды сами начинают новый эпизод (как в Gymnasium VectorEnv).

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
        // Вызывающий передаёт буфер длиной cap
        unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), buf, n); }
    }
    s.len()
}

#[no_mangle]
pub extern "C" fn ws_last_error(buf: *mut u8, cap: usize) -> usize { LAST_ERROR.with(|l| write_str(&l.borrow(), buf, cap)) }

#[allow(clippy::too_many_arguments)]
fn build(task: &str, worlds: &str, n: usize, seed: u64, noise: f64, randomize: bool, frame_skip: usize, max_steps: usize, rotor_scale: f64, gust: f64, lag: f64, fault: f64, threads: usize) -> Result<VecEnv, String> {
    if n == 0 { return Err("нужна хотя бы одна среда".into()); }
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
    let proto = Env::new(cfg)?;
    let (nobs, nact, nstate) = (proto.obs_names().len(), proto.act_names().len(), proto.state_names().len());
    Ok(VecEnv { envs: vec![proto; n], seed, episodes: vec![0; n], threads: threads.max(1), nobs, nact, nstate })
}

/// Возвращает указатель или null; причину даёт `ws_last_error`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn ws_vec_new(task: *const c_char, worlds: *const c_char, n: usize, seed: u64, noise: f64, randomize: bool, frame_skip: usize, max_steps: usize, rotor_scale: f64, gust: f64, lag: f64, fault: f64, threads: usize) -> *mut VecEnv {
    let text = |p: *const c_char| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    match build(&text(task), &text(worlds), n, seed, noise, randomize, frame_skip, max_steps, rotor_scale, gust, lag, fault, threads) {
        Ok(v) => Box::into_raw(Box::new(v)),
        Err(e) => { set_error(e); std::ptr::null_mut() }
    }
}

#[no_mangle]
pub extern "C" fn ws_vec_free(p: *mut VecEnv) { if !p.is_null() { drop(unsafe { Box::from_raw(p) }); } }

/// Описание среды в JSON: размеры, имена, границы действий.
#[no_mangle]
pub extern "C" fn ws_vec_spec(p: *mut VecEnv, buf: *mut u8, cap: usize) -> usize {
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

/// Выполнить `f` для каждой среды, параллельно по кускам. Результат не зависит от числа потоков: у сред нет общего состояния.
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
pub extern "C" fn ws_vec_reset(p: *mut VecEnv, obs: *mut f32) {
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

/// `final_obs` — наблюдение в момент конца эпизода: у завершившихся сред `obs` уже от нового эпизода.
#[no_mangle]
pub extern "C" fn ws_vec_step(p: *mut VecEnv, act: *const f32, obs: *mut f32, reward: *mut f32, terminated: *mut u8, truncated: *mut u8, final_obs: *mut f32) {
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
        pub extern "C" fn $name(p: *mut VecEnv, out: *mut f32) {
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
