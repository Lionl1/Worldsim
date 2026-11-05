//! External software in the loop over TCP: flight software or a controller on the same machine drives the simulation.
//!
//! One JSON object per line. The server greets with the environment description and the first observation;
//! the client answers `{"action":[...]}`, `{"reset":seed}` or `{"bye":true}`.
//! In real time a step happens on every control tick: a command that misses the tick is counted, and the previous one stays active.
//! The timing comes from the operating system, so this is soft real time: for integration work, not for hard timing evidence.
//! Listens on 127.0.0.1 only: the protocol has no access control.

use crate::env::{Env, EnvConfig};
use crate::json::J;
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

const MAX_LINE: usize = 1 << 16;

pub struct LoopOpts { pub port: u16, pub realtime: bool, pub seed: u64, pub once: bool }

fn write_line(s: &mut TcpStream, j: &J) -> std::io::Result<()> { s.write_all((j.render() + "\n").as_bytes()) }

/// Reads a line until the deadline. None means no complete line yet; a partial line stays in `buf`.
fn read_line(r: &mut BufReader<TcpStream>, buf: &mut Vec<u8>, deadline: Option<Instant>) -> std::io::Result<Option<Vec<u8>>> {
    loop {
        let wait = match deadline {
            Some(d) => { let left = d.saturating_duration_since(Instant::now()); if left.is_zero() { return Ok(None); } Some(left) }
            None => None,
        };
        r.get_ref().set_read_timeout(wait)?;
        let (used, found) = match r.fill_buf() {
            Ok([]) => return Err(ErrorKind::UnexpectedEof.into()),
            Ok(data) => match data.iter().position(|b| *b == b'\n') {
                Some(i) => { buf.extend_from_slice(&data[..i]); (i + 1, true) }
                None => { buf.extend_from_slice(data); (data.len(), false) }
            },
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => return Ok(None),
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        r.consume(used);
        if buf.len() > MAX_LINE { return Err(std::io::Error::new(ErrorKind::InvalidData, "line too long")); }
        if found { return Ok(Some(std::mem::take(buf))); }
    }
}

enum Msg { Action(Vec<f64>), Reset(u64), Bye }

fn parse(line: &[u8], n_act: usize) -> Result<Msg, String> {
    let j = J::parse(&String::from_utf8_lossy(line))?;
    if j.get("bye").is_some() { return Ok(Msg::Bye); }
    if let Some(s) = j.get("reset") { return Ok(Msg::Reset(s.as_f64().ok_or("reset: a number is required")? as u64)); }
    let a = j.get("action").and_then(|x| x.as_arr()).ok_or("expected action, reset or bye")?;
    if a.len() != n_act { return Err(format!("the action must have {n_act} numbers, got {}", a.len())); }
    Ok(Msg::Action(a.iter().map(|x| x.as_f64().ok_or("action: only numbers".to_string())).collect::<Result<_, _>>()?))
}

fn step_reply(env: &mut Env, a: &[f64], late: bool, misses: usize) -> J {
    let (obs, reward, terminated, truncated) = env.step(a);
    J::obj().set("obs", obs).set("reward", reward).set("terminated", terminated).set("truncated", truncated).set("t", env.t)
        .set("reason", env.info.reason.clone()).set("success", env.info.success).set("late", late).set("misses", misses)
}

fn session(stream: TcpStream, cfg: &EnvConfig, o: &LoopOpts) -> std::io::Result<()> {
    stream.set_nodelay(true)?;
    let mut w = stream.try_clone()?;
    let mut r = BufReader::new(stream);
    let bad = |e: String| std::io::Error::new(ErrorKind::InvalidInput, e);
    let mut env = Env::new(cfg.clone()).map_err(bad)?;
    env.reset(o.seed);
    let (lo, hi) = env.act_bounds();
    let n_act = lo.len();
    let dt = env.control_dt();
    write_line(&mut w, &J::obj().set("task", env.cfg.task.name()).set("obs_names", env.obs_names()).set("act_names", env.act_names()).set("act_low", lo).set("act_high", hi)
        .set("control_dt", dt).set("realtime", o.realtime).set("obs", env.last_obs.clone()))?;
    let (mut buf, mut misses, mut held) = (vec![], 0usize, vec![0.0; n_act]);
    let mut tick = Instant::now() + Duration::from_secs_f64(dt);
    loop {
        let deadline = o.realtime.then_some(tick);
        let line = match read_line(&mut r, &mut buf, deadline) {
            Ok(x) => x,
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        };
        let Some(line) = line else {
            // the tick has come and no command has arrived
            if !env.done {
                misses += 1;
                let reply = step_reply(&mut env, &held, true, misses);
                write_line(&mut w, &reply)?;
            }
            tick += Duration::from_secs_f64(dt);
            continue;
        };
        match parse(&line, n_act) {
            Err(e) => write_line(&mut w, &J::obj().set("error", e))?,
            Ok(Msg::Bye) => return Ok(()),
            Ok(Msg::Reset(seed)) => {
                env.reset(seed);
                misses = 0;
                held = vec![0.0; n_act];
                write_line(&mut w, &J::obj().set("obs", env.last_obs.clone()).set("t", env.t))?;
                tick = Instant::now() + Duration::from_secs_f64(dt);
            }
            Ok(Msg::Action(_)) if env.done => write_line(&mut w, &J::obj().set("error", "the episode has ended, send reset"))?,
            Ok(Msg::Action(a)) => {
                held = a;
                if o.realtime {
                    // a command that came early acts on its tick
                    std::thread::sleep(tick.saturating_duration_since(Instant::now()));
                    tick += Duration::from_secs_f64(dt);
                }
                let reply = step_reply(&mut env, &held, false, misses);
                write_line(&mut w, &reply)?;
            }
        }
    }
}

/// Serves clients one at a time; each gets a new environment.
pub fn serve(cfg: EnvConfig, o: LoopOpts) -> Result<(), String> {
    Env::new(cfg.clone())?;
    let l = TcpListener::bind(("127.0.0.1", o.port)).map_err(|e| format!("port {}: {e}", o.port))?;
    println!("worldsim loop: 127.0.0.1:{} ({}, {})", o.port, cfg.task.name(), if o.realtime { "real time" } else { "lockstep" });
    std::io::stdout().flush().ok();
    for s in l.incoming() {
        let s = s.map_err(|e| e.to_string())?;
        if let Err(e) = session(s, &cfg, &o) { eprintln!("session: {e}"); }
        if o.once { break; }
    }
    Ok(())
}
