use worldsim::env::{Env, EnvConfig};
use std::time::Instant;
fn main() {
    for task in ["lander", "drone", "rover"] {
        for w in ["earth", "moon", "mars"] {
            let mut cfg = EnvConfig::new(task, w).unwrap();
            cfg.randomize = true;
            if let Ok(v) = std::env::var("ROTOR") { if task == "drone" && w == "mars" { cfg.rotor_scale = v.parse().unwrap(); } }
            let mut env = Env::new(cfg).unwrap();
            let (mut ok, mut steps, mut ret) = (0, 0usize, 0.0);
            let t0 = Instant::now();
            let n = 10;
            let mut reasons = vec![];
            for ep in 0..n {
                env.reset(ep as u64 + 1);
                loop {
                    let a = env.autopilot();
                    let (_, _, term, trunc) = env.step(&a);
                    steps += 1;
                    if term || trunc { break; }
                }
                if env.info.success { ok += 1; }
                ret += env.ep_return;
                if reasons.len() < 3 { reasons.push(env.info.reason.clone()); }
            }
            let el = t0.elapsed().as_secs_f64();
            println!("{task:6} {w:5}: success {ok}/{n}, avg return {:.1}, {:.0} steps/s | {:?}", ret / n as f64, steps as f64 / el, reasons);
        }
    }
}
