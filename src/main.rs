use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::ExitCode;

use worldsim::ballistics::{self, Launch, Options, Projectile};
use worldsim::collect;
use worldsim::constants::{AU, DAY, YEAR};
use worldsim::env::{param_default, EnvConfig, Task};
use worldsim::datagen::{self, DatasetConfig};
use worldsim::magneto::{self, ParticleLaunch};
use worldsim::mission;
use worldsim::nbody::{Integrator, NBody};
use worldsim::presets;
use worldsim::thermal::{self, ThermalOptions};
use worldsim::validate;
use worldsim::world::{World, WorldSpec};

const HELP: &str = "worldsim — physical models of worlds

USAGE: worldsim <command> [--key value ...]

COMMANDS
  presets                       list physics / atmosphere / surface presets
  info      [world]               world parameters and derived quantities
  throw     [world] --speed 300 --elev 45 --az 90 --lat 0 --radius 0.05 --density 7800
            [--no-drag] [--no-rotation] [--no-j2] [--out traj.csv]
  thermal   [world] --lat 0 --dec 0 --days 15 [--out temp.csv]
  particle  [world] --species proton --mev 1 --L 3 --pitch 40 --time 20 [--out p.csv]
  system    --bodies sun,earth,moon,mars --years 1 --dt-hours 1 [--gr] [--leapfrog] [--out sys.csv]
  dataset   --task projectile|orbit|thermal|particle|system --n 1000
            --worlds earth,moon,mars --out data/ --seed 42 [--points 100] [--noise 0.01] [--strict]
  serve     [--host 127.0.0.1] [--port 8080] [--no-open] [--token T] [--data-dir runs] [--cors] [--log] [--insecure]
            interface and API. An external address needs a token (--token or WORLDSIM_TOKEN)
  health    [--port 8080]               server liveness check, exit code 0 or 1
  version                               version
  validate                      run all law checks
  params    [--task lander|drone|rover]  all configurable parameters: name, units, range, default value
  mission   file.json [--out dir] [--threads N]
            Mission Lab: requirements scorecard, main drivers, value of technologies (see missions/)
  collect   --task lander|drone|rover --worlds moon,mars:earth:moon --episodes 1000 --out dir
            [--policies smooth,hold,auto_noise] [--noise 1] [--randomize] [--frame-skip 4] [--max-steps N]
            [--gust 2] [--lag 0.05] [--fault 0.2] [--rotor-scale 1] [--threads N] [--seed 42]
            [--terrain 33] [--camera 64 --camera-every 5] [--counterfactuals 3 (per 1000 steps)] [--set throttle_min=0.1,gyro_bias=0.01]
            [--terrain-file site.asc] [--terrain-size 2000]  terrain of a real site (.asc, .csv, .npy)
            transitions for world model training: .npy + meta.json (see README)

WORLD (for any command)
  --world mars                  everything from Mars
  --physics mars --atmosphere earth --surface moon   mix presets
  --gravity-scale 1.5           GM scale;  --distance-au 1.2  distance to the star
  In dataset: --worlds earth,moon,mars:earth:moon  (physics:atmosphere:surface)
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(|a| a == "-h" || a == "--help").unwrap_or(false) { print!("{HELP}"); return ExitCode::SUCCESS; }
    // no arguments: start the interface right away
    let args = if args.is_empty() { vec!["serve".to_string()] } else { args };
    let cmd = args[0].clone();
    let (pos, kv) = parse(&args[1..]);
    match run(&cmd, &pos, &kv) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
    }
}

fn parse(a: &[String]) -> (Vec<String>, HashMap<String, String>) {
    let mut pos = Vec::new();
    let mut kv = HashMap::new();
    let mut i = 0;
    while i < a.len() {
        if let Some(k) = a[i].strip_prefix("--") {
            if i + 1 < a.len() && !a[i + 1].starts_with("--") { kv.insert(k.to_string(), a[i + 1].clone()); i += 2; }
            else { kv.insert(k.to_string(), "true".into()); i += 1; }
        } else { pos.push(a[i].clone()); i += 1; }
    }
    (pos, kv)
}

fn num(kv: &HashMap<String, String>, k: &str, d: f64) -> Result<f64, String> {
    kv.get(k).map(|v| v.parse::<f64>().map_err(|_| format!("--{k}: expected a number, got '{v}'"))).unwrap_or(Ok(d))
}
fn flag(kv: &HashMap<String, String>, k: &str) -> bool { kv.contains_key(k) }

fn world_spec(pos: &[String], kv: &HashMap<String, String>) -> Result<WorldSpec, String> {
    let base = kv.get("world").or(kv.get("physics")).or(pos.first()).cloned().unwrap_or("earth".into());
    Ok(WorldSpec {
        physics: base,
        atmosphere: kv.get("atmosphere").cloned(),
        surface: kv.get("surface").cloned(),
        gravity_scale: num(kv, "gravity-scale", 1.0)?,
        distance_au: kv.get("distance-au").map(|v| v.parse().map_err(|_| "--distance-au".to_string())).transpose()?,
    })
}

/// Which keys a command understands: a typo must not silently become the default value.
fn allowed_flags(cmd: &str) -> Option<&'static [&'static str]> {
    Some(match cmd {
        "presets" | "validate" => &[],
        "info" => &[],
        "throw" => &["speed", "elev", "az", "lat", "lon", "alt", "radius", "density", "cd", "no-drag", "no-rotation", "no-j2", "sample-dt", "max-time", "out"],
        "thermal" => &["lat", "dec", "days", "sample-days", "out"],
        "particle" => &["species", "mev", "L", "pitch", "time", "out", "lon"],
        "system" => &["bodies", "years", "dt-hours", "gr", "leapfrog", "out"],
        "dataset" => &["task", "n", "worlds", "out", "seed", "points", "noise", "strict"],
        "serve" => &["host", "port", "no-open", "token", "data-dir", "cors", "log", "insecure"],
        "health" => &["port"],
        "version" => &[],
        "mission" => &["out", "threads"],
        "params" => &["task"],
        "collect" => &["task", "worlds", "episodes", "policies", "noise", "randomize", "frame-skip", "max-steps", "gust", "lag", "fault", "rotor-scale", "threads", "seed", "terrain", "camera", "camera-every", "counterfactuals", "set", "terrain-file", "terrain-size", "out"],
        _ => return None,
    })
}

fn check_flags(cmd: &str, kv: &HashMap<String, String>) -> Result<(), String> {
    let Some(allowed) = allowed_flags(cmd) else { return Ok(()) };
    let takes_world = matches!(cmd, "info" | "throw" | "thermal" | "particle");
    const WORLD: &[&str] = &["world", "physics", "atmosphere", "surface", "gravity-scale", "distance-au"];
    let mut bad: Vec<&str> = kv.keys().map(|k| k.as_str()).filter(|k| !allowed.contains(k) && !(takes_world && WORLD.contains(k))).collect();
    bad.sort();
    if let Some(k) = bad.first() { return Err(format!("unknown key --{k} for command '{cmd}'. worldsim --help")); }
    Ok(())
}

fn task_bit(t: Task) -> u8 { match t { Task::Lander => 1, Task::Drone => 2, Task::Rover => 4 } }
fn task_index(t: Task) -> usize { match t { Task::Lander => 0, Task::Drone => 1, Task::Rover => 2 } }

fn out_file(kv: &HashMap<String, String>) -> Result<Option<BufWriter<File>>, String> {
    kv.get("out").map(|p| File::create(p).map(BufWriter::new).map_err(|e| format!("{p}: {e}"))).transpose()
}

fn run(cmd: &str, pos: &[String], kv: &HashMap<String, String>) -> Result<(), String> {
    check_flags(cmd, kv)?;
    match cmd {
        "presets" => {
            println!("physics:     {}", presets::PHYSICS_NAMES.join(", "));
            println!("atmosphere:  {}", presets::ATMOSPHERE_NAMES.join(", "));
            println!("surface:{}", presets::SURFACE_NAMES.join(", "));
        }
        "info" => info(&World::build(&world_spec(pos, kv)?)?)?,
        "throw" => {
            let w = World::build(&world_spec(pos, kv)?)?;
            if num(kv, "speed", 300.0)? < 0.0 { return Err("--speed: speed cannot be negative, direction comes from --elev and --az".into()); }
            if num(kv, "radius", 0.05)? <= 0.0 || num(kv, "density", 7800.0)? <= 0.0 { return Err("--radius and --density must be greater than zero".into()); }
            let p = Projectile::sphere(num(kv, "radius", 0.05)?, num(kv, "density", 7800.0)?);
            let p = Projectile { cd: num(kv, "cd", p.cd)?, ..p };
            let l = Launch { lat: num(kv, "lat", 0.0)?, lon: num(kv, "lon", 0.0)?, altitude: num(kv, "alt", 0.0)?,
                speed: num(kv, "speed", 300.0)?, elevation: num(kv, "elev", 45.0)?, azimuth: num(kv, "az", 90.0)? };
            let o = Options { rotating: !flag(kv, "no-rotation"), j2: !flag(kv, "no-j2"), drag: !flag(kv, "no-drag"),
                buoyancy: !flag(kv, "no-drag"), sample_dt: num(kv, "sample-dt", 0.1)?, max_time: num(kv, "max-time", 1e5)?, ..Default::default() };
            let r = ballistics::simulate(&w, p, l, o);
            println!("{}", w.name());
            println!("projectile: mass {:.3} kg, cross-section {:.4} m², Cd {}", p.mass, p.area, p.cd);
            println!("flight time      {:.3} s{}", r.flight_time, if r.landed { "" } else { " (did not land)" });
            println!("range         {:.3} m", r.downrange);
            println!("max altitude      {:.3} m", r.max_altitude);
            println!("impact speed    {:.3} m/s", r.impact_speed);
            println!("dissipated as heat  {:.3e} J", r.dissipated_energy);
            println!("energy balance error {:.2e}", r.energy_error);
            if let Some(mut f) = out_file(kv)? {
                writeln!(f, "t,lat,lon,altitude,downrange,speed,mach,x,y,z,vx,vy,vz").map_err(|e| e.to_string())?;
                for s in &r.samples {
                    writeln!(f, "{},{},{},{},{},{},{},{},{},{},{},{},{}", s.t, s.lat, s.lon, s.altitude, s.downrange, s.speed, s.mach,
                        s.pos.x, s.pos.y, s.pos.z, s.vel.x, s.vel.y, s.vel.z).map_err(|e| e.to_string())?;
                }
            }
        }
        "thermal" => {
            let w = World::build(&world_spec(pos, kv)?)?;
            let r = thermal::simulate(&w, ThermalOptions { latitude: num(kv, "lat", 0.0)?, declination: num(kv, "dec", 0.0)?,
                spinup_days: num(kv, "days", 15.0)? as usize, ..Default::default() });
            println!("{}", w.name());
            println!("latitude {}°, skin layer {:.3} m", num(kv, "lat", 0.0)?, r.skin_depth);
            println!("T surface: min {:.1} K, max {:.1} K, mean {:.1} K", r.t_min, r.t_max, r.t_mean);
            println!("energy balance residual {:.2e}", r.energy_residual);
            for h in [0usize, 3, 6, 9, 12, 15, 18, 21] {
                let i = h * r.surface_t.len() / 24;
                println!("  {:>2}:00  {:7.1} K", h, r.surface_t[i.min(r.surface_t.len() - 1)]);
            }
            if let Some(mut f) = out_file(kv)? {
                writeln!(f, "local_time_h,surface_T").map_err(|e| e.to_string())?;
                for (t, s) in r.local_time_h.iter().zip(&r.surface_t) { writeln!(f, "{t},{s}").map_err(|e| e.to_string())?; }
            }
        }
        "particle" => {
            let w = World::build(&world_spec(pos, kv)?)?;
            if w.physics.dipole_moment <= 0.0 { return Err(format!("at {} has no global magnetic field: particles are not trapped", w.physics.name)); }
            let sp = magneto::species(kv.get("species").map(|s| s.as_str()).unwrap_or("proton"))?;
            let r = magneto::simulate(&w, sp, ParticleLaunch { l_shell: num(kv, "L", 3.0)?, energy_mev: num(kv, "mev", 1.0)?, pitch_angle: num(kv, "pitch", 40.0)? },
                num(kv, "time", 20.0)?, 2000);
            println!("mirror latitude: simulation {:.2}°, theory {:.2}°", r.max_abs_lat, r.mirror_lat_theory);
            println!("longitude drift {:.2}°, |ΔE/E| = {:.2e}, trapped: {}", r.drift_longitude, r.energy_drift, r.trapped);
            if let Some(mut f) = out_file(kv)? {
                writeln!(f, "t,x,y,z").map_err(|e| e.to_string())?;
                for (t, p) in r.t.iter().zip(&r.pos) { writeln!(f, "{t},{},{},{}", p.x, p.y, p.z).map_err(|e| e.to_string())?; }
            }
        }
        "system" => {
            let names: Vec<String> = kv.get("bodies").cloned().unwrap_or("sun,mercury,venus,earth,moon,mars,jupiter".into()).split(',').map(|s| s.trim().to_string()).collect();
            let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            let mut s = NBody::from_presets(&refs)?;
            if flag(kv, "gr") { s.enable_relativity("sun")?; }
            if flag(kv, "leapfrog") { s.integrator = Integrator::Leapfrog; }
            let dt = num(kv, "dt-hours", 1.0)? * 3600.0;
            let total = num(kv, "years", 1.0)? * YEAR;
            let every = (num(kv, "sample-days", 1.0)? * DAY / dt).max(1.0) as usize;
            let mut f = out_file(kv)?;
            if let Some(f) = &mut f { writeln!(f, "t_days,body,x_au,y_au,z_au,vx,vy,vz").map_err(|e| e.to_string())?; }
            let i0 = s.invariants();
            let steps = (total / dt) as usize;
            let mut max_de: f64 = 0.0;
            for k in 0..=steps {
                if k > 0 { s.step(dt); }
                if k % every == 0 {
                    max_de = max_de.max(((s.invariants().energy - i0.energy) / i0.energy).abs());
                    if let Some(f) = &mut f {
                        for b in &s.bodies {
                            writeln!(f, "{},{},{},{},{},{},{},{}", s.t / DAY, b.name, b.r.x / AU, b.r.y / AU, b.r.z / AU, b.v.x, b.v.y, b.v.z).map_err(|e| e.to_string())?;
                        }
                    }
                }
            }
            let i1 = s.invariants();
            println!("bodies: {}, steps: {steps}, t = {:.3} yr", s.bodies.len(), s.t / YEAR);
            println!("max |ΔE/E| = {max_de:.2e}");
            println!("|ΔL|/|L|   = {:.2e}", (i1.angular_momentum - i0.angular_momentum).norm() / i0.angular_momentum.norm());
            for b in &s.bodies { println!("  {:8} r = {:.4} AU", b.name, b.r.norm() / AU); }
        }
        "dataset" => {
            let worlds: Vec<WorldSpec> = kv.get("worlds").cloned().unwrap_or("earth,moon,mars".into()).split(',').map(|s| WorldSpec::parse(s.trim())).collect();
            let cfg = DatasetConfig {
                task: kv.get("task").cloned().unwrap_or("projectile".into()),
                n: num(kv, "n", 100.0)? as usize,
                worlds, seed: num(kv, "seed", 42.0)? as u64,
                out_dir: kv.get("out").cloned().unwrap_or("data".into()),
                noise: num(kv, "noise", 0.0)?, strict: flag(kv, "strict"), points: num(kv, "points", 100.0)? as usize,
            };
            let t0 = std::time::Instant::now();
            let st = datagen::generate(&cfg)?;
            println!("task {}: created {} samples, valid {} ({:.1}%), max error {:.2e}, {:.1} s → {}/",
                cfg.task, st.generated, st.valid, 100.0 * st.valid as f64 / st.generated.max(1) as f64, st.max_error, t0.elapsed().as_secs_f64(), cfg.out_dir);
        }
        "serve" => {
            let port = num(kv, "port", 8080.0)?;
            if port.fract() != 0.0 || !(1.0..=65535.0).contains(&port) { return Err(format!("--port: need a port from 1 to 65535, got {port}")); }
            worldsim::server::serve_opts(worldsim::server::ServerOpts {
                host: kv.get("host").cloned().unwrap_or("127.0.0.1".into()),
                port: port as u16,
                open: !flag(kv, "no-open"),
                // a token on the command line is visible in the process list, so the environment variable is preferred
                token: kv.get("token").cloned().or(std::env::var("WORLDSIM_TOKEN").ok()).filter(|t| !t.is_empty()),
                data_dir: kv.get("data-dir").cloned().or(std::env::var("WORLDSIM_DATA").ok()).unwrap_or("runs".into()).into(),
                cors: flag(kv, "cors"),
                log: flag(kv, "log"),
                insecure: flag(kv, "insecure"),
            })?
        }
        "health" => {
            use std::io::{Read, Write as _};
            let port = num(kv, "port", 8080.0)? as u16;
            let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).map_err(|e| format!("port {port}: {e}"))?;
            c.write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n").map_err(|e| e.to_string())?;
            let mut r = String::new();
            c.read_to_string(&mut r).map_err(|e| e.to_string())?;
            if !r.starts_with("HTTP/1.1 200") { return Err(format!("server replied: {}", r.lines().next().unwrap_or(""))); }
            println!("ok");
        }
        "version" => println!("worldsim {}", env!("CARGO_PKG_VERSION")),
        "params" => {
            let task = EnvConfig::new(kv.get("task").map(|s| s.as_str()).unwrap_or("lander"), "earth")?.task;
            let mut group = "";
            for d in worldsim::env::PARAMS.iter().filter(|d| param_default(task, d.name).is_some() && d.tasks & task_bit(task) != 0) {
                if d.group != group { group = d.group; println!("\n{group}"); }
                println!("  {:<20} {:<34} {:>9} {:<6} [{}, {}]", d.name, d.label, d.defaults[task_index(task)], d.unit, d.min, d.max);
            }
        }
        "mission" => {
            let path = pos.first().ok_or("specify the mission file: worldsim mission missions/lunar_lander.json --out report")?;
            let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            let threads = num(kv, "threads", std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1) as f64)? as usize;
            let report = mission::run(&mission::MissionSpec::parse_local(&text)?, threads)?.set("spec_sha256", worldsim::store::sha256(text.as_bytes()));
            if let Some(out) = kv.get("out") {
                std::fs::create_dir_all(out).map_err(|e| format!("{out}: {e}"))?;
                std::fs::write(format!("{out}/report.json"), report.render()).map_err(|e| e.to_string())?;
                std::fs::write(format!("{out}/report.md"), mission::markdown(&report)).map_err(|e| e.to_string())?;
                std::fs::write(format!("{out}/spec.json"), &text).map_err(|e| e.to_string())?;
            }
            print!("{}", mission::markdown(&report));
        }
        "collect" => {
            let cfg = collect::CollectConfig {
                task: kv.get("task").cloned().unwrap_or("lander".into()),
                worlds: kv.get("worlds").cloned().unwrap_or("earth,moon,mars".into()).split(',').map(|s| WorldSpec::parse(s.trim())).collect(),
                policies: collect::Policy::parse_list(kv.get("policies").map(|s| s.as_str()).unwrap_or("smooth,hold,auto_noise"))?,
                episodes: num(kv, "episodes", 100.0)? as usize,
                seed: num(kv, "seed", 42.0)? as u64,
                noise: num(kv, "noise", 1.0)?,
                randomize: flag(kv, "randomize"),
                frame_skip: kv.contains_key("frame-skip").then(|| num(kv, "frame-skip", 4.0)).transpose()?.map(|x| x as usize),
                max_steps: kv.contains_key("max-steps").then(|| num(kv, "max-steps", 1.0)).transpose()?.map(|x| x as usize),
                rotor_scale: num(kv, "rotor-scale", 1.0)?,
                gust: num(kv, "gust", 0.0)?,
                lag: num(kv, "lag", 0.0)?,
                fault: num(kv, "fault", 0.0)?,
                threads: num(kv, "threads", std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1) as f64)? as usize,
                out_dir: kv.get("out").cloned().ok_or("--out: specify the data folder")?,
                terrain_n: num(kv, "terrain", 0.0)? as usize,
                camera_n: num(kv, "camera", 0.0)? as usize,
                camera_every: num(kv, "camera-every", 1.0)? as usize,
                counterfactuals: num(kv, "counterfactuals", 0.0)? as usize,
                params: kv.get("set").cloned().unwrap_or_default(),
                custom_terrain: kv.get("terrain-file").map(|p| worldsim::dem::load(p, kv.get("terrain-size").and_then(|v| v.parse().ok()), 0.6).map(std::sync::Arc::new)).transpose()?,
            };
            let st = collect::run(&cfg)?;
            println!("{}: {} episodes, {} transitions, successful {:.1}%, {:.0} steps/s → {}/", cfg.task, st.episodes, st.transitions,
                100.0 * st.successes as f64 / st.episodes.max(1) as f64, st.transitions as f64 / st.seconds.max(1e-9), cfg.out_dir);
            if st.non_finite > 0 { return Err(format!("found {} non-numeric values", st.non_finite)); }
        }
        "validate" => {
            let t0 = std::time::Instant::now();
            let checks = validate::all();
            let ok = checks.iter().filter(|c| c.pass).count();
            for c in &checks {
                println!("{} {:<38} {:<58} {:>18}  expected {}", if c.pass { "✔" } else { "✘" }, c.law, c.test, c.value, c.expected);
            }
            println!("\npassed {ok}/{} in {:.1} s", checks.len(), t0.elapsed().as_secs_f64());
            if ok != checks.len() { return Err("some checks failed".into()); }
        }
        _ => return Err(format!("unknown command '{cmd}'. worldsim --help")),
    }
    Ok(())
}

fn info(w: &World) -> Result<(), String> {
    let p = &w.physics;
    println!("{}", w.name());
    println!("— physics —");
    println!("  mass            {:.4e} kg", p.gm / worldsim::constants::G);
    println!("  radius           {:.1} km", p.radius / 1e3);
    println!("  g at surface  {:.4} m/s²", w.g_surface());
    println!("  escape velocity  {:.3} km/s", w.escape_velocity() / 1e3);
    println!("  orbital velocity  {:.3} km/s", w.circular_velocity(0.0) / 1e3);
    println!("  solar day (synodic.)   {:.3} h", p.solar_day / 3600.0);
    println!("  J2               {:.3e}", p.j2);
    println!("  magnetic dipole     {:.2e} A·m²", p.dipole_moment);
    println!("  to the star        {:.4} AU, insolation {:.1} W/m²", w.star_distance / AU, w.insolation());
    println!("  T equilibrium    {:.1} K", w.equilibrium_temperature());
    match &w.atmosphere {
        None => println!("— atmosphere: none (vacuum)"),
        Some(a) => {
            println!("— atmosphere ({}) —", a.preset.name);
            println!("  composition           {}", a.preset.composition.iter().map(|(g, x)| format!("{g} {:.3}%", x * 100.0)).collect::<Vec<_>>().join(", "));
            println!("  M = {:.2} g/mol, γ = {:.3}, τ_IR = {:.3}", a.molar_mass * 1e3, a.gamma, w.greenhouse_tau());
            println!("  scale height      {:.2} km", a.scale_height(w.g_surface()) / 1e3);
            println!("  {:>8} {:>10} {:>12} {:>12} {:>9}", "h, km", "T, K", "P, Pa", "ρ, kg/m³", "c, m/s");
            for h in [0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0] {
                let s = a.at(h * 1e3);
                println!("  {:>8.0} {:>10.1} {:>12.4e} {:>12.4e} {:>9.1}", h, s.temperature, s.pressure, s.density, s.sound_speed);
            }
        }
    }
    println!("  T surface (mean.) {:.1} K", w.mean_surface_temperature());
    let s = &w.surface;
    println!("— surface ({}) — {}", s.name, s.description);
    println!("  oxides, mass.%   {}", presets::normalized(&s.oxides_wt).iter().map(|(o, x)| format!("{o} {:.1}", x * 100.0)).collect::<Vec<_>>().join(", "));
    println!("  elements, mass.% {}", presets::oxides_to_elements(&s.oxides_wt)?.iter().map(|(e, x)| format!("{e} {:.1}", x * 100.0)).collect::<Vec<_>>().join(", "));
    println!("  thermal inertia {:.0}, thermal conductivity {:.4} W/(m·K)", s.thermal_inertia, s.conductivity());
    Ok(())
}
