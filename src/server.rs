//! Локальный HTTP-сервер (только std): отдаёт 3D-интерфейс и API ко всем симуляциям.
//! `worldsim serve --port 8080` → http://localhost:8080

use crate::ballistics::{self, Launch, Options, Projectile};
use crate::climate::{self, ClimateOptions};
use crate::constants::{AU, DAY, G};
use crate::cosmology::{self, Cosmology, PmSim};
use crate::galaxy::{self, Galaxy, TIME_GYR};
use crate::json::J;
use crate::nbody::NBody;
use crate::presets;
use crate::stars::{self, Star};
use crate::starsystem::{StarSystem, GM_EARTH, R_EARTH};
use crate::validate;
use crate::world::{World, WorldSpec};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const INDEX: &str = include_str!("../web/index.html");
const STARS: &[u8] = include_bytes!("../data/stars_hyg.json");
const EXOPLANETS: &[u8] = include_bytes!("../data/exoplanets_oec.json");

enum Sim {
    NBody { sim: NBody, e0: f64 },
    Galaxy(Galaxy),
    Pm(PmSim),
}

#[derive(Clone)]
struct State {
    sims: Arc<Mutex<HashMap<u64, Arc<Mutex<Sim>>>>>,
    next: Arc<AtomicU64>,
}

type Q = HashMap<String, String>;

struct Resp { status: u16, ctype: &'static str, body: Vec<u8>, headers: Vec<(String, String)> }

impl Resp {
    fn json(j: J) -> Resp { Resp { status: 200, ctype: "application/json; charset=utf-8", body: j.render().into_bytes(), headers: vec![] } }
    fn bin(b: Vec<u8>, headers: Vec<(String, String)>) -> Resp { Resp { status: 200, ctype: "application/octet-stream", body: b, headers } }
    fn err(code: u16, msg: &str) -> Resp { Resp { status: code, ctype: "application/json; charset=utf-8", body: J::obj().set("error", msg).render().into_bytes(), headers: vec![] } }
}

pub fn serve(port: u16, open_browser: bool) -> Result<(), String> {
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| format!("порт {port}: {e}"))?;
    let url = format!("http://localhost:{port}");
    println!("worldsim: интерфейс на {url}  (Ctrl+C — остановить)");
    if open_browser {
        let cmd = if cfg!(target_os = "macos") { Some("open") } else if cfg!(target_os = "linux") { Some("xdg-open") } else { None };
        if let Some(c) = cmd { let _ = std::process::Command::new(c).arg(&url).spawn(); }
        #[cfg(target_os = "windows")]
        { let _ = std::process::Command::new("cmd").args(["/C", "start", &url]).spawn(); }
    }
    let state = State { sims: Arc::new(Mutex::new(HashMap::new())), next: Arc::new(AtomicU64::new(1)) };
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let st = state.clone();
        std::thread::spawn(move || { let _ = handle(stream, st); });
    }
    Ok(())
}

fn handle(mut stream: TcpStream, st: State) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut content_len = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" { break; }
        if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") { content_len = v.trim().parse().unwrap_or(0); }
    }
    if content_len > 0 { let mut b = vec![0; content_len.min(1 << 20)]; reader.read_exact(&mut b)?; }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("/");
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let q = parse_query(query);
    let resp = if method == "GET" || method == "POST" {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| route(path, &q, &st))) {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => Resp::err(400, &e),
            Err(_) => Resp::err(500, "внутренняя ошибка симуляции"),
        }
    } else { Resp::err(405, "только GET") };
    let reason = match resp.status { 200 => "OK", 400 => "Bad Request", 404 => "Not Found", 405 => "Method Not Allowed", _ => "Internal Server Error" };
    let mut head = format!("HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Expose-Headers: *\r\nCache-Control: no-store\r\nConnection: close\r\n",
        resp.status, reason, resp.ctype, resp.body.len());
    for (k, v) in &resp.headers { head.push_str(&format!("{k}: {v}\r\n")); }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(&resp.body)?;
    stream.flush()
}

fn parse_query(s: &str) -> Q {
    s.split('&').filter(|p| !p.is_empty()).map(|p| {
        let (k, v) = p.split_once('=').unwrap_or((p, ""));
        (decode(k), decode(v))
    }).collect()
}

fn decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'+' {
            out.push(b' ');
            i += 1;
        } else if b[i] == b'%' && i + 2 < b.len() {
            match std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                Some(v) => { out.push(v); i += 3; }
                None => { out.push(b'%'); i += 1; }
            }
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn f(q: &Q, k: &str, d: f64) -> f64 { q.get(k).and_then(|v| v.parse().ok()).unwrap_or(d) }
fn s<'a>(q: &'a Q, k: &str, d: &'a str) -> &'a str { q.get(k).map(|v| v.as_str()).filter(|v| !v.is_empty()).unwrap_or(d) }

fn route(path: &str, q: &Q, st: &State) -> Result<Resp, String> {
    Ok(match path {
        "/" | "/index.html" => Resp { status: 200, ctype: "text/html; charset=utf-8", body: INDEX.as_bytes().to_vec(), headers: vec![] },
        "/data/stars.json" => Resp { status: 200, ctype: "application/json", body: STARS.to_vec(), headers: vec![] },
        "/data/exoplanets.json" => Resp { status: 200, ctype: "application/json", body: EXOPLANETS.to_vec(), headers: vec![] },
        "/api/presets" => Resp::json(presets_json()),
        "/api/world" => Resp::json(world_json(&world_from(q)?)?),
        "/api/climate" => Resp::json(climate_json(q)?),
        "/api/throw" => Resp::json(throw_json(q)?),
        "/api/system" => Resp::json(system_json(&system_from(q)?)),
        "/api/nbody/new" => nbody_new(q, st)?,
        "/api/nbody/step" => nbody_step(q, st)?,
        "/api/galaxy/new" => galaxy_new(q, st)?,
        "/api/galaxy/step" => galaxy_step(q, st)?,
        "/api/galaxy/rotation" => galaxy_rotation(q, st)?,
        "/api/cosmo/new" => cosmo_new(q, st)?,
        "/api/cosmo/step" => cosmo_step(q, st)?,
        "/api/cosmo/power" => cosmo_power(q, st)?,
        "/api/cosmo/background" => Resp::json(cosmo_background()),
        "/api/validate" => Resp::json(J::Arr(validate::all().into_iter().map(|c| J::obj().set("law", c.law).set("test", c.test).set("value", c.value).set("expected", c.expected).set("pass", c.pass)).collect())),
        _ => Resp::err(404, "нет такого пути"),
    })
}

// ------------------------------------------------------------------ миры

fn presets_json() -> J {
    let strs = |v: &[&str]| J::Arr(v.iter().map(|x| J::from(*x)).collect());
    J::obj().set("physics", strs(presets::PHYSICS_NAMES)).set("atmosphere", strs(presets::ATMOSPHERE_NAMES))
        .set("surface", strs(presets::SURFACE_NAMES)).set("stars", strs(stars::STAR_NAMES))
        .set("galaxy", strs(galaxy::PRESETS)).set("cosmology", strs(cosmology::PRESETS))
}

fn star_from(q: &Q) -> Result<Star, String> {
    if q.contains_key("ts") {
        let ts = f(q, "ts", 5772.0);
        let rs = f(q, "rs", 1.0);
        let ms = f(q, "ms", 1.0);
        Ok(Star { name: s(q, "star_name", "звезда").to_string(), mass: ms, radius: rs, luminosity: rs * rs * (ts / 5772.0).powi(4), teff: ts })
    } else {
        stars::star(s(q, "star", "sun"))
    }
}

fn system_from(q: &Q) -> Result<StarSystem, String> {
    let seed = f(q, "seed", 1.0) as u64;
    match s(q, "star", "random") {
        "solar" => Ok(StarSystem::solar()),
        "random" if !q.contains_key("ts") => StarSystem::generate(seed, None),
        _ => regenerate_with(star_from(q)?, seed),
    }
}

fn regenerate_with(star: Star, seed: u64) -> Result<StarSystem, String> {
    // генерация с произвольной звездой: передаём её через пресет-строку ms:<масса>, затем подменяем измеренными параметрами
    let mut sys = StarSystem::generate(seed, Some(&format!("ms:{}", star.mass)))?;
    sys.star = star;
    sys.hz = sys.star.habitable_zone();
    for p in &mut sys.planets {
        p.t_eq = sys.star.teff * (sys.star.radius * stars::R_SUN / (2.0 * p.a * AU)).sqrt() * (1.0 - p.albedo).powf(0.25);
        p.in_hz = p.a >= sys.hz.0 && p.a <= sys.hz.1;
        p.kind = crate::starsystem::classify(p.mass, p.t_eq);
    }
    Ok(sys)
}

/// Мир либо из пресетов (physics/atmosphere/surface), либо из параметров планеты (m, r, a, + звезда).
fn world_from(q: &Q) -> Result<World, String> {
    if q.contains_key("m") {
        let star = star_from(q)?;
        let r = q.get("r").and_then(|v| v.parse().ok());
        let sys = StarSystem::single(star, f(q, "m", 1.0), r, f(q, "a", 1.0), f(q, "e", 0.0), s(q, "name", "планета"));
        return sys.world(0);
    }
    if q.contains_key("seed") && q.contains_key("planet") {
        let sys = system_from(q)?;
        let i = (f(q, "planet", 0.0) as usize).min(sys.planets.len() - 1);
        return sys.world(i);
    }
    World::build(&WorldSpec {
        physics: s(q, "physics", "earth").into(),
        atmosphere: q.get("atmosphere").filter(|v| !v.is_empty() && *v != "auto").cloned(),
        surface: q.get("surface").filter(|v| !v.is_empty() && *v != "auto").cloned(),
        gravity_scale: f(q, "gravity_scale", 1.0),
        distance_au: q.get("distance_au").and_then(|v| v.parse().ok()),
    })
}

fn world_json(w: &World) -> Result<J, String> {
    let p = &w.physics;
    let mut j = J::obj().set("name", w.name()).set("physics", p.name.clone())
        .set("mass_kg", p.gm / G).set("mass_earth", p.gm / GM_EARTH).set("radius_km", p.radius / 1e3).set("radius_earth", p.radius / R_EARTH)
        .set("g", w.g_surface()).set("v_esc", w.escape_velocity()).set("v_orb", w.circular_velocity(0.0))
        .set("day_h", p.solar_day / 3600.0).set("j2", p.j2).set("obliquity", p.obliquity).set("dipole", p.dipole_moment)
        .set("distance_au", w.star_distance / AU).set("insolation", w.insolation()).set("albedo", p.bond_albedo)
        .set("t_eq", w.equilibrium_temperature()).set("t_mean", w.mean_surface_temperature()).set("tau", w.greenhouse_tau())
        .set("has_surface", p.has_surface).set("year_days", climate::orbital_period(w) / DAY);
    j = match &w.atmosphere {
        None => j.set("atmosphere", J::Null),
        Some(a) => {
            let mut hs = Vec::new(); let mut ts = Vec::new(); let mut ps = Vec::new(); let mut rs = Vec::new(); let mut cs = Vec::new();
            let top = a.top().min(120e3);
            let mut h = 0.0;
            while h <= top { let s = a.at(h); hs.push(h / 1e3); ts.push(s.temperature); ps.push(s.pressure); rs.push(s.density); cs.push(s.sound_speed); h += top / 120.0; }
            j.set("atmosphere", J::obj().set("name", a.preset.name.clone())
                .set("composition", J::Arr(a.preset.composition.iter().map(|(g, x)| J::Arr(vec![J::from(g.clone()), J::from(*x)])).collect()))
                .set("pressure", a.preset.surface_pressure).set("molar_mass", a.molar_mass).set("gamma", a.gamma)
                .set("scale_height_km", a.scale_height(w.g_surface()) / 1e3).set("column_kg_m2", a.preset.surface_pressure / w.g_surface())
                .set("profile", J::obj().set("h_km", hs).set("T", ts).set("P", ps).set("rho", rs).set("c", cs)))
        }
    };
    let sf = &w.surface;
    let ox = presets::normalized(&sf.oxides_wt);
    let el = presets::oxides_to_elements(&sf.oxides_wt)?;
    j = j.set("surface", J::obj().set("name", sf.name.clone()).set("description", sf.description.clone())
        .set("oxides", J::Arr(ox.into_iter().map(|(k, v)| J::Arr(vec![J::from(k), J::from(v)])).collect()))
        .set("elements", J::Arr(el.into_iter().map(|(k, v)| J::Arr(vec![J::from(k), J::from(v)])).collect()))
        .set("thermal_inertia", sf.thermal_inertia).set("density", sf.density));
    Ok(j)
}

fn climate_json(q: &Q) -> Result<J, String> {
    let w = world_from(q)?;
    let o = ClimateOptions {
        solar_scale: f(q, "solar", 1.0), tau_scale: f(q, "tau", 1.0),
        obliquity: q.get("obliquity").and_then(|v| v.parse().ok()),
        eccentricity: q.get("ecc").and_then(|v| v.parse().ok()),
        ocean_fraction: q.get("ocean").and_then(|v| v.parse().ok()),
        initial_t: if s(q, "cold", "0") == "1" { Some(200.0) } else { None },
        years: f(q, "years", 30.0) as usize, bands: 60, samples_per_year: 48, ..Default::default()
    };
    let r = climate::simulate(&w, &o);
    Ok(J::obj().set("world", w.name()).set("lat", r.lat).set("time", r.time_frac)
        .set("temp", J::Arr(r.temp.into_iter().map(J::from).collect())).set("annual", r.annual_mean)
        .set("global", r.global_mean).set("ice_edge", r.ice_edge).set("ice_t", r.ice_temperature)
        .set("D", r.d).set("C", r.heat_capacity).set("residual", r.energy_residual).set("imbalance", r.toa_imbalance)
        .set("year_days", climate::orbital_period(&w) / DAY))
}

fn throw_json(q: &Q) -> Result<J, String> {
    let w = world_from(q)?;
    let p = Projectile::sphere(f(q, "radius", 0.1), f(q, "density", 2000.0));
    let speed = f(q, "speed", 300.0);
    let l = Launch { lat: f(q, "lat", 30.0), lon: 0.0, altitude: f(q, "alt", 0.0), speed, elevation: f(q, "elev", 45.0), azimuth: f(q, "az", 90.0) };
    let t_guess = (2.0 * speed * l.elevation.to_radians().sin().max(0.05) / w.g_surface()).max(1.0);
    let o = Options { drag: s(q, "drag", "1") == "1", buoyancy: s(q, "drag", "1") == "1", sample_dt: t_guess / 300.0, max_time: f(q, "max_time", t_guess * 50.0 + 100.0), ..Default::default() };
    let r = ballistics::simulate(&w, p, l, o);
    let rr = w.physics.radius;
    let pts: Vec<f64> = r.samples.iter().flat_map(|s| [s.pos.x / rr, s.pos.z / rr, -s.pos.y / rr]).collect();
    let alt: Vec<f64> = r.samples.iter().map(|s| s.altitude).collect();
    let dr: Vec<f64> = r.samples.iter().map(|s| s.downrange).collect();
    let sp: Vec<f64> = r.samples.iter().map(|s| s.speed).collect();
    Ok(J::obj().set("points", pts).set("altitude", alt).set("downrange", dr).set("speed", sp)
        .set("flight_time", r.flight_time).set("range", r.downrange).set("max_alt", r.max_altitude).set("impact_speed", r.impact_speed)
        .set("landed", r.landed).set("energy_error", r.energy_error).set("dissipated", r.dissipated_energy).set("mass", p.mass))
}

// ------------------------------------------------------------------ системы и N тел

fn star_json(st: &Star) -> J {
    let rgb = st.rgb();
    let hz = st.habitable_zone();
    J::obj().set("name", st.name.clone()).set("mass", st.mass).set("radius", st.radius).set("luminosity", st.luminosity)
        .set("teff", st.teff).set("class", st.spectral_class()).set("lifetime_gyr", st.lifetime_years() / 1e9)
        .set("rgb", rgb.to_vec()).set("hz", vec![hz.0, hz.1])
}

fn system_json(sys: &StarSystem) -> J {
    J::obj().set("seed", sys.seed).set("star", star_json(&sys.star)).set("planets", J::Arr(sys.planets.iter().map(|p| {
        J::obj().set("name", p.name.clone()).set("mass", p.mass).set("radius", p.radius).set("a", p.a).set("e", p.e).set("inc", p.inc)
            .set("t_eq", p.t_eq).set("kind", p.kind).set("in_hz", p.in_hz)
            .set("period_days", 365.25 * (p.a.powi(3) / sys.star.mass).sqrt())
    }).collect()))
}

fn store(st: &State, sim: Sim) -> u64 {
    let id = st.next.fetch_add(1, Ordering::SeqCst);
    let mut m = st.sims.lock().unwrap();
    if m.len() > 24 { if let Some(&k) = m.keys().min() { m.remove(&k); } }
    m.insert(id, Arc::new(Mutex::new(sim)));
    id
}

fn get(st: &State, q: &Q) -> Result<Arc<Mutex<Sim>>, String> {
    let id = f(q, "id", 0.0) as u64;
    st.sims.lock().unwrap().get(&id).cloned().ok_or_else(|| "сессия симуляции не найдена — создайте заново".to_string())
}

const BODY_COLORS: &[(&str, &str)] = &[("sun", "#ffd27a"), ("mercury", "#9c8f86"), ("venus", "#e8cf8e"), ("earth", "#4f8bd6"),
    ("moon", "#bdbdbd"), ("mars", "#c9562e"), ("jupiter", "#d8b48a")];

fn nbody_new(q: &Q, st: &State) -> Result<Resp, String> {
    let kind = s(q, "kind", "solar");
    let (mut sim, meta) = if kind == "solar" {
        let names: Vec<String> = s(q, "bodies", "sun,mercury,venus,earth,moon,mars,jupiter").split(',').map(|x| x.trim().to_string()).collect();
        let refs: Vec<&str> = names.iter().map(|x| x.as_str()).collect();
        let sim = NBody::from_presets(&refs)?;
        let meta = J::Arr(names.iter().map(|n| {
            let p = presets::physics(n).unwrap();
            let col = BODY_COLORS.iter().find(|c| c.0 == n).map(|c| c.1).unwrap_or("#ffffff");
            J::obj().set("name", n.clone()).set("radius_km", p.radius / 1e3).set("color", col).set("star", n == "sun")
        }).collect());
        (sim, J::obj().set("bodies", meta))
    } else {
        let sys = system_from(q)?;
        let sim = sys.nbody();
        let mut meta = vec![J::obj().set("name", sys.star.name.clone()).set("radius_km", sys.star.radius * stars::R_SUN / 1e3).set("star", true)
            .set("color", format!("rgb({},{},{})", (sys.star.rgb()[0] * 255.0) as u8, (sys.star.rgb()[1] * 255.0) as u8, (sys.star.rgb()[2] * 255.0) as u8))];
        for p in &sys.planets {
            let col = if p.in_hz { "#5fd38d" } else if p.mass > 15.0 { "#d9b38c" } else if p.t_eq > 400.0 { "#e07b4f" } else { "#9fb4c9" };
            meta.push(J::obj().set("name", p.name.clone()).set("radius_km", p.radius * R_EARTH / 1e3).set("color", col).set("star", false));
        }
        (sim, J::obj().set("bodies", J::Arr(meta)).set("system", system_json(&sys)))
    };
    if s(q, "gr", "0") == "1" { if let Some(i) = sim.index("sun") { sim.relativity_center = Some(i); } }
    let e0 = sim.invariants().energy;
    let id = store(st, Sim::NBody { sim, e0 });
    Ok(Resp::json(meta.set("id", id)))
}

fn nbody_step(q: &Q, st: &State) -> Result<Resp, String> {
    let h = get(st, q)?;
    let mut g = h.lock().unwrap();
    let Sim::NBody { sim, e0 } = &mut *g else { return Err("не N-тел сессия".into()) };
    let total = f(q, "days", 1.0).clamp(0.0, 3650.0) * DAY;
    let dt = (f(q, "dt_hours", 2.0) * 3600.0).clamp(60.0, 30.0 * DAY);
    let n = (total / dt).ceil().max(1.0) as usize;
    for _ in 0..n.min(200_000) { sim.step(total / n as f64); }
    let pos: Vec<f64> = sim.bodies.iter().flat_map(|b| [b.r.x / AU, b.r.y / AU, b.r.z / AU]).collect();
    let err = ((sim.invariants().energy - *e0) / *e0).abs();
    Ok(Resp::json(J::obj().set("t_days", sim.t / DAY).set("pos", pos).set("energy_error", err)))
}

// ------------------------------------------------------------------ галактики

fn f32_bytes(it: impl Iterator<Item = f64>) -> Vec<u8> { it.flat_map(|x| (x as f32).to_le_bytes()).collect() }

fn galaxy_new(q: &Q, st: &State) -> Result<Resp, String> {
    let n = (f(q, "n", 20000.0) as usize).clamp(500, 200_000);
    let g = galaxy::preset(s(q, "preset", "milky_way"), n, f(q, "seed", 1.0) as u64)?;
    let kinds: String = g.kind.iter().map(|k| char::from(b'0' + *k as u8)).collect();
    let n = g.len();
    let scale = match s(q, "preset", "milky_way") { "plummer" => 1.0, "globular" => 0.003, _ => 10.0 };
    let id = store(st, Sim::Galaxy(g));
    Ok(Resp::json(J::obj().set("id", id).set("n", n).set("kinds", kinds).set("scale_kpc", scale)))
}

fn galaxy_step(q: &Q, st: &State) -> Result<Resp, String> {
    let h = get(st, q)?;
    let mut g = h.lock().unwrap();
    let Sim::Galaxy(gal) = &mut *g else { return Err("не галактическая сессия".into()) };
    let steps = (f(q, "steps", 1.0) as usize).min(50);
    let dt = f(q, "dt", 0.002);
    for _ in 0..steps { gal.step(dt); }
    let body = f32_bytes(gal.pos.iter().flat_map(|p| [p.x, p.z, -p.y]));
    Ok(Resp::bin(body, vec![("X-T-Gyr".into(), format!("{:.4}", gal.t * TIME_GYR))]))
}

fn galaxy_rotation(q: &Q, st: &State) -> Result<Resp, String> {
    let h = get(st, q)?;
    let g = h.lock().unwrap();
    let Sim::Galaxy(gal) = &*g else { return Err("не галактическая сессия".into()) };
    let rc = gal.rotation_curve(25.0, 25);
    let (k, w) = gal.energy();
    Ok(Resp::json(J::obj().set("r", rc.iter().map(|x| x.0).collect::<Vec<_>>()).set("v", rc.iter().map(|x| x.1).collect::<Vec<_>>())
        .set("virial", 2.0 * k / w.abs()).set("energy", k + w)))
}

// ------------------------------------------------------------------ космология

fn cosmo_new(q: &Q, st: &State) -> Result<Resp, String> {
    let c = Cosmology::preset(s(q, "preset", "planck2018"))?;
    let n = match f(q, "n", 32.0) as usize { x if x >= 64 => 64, x if x >= 32 => 32, _ => 16 };
    let bx = f(q, "box", 200.0).clamp(10.0, 2000.0);
    let a0 = f(q, "a0", 0.02).clamp(0.005, 0.5);
    let pm = PmSim::new(c, n, bx, a0, f(q, "seed", 1.0) as u64);
    let id = store(st, Sim::Pm(pm));
    Ok(Resp::json(J::obj().set("id", id).set("n", n).set("box", bx).set("a", a0).set("age_gyr", c.age_gyr(a0)).set("age_now", c.age_gyr(1.0))))
}

fn cosmo_step(q: &Q, st: &State) -> Result<Resp, String> {
    let h = get(st, q)?;
    let mut g = h.lock().unwrap();
    let Sim::Pm(pm) = &mut *g else { return Err("не космологическая сессия".into()) };
    let da = f(q, "da", 0.01).clamp(0.0, 0.05);
    let steps = (f(q, "steps", 1.0) as usize).clamp(0, 10);
    for _ in 0..steps { if pm.a < 1.0 { pm.step(da.min(1.0 - pm.a).max(1e-6)); } }
    let rms = pm.delta_rms();
    // плотность в ячейке частицы — для раскраски
    let n = pm.n;
    let cell = pm.box_size / n as f64;
    let half = pm.box_size / 2.0;
    let mut out = Vec::with_capacity(pm.pos.len() * 16);
    for p in &pm.pos {
        let idx = |c: f64| ((c / cell) as usize).min(n - 1);
        let d = pm.density[(idx(p[0]) * n + idx(p[1])) * n + idx(p[2])];
        for v in [p[0] - half, p[1] - half, p[2] - half, d] { out.extend_from_slice(&(v as f32).to_le_bytes()); }
    }
    let a = pm.a;
    let c = pm.cosmo;
    Ok(Resp::bin(out, vec![("X-A".into(), format!("{a:.5}")), ("X-Z".into(), format!("{:.3}", 1.0 / a - 1.0)),
        ("X-Age-Gyr".into(), format!("{:.4}", c.age_gyr(a))), ("X-Delta-Rms".into(), format!("{rms:.4}")),
        ("X-Growth".into(), format!("{:.5}", c.growth(a)))]))
}

fn cosmo_power(q: &Q, st: &State) -> Result<Resp, String> {
    let h = get(st, q)?;
    let mut g = h.lock().unwrap();
    let Sim::Pm(pm) = &mut *g else { return Err("не космологическая сессия".into()) };
    let p = pm.measure_power(16);
    let c = pm.cosmo;
    let d2 = c.growth(pm.a).powi(2);
    Ok(Resp::json(J::obj().set("k", p.iter().map(|x| x.0).collect::<Vec<_>>()).set("p", p.iter().map(|x| x.1).collect::<Vec<_>>())
        .set("p_lin", p.iter().map(|x| c.power(x.0) * d2).collect::<Vec<_>>()).set("a", pm.a)))
}

fn cosmo_background() -> J {
    let mut out = J::obj();
    for name in cosmology::PRESETS {
        let c = Cosmology::preset(name).unwrap();
        let mut ts = Vec::new(); let mut as_ = Vec::new(); let mut ds = Vec::new(); let mut zs = Vec::new(); let mut dl = Vec::new();
        for i in 0..=80 {
            let a = 0.01 + (2.5 - 0.01) * (i as f64 / 80.0).powi(2);
            as_.push(a);
            ts.push(c.age_gyr(a));
            ds.push(c.growth(a));
        }
        for i in 0..=40 { let z = 3.0 * i as f64 / 40.0; zs.push(z); dl.push(c.luminosity_distance(z)); }
        out = out.set(name, J::obj().set("a", as_).set("t_gyr", ts).set("D", ds).set("z", zs).set("dl_mpc", dl)
            .set("age", c.age_gyr(1.0)).set("om", c.omega_m).set("ol", c.omega_l).set("h", c.h));
    }
    out
}

