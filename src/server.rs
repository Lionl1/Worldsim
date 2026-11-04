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
const MISSION_EXAMPLE: &str = include_str!("../missions/lunar_lander.json");

/// Каталоги читаются с диска: ./data или папка проекта. Собрать: tools/fetch_data.sh
fn data_file(name: &str) -> Option<Vec<u8>> {
    let candidates = [std::path::PathBuf::from("data").join(name), std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data").join(name)];
    candidates.iter().find_map(|p| std::fs::read(p).ok())
}

fn data_resp(name: &str) -> Resp {
    match data_file(name) {
        Some(b) => Resp { status: 200, ctype: "application/json", body: b, headers: vec![] },
        None => Resp::err(404, "каталоги не скачаны: выполните ./tools/fetch_data.sh в папке проекта"),
    }
}

enum Sim {
    Env(Box<crate::env::Env>),
    NBody { sim: NBody, e0: f64 },
    Galaxy(Galaxy),
    Pm(PmSim),
}

struct Slot { sim: Arc<Mutex<Sim>>, last_used: u64, is_env: bool }

/// Сколько сессий держим: среды лёгкие и нужны сотнями, тяжёлые симуляции — десятками.
const MAX_ENV_SESSIONS: usize = 1024;
const MAX_OTHER_SESSIONS: usize = 64;

/// Настройки сервера.
#[derive(Clone, Debug)]
pub struct ServerOpts { pub host: String, pub port: u16, pub open: bool, pub token: Option<String>, pub data_dir: std::path::PathBuf, pub cors: bool, pub log: bool, pub insecure: bool }

const MAX_CONNECTIONS: usize = 512;
const MAX_BODY: usize = 1 << 20;
const MAX_LINE: u64 = 16 * 1024;
const MAX_HEADERS: usize = 100;
const MAX_ACTIVE_JOBS: usize = 4;
const MAX_KEPT_JOBS: usize = 200;

/// Фоновая задача.
struct Job { id: String, name: String, spec: String, progress: crate::mission::Progress, state: Mutex<JobState> }
#[derive(Clone)]
enum JobState { Running, Done(String), Failed(String), Cancelled }

#[derive(Clone)]
struct State {
    sims: Arc<Mutex<HashMap<u64, Slot>>>,
    next: Arc<AtomicU64>,
    tick: Arc<AtomicU64>,
    opts: Arc<ServerOpts>,
    active: Arc<std::sync::atomic::AtomicUsize>,
    jobs: Arc<Mutex<Vec<Arc<Job>>>>,
}

/// Снимает счётчик соединений даже при панике в потоке.
struct ConnGuard(Arc<std::sync::atomic::AtomicUsize>);
impl Drop for ConnGuard { fn drop(&mut self) { self.0.fetch_sub(1, Ordering::SeqCst); } }

type Q = HashMap<String, String>;

struct Resp { status: u16, ctype: &'static str, body: Vec<u8>, headers: Vec<(String, String)> }

impl Resp {
    fn json(j: J) -> Resp { Resp { status: 200, ctype: "application/json; charset=utf-8", body: j.render().into_bytes(), headers: vec![] } }
    fn bin(b: Vec<u8>, headers: Vec<(String, String)>) -> Resp { Resp { status: 200, ctype: "application/octet-stream", body: b, headers } }
    fn err(code: u16, msg: &str) -> Resp { Resp { status: code, ctype: "application/json; charset=utf-8", body: J::obj().set("error", msg).render().into_bytes(), headers: vec![] } }
}

pub fn serve(port: u16, open_browser: bool) -> Result<(), String> {
    serve_opts(ServerOpts { host: "127.0.0.1".into(), port, open: open_browser, token: None, data_dir: "runs".into(), cors: false, log: false, insecure: false })
}

fn is_loopback(host: &str) -> bool { matches!(host, "127.0.0.1" | "localhost" | "::1" | "[::1]") }

pub fn serve_opts(opts: ServerOpts) -> Result<(), String> {
    if !is_loopback(&opts.host) && opts.token.is_none() && !opts.insecure {
        return Err(format!("адрес {} виден из сети: задайте токен через WORLDSIM_TOKEN или --token. Запустить без защиты можно только явно: --insecure", opts.host));
    }
    let listener = TcpListener::bind((opts.host.as_str(), opts.port)).map_err(|e| format!("порт {}: {e}", opts.port))?;
    let url = format!("http://{}:{}", if is_loopback(&opts.host) { "localhost" } else { &opts.host }, opts.port);
    println!("worldsim {}: {url}  (Ctrl+C — остановить){}", env!("CARGO_PKG_VERSION"), if opts.token.is_some() { ", нужен токен" } else { "" });
    if opts.open {
        let cmd = if cfg!(target_os = "macos") { Some("open") } else if cfg!(target_os = "linux") { Some("xdg-open") } else { None };
        if let Some(c) = cmd { let _ = std::process::Command::new(c).arg(&url).spawn(); }
        #[cfg(target_os = "windows")]
        { let _ = std::process::Command::new("cmd").args(["/C", "start", &url]).spawn(); }
    }
    let state = State { sims: Arc::new(Mutex::new(HashMap::new())), next: Arc::new(AtomicU64::new(1)), tick: Arc::new(AtomicU64::new(1)), opts: Arc::new(opts), active: Default::default(), jobs: Default::default() };
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        if state.active.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
            state.active.fetch_sub(1, Ordering::SeqCst);
            let _ = stream.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            continue;
        }
        let st = state.clone();
        std::thread::spawn(move || { let _g = ConnGuard(st.active.clone()); let _ = handle(stream, st); });
    }
    Ok(())
}

/// Строка заголовка ограниченной длины: нельзя занять память бесконечной строкой.
fn read_line_limited(r: &mut BufReader<TcpStream>, out: &mut String) -> std::io::Result<usize> {
    let mut lim = r.by_ref().take(MAX_LINE);
    let mut raw = Vec::new();
    let n = lim.read_until(b'\n', &mut raw)?;
    out.push_str(&String::from_utf8_lossy(&raw));
    Ok(n)
}

/// Сравнение токена за постоянное время: по времени ответа нельзя подбирать символы.
fn token_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() { return false; }
    a.bytes().zip(b.bytes()).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

fn send(stream: &mut TcpStream, st: &State, resp: &Resp, close: bool, head_only: bool) -> std::io::Result<()> {
    let reason = match resp.status { 200 => "OK", 400 => "Bad Request", 401 => "Unauthorized", 403 => "Forbidden", 404 => "Not Found", 405 => "Method Not Allowed", 413 => "Payload Too Large", 429 => "Too Many Requests", 431 => "Request Header Fields Too Large", _ => "Internal Server Error" };
    let mut head = format!("HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: {}\r\n",
        resp.status, reason, resp.ctype, resp.body.len(), if close { "close" } else { "keep-alive" });
    if st.opts.cors { head.push_str("Access-Control-Allow-Origin: *\r\nAccess-Control-Expose-Headers: *\r\n"); }
    if resp.status == 401 { head.push_str("WWW-Authenticate: Bearer\r\n"); }
    for (k, v) in &resp.headers { head.push_str(&format!("{k}: {v}\r\n")); }
    head.push_str("\r\n");
    let mut out = head.into_bytes();
    if !head_only { out.extend_from_slice(&resp.body); }
    stream.write_all(&out)?;
    stream.flush()
}

fn handle(mut stream: TcpStream, st: State) -> std::io::Result<()> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(60)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(60)))?;
    let _ = stream.set_nodelay(true);
    let mut reader = BufReader::new(stream.try_clone()?);
    // keep-alive: много запросов в одном соединении (важно для обучения агентов)
    loop {
        let mut line = String::new();
        if read_line_limited(&mut reader, &mut line)? == 0 { return Ok(()); }
        if line.trim().is_empty() { continue; }
        if !line.ends_with('\n') { return send(&mut stream, &st, &Resp::err(431, "слишком длинная строка запроса"), true, false); }
        let started = std::time::Instant::now();
        let mut content_len = 0usize;
        let mut close = false;
        let (mut auth, mut origin, mut host) = (String::new(), String::new(), String::new());
        let mut headers = 0;
        loop {
            let mut h = String::new();
            if read_line_limited(&mut reader, &mut h)? == 0 || h == "\r\n" || h == "\n" { break; }
            headers += 1;
            if headers > MAX_HEADERS || !h.ends_with('\n') { return send(&mut stream, &st, &Resp::err(431, "слишком большие заголовки"), true, false); }
            let hl = h.to_ascii_lowercase();
            if let Some(v) = hl.strip_prefix("content-length:") { content_len = v.trim().parse().unwrap_or(usize::MAX); }
            if hl.starts_with("connection:") && hl.contains("close") { close = true; }
            if hl.starts_with("authorization:") { auth = h["authorization:".len()..].trim().to_string(); }
            if let Some(v) = hl.strip_prefix("origin:") { origin = v.trim().to_string(); }
            if let Some(v) = hl.strip_prefix("host:") { host = v.trim().to_string(); }
        }
        // тело больше лимита не читаем и не обрезаем: обрезанное описание миссии дало бы другой результат
        if content_len > MAX_BODY {
            // дочитываем в разумных пределах, иначе клиент получит обрыв вместо понятного ответа
            let _ = std::io::copy(&mut reader.by_ref().take((content_len as u64).min(16 << 20)), &mut std::io::sink());
            return send(&mut stream, &st, &Resp::err(413, "тело запроса больше 1 МБ"), true, false);
        }
        let mut body = Vec::new();
        if content_len > 0 { body = vec![0; content_len]; reader.read_exact(&mut body)?; }
        let mut parts = line.split_whitespace();
        let method = parts.next().unwrap_or("").to_string();
        let target = parts.next().unwrap_or("/");
        let (path, query) = target.split_once('?').unwrap_or((target, ""));
        let path = path.to_string();
        let mut q = parse_query(query);
        // POST с телом вида a=1&b=2 (или JSON-массив действия) — для длинных действий
        if method == "POST" && !body.is_empty() {
            let b = String::from_utf8_lossy(&body).to_string();
            let t = b.trim();
            if path == "/api/mission/run" || path == "/api/jobs/mission" { q.insert("spec".into(), t.to_string()); }
            else if t.starts_with('[') { q.insert("action".into(), t.trim_matches(|c| c == '[' || c == ']').to_string()); }
            else { q.extend(parse_query(t)); }
        }
        let resp = if !(method == "GET" || method == "POST" || method == "HEAD") {
            Resp::err(405, "только GET/POST/HEAD")
        } else if let Some(r) = guard(&st, &path, &auth, &origin, &host) {
            r
        } else if method != "POST" && (path == "/api/mission/run" || path == "/api/jobs/mission" || (path.starts_with("/api/jobs/") && path.ends_with("/cancel"))) {
            // запуск и отмена меняют состояние: случайная ссылка или предзагрузка не должны их вызывать
            Resp::err(405, "этот запрос выполняется методом POST")
        } else {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| route(&path, &q, &st))) {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => Resp::err(if e.contains("не найден") { 404 } else if e.starts_with("слишком много") { 429 } else { 400 }, &e),
                Err(_) => Resp::err(500, "внутренняя ошибка симуляции"),
            }
        };
        if st.opts.log {
            // строку запроса не пишем: в параметрах могут быть данные заказчика
            eprintln!("{} {} {} {}b {:.1}ms", method, path, resp.status, resp.body.len(), started.elapsed().as_secs_f64() * 1e3);
        }
        send(&mut stream, &st, &resp, close, method == "HEAD")?;
        if close { return Ok(()); }
    }
}

/// Источник и доступ. None — запрос можно обрабатывать.
fn guard(st: &State, path: &str, auth: &str, origin: &str, host: &str) -> Option<Resp> {
    if path == "/healthz" { return None; }
    if !st.opts.cors {
        // браузер присылает Origin при запросе со страницы: чужому сайту нельзя обращаться к локальному серверу
        if !origin.is_empty() {
            let o = origin.split("://").nth(1).unwrap_or(origin);
            if o != host { return Some(Resp::err(403, "запрос с другого сайта отклонён; разрешить можно ключом --cors")); }
        }
        // без токена отвечаем только на локальные имена: защита от подмены DNS
        if st.opts.token.is_none() && is_loopback(&st.opts.host) {
            let name = if host.starts_with('[') { host.split(']').next().map(|x| format!("{x}]")).unwrap_or_default() } else { host.split(':').next().unwrap_or("").to_string() };
            if !name.is_empty() && !is_loopback(&name) { return Some(Resp::err(403, "неизвестное имя сервера")); }
        }
    }
    if let Some(t) = &st.opts.token {
        if path == "/" || path == "/index.html" { return None; } // страница без данных: токен спрашивает она сама
        let given = auth.strip_prefix("Bearer ").unwrap_or("");
        if !token_eq(given, t) { return Some(Resp::err(401, "нужен токен: Authorization: Bearer <токен>")); }
    }
    None
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

/// Допустимые диапазоны числовых параметров: мусор должен давать понятную ошибку, а не NaN в ответе и не молчаливую подмену.
const PARAM_RANGES: &[(&str, f64, f64)] = &[
    ("id", 0.0, 1e15), ("seed", 0.0, 1e15), ("n", 0.0, 1e9), ("steps", 0.0, 1e6), ("repeat", 1.0, 1e6), ("planet", 0.0, 1e6),
    ("gravity_scale", 1e-15, 100.0), ("distance_au", 1e-6, 1e6), ("solar", 0.0, 100.0), ("tau", 0.0, 100.0), ("obliquity", -180.0, 180.0),
    ("ecc", 0.0, 0.99), ("e", 0.0, 0.99), ("ocean", 0.0, 1.0), ("years", 1.0, 2000.0), ("speed", 0.0, 1e7), ("elev", -90.0, 90.0), ("az", -360.0, 360.0),
    ("lat", -90.0, 90.0), ("alt", 0.0, 1e8), ("radius", 1e-6, 1e3), ("density", 1e-3, 1e6), ("max_time", 1e-3, 1e9), ("days", 0.0, 3650.0), ("dt_hours", 1e-3, 1e4),
    ("dt", 1e-12, 1.0), ("box", 10.0, 2000.0), ("a0", 0.005, 0.5), ("da", 0.0, 0.05), ("m", 1e-6, 1e6), ("a", 1e-4, 1e5), ("ms", 1e-3, 1000.0), ("rs", 1e-3, 1e4), ("ts", 1.0, 1e6),
    ("noise", 0.0, 100.0), ("rotor_scale", 1e-3, 1e4), ("max_steps", 1.0, 1e9), ("frame_skip", 1.0, 100.0), ("gust", 0.0, 100.0), ("lag", 0.0, 10.0), ("fault", 0.0, 1.0),
    ("w", 1.0, 1e5), ("h", 1.0, 1e5), ("fov", 1.0, 179.0),
];

fn check_params(q: &Q) -> Result<(), String> {
    for (k, lo, hi) in PARAM_RANGES {
        let Some(v) = q.get(*k).filter(|v| !v.is_empty()) else { continue };
        let x: f64 = v.parse().map_err(|_| format!("{k}: ожидалось число, получено '{v}'"))?;
        if !x.is_finite() || x < *lo || x > *hi { return Err(format!("{k} = {v} вне диапазона [{lo}, {hi}]")); }
    }
    Ok(())
}

fn route(path: &str, q: &Q, st: &State) -> Result<Resp, String> {
    // у лаборатории и чёрных дыр свои диапазоны: там `a` — спин и ширина барьера, а не полуось орбиты
    if !path.starts_with("/api/lab/") && !path.starts_with("/api/blackhole/") && !path.starts_with("/api/runs") { check_params(q)?; }
    Ok(match path {
        "/" | "/index.html" => Resp { status: 200, ctype: "text/html; charset=utf-8", body: INDEX.as_bytes().to_vec(),
            headers: vec![("Content-Security-Policy".into(), "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'".into())] },
        "/healthz" => Resp::json(J::obj().set("status", "ok")),
        "/api/version" => Resp::json(J::obj().set("name", "worldsim").set("version", env!("CARGO_PKG_VERSION")).set("params", crate::env::PARAMS.len()).set("auth", st.opts.token.is_some())),
        "/api/openapi.json" => Resp { status: 200, ctype: "application/json; charset=utf-8", body: include_bytes!("../docs/openapi.json").to_vec(), headers: vec![] },
        "/api/jobs/mission" => job_start(q, st)?,
        "/api/jobs" => Resp::json(J::Arr(st.jobs.lock().unwrap_or_else(|e| e.into_inner()).iter().rev().map(|j| job_json(j)).collect())),
        "/api/runs" => Resp::json(crate::store::list_runs(&st.opts.data_dir)),
        "/api/runs/compare" => Resp::json(crate::store::compare(&st.opts.data_dir, s(q, "a", ""), s(q, "b", ""))?),
        p if p.starts_with("/api/runs/") => Resp::json(crate::store::load_report(&st.opts.data_dir, &p["/api/runs/".len()..])?),
        p if p.starts_with("/api/jobs/") => job_route(&p["/api/jobs/".len()..], st)?,
        "/data/stars.json" => data_resp("stars_hyg.json"),
        "/data/exoplanets.json" => data_resp("exoplanets_oec.json"),
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
        "/api/env/tasks" => Resp::json(J::Arr(crate::env::TASKS.iter().map(|(n, d)| J::obj().set("name", *n).set("description", *d)).collect())),
        "/api/env/make" => env_make(q, st)?,
        "/api/env/reset" => env_reset(q, st)?,
        "/api/env/step" => env_step(q, st)?,
        "/api/env/state" => env_state(q, st)?,
        "/api/env/terrain" => env_terrain(q, st)?,
        "/api/env/camera" => env_camera(q, st)?,
        "/api/env/list" => env_list(st),
        "/api/mission/example" => match s(q, "name", "lunar_lander") {
            "mars_helicopter" => Resp::json(J::parse(include_str!("../missions/mars_helicopter.json"))?),
            "lunar_rover" => Resp::json(J::parse(include_str!("../missions/lunar_rover.json"))?),
            _ => Resp::json(J::parse(MISSION_EXAMPLE)?),
        },
        "/api/mission/run" => {
            // описание приходит телом POST как JSON; handle() кладёт его в spec
            let spec = q.get("spec").ok_or("нет описания миссии")?;
            let report = crate::mission::run(&crate::mission::MissionSpec::parse(spec)?, std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1))?;
            let md = crate::mission::markdown(&report);
            Resp::json(report.set("markdown", md))
        }
        "/api/env/schema" => Resp::json(param_schema()),
        "/api/blackhole/props" => Resp::json(bh_props(q)?),
        "/api/blackhole/render" => bh_render(q)?,
        p if p.starts_with("/api/lab/") => Resp::json(crate::lab::run(&p["/api/lab/".len()..], q)?),
        "/api/env/fork" => env_fork(q, st)?,
        "/api/env/close" => env_close(q, st)?,
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
        .set("has_surface", p.has_surface).set("year_days", climate::orbital_period(w) / DAY)
        // приливный захват: у таких миров sol_day — условное «очень много», показывать его как сутки нельзя
        .set("tidally_locked", p.solar_day > 100.0 * climate::orbital_period(w));
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
        years: f(q, "years", 30.0) as usize, samples_per_year: 48, ..Default::default()
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
    let is_env = matches!(sim, Sim::Env(_));
    let cap = if is_env { MAX_ENV_SESSIONS } else { MAX_OTHER_SESSIONS };
    let mut m = st.sims.lock().unwrap_or_else(|e| e.into_inner());
    // вытесняем самую давно не использованную сессию того же рода, а не самую старую
    if m.values().filter(|s| s.is_env == is_env).count() >= cap {
        if let Some(k) = m.iter().filter(|(_, s)| s.is_env == is_env).min_by_key(|(_, s)| s.last_used).map(|(k, _)| *k) { m.remove(&k); }
    }
    m.insert(id, Slot { sim: Arc::new(Mutex::new(sim)), last_used: st.tick.fetch_add(1, Ordering::Relaxed), is_env });
    id
}

fn get(st: &State, q: &Q) -> Result<Arc<Mutex<Sim>>, String> {
    let id = f(q, "id", 0.0) as u64;
    let mut m = st.sims.lock().unwrap_or_else(|e| e.into_inner());
    let slot = m.get_mut(&id).ok_or_else(|| "сессия симуляции не найдена — создайте заново".to_string())?;
    slot.last_used = st.tick.fetch_add(1, Ordering::Relaxed);
    Ok(slot.sim.clone())
}

const BODY_COLORS: &[(&str, &str)] = &[("sun", "#ffd27a"), ("mercury", "#9c8f86"), ("venus", "#e8cf8e"), ("earth", "#4f8bd6"),
    ("moon", "#bdbdbd"), ("mars", "#c9562e"), ("jupiter", "#d8b48a")];

fn nbody_new(q: &Q, st: &State) -> Result<Resp, String> {
    let kind = s(q, "kind", "solar");
    let (mut sim, meta) = if kind == "solar" {
        let names: Vec<String> = s(q, "bodies", "sun,mercury,venus,earth,moon,mars,jupiter").split(',').map(|x| x.trim().to_string()).collect();
        let refs: Vec<&str> = names.iter().map(|x| x.as_str()).collect();
        // два одинаковых тела оказались бы в одной точке: сила бесконечна
        if let Some(d) = names.iter().enumerate().find(|(i, n)| names[..*i].contains(n)) { return Err(format!("тело '{}' указано дважды", d.1)); }
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
    let mut g = h.lock().unwrap_or_else(|e| e.into_inner());
    let Sim::NBody { sim, e0 } = &mut *g else { return Err("не N-тел сессия".into()) };
    let total = f(q, "days", 1.0).clamp(0.0, 3650.0) * DAY;
    let dt = (f(q, "dt_hours", 2.0) * 3600.0).clamp(60.0, 30.0 * DAY);
    let n = (total / dt).ceil().max(1.0) as usize;
    for _ in 0..n.min(200_000) { sim.step(total / n as f64); }
    let pos: Vec<f64> = sim.bodies.iter().flat_map(|b| [b.r.x / AU, b.r.y / AU, b.r.z / AU]).collect();
    let err = if e0.abs() > 0.0 { ((sim.invariants().energy - *e0) / *e0).abs() } else { 0.0 };
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
    let mut g = h.lock().unwrap_or_else(|e| e.into_inner());
    let Sim::Galaxy(gal) = &mut *g else { return Err("не галактическая сессия".into()) };
    let steps = (f(q, "steps", 1.0) as usize).min(50);
    let dt = f(q, "dt", 0.002);
    for _ in 0..steps { gal.step(dt); }
    let body = f32_bytes(gal.pos.iter().flat_map(|p| [p.x, p.z, -p.y]));
    Ok(Resp::bin(body, vec![("X-T-Gyr".into(), format!("{:.6e}", gal.t * TIME_GYR))]))
}

fn galaxy_rotation(q: &Q, st: &State) -> Result<Resp, String> {
    let h = get(st, q)?;
    let g = h.lock().unwrap_or_else(|e| e.into_inner());
    let Sim::Galaxy(gal) = &*g else { return Err("не галактическая сессия".into()) };
    let rc = gal.rotation_curve(gal.rotation_range(), 25);
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
    let mut g = h.lock().unwrap_or_else(|e| e.into_inner());
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
    let mut g = h.lock().unwrap_or_else(|e| e.into_inner());
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


// ------------------------------------------------------------------ среда для агентов

use crate::env::{Env, EnvConfig};

fn env_make(q: &Q, st: &State) -> Result<Resp, String> {
    let mut cfg = EnvConfig::new(s(q, "task", "lander"), s(q, "physics", "moon"))?;
    cfg.world.atmosphere = q.get("atmosphere").filter(|v| !v.is_empty() && *v != "auto").cloned();
    cfg.world.surface = q.get("surface").filter(|v| !v.is_empty() && *v != "auto").cloned();
    cfg.world.gravity_scale = f(q, "gravity_scale", 1.0);
    cfg.randomize = s(q, "randomize", "0") == "1";
    cfg.noise = f(q, "noise", 1.0);
    cfg.rotor_scale = f(q, "rotor_scale", 1.0);
    if let Some(m) = q.get("max_steps").and_then(|v| v.parse().ok()) { cfg.max_steps = m; }
    if let Some(k) = q.get("frame_skip").and_then(|v| v.parse().ok()) { cfg.frame_skip = k; }
    cfg.gust = f(q, "gust", 0.0);
    cfg.actuator_lag = f(q, "lag", 0.0);
    cfg.fault_prob = f(q, "fault", 0.0);
    if let Some(p) = q.get("params") { cfg.set_list(p)?; }
    if let Some(w) = q.get("worlds").filter(|v| !v.is_empty()) { cfg.worlds = w.split(',').map(|x| WorldSpec::parse(x.trim())).collect(); }
    let mut env = Env::new(cfg)?;
    let obs = env.reset(f(q, "seed", 1.0) as u64);
    let (lo, hi) = env.act_bounds();
    let meta = J::obj().set("task", s(q, "task", "lander")).set("world", env.world.name())
        .set("obs_names", env.obs_names()).set("act_names", env.act_names()).set("act_low", lo).set("act_high", hi)
        .set("obs_dim", obs.len()).set("act_dim", env.act_names().len()).set("control_dt", env.control_dt()).set("max_steps", env.cfg.max_steps)
        .set("state_names", str_list(&env.state_names())).set("param_names", str_list(crate::env::PARAM_NAMES)).set("params", env.params())
        .set("obs", obs).set("text", env.describe());
    let id = store(st, Sim::Env(Box::new(env)));
    Ok(Resp::json(meta.set("id", id)))
}

fn job_json(j: &Job) -> J {
    let (done, total) = (j.progress.done.load(Ordering::Relaxed), j.progress.total.load(Ordering::Relaxed));
    let st = j.state.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let (status, run, error) = match st {
        JobState::Running => ("running", J::Null, J::Null),
        JobState::Done(id) => ("done", J::from(id), J::Null),
        JobState::Failed(e) => ("failed", J::Null, J::from(e)),
        JobState::Cancelled => ("cancelled", J::Null, J::Null),
    };
    J::obj().set("id", j.id.clone()).set("name", j.name.clone()).set("status", status).set("done", done).set("total", total)
        .set("fraction", if total > 0 { (done as f64 / total as f64).min(1.0) } else { 0.0 }).set("run", run).set("error", error)
}

fn job_start(q: &Q, st: &State) -> Result<Resp, String> {
    let text = q.get("spec").ok_or("нет описания миссии")?.clone();
    let spec = crate::mission::MissionSpec::parse(&text)?;
    let mut jobs = st.jobs.lock().unwrap_or_else(|e| e.into_inner());
    let running = jobs.iter().filter(|j| matches!(*j.state.lock().unwrap_or_else(|e| e.into_inner()), JobState::Running)).count();
    if running >= MAX_ACTIVE_JOBS { return Err(format!("слишком много задач: выполняется {running}, дождитесь завершения")); }
    if jobs.len() >= MAX_KEPT_JOBS { jobs.remove(0); }
    let id = format!("{:x}", st.next.fetch_add(1, Ordering::SeqCst));
    let job = Arc::new(Job { id: id.clone(), name: spec.name.clone(), spec: text, progress: Default::default(), state: Mutex::new(JobState::Running) });
    jobs.push(job.clone());
    drop(jobs);
    let root = st.opts.data_dir.clone();
    std::thread::spawn(move || {
        let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
        // паника внутри расчёта не должна оставить задачу в состоянии «идёт» навсегда
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| crate::mission::run_with(&spec, threads, &job.progress)));
        let next = match result {
            Ok(Ok(r)) => {
                let md = crate::mission::markdown(&r);
                let _ = md;
                match crate::store::save_run(&root, &job.spec, &r.set("spec_sha256", crate::store::sha256(job.spec.as_bytes()))) { Ok(id) => JobState::Done(id), Err(e) => JobState::Failed(e) }
            }
            Ok(Err(_)) if job.progress.cancel.load(Ordering::Relaxed) => JobState::Cancelled,
            Ok(Err(e)) => JobState::Failed(e),
            Err(_) => JobState::Failed("внутренняя ошибка расчёта".into()),
        };
        *job.state.lock().unwrap_or_else(|e| e.into_inner()) = next;
    });
    Ok(Resp::json(J::obj().set("id", id)))
}

fn job_route(rest: &str, st: &State) -> Result<Resp, String> {
    let (id, cancel) = match rest.strip_suffix("/cancel") { Some(x) => (x, true), None => (rest, false) };
    let job = st.jobs.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|j| j.id == id).cloned().ok_or_else(|| format!("задача {id} не найдена"))?;
    if cancel { job.progress.cancel.store(true, Ordering::Relaxed); }
    Ok(Resp::json(job_json(&job)))
}

fn bh_props(q: &Q) -> Result<J, String> {
    use crate::blackhole as b;
    let m = crate::lab::num(q, "m_sun", 10.0, 1e-20, 1e12)? * crate::quantum::M_SUN;
    let a = crate::lab::num(q, "a", 0.0, 0.0, 0.9999)?.max(1e-9);
    Ok(J::obj().set("rs", b::schwarzschild_radius(m)).set("horizon", b::kerr_horizon(a)).set("isco", b::kerr_isco(a)).set("photon", b::kerr_photon_orbit(a))
        .set("shadow", 27f64.sqrt()).set("efficiency", b::radiative_efficiency(a)).set("t_hawking", b::hawking_temperature(m)).set("evaporation_yr", b::evaporation_years(m)).set("tide", b::horizon_tide(m)))
}

fn bh_render(q: &Q) -> Result<Resp, String> {
    use crate::lab::num;
    let (w, h) = (num(q, "w", 480.0, 16.0, 1600.0)? as usize, num(q, "h", 300.0, 16.0, 1200.0)? as usize);
    let o = crate::blackhole::RenderOpts { w, h, incl_deg: num(q, "incl", 78.0, 0.0, 90.0)?, r_in: 6.0, r_out: num(q, "r_out", 18.0, 6.5, 500.0)?, distance: num(q, "dist", 45.0, 8.0, 500.0)?, fov_deg: 34.0, doppler: s(q, "doppler", "1") == "1" };
    let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
    Ok(Resp::bin(crate::blackhole::render(&o, threads), vec![("X-Width".into(), w.to_string()), ("X-Height".into(), h.to_string())]))
}

/// Описание всех параметров для интерфейса и клиентов: форму можно построить без знания о конкретных полях.
fn param_schema() -> J {
    J::Arr(crate::env::PARAMS.iter().map(|d| {
        let tasks: Vec<String> = [(1u8, "lander"), (2, "drone"), (4, "rover")].iter().filter(|(b, _)| d.tasks & b != 0).map(|(_, n)| n.to_string()).collect();
        J::obj().set("name", d.name).set("group", d.group).set("label", d.label).set("unit", d.unit).set("min", d.min).set("max", d.max)
            .set("tasks", tasks).set("defaults", J::obj().set("lander", d.defaults[0]).set("drone", d.defaults[1]).set("rover", d.defaults[2])).set("help", d.help)
    }).collect())
}

fn str_list(v: &[&str]) -> J { J::Arr(v.iter().map(|x| J::from(*x)).collect()) }

fn with_env<T>(q: &Q, st: &State, f: impl FnOnce(&mut Env) -> Result<T, String>) -> Result<T, String> {
    let h = get(st, q)?;
    let mut g = h.lock().unwrap_or_else(|e| e.into_inner());
    match &mut *g { Sim::Env(e) => f(e), _ => Err("не сессия среды".into()) }
}

fn info_json(e: &Env) -> J {
    J::obj().set("success", e.info.success).set("crashed", e.info.crashed).set("reason", e.info.reason.clone())
        .set("t", e.t).set("steps", e.steps).set("episode_return", e.ep_return).set("fuel_used", e.info.fuel_used)
        .set("energy_used", e.info.energy_used).set("resource_frac", e.battery_frac())
}

fn env_reset(q: &Q, st: &State) -> Result<Resp, String> {
    let seed = f(q, "seed", 1.0) as u64;
    with_env(q, st, |e| {
        let obs = e.reset(seed);
        Ok(Resp::json(J::obj().set("obs", obs).set("info", info_json(e)).set("text", e.describe())))
    })
}

fn env_step(q: &Q, st: &State) -> Result<Resp, String> {
    let want_text = s(q, "text", "0") == "1";
    let want_traj = s(q, "traj", "0") == "1";
    let mode = s(q, "mode", "").to_string();
    let action: Vec<f64> = q.get("action").map(|a| a.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    let n = (f(q, "repeat", 1.0) as usize).clamp(1, 100);
    with_env(q, st, |e| {
        let mut total = 0.0;
        let mut res = (e.last_obs.clone(), 0.0, e.done, false);
        let mut traj: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::new();
        for _ in 0..n {
            let a = match mode.as_str() {
                "auto" => e.autopilot(),
                "random" => { let (lo, hi) = e.act_bounds(); let mut r = crate::rng::Rng::new(e.steps as u64 * 7919 + e.seed); lo.iter().zip(&hi).map(|(l, h)| r.uniform(*l, *h)).collect() }
                _ => action.clone(),
            };
            if a.len() != e.act_names().len() { return Err(format!("действие должно иметь {} чисел: {:?}", e.act_names().len(), e.act_names())); }
            res = e.step(&a);
            total += res.1;
            if want_traj { traj.push((a, res.0.clone(), res.1)); }
            if res.2 || res.3 { break; }
        }
        let mut j = J::obj().set("obs", res.0).set("reward", total).set("terminated", res.2).set("truncated", res.3).set("info", info_json(e));
        if want_text { j = j.set("text", e.describe()); }
        // при repeat > 1 без этого промежуточные переходы теряются
        if want_traj {
            j = j.set("traj_actions", J::Arr(traj.iter().map(|t| J::from(t.0.clone())).collect()))
                .set("traj_obs", J::Arr(traj.iter().map(|t| J::from(t.1.clone())).collect()))
                .set("traj_rewards", traj.iter().map(|t| t.2).collect::<Vec<f64>>());
        }
        Ok(Resp::json(j))
    })
}

fn v3(v: crate::vec3::Vec3) -> J { J::from(vec![v.x, v.y, v.z]) }

fn env_state(q: &Q, st: &State) -> Result<Resp, String> {
    with_env(q, st, |e| {
        let (p, quat) = e.pose();
        let lidar: Vec<J> = e.last_lidar.iter().map(|(a, b)| J::from(vec![a.x, a.y, a.z, b.x, b.y, b.z])).collect();
        Ok(Resp::json(J::obj().set("task", format!("{:?}", e.cfg.task).to_lowercase()).set("pos", v3(p)).set("quat", vec![quat.w, quat.x, quat.y, quat.z])
            .set("vel", v3(e.velocity())).set("goal", v3(e.goal)).set("wind", v3(e.wind())).set("lidar", J::Arr(lidar))
            .set("action", e.last_action.clone()).set("obs", e.last_obs.clone()).set("obs_names", e.obs_names()).set("done", e.done)
            .set("info", info_json(e)).set("text", e.describe()).set("speed", e.rover_speed()).set("world", e.world.name())
            .set("state_names", str_list(&e.state_names())).set("state", e.state()).set("param_names", str_list(crate::env::PARAM_NAMES)).set("params", e.params())))
    })
}

fn env_terrain(q: &Q, st: &State) -> Result<Resp, String> {
    let n = (f(q, "n", 129.0) as usize).clamp(17, 257);
    with_env(q, st, |e| {
        let t = &e.terrain;
        let mut h = Vec::with_capacity(n * n);
        for j in 0..n { for i in 0..n {
            let x = -t.size / 2.0 + t.size * i as f64 / (n - 1) as f64;
            let y = -t.size / 2.0 + t.size * j as f64 / (n - 1) as f64;
            h.push(t.height(x, y));
        }}
        Ok(Resp::json(J::obj().set("n", n).set("size", t.size).set("h", h).set("surface", e.world.surface.name.clone())
            .set("oxides", J::Arr(presets::normalized(&e.world.surface.oxides_wt).into_iter().map(|(k, v)| J::Arr(vec![J::from(k), J::from(v)])).collect()))
            .set("has_air", e.world.atmosphere.is_some())))
    })
}

fn env_camera(q: &Q, st: &State) -> Result<Resp, String> {
    let (w, h) = ((f(q, "w", 48.0) as usize).clamp(4, 256), (f(q, "h", 32.0) as usize).clamp(4, 256));
    let fov = f(q, "fov", 70.0);
    if s(q, "mode", "depth") == "rgb" {
        return with_env(q, st, |e| Ok(Resp::bin(e.rgb_camera(w, h, fov), vec![("X-Width".into(), w.to_string()), ("X-Height".into(), h.to_string())])));
    }
    with_env(q, st, |e| Ok(Resp::json(J::obj().set("w", w).set("h", h).set("depth", e.depth_camera(w, h, fov)))))
}

/// Копия среды: то же состояние и тот же ГПСЧ. Можно проиграть разные действия из одной точки и сравнить с «воображением» модели.
fn env_fork(q: &Q, st: &State) -> Result<Resp, String> {
    let copy = with_env(q, st, |e| Ok(Box::new(e.clone())))?;
    let id = store(st, Sim::Env(copy));
    Ok(Resp::json(J::obj().set("id", id)))
}

fn env_close(q: &Q, st: &State) -> Result<Resp, String> {
    let id = f(q, "id", 0.0) as u64;
    let closed = st.sims.lock().unwrap_or_else(|e| e.into_inner()).remove(&id).is_some();
    Ok(Resp::json(J::obj().set("closed", closed)))
}

fn env_list(st: &State) -> Resp {
    let m = st.sims.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::new();
    for (id, s) in m.iter() {
        if let Ok(g) = s.sim.try_lock() {
            if let Sim::Env(e) = &*g {
                out.push(J::obj().set("id", *id).set("task", format!("{:?}", e.cfg.task).to_lowercase()).set("world", e.world.name()).set("steps", e.steps).set("episode_return", e.ep_return));
            }
        }
    }
    Resp::json(J::Arr(out))
}
