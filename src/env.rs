//! Среда для обучения ИИ-агентов и роботов (в стиле Gymnasium).
//!
//! Агент получает **только данные сенсоров** (с шумом, как у настоящего робота) и выдаёт
//! непрерывные команды приводам. Физика — та же, что во всём worldsim: тяготение мира,
//! плотность атмосферы из гидростатики, ветер, сопротивление, закон Архимеда, уравнение
//! Циолковского для двигателя, импульсная теория винта для дрона, сцепление колёс по Кулону.
//!
//! Задачи:
//!   * `lander` — мягкая посадка на площадку (тяга + ориентация, расход топлива по Isp)
//!   * `drone`  — квадрокоптер долетает до точки; тяга ∝ ρ, мощность ∝ T^{3/2}/√(2ρA)
//!   * `rover`  — ровер едет к цели по рельефу; солнечная батарея, склоны, лидар
//!
//! Один и тот же агент можно учить на Земле, Луне, Марсе и смешанных мирах (domain randomization).

use crate::rng::Rng;
use crate::vec3::Vec3;
use crate::world::{World, WorldSpec};
use std::f64::consts::PI;
use std::sync::Arc;

pub const G0: f64 = 9.80665;
const LEG_K: f64 = 6.0e4;   // жёсткость ноги, Н/м
const LEG_C: f64 = 6.0e3;   // демпфирование ноги, Н·с/м
const LEG_CT: f64 = 2.0e3;  // вязкое приближение трения покоя, Н·с/м
pub const ROVER_TIP_LAT: f64 = 1.0;   // колея/2h = 1.2 м / (2·0.6 м)
pub const ROVER_TIP_LONG: f64 = 1.33; // база/2h = 1.6 м / (2·0.6 м)
const ROVER_V_MAX: f64 = 2.5;         // скорость холостого хода моторов, м/с

// ================================================================ рельеф

#[derive(Clone, Debug)]
pub struct Terrain {
    pub n: usize,     // узлов на сторону
    pub size: f64,    // м
    pub h: Vec<f64>,  // n×n, строка = y
    pub friction: f64,
}

fn hash2(x: i64, y: i64, seed: u64) -> f64 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B1_85EB_CA87) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 31;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29;
    (h >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn value_noise(x: f64, y: f64, seed: u64) -> f64 {
    let (xi, yi) = (x.floor(), y.floor());
    let (fx, fy) = (x - xi, y - yi);
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let (ux, uy) = (s(fx), s(fy));
    let (xi, yi) = (xi as i64, yi as i64);
    let a = hash2(xi, yi, seed);
    let b = hash2(xi + 1, yi, seed);
    let c = hash2(xi, yi + 1, seed);
    let d = hash2(xi + 1, yi + 1, seed);
    a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy
}

impl Terrain {
    /// Процедурный рельеф под тип поверхности мира: кратеры на безвоздушных телах, холмы на Земле.
    pub fn generate(w: &World, seed: u64, size: f64, n: usize, pad_radius: f64) -> Terrain {
        let surf = w.surface.name.as_str();
        let airless = w.atmosphere.is_none();
        let (amp, craters, friction) = match surf {
            "moon" | "moon_mare" | "mercury" => (18.0, 30, 0.6),
            "mars" => (25.0, 14, 0.55),
            "venus" => (12.0, 4, 0.6),
            _ => (15.0, if airless { 20 } else { 0 }, 0.8),
        };
        let mut h = vec![0.0; n * n];
        let cell = size / (n - 1) as f64;
        for j in 0..n {
            for i in 0..n {
                let (x, y) = (i as f64 * cell - size / 2.0, j as f64 * cell - size / 2.0);
                let mut s = 0.0;
                let mut a = 1.0;
                let mut f = 1.0 / 180.0;
                for o in 0..6 {
                    s += a * value_noise(x * f, y * f, seed.wrapping_add(o));
                    a *= 0.5;
                    f *= 2.07;
                }
                h[j * n + i] = amp * s;
            }
        }
        let mut rng = Rng::new(seed ^ 0xC0FFEE);
        for _ in 0..craters {
            let (cx, cy) = (rng.uniform(-size / 2.0, size / 2.0), rng.uniform(-size / 2.0, size / 2.0));
            let r = rng.log_uniform(6.0, size / 8.0);
            let depth = 0.2 * r;
            for j in 0..n {
                for i in 0..n {
                    let (x, y) = (i as f64 * cell - size / 2.0, j as f64 * cell - size / 2.0);
                    let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() / r;
                    if d < 2.0 {
                        let bowl = if d < 1.0 { -depth * (1.0 - d * d) } else { 0.0 };
                        let rim = 0.25 * depth * (-((d - 1.0) / 0.25).powi(2)).exp();
                        h[j * n + i] += bowl + rim;
                    }
                }
            }
        }
        // посадочная площадка / старт: ровный круг в центре
        if pad_radius > 0.0 {
            let mut t = Terrain { n, size, h, friction };
            let h0 = t.height(0.0, 0.0);
            for j in 0..n {
                for i in 0..n {
                    let (x, y) = (i as f64 * cell - size / 2.0, j as f64 * cell - size / 2.0);
                    let d = (x * x + y * y).sqrt();
                    let w8 = ((d - pad_radius) / (pad_radius * 0.6)).clamp(0.0, 1.0);
                    let w8 = w8 * w8 * (3.0 - 2.0 * w8);
                    t.h[j * n + i] = h0 * (1.0 - w8) + t.h[j * n + i] * w8;
                }
            }
            return t;
        }
        Terrain { n, size, h, friction }
    }

    /// Наклонная плоскость (для проверок): высота = grade · x.
    pub fn plane(grade: f64, friction: f64) -> Terrain {
        let n = 33;
        let size = 2000.0;
        let cell = size / (n - 1) as f64;
        let h = (0..n * n).map(|k| grade * ((k % n) as f64 * cell - size / 2.0)).collect();
        Terrain { n, size, h, friction }
    }

    pub fn height(&self, x: f64, y: f64) -> f64 {
        let cell = self.size / (self.n - 1) as f64;
        let fx = ((x + self.size / 2.0) / cell).clamp(0.0, (self.n - 1) as f64 - 1e-9);
        let fy = ((y + self.size / 2.0) / cell).clamp(0.0, (self.n - 1) as f64 - 1e-9);
        let (i, j) = (fx.floor() as usize, fy.floor() as usize);
        let (u, v) = (fx - i as f64, fy - j as f64);
        let at = |i: usize, j: usize| self.h[j * self.n + i];
        at(i, j) * (1.0 - u) * (1.0 - v) + at(i + 1, j) * u * (1.0 - v) + at(i, j + 1) * (1.0 - u) * v + at(i + 1, j + 1) * u * v
    }

    pub fn gradient(&self, x: f64, y: f64) -> (f64, f64) {
        let e = 0.5;
        ((self.height(x + e, y) - self.height(x - e, y)) / (2.0 * e), (self.height(x, y + e) - self.height(x, y - e)) / (2.0 * e))
    }

    /// Пересечение луча с рельефом: расстояние до поверхности или max_range.
    pub fn raycast(&self, o: Vec3, d: Vec3, max_range: f64) -> f64 {
        let d = d.unit();
        let mut t = 0.0;
        let mut prev = 0.0;
        let step = (self.size / (self.n - 1) as f64 * 0.5).max(0.25);
        while t < max_range {
            let p = o + d * t;
            if p.z <= self.height(p.x, p.y) {
                let (mut lo, mut hi) = (prev, t);
                for _ in 0..30 {
                    let m = 0.5 * (lo + hi);
                    let q = o + d * m;
                    if q.z <= self.height(q.x, q.y) { hi = m } else { lo = m }
                }
                return hi;
            }
            prev = t;
            t += step;
        }
        max_range
    }
}

// ================================================================ ориентация

#[derive(Clone, Copy, Debug)]
pub struct Quat { pub w: f64, pub x: f64, pub y: f64, pub z: f64 }

impl Quat {
    pub const ID: Quat = Quat { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };
    pub fn from_axis_angle(a: Vec3, ang: f64) -> Quat {
        let a = a.unit();
        let s = (ang / 2.0).sin();
        Quat { w: (ang / 2.0).cos(), x: a.x * s, y: a.y * s, z: a.z * s }
    }
    pub fn mul(self, o: Quat) -> Quat {
        Quat {
            w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        }
    }
    pub fn conj(self) -> Quat { Quat { w: self.w, x: -self.x, y: -self.y, z: -self.z } }
    /// тело → мир
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let p = self.mul(Quat { w: 0.0, x: v.x, y: v.y, z: v.z }).mul(self.conj());
        Vec3::new(p.x, p.y, p.z)
    }
    pub fn inv_rotate(self, v: Vec3) -> Vec3 { self.conj().rotate(v) }
    pub fn normalize(self) -> Quat {
        let n = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        Quat { w: self.w / n, x: self.x / n, y: self.y / n, z: self.z / n }
    }
    /// поворот на угловую скорость ω (в осях тела) за dt
    pub fn integrate(self, w_body: Vec3, dt: f64) -> Quat {
        let ang = w_body.norm() * dt;
        if ang < 1e-12 { return self; }
        self.mul(Quat::from_axis_angle(w_body, ang)).normalize()
    }
}

// ================================================================ описание задач

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Task { Lander, Drone, Rover }

pub const TASKS: &[(&str, &str)] = &[
    ("lander", "Посадочный модуль: мягко сесть на площадку. Действия: тяга 0…1, моменты ориентации по 3 осям. Топливо тратится по уравнению Циолковского."),
    ("drone", "Квадрокоптер: долететь до точки и зависнуть. Действия: 4 винта 0…1. Тяга ∝ плотности воздуха — на Марсе нужны огромные винты, на Луне полёт невозможен."),
    ("rover", "Ровер: доехать до цели по рельефу. Действия: моменты левых и правых колёс −1…1. Сцепление по Кулону, склоны, солнечная батарея."),
];

#[derive(Clone, Debug)]
pub struct EnvConfig {
    pub task: Task,
    pub world: WorldSpec,
    pub randomize: bool,   // рандомизация параметров (масса, двигатель, ветер, мир)
    pub noise: f64,        // множитель шума сенсоров (0 — идеальные сенсоры)
    pub frame_skip: usize, // шагов физики на одно действие
    pub max_steps: usize,
    pub rotor_scale: f64,  // для дрона: множитель площади винтов (Ingenuity ≈ 60)
    pub worlds: Vec<WorldSpec>, // если не пусто — мир выбирается заново в каждом эпизоде
    pub gust: f64,         // СКО порывов ветра, м/с (процесс Орнштейна–Уленбека, τ = 2 с)
    pub actuator_lag: f64, // постоянная времени приводов, с (0 — мгновенно)
    pub fault_prob: f64,   // вероятность, что в эпизоде один привод ослаблен (скрытый параметр)
    pub vehicle: Vec<(String, f64)>, // переопределение параметров аппарата, см. VEHICLE_PARAMS
}

/// Что можно задать в описании аппарата.
pub const VEHICLE_PARAMS: &[&str] = &["dry_mass", "fuel", "isp", "t_max", "torque_max", "drag_cd", "drag_area", "rover_mass", "motor_force", "panel_area", "battery_kj"];

impl EnvConfig {
    pub fn new(task: &str, physics: &str) -> Result<EnvConfig, String> {
        let task = match task {
            "lander" => Task::Lander,
            "drone" => Task::Drone,
            "rover" => Task::Rover,
            _ => return Err(format!("неизвестная задача '{task}': lander, drone, rover")),
        };
        Ok(EnvConfig { task, world: WorldSpec::preset(physics), randomize: false, noise: 1.0, frame_skip: 4,
            max_steps: match task { Task::Lander => 5000, Task::Drone => 2500, Task::Rover => 4000 }, rotor_scale: 1.0,
            worlds: vec![], gust: 0.0, actuator_lag: 0.0, fault_prob: 0.0, vehicle: vec![] })
    }
}

#[derive(Clone, Debug)]
struct Body {
    pos: Vec3, vel: Vec3, q: Quat, w: Vec3,
    mass: f64, inertia: Vec3,
}

#[derive(Clone, Debug)]
struct Params {
    dry_mass: f64, fuel: f64, isp: f64, t_max: f64, torque_max: f64, cd: f64, area: f64, volume: f64, leg: f64,
    // дрон
    arm: f64, rotor_area: f64, k_yaw: f64, battery_j: f64,
    // ровер
    track: f64, crr: f64, motor_force: f64, panel_area: f64, rover_mass: f64,
}

#[derive(Clone, Debug, Default)]
pub struct StepInfo {
    pub success: bool,
    pub crashed: bool,
    pub reason: String,
    pub fuel_used: f64,
    pub energy_used: f64,
    pub touch_speed: f64,  // скорость в момент первого касания, м/с
    pub touch_tilt: f64,   // наклон в момент первого касания, °
    pub touch_offset: f64, // промах от площадки или цели, м
}

#[derive(Clone)]
pub struct Env {
    pub cfg: EnvConfig,
    pub world: World,
    pub terrain: Arc<Terrain>,
    pub t: f64,
    pub steps: usize,
    pub dt: f64,
    pub goal: Vec3,
    pub wind: Vec3,
    pub done: bool,
    pub ep_return: f64,
    pub last_action: Vec<f64>,
    pub last_obs: Vec<f64>,
    pub last_lidar: Vec<(Vec3, Vec3)>, // (начало, точка попадания) для визуализации
    pub info: StepInfo,
    pub seed: u64,
    body: Body,
    p: Params,
    rng: Rng,
    battery: f64,
    heading: f64, // ровер
    speed: f64,
    yaw_rate: f64,
    sun_elev: f64,
    prev_shaping: f64,
    base_world: World,
    hover_time: f64,
    pub world_index: usize,
    act_state: [f64; 4],  // положение приводов (с учётом задержки)
    act_gain: [f64; 4],   // скрытая деградация приводов
    fault: Option<(usize, f64)>,
    gust_v: Vec3,
    dyn_rng: Rng,         // отдельный поток для возмущений: шум сенсоров на них не влияет
    in_contact: bool,
    settle_t: f64,
    ground_t: f64,
    touch_msg: String,
    touch_bonus: f64,
    touched: bool,
}

/// Имена скрытых параметров эпизода (см. `Env::params`): то, что модель мира должна вывести сама.
pub const PARAM_NAMES: &[&str] = &["g", "rho0", "p0_kpa", "t0_k", "friction", "dry_mass", "fuel_mass", "isp", "t_max", "torque_max", "rotor_area", "rover_mass",
    "wind_x", "wind_y", "gust", "actuator_lag", "fault_actuator", "fault_gain", "world_index", "goal_x", "goal_y", "goal_z", "control_dt"];

impl Env {
    pub fn new(cfg: EnvConfig) -> Result<Env, String> {
        let world = World::build(&cfg.world)?;
        let terrain = Arc::new(Terrain::plane(0.0, 0.8));
        let mut e = Env {
            world: world.clone(), base_world: world, terrain, t: 0.0, steps: 0, dt: 0.005, goal: Vec3::ZERO, wind: Vec3::ZERO,
            done: true, ep_return: 0.0, last_action: vec![], last_obs: vec![], last_lidar: vec![], info: StepInfo::default(), seed: 0,
            body: Body { pos: Vec3::ZERO, vel: Vec3::ZERO, q: Quat::ID, w: Vec3::ZERO, mass: 1.0, inertia: Vec3::new(1.0, 1.0, 1.0) },
            p: Params { dry_mass: 0.0, fuel: 0.0, isp: 0.0, t_max: 0.0, torque_max: 0.0, cd: 0.0, area: 0.0, volume: 0.0, leg: 0.0,
                arm: 0.0, rotor_area: 0.0, k_yaw: 0.0, battery_j: 0.0, track: 0.0, crr: 0.0, motor_force: 0.0, panel_area: 0.0, rover_mass: 0.0 },
            rng: Rng::new(0), battery: 0.0, heading: 0.0, speed: 0.0, yaw_rate: 0.0, sun_elev: 0.5, prev_shaping: 0.0, hover_time: 0.0, cfg,
            world_index: 0, act_state: [0.0; 4], act_gain: [1.0; 4], fault: None, gust_v: Vec3::ZERO, dyn_rng: Rng::new(0),
            in_contact: false, settle_t: 0.0, ground_t: 0.0, touch_msg: String::new(), touch_bonus: 0.0, touched: false,
        };
        for spec in &e.cfg.worlds { World::build(spec)?; }
        e.reset(1);
        Ok(e)
    }

    pub fn obs_names(&self) -> Vec<String> {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        match self.cfg.task {
            Task::Lander => {
                let mut v = s(&["gps_dx", "gps_dy", "gps_dz", "vel_x", "vel_y", "vel_z", "up_bx", "up_by", "up_bz", "gyro_x", "gyro_y", "gyro_z",
                    "radar_alt", "accel_bx", "accel_by", "accel_bz", "fuel_frac", "pressure_kpa"]);
                for k in 0..8 { v.push(format!("lidar_{k}")); }
                v.push("radar_hit".into());
                for k in 0..8 { v.push(format!("lidar_hit_{k}")); }
                v
            }
            Task::Drone => s(&["goal_dx", "goal_dy", "goal_dz", "vel_x", "vel_y", "vel_z", "up_bx", "up_by", "up_bz", "gyro_x", "gyro_y", "gyro_z",
                "radar_alt", "accel_bx", "accel_by", "accel_bz", "battery_frac", "pressure_kpa", "air_density", "radar_hit"]),
            Task::Rover => {
                let mut v = s(&["goal_dx", "goal_dy", "head_cos", "head_sin", "speed", "yaw_rate", "pitch", "roll", "battery_frac", "sun_elev", "temp_k", "pressure_kpa"]);
                for k in 0..16 { v.push(format!("lidar_{k}")); }
                for k in 0..16 { v.push(format!("lidar_hit_{k}")); }
                v
            }
        }
    }

    pub fn act_names(&self) -> Vec<String> {
        match self.cfg.task {
            Task::Lander => vec!["throttle".into(), "torque_x".into(), "torque_y".into(), "torque_z".into()],
            Task::Drone => (1..=4).map(|i| format!("rotor_{i}")).collect(),
            Task::Rover => vec!["left".into(), "right".into()],
        }
    }

    pub fn act_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        match self.cfg.task {
            Task::Lander => (vec![0.0, -1.0, -1.0, -1.0], vec![1.0; 4]),
            Task::Drone => (vec![0.0; 4], vec![1.0; 4]),
            Task::Rover => (vec![-1.0; 2], vec![1.0; 2]),
        }
    }

    pub fn control_dt(&self) -> f64 { self.dt * self.cfg.frame_skip as f64 }

    fn g(&self) -> f64 { self.world.g_at(self.body.pos.z.max(0.0)) }
    fn air(&self, z: f64) -> (f64, f64, f64) {
        match &self.world.atmosphere {
            Some(a) => { let s = a.at(z.max(0.0)); (s.density, s.pressure, s.temperature) }
            None => (0.0, 0.0, self.world.mean_surface_temperature()),
        }
    }

    pub fn reset(&mut self, seed: u64) -> Vec<f64> {
        self.seed = seed;
        self.rng = Rng::new(seed);
        // рандомизация мира: множитель g ±20%, расстояние до звезды
        self.world = self.base_world.clone();
        self.world_index = 0;
        let mut spec = self.cfg.world.clone();
        if !self.cfg.worlds.is_empty() {
            self.world_index = (Rng::new(seed ^ 0x11C3).next_u64() % self.cfg.worlds.len() as u64) as usize;
            spec = self.cfg.worlds[self.world_index].clone();
            if let Ok(w) = World::build(&spec) { self.world = w; }
        }
        if self.cfg.randomize {
            spec.gravity_scale *= self.rng.uniform(0.85, 1.15);
            if let Ok(w) = World::build(&spec) { self.world = w; }
        }
        self.t = 0.0;
        self.steps = 0;
        self.done = false;
        self.ep_return = 0.0;
        self.info = StepInfo::default();
        self.dt = if self.cfg.task == Task::Rover { 0.025 } else { 0.005 };
        self.hover_time = 0.0;
        self.in_contact = false;
        self.touched = false;
        self.settle_t = 0.0;
        self.ground_t = 0.0;
        self.touch_msg.clear();
        self.touch_bonus = 0.0;
        self.act_state = [0.0; 4];
        self.act_gain = [1.0; 4];
        self.fault = None;
        self.gust_v = Vec3::ZERO;
        self.dyn_rng = Rng::new(seed ^ 0xD15C);
        let pad = match self.cfg.task { Task::Lander => 25.0, Task::Rover => 8.0, Task::Drone => 15.0 };
        let size = match self.cfg.task { Task::Rover => 600.0, _ => 1200.0 };
        self.terrain = Arc::new(Terrain::generate(&self.world, seed, size, 161, pad));
        let r = |rng: &mut Rng, s: f64| if s > 0.0 { rng.uniform(1.0 - s, 1.0 + s) } else { 1.0 };
        let rs = if self.cfg.randomize { 1.0 } else { 0.0 };
        self.p = Params {
            dry_mass: 1000.0 * r(&mut self.rng, 0.1 * rs), fuel: 450.0, isp: 300.0 * r(&mut self.rng, 0.05 * rs), t_max: 22_000.0, torque_max: 3000.0,
            cd: 1.0, area: 4.0, volume: 6.0, leg: 1.6,
            arm: 0.25, rotor_area: 4.0 * PI * 0.12f64.powi(2) * self.cfg.rotor_scale, k_yaw: 0.02, battery_j: 1.5e5 * self.cfg.rotor_scale.max(1.0),
            track: 1.2, crr: 0.05, motor_force: 900.0, panel_area: 1.5, rover_mass: 180.0 * r(&mut self.rng, 0.1 * rs),
        };
        self.apply_vehicle();
        let (rho, _, _) = self.air(0.0);
        self.wind = if rho > 0.0 && self.cfg.randomize { Vec3::new(self.rng.normal(), self.rng.normal(), 0.0) * 4.0 } else { Vec3::ZERO };
        let h0 = self.terrain.height(0.0, 0.0);
        match self.cfg.task {
            Task::Lander => {
                let mut rng = Rng::new(seed ^ 77);
                let alt = rng.uniform(300.0, 700.0);
                let off = Vec3::new(rng.uniform(-200.0, 200.0), rng.uniform(-200.0, 200.0), 0.0);
                let tilt = Quat::from_axis_angle(Vec3::new(rng.normal(), rng.normal(), 0.0), rng.uniform(0.0, 0.15));
                let m = self.p.dry_mass + self.p.fuel;
                self.body = Body { pos: Vec3::new(off.x, off.y, h0 + alt), vel: Vec3::new(rng.uniform(-15.0, 15.0), rng.uniform(-15.0, 15.0), rng.uniform(-35.0, -10.0)),
                    q: tilt, w: Vec3::new(rng.normal(), rng.normal(), rng.normal()) * 0.02, mass: m,
                    inertia: Vec3::new(m * (3.0 * 1.5f64.powi(2) + 3.0f64.powi(2)) / 12.0, m * (3.0 * 1.5f64.powi(2) + 3.0f64.powi(2)) / 12.0, m * 1.5f64.powi(2) / 2.0) };
                self.goal = Vec3::new(0.0, 0.0, h0);
            }
            Task::Drone => {
                let mut rng = Rng::new(seed ^ 99);
                let m = 1.5;
                let ang = rng.uniform(0.0, 2.0 * PI);
                let d = rng.uniform(20.0, 60.0);
                let (gx, gy) = (d * ang.cos(), d * ang.sin());
                self.goal = Vec3::new(gx, gy, self.terrain.height(gx, gy) + rng.uniform(5.0, 20.0));
                // старт в воздухе: иначе без автопилота эпизод кончается за доли секунды и данных нет
                self.body = Body { pos: Vec3::new(0.0, 0.0, h0 + rng.uniform(8.0, 30.0)), vel: Vec3::ZERO, q: Quat::ID, w: Vec3::ZERO, mass: m, inertia: Vec3::new(0.02, 0.02, 0.04) };
                self.battery = self.p.battery_j;
            }
            Task::Rover => {
                let mut rng = Rng::new(seed ^ 55);
                let ang = rng.uniform(0.0, 2.0 * PI);
                let d = rng.uniform(80.0, 220.0);
                let (gx, gy) = (d * ang.cos(), d * ang.sin());
                self.goal = Vec3::new(gx, gy, self.terrain.height(gx, gy));
                self.heading = rng.uniform(-PI, PI);
                self.speed = 0.0;
                self.yaw_rate = 0.0;
                self.sun_elev = rng.uniform(0.15, 1.2);
                self.battery = 0.6 * 3.6e6; // 0.6 кВт·ч
                self.body = Body { pos: Vec3::new(0.0, 0.0, h0), vel: Vec3::ZERO, q: Quat::ID, w: Vec3::ZERO, mass: self.p.rover_mass, inertia: Vec3::new(1.0, 1.0, 40.0) };
            }
        }
        let n_act = self.act_names().len();
        if self.cfg.fault_prob > 0.0 && self.dyn_rng.f64() < self.cfg.fault_prob {
            let i = (self.dyn_rng.next_u64() % n_act as u64) as usize;
            let g = self.dyn_rng.uniform(0.3, 0.8);
            self.act_gain[i] = g;
            self.fault = Some((i, g));
        }
        self.prev_shaping = self.shaping();
        self.last_action = vec![0.0; n_act];
        self.last_obs = self.observe();
        self.last_obs.clone()
    }

    // ------------------------------------------------------------ сенсоры

    fn n(&mut self, sigma: f64) -> f64 { sigma * self.cfg.noise * self.rng.normal() }

    fn observe(&mut self) -> Vec<f64> {
        let b = self.body.clone();
        let (rho, p, temp) = self.air(b.pos.z);
        self.last_lidar.clear();
        match self.cfg.task {
            Task::Lander | Task::Drone => {
                let up_b = b.q.inv_rotate(Vec3::new(0.0, 0.0, 1.0));
                let down_w = b.q.rotate(Vec3::new(0.0, 0.0, -1.0));
                let radar = self.terrain.raycast(b.pos, down_w, 2000.0);
                self.last_lidar.push((b.pos, b.pos + down_w * radar));
                let acc = self.specific_force();
                let rel = if self.cfg.task == Task::Lander { b.pos - self.goal } else { self.goal - b.pos };
                let mut hits = vec![if radar < 2000.0 { 1.0 } else { 0.0 }];
                let mut o = vec![
                    rel.x + self.n(1.0), rel.y + self.n(1.0), rel.z + self.n(1.0),
                    b.vel.x + self.n(0.1), b.vel.y + self.n(0.1), b.vel.z + self.n(0.1),
                    up_b.x + self.n(0.005), up_b.y + self.n(0.005), up_b.z + self.n(0.005),
                    b.w.x + self.n(0.005), b.w.y + self.n(0.005), b.w.z + self.n(0.005),
                    radar + self.n(0.3),
                    acc.x + self.n(0.05), acc.y + self.n(0.05), acc.z + self.n(0.05),
                ];
                if self.cfg.task == Task::Lander {
                    o.push((b.mass - self.p.dry_mass) / self.p.fuel);
                    o.push(p / 1e3 * (1.0 + self.n(0.002)));
                    for k in 0..8 {
                        // конус лучей вниз-вбок, 30° от вертикали тела
                        let a = 2.0 * PI * k as f64 / 8.0;
                        let dir_b = Vec3::new(0.5 * a.cos(), 0.5 * a.sin(), -0.866);
                        let d = b.q.rotate(dir_b);
                        let r = self.terrain.raycast(b.pos, d, 1000.0);
                        self.last_lidar.push((b.pos, b.pos + d.unit() * r));
                        o.push(r + self.n(0.2));
                        hits.push(if r < 1000.0 { 1.0 } else { 0.0 });
                    }
                } else {
                    o.push(self.battery / self.p.battery_j);
                    o.push(p / 1e3 * (1.0 + self.n(0.002)));
                    o.push(rho);
                }
                o.extend(hits);
                o
            }
            Task::Rover => {
                let rel = self.goal - b.pos;
                let (gx, gy) = self.terrain.gradient(b.pos.x, b.pos.y);
                let (c, s) = (self.heading.cos(), self.heading.sin());
                let pitch = (gx * c + gy * s).atan();
                let roll = (-gx * s + gy * c).atan();
                let mut o = vec![
                    rel.x + self.n(0.5), rel.y + self.n(0.5), c + self.n(0.01), s + self.n(0.01),
                    self.speed + self.n(0.02), self.yaw_rate + self.n(0.005), pitch + self.n(0.005), roll + self.n(0.005),
                    self.battery / (0.6 * 3.6e6), self.sun_elev, temp + self.n(0.5), p / 1e3,
                ];
                let eye = b.pos + Vec3::new(0.0, 0.0, 1.0);
                let mut hits = Vec::with_capacity(16);
                for k in 0..16 {
                    let a = self.heading + 2.0 * PI * k as f64 / 16.0;
                    let d = Vec3::new(a.cos(), a.sin(), -0.06);
                    let r = self.terrain.raycast(eye, d, 40.0);
                    self.last_lidar.push((eye, eye + d.unit() * r));
                    o.push(r + self.n(0.05));
                    hits.push(if r < 40.0 { 1.0 } else { 0.0 });
                }
                o.extend(hits);
                o
            }
        }
    }

    /// Акселерометр меряет удельную силу (всё, кроме тяготения), в осях тела.
    fn specific_force(&self) -> Vec3 {
        let eff = self.effective_action();
        let f = self.forces(&self.body, &eff).0;
        self.body.q.inv_rotate(f / self.body.mass + Vec3::new(0.0, 0.0, self.g()))
    }

    /// Камера глубины w×h, взгляд вдоль оси тела (для лендера/дрона — вниз-вперёд, для ровера — вперёд).
    pub fn depth_camera(&self, w: usize, h: usize, fov_deg: f64) -> Vec<f64> {
        let (origin, rays) = self.camera_rays(w, h, fov_deg);
        rays.iter().map(|d| self.terrain.raycast(origin, *d, 400.0)).collect()
    }

    /// Простой рендер RGB (w×h×3, по строкам): рельеф по Ламберту, цвет грунта и неба из мира, дымка ∝ плотности воздуха.
    pub fn rgb_camera(&self, w: usize, h: usize, fov_deg: f64) -> Vec<u8> {
        let (origin, rays) = self.camera_rays(w, h, fov_deg);
        let surf = match self.world.surface.name.as_str() {
            "moon" | "moon_mare" | "mercury" => [0.62, 0.60, 0.57],
            "mars" => [0.72, 0.42, 0.26],
            "venus" => [0.74, 0.60, 0.36],
            _ => [0.42, 0.46, 0.30],
        };
        let (rho, _, _) = self.air(origin.z);
        let sky = match self.world.atmosphere.as_ref().map(|a| a.preset.name.as_str()) {
            None => [0.01, 0.01, 0.02],
            Some("mars") => [0.80, 0.60, 0.45],
            Some("venus") => [0.85, 0.72, 0.42],
            Some("jupiter") => [0.78, 0.66, 0.50],
            Some(_) => [0.48, 0.66, 0.92],
        };
        let elev = if self.cfg.task == Task::Rover { self.sun_elev } else { 0.7 };
        let sun = Vec3::new(elev.cos() * 0.6, elev.cos() * 0.8, elev.sin()).unit();
        let haze = 0.004 * rho.min(5.0);
        let mut out = Vec::with_capacity(w * h * 3);
        let range = if self.cfg.task == Task::Rover { 400.0 } else { 1500.0 };
        for d in &rays {
            let r = self.terrain.raycast(origin, *d, range);
            let c = if r >= range { sky } else {
                let p = origin + d.unit() * r;
                let (gx, gy) = self.terrain.gradient(p.x, p.y);
                let lit = 0.15 + 0.85 * Vec3::new(-gx, -gy, 1.0).unit().dot(sun).max(0.0);
                let fog = 1.0 - (-haze * r).exp();
                [surf[0] * lit * (1.0 - fog) + sky[0] * fog, surf[1] * lit * (1.0 - fog) + sky[1] * fog, surf[2] * lit * (1.0 - fog) + sky[2] * fog]
            };
            for x in c { out.push((x.clamp(0.0, 1.0) * 255.0) as u8); }
        }
        out
    }

    fn camera_rays(&self, w: usize, h: usize, fov_deg: f64) -> (Vec3, Vec<Vec3>) {
        let b = &self.body;
        let (origin, fwd, right, up) = match self.cfg.task {
            Task::Rover => {
                let f = Vec3::new(self.heading.cos(), self.heading.sin(), -0.15).unit();
                (b.pos + Vec3::new(0.0, 0.0, 1.2), f, Vec3::new(self.heading.sin(), -self.heading.cos(), 0.0), Vec3::new(0.0, 0.0, 1.0))
            }
            _ => {
                let f = b.q.rotate(Vec3::new(0.35, 0.0, -1.0).unit());
                (b.pos, f, b.q.rotate(Vec3::new(0.0, -1.0, 0.0)), b.q.rotate(Vec3::new(1.0, 0.0, 0.3)).unit())
            }
        };
        let tan = (fov_deg.to_radians() / 2.0).tan();
        let mut out = Vec::with_capacity(w * h);
        for j in 0..h {
            for i in 0..w {
                let u = (2.0 * (i as f64 + 0.5) / w as f64 - 1.0) * tan;
                let v = (1.0 - 2.0 * (j as f64 + 0.5) / h as f64) * tan * h as f64 / w as f64;
                out.push(fwd + right * u + up * v);
            }
        }
        (origin, out)
    }

    // ------------------------------------------------------------ физика

    /// (сила в мире, момент в осях тела, расход массы, мощность)
    fn forces(&self, b: &Body, a: &[f64]) -> (Vec3, Vec3, f64, f64) {
        let g = self.world.g_at(b.pos.z.max(0.0));
        let (rho, _, _) = self.air(b.pos.z);
        let mut f = Vec3::new(0.0, 0.0, -b.mass * g);
        let mut tq = Vec3::ZERO;
        let mut mdot = 0.0;
        let mut power = 0.0;
        let v_rel = b.vel - self.wind - self.gust_v;
        match self.cfg.task {
            Task::Lander => {
                let fuel_left = b.mass - self.p.dry_mass;
                let thr = if fuel_left > 0.0 { a.first().copied().unwrap_or(0.0).clamp(0.0, 1.0) * self.p.t_max } else { 0.0 };
                f += b.q.rotate(Vec3::new(0.0, 0.0, thr));
                mdot = thr / (self.p.isp * G0); // Циолковский: ṁ = T / (Isp·g₀)
                let rcs = |k: usize| a.get(k).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
                tq = Vec3::new(rcs(1), rcs(2), rcs(3)) * self.p.torque_max;
                f += v_rel * (-0.5 * rho * self.p.cd * self.p.area * v_rel.norm());
                f += Vec3::new(0.0, 0.0, rho * self.p.volume * g); // Архимед
            }
            Task::Drone => {
                // импульсная теория: T_max ∝ ρ·A (при фикс. окружной скорости лопасти), P_инд = T^{3/2}/√(2ρA)
                let a_rot = self.p.rotor_area / 4.0;
                let t_max_each = 2.5 * 1.5 * G0 / 4.0 * (rho / 1.225) * (a_rot / (PI * 0.12f64.powi(2)));
                let powered = self.battery > 0.0;
                let mut ts = [0.0; 4];
                for i in 0..4 {
                    ts[i] = if powered { a.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0) * t_max_each } else { 0.0 };
                    if rho > 0.0 && ts[i] > 0.0 { power += ts[i].powf(1.5) / (2.0 * rho * a_rot).sqrt() / 0.7; }
                }
                let total: f64 = ts.iter().sum();
                f += b.q.rotate(Vec3::new(0.0, 0.0, total));
                // винты: 1 (+x), 2 (+y), 3 (−x), 4 (−y); 1,3 — по часовой, 2,4 — против
                let l = self.p.arm;
                tq = Vec3::new(l * (ts[1] - ts[3]), l * (ts[2] - ts[0]), self.p.k_yaw * (ts[1] + ts[3] - ts[0] - ts[2]));
                f += v_rel * (-0.5 * rho * 1.0 * 0.05 * v_rel.norm());
                tq = tq - b.w * (0.5 * rho / 1.225 * 0.002 + 1e-4);
            }
            Task::Rover => {}
        }
        if self.cfg.task == Task::Lander {
            let (cf, ct) = self.leg_forces(b);
            f += cf;
            tq = tq + ct;
        }
        (f, tq, mdot, power)
    }

    /// Опоры: четыре ноги как пружина с демпфером и трение Кулона. Без этого модель никогда не видит контакт.
    fn leg_forces(&self, b: &Body) -> (Vec3, Vec3) {
        let (mut f, mut tq) = (Vec3::ZERO, Vec3::ZERO);
        if b.pos.z - self.terrain.height(b.pos.x, b.pos.y) > self.p.leg + 4.0 { return (f, tq); }
        for off in self.feet() {
            let p = b.pos + b.q.rotate(off);
            let pen = self.terrain.height(p.x, p.y) - p.z;
            if pen <= 0.0 { continue; }
            let v = b.vel + b.q.rotate(b.w.cross(off));
            let n = (LEG_K * pen - LEG_C * v.z).max(0.0);
            let fr = Vec3::new(-v.x, -v.y, 0.0) * LEG_CT;
            let lim = self.terrain.friction * n;
            let fr = if fr.norm() > lim { fr.unit() * lim } else { fr };
            let fw = fr + Vec3::new(0.0, 0.0, n);
            f += fw;
            tq = tq + off.cross(b.q.inv_rotate(fw));
        }
        (f, tq)
    }

    /// Параметры из описания аппарата; случайный разброс эпизода для них не применяется.
    fn apply_vehicle(&mut self) {
        for (k, v) in &self.cfg.vehicle {
            let p = &mut self.p;
            match k.as_str() {
                "dry_mass" => p.dry_mass = *v,
                "fuel" => p.fuel = *v,
                "isp" => p.isp = *v,
                "t_max" => p.t_max = *v,
                "torque_max" => p.torque_max = *v,
                "drag_cd" => p.cd = *v,
                "drag_area" => p.area = *v,
                "rover_mass" => p.rover_mass = *v,
                "motor_force" => p.motor_force = *v,
                "panel_area" => p.panel_area = *v,
                "battery_kj" => p.battery_j = *v * 1e3,
                _ => {}
            }
        }
    }

    fn feet(&self) -> [Vec3; 4] {
        let l = self.p.leg;
        [Vec3::new(1.5, 0.0, -l), Vec3::new(-1.5, 0.0, -l), Vec3::new(0.0, 1.5, -l), Vec3::new(0.0, -1.5, -l)]
    }

    fn feet_touching(&self) -> bool {
        let b = &self.body;
        self.feet().iter().any(|off| { let p = b.pos + b.q.rotate(*off); p.z <= self.terrain.height(p.x, p.y) })
    }

    /// Команда с учётом деградации приводов.
    fn effective_action(&self) -> [f64; 4] {
        let mut e = [0.0; 4];
        for i in 0..4 { e[i] = self.act_state[i] * self.act_gain[i]; }
        e
    }

    /// Приводы: апериодическое звено 1-го порядка (τ = actuator_lag) за шаг физики.
    fn update_actuators(&mut self, cmd: &[f64]) {
        let k = if self.cfg.actuator_lag > 0.0 { 1.0 - (-self.dt / self.cfg.actuator_lag).exp() } else { 1.0 };
        for (i, c) in cmd.iter().enumerate().take(4) { self.act_state[i] += (c - self.act_state[i]) * k; }
    }

    /// Порывы: процесс Орнштейна–Уленбека со стационарным СКО `gust` и τ = 2 с.
    fn update_gust(&mut self, dt: f64) {
        if self.cfg.gust <= 0.0 || self.air(self.body.pos.z).0 <= 0.0 { return; }
        let tau = 2.0;
        let s = self.cfg.gust * (2.0 * dt / tau).sqrt();
        let n = Vec3::new(self.dyn_rng.normal(), self.dyn_rng.normal(), 0.3 * self.dyn_rng.normal());
        self.gust_v = self.gust_v * (1.0 - dt / tau) + n * s;
    }

    fn physics_step(&mut self, cmd: &[f64]) {
        self.update_actuators(cmd);
        let eff = self.effective_action();
        let a: &[f64] = &eff[..cmd.len().min(4)];
        let dt = self.dt;
        match self.cfg.task {
            Task::Lander | Task::Drone => {
                let (f, tq, mdot, power) = self.forces(&self.body, a);
                let b = &mut self.body;
                // полу-неявный Эйлер (симплектический для поступательного движения)
                b.vel += f * (dt / b.mass);
                b.pos += b.vel * dt;
                // уравнения Эйлера для вращения: I ω̇ = τ − ω × Iω
                let iw = Vec3::new(b.inertia.x * b.w.x, b.inertia.y * b.w.y, b.inertia.z * b.w.z);
                let wdot = tq - b.w.cross(iw);
                b.w += Vec3::new(wdot.x / b.inertia.x, wdot.y / b.inertia.y, wdot.z / b.inertia.z) * dt;
                b.q = b.q.integrate(b.w, dt);
                if self.cfg.task == Task::Lander {
                    let dm = (mdot * dt).min(b.mass - self.p.dry_mass);
                    b.mass -= dm;
                    self.info.fuel_used += dm;
                } else {
                    self.battery = (self.battery - power * dt).max(0.0);
                    self.info.energy_used += power * dt;
                }
            }
            Task::Rover => {
                let g = self.world.g_at(0.0);
                let (rho, _, _) = self.air(self.body.pos.z);
                let m = self.p.rover_mass;
                let (gx, gy) = self.terrain.gradient(self.body.pos.x, self.body.pos.y);
                let (c, s) = (self.heading.cos(), self.heading.sin());
                let slope_along = gx * c + gy * s;
                let cos_t = 1.0 / (1.0 + gx * gx + gy * gy).sqrt();
                let n_force = m * g * cos_t;
                let powered = self.battery > 0.0;
                let mut side = [0.0; 2];
                for k in 0..2 {
                    let cmd = if powered { a.get(k).copied().unwrap_or(0.0).clamp(-1.0, 1.0) } else { 0.0 };
                    // характеристика мотора: сила падает с ростом скорости до нуля на v_max (противо-ЭДС)
                    let along = if cmd * self.speed > 0.0 { (1.0 - self.speed.abs() / ROVER_V_MAX).max(0.0) } else { 1.0 };
                    // сцепление: |F| ≤ μ·N/2 (закон Кулона), иначе колёса буксуют
                    side[k] = (cmd * along * self.p.motor_force / 2.0).clamp(-self.terrain.friction * n_force / 2.0, self.terrain.friction * n_force / 2.0);
                }
                let f_grav = -m * g * slope_along * cos_t; // компонента тяжести вдоль склона
                let f_rr = -self.p.crr * n_force * self.speed.signum() * (self.speed.abs() / 0.05).min(1.0);
                let f_drag = -0.5 * rho * 1.0 * 1.2 * self.speed * self.speed.abs();
                // на склоне круче угла трения ровер сползает, даже с заблокированными колёсами
                let f_total = side[0] + side[1] + f_grav + f_rr + f_drag;
                self.speed += f_total / m * dt;
                let i_z = self.body.inertia.z;
                let yaw_tq = (side[1] - side[0]) * self.p.track / 2.0 - 120.0 * self.yaw_rate;
                self.yaw_rate += yaw_tq / i_z * dt;
                self.heading += self.yaw_rate * dt;
                let ground_v = self.speed * cos_t;
                self.body.pos.x += ground_v * c * dt;
                self.body.pos.y += ground_v * s * dt;
                self.body.pos.z = self.terrain.height(self.body.pos.x, self.body.pos.y);
                self.body.vel = Vec3::new(ground_v * c, ground_v * s, 0.0);
                let p_mech: f64 = side.iter().map(|f| (f * self.speed).abs()).sum::<f64>() / 0.75 + 15.0;
                let insol = self.world.insolation() * self.sun_elev.sin() * if rho > 0.0 { 0.75 } else { 1.0 };
                let p_solar = self.p.panel_area * 0.25 * insol;
                self.battery = (self.battery + (p_solar - p_mech) * dt).clamp(0.0, 0.6 * 3.6e6);
                self.info.energy_used += p_mech * dt;
            }
        }
        self.t += dt;
    }

    fn shaping(&self) -> f64 {
        let b = &self.body;
        match self.cfg.task {
            Task::Lander => {
                let d = b.pos - self.goal;
                let tilt = b.q.rotate(Vec3::new(0.0, 0.0, 1.0)).z;
                -0.3 * (d.x * d.x + d.y * d.y).sqrt() - 0.1 * d.z.abs() - 1.0 * b.vel.norm() + 20.0 * tilt
            }
            Task::Drone => -(self.goal - b.pos).norm() - 0.3 * b.vel.norm(),
            Task::Rover => -(Vec3::new(self.goal.x - b.pos.x, self.goal.y - b.pos.y, 0.0)).norm(),
        }
    }

    /// Шаг среды: действие → (наблюдение, награда, завершено, обрезано).
    pub fn step(&mut self, action: &[f64]) -> (Vec<f64>, f64, bool, bool) {
        if self.done { return (self.last_obs.clone(), 0.0, true, false); }
        let (lo, hi) = self.act_bounds();
        let a: Vec<f64> = action.iter().enumerate().map(|(i, x)| if x.is_finite() { x.clamp(lo[i], hi[i]) } else { 0.0 }).collect();
        self.last_action = a.clone();
        let fuel0 = self.info.fuel_used;
        let en0 = self.info.energy_used;
        let speed0 = self.speed;
        self.update_gust(self.control_dt());
        let mut terminated = false;
        let mut reward = 0.0;
        for _ in 0..self.cfg.frame_skip {
            self.physics_step(&a);
            if let Some((r, msg, ok)) = self.check_contact() {
                reward += r;
                terminated = true;
                self.info.success = ok;
                self.info.crashed = !ok;
                self.info.reason = msg;
                break;
            }
        }
        self.steps += 1;
        let sh = self.shaping();
        reward += sh - self.prev_shaping;
        self.prev_shaping = sh;
        match self.cfg.task {
            Task::Lander => reward -= 0.02 * (self.info.fuel_used - fuel0),
            Task::Drone => {
                reward -= 1e-4 * (self.info.energy_used - en0);
                self.info.touch_offset = (self.goal - self.body.pos).norm();
                if (self.goal - self.body.pos).norm() < 2.0 && self.body.vel.norm() < 1.0 {
                    reward += 0.5;
                    self.hover_time += self.control_dt();
                    if self.hover_time > 3.0 && !terminated {
                        reward += 100.0; terminated = true; self.info.success = true;
                        self.info.reason = format!("зависание у цели 3 с, израсходовано {:.1} кДж", self.info.energy_used / 1e3);
                    }
                } else { self.hover_time = 0.0; }
                if self.battery <= 0.0 && !terminated { self.info.reason = "батарея разряжена".into(); }
                if self.in_contact { self.ground_t += self.control_dt(); } else { self.ground_t = 0.0; }
                if self.ground_t > 3.0 && !terminated {
                    reward -= 20.0; terminated = true; self.info.crashed = true;
                    self.info.reason = "не взлетел: 3 с на земле".into();
                }
            }
            Task::Rover => {
                reward -= 2e-5 * (self.info.energy_used - en0);
                let d = Vec3::new(self.goal.x - self.body.pos.x, self.goal.y - self.body.pos.y, 0.0).norm();
                self.info.touch_offset = d;
                if d < 3.0 { reward += 100.0; terminated = true; self.info.success = true; self.info.reason = "цель достигнута".into(); }
                let (gx, gy) = self.terrain.gradient(self.body.pos.x, self.body.pos.y);
                let (c, s) = (self.heading.cos(), self.heading.sin());
                let cos_t = 1.0 / (1.0 + gx * gx + gy * gy).sqrt();
                let g_n = (self.world.g_at(0.0) * cos_t).max(1e-6);
                // опрокидывание: тангенс наклона + инерция против запаса устойчивости (колея/2h, база/2h)
                let lateral = -gx * s + gy * c + self.speed * self.yaw_rate / g_n;
                let longitudinal = gx * c + gy * s + (self.speed - speed0) / self.control_dt() / g_n;
                if !terminated && (lateral.abs() > ROVER_TIP_LAT || longitudinal.abs() > ROVER_TIP_LONG) {
                    reward -= 50.0; terminated = true; self.info.crashed = true; self.info.reason = "опрокинулся на склоне".into();
                }
                if self.battery <= 0.0 && self.sun_elev < 0.05 { terminated = true; self.info.reason = "батарея разряжена".into(); }
            }
        }
        let half = self.terrain.size / 2.0;
        if !terminated && (self.body.pos.x.abs() > half || self.body.pos.y.abs() > half || self.body.pos.z > self.terrain.height(0.0, 0.0) + 50_000.0) {
            terminated = true;
            reward -= 50.0;
            self.info.reason = "вылетел за пределы карты".into();
        }
        let truncated = !terminated && self.steps >= self.cfg.max_steps;
        if truncated { self.info.reason = "лимит времени".into(); }
        self.done = terminated || truncated;
        self.ep_return += reward;
        self.last_obs = self.observe();
        (self.last_obs.clone(), reward, terminated, truncated)
    }

    /// Контакт с поверхностью: (награда, сообщение, успех), если эпизод на этом кончился.
    fn check_contact(&mut self) -> Option<(f64, String, bool)> {
        match self.cfg.task {
            Task::Lander => {
                let touching = self.feet_touching();
                let b = &self.body;
                let tilt = b.q.rotate(Vec3::new(0.0, 0.0, 1.0)).z.clamp(-1.0, 1.0).acos().to_degrees();
                let v = b.vel.norm();
                if touching && !self.in_contact {
                    let (gx, gy) = self.terrain.gradient(b.pos.x, b.pos.y);
                    let slope = (gx * gx + gy * gy).sqrt().atan().to_degrees();
                    if !(v < 2.5 && tilt < 15.0 && slope < 12.0) {
                        if !self.touched {
                            self.info.touch_speed = v;
                            self.info.touch_tilt = tilt;
                            self.info.touch_offset = ((b.pos.x - self.goal.x).powi(2) + (b.pos.y - self.goal.y).powi(2)).sqrt();
                        }
                        return Some((-100.0, format!("крушение: {v:.1} м/с, наклон {tilt:.0}°, склон {slope:.0}°"), false));
                    }
                    if !self.touched {
                        let off = ((b.pos.x - self.goal.x).powi(2) + (b.pos.y - self.goal.y).powi(2)).sqrt();
                        self.touch_bonus = if off < 25.0 { 100.0 } else { 40.0 };
                        self.touch_msg = format!("мягкая посадка: {v:.2} м/с, наклон {tilt:.1}°, {off:.0} м от центра площадки");
                        self.info.touch_speed = v;
                        self.info.touch_tilt = tilt;
                        self.info.touch_offset = off;
                    }
                    self.touched = true;
                }
                self.in_contact = touching;
                if !touching { self.settle_t = 0.0; return None; }
                if tilt > 35.0 { return Some((-100.0, format!("опрокинулся после касания: наклон {tilt:.0}°"), false)); }
                if v < 0.2 && self.body.w.norm() < 0.15 { self.settle_t += self.dt; } else { self.settle_t = 0.0; }
                if self.settle_t > 0.5 { return Some((self.touch_bonus, self.touch_msg.clone(), true)); }
                None
            }
            Task::Drone => {
                let b = &self.body;
                let ground = self.terrain.height(b.pos.x, b.pos.y) + 0.05;
                if b.pos.z > ground { self.in_contact = false; return None; }
                let v = b.vel.norm();
                if !self.in_contact && v > 3.0 { return Some((-100.0, format!("удар о землю на {v:.1} м/с"), false)); }
                let tilt = b.q.rotate(Vec3::new(0.0, 0.0, 1.0)).z.clamp(-1.0, 1.0).acos().to_degrees();
                if tilt > 60.0 { return Some((-100.0, format!("опрокинулся на земле: наклон {tilt:.0}°"), false)); }
                // неупругий контакт: дрон стоит на земле и может снова взлететь
                self.in_contact = true;
                let b = &mut self.body;
                b.pos.z = ground;
                b.vel.z = b.vel.z.max(0.0);
                b.vel.x *= 0.9;
                b.vel.y *= 0.9;
                b.w = b.w * 0.9;
                None
            }
            Task::Rover => None,
        }
    }

    // ------------------------------------------------------------ для людей и языковых моделей

    /// Текстовое описание сенсоров — вход для LLM/VLM-агента.
    pub fn describe(&self) -> String {
        let b = &self.body;
        let (rho, p, t) = self.air(b.pos.z);
        let wname = &self.world.physics.name;
        match self.cfg.task {
            Task::Lander => {
                let d = b.pos - self.goal;
                let tilt = b.q.rotate(Vec3::new(0.0, 0.0, 1.0)).z.clamp(-1.0, 1.0).acos().to_degrees();
                format!("Мир: {wname}, g = {:.2} м/с², давление {:.0} Па. Модуль на высоте {:.0} м над площадкой, смещение {:.0} м на восток и {:.0} м на север. \
Скорость: вниз {:.1} м/с, по горизонтали {:.1} м/с. Наклон от вертикали {:.1}°. Топливо {:.0}%. Радиовысотомер: {:.0} м до грунта. \
Действия: throttle 0…1 (макс. тяга {:.0} кН), torque_x/y/z −1…1.",
                    self.g(), p, d.z, d.x, d.y, -b.vel.z, (b.vel.x.powi(2) + b.vel.y.powi(2)).sqrt(), tilt,
                    100.0 * (b.mass - self.p.dry_mass) / self.p.fuel, self.last_obs.get(12).copied().unwrap_or(0.0), self.p.t_max / 1e3)
            }
            Task::Drone => {
                let d = self.goal - b.pos;
                format!("Мир: {wname}, плотность воздуха {rho:.4} кг/м³, g = {:.2} м/с². Цель: {:.1} м на восток, {:.1} м на север, {:.1} м вверх. \
Скорость {:.1} м/с, батарея {:.0}%. Действия: 4 винта 0…1.",
                    self.g(), d.x, d.y, d.z, b.vel.norm(), 100.0 * self.battery / self.p.battery_j)
            }
            Task::Rover => {
                let d = self.goal - b.pos;
                let ahead = self.last_obs.get(12).copied().unwrap_or(40.0);
                format!("Мир: {wname}, T = {t:.0} К. Цель в {:.0} м (на восток {:.0}, на север {:.0}), курс {:.0}°. Скорость {:.2} м/с, батарея {:.0}%, \
Солнце на высоте {:.0}°. Лидар вперёд: {:.1} м. Действия: left, right −1…1.",
                    (d.x * d.x + d.y * d.y).sqrt(), d.x, d.y, self.heading.to_degrees().rem_euclid(360.0), self.speed,
                    100.0 * self.battery / (0.6 * 3.6e6), self.sun_elev.to_degrees(), ahead)
            }
        }
    }

    /// Встроенный автопилот (ПД-регулятор) — базовая линия для сравнения с обученными агентами.
    pub fn autopilot(&self) -> Vec<f64> { self.autopilot_nav(Vec3::ZERO, Vec3::ZERO) }

    /// Автопилот с ошибкой навигации: показывает, как качество сенсоров влияет на посадку.
    /// `pos_err` и `vel_err` — насколько оценка отличается от истины (ориентация считается известной).
    pub fn autopilot_nav(&self, pos_err: Vec3, vel_err: Vec3) -> Vec<f64> {
        let mut nav = self.body.clone();
        nav.pos = nav.pos + pos_err;
        nav.vel = nav.vel + vel_err;
        let b = &nav;
        let g = self.g();
        match self.cfg.task {
            Task::Lander => {
                if self.touched && self.in_contact { return vec![0.0; 4]; }
                let d = b.pos - self.goal;
                // высота над рельефом, а не над площадкой: иначе модуль влетает в борт кратера
                let alt = (b.pos.z - self.terrain.height(b.pos.x, b.pos.y)).min(d.z) - self.p.leg;
                let off = (d.x * d.x + d.y * d.y).sqrt();
                // желаемая горизонтальная скорость к площадке и вертикальная по высоте
                let vh_target = Vec3::new(-d.x, -d.y, 0.0) * 0.12;
                let vh_target = if vh_target.norm() > 15.0 { vh_target.unit() * 15.0 } else { vh_target };
                // запас по тяге задаёт, с какой скоростью можно снижаться и ещё успеть затормозить
                let brake = (self.p.t_max / b.mass * self.act_gain[0] - g).max(0.3);
                let vz_stop = (1.4 * brake * alt.max(0.0)).sqrt();
                // пока не над площадкой — не снижаемся ниже 40 м: садиться можно только на ровное
                let vz_target = if off > 12.0 && alt < 40.0 { (40.0 - alt) * 0.3 } else { -(1.0 + 0.12 * alt).min(vz_stop + 1.0).min(50.0) };
                let acc_des = Vec3::new((vh_target.x - b.vel.x) * 0.4, (vh_target.y - b.vel.y) * 0.4, (vz_target - b.vel.z) * 1.2 + g);
                let want_up = acc_des.unit();
                let up_w = b.q.rotate(Vec3::new(0.0, 0.0, 1.0));
                let err_w = up_w.cross(want_up);
                let err_b = b.q.inv_rotate(err_w);
                let kp = 6.0;
                let kd = 4.0;
                let t = |e: f64, w: f64, i: f64| ((kp * e - kd * w) * i / self.p.torque_max).clamp(-1.0, 1.0);
                let thr = (acc_des.norm() * b.mass / self.p.t_max * up_w.dot(want_up).max(0.0)).clamp(0.0, 1.0);
                vec![thr, t(err_b.x, b.w.x, b.inertia.x), t(err_b.y, b.w.y, b.inertia.y), t(0.0, b.w.z, b.inertia.z)]
            }
            Task::Drone => {
                let (rho, _, _) = self.air(b.pos.z);
                let d = self.goal - b.pos;
                let v_des = if d.norm() > 1e-6 { d.unit() * (d.norm() * 0.6).min(6.0) } else { Vec3::ZERO };
                let acc_des = (v_des - b.vel) * 1.5 + Vec3::new(0.0, 0.0, g);
                let acc_des = Vec3::new(acc_des.x.clamp(-4.0, 4.0), acc_des.y.clamp(-4.0, 4.0), acc_des.z.max(0.2 * g));
                let want_up = acc_des.unit();
                let up_w = b.q.rotate(Vec3::new(0.0, 0.0, 1.0));
                let err_b = b.q.inv_rotate(up_w.cross(want_up));
                let a_rot = self.p.rotor_area / 4.0;
                let tmax = 2.5 * 1.5 * G0 / 4.0 * (rho / 1.225) * (a_rot / (PI * 0.12f64.powi(2)));
                if tmax <= 0.0 { return vec![0.0; 4]; }
                let base = acc_des.norm() * b.mass / 4.0 / tmax;
                let tx = 0.6 * err_b.x - 0.15 * b.w.x;
                let ty = 0.6 * err_b.y - 0.15 * b.w.y;
                let tz = -0.05 * b.w.z;
                let l = self.p.arm;
                // распределение моментов по винтам (обратно к матрице смешивания)
                let dx = tx / (2.0 * l) / tmax;
                let dy = ty / (2.0 * l) / tmax;
                let dz = tz / (4.0 * self.p.k_yaw) / tmax;
                vec![(base - dy - dz).clamp(0.0, 1.0), (base + dx + dz).clamp(0.0, 1.0), (base + dy - dz).clamp(0.0, 1.0), (base - dx + dz).clamp(0.0, 1.0)]
            }
            Task::Rover => {
                let d = self.goal - b.pos;
                let want = d.y.atan2(d.x);
                // карта опасностей: из курсов рядом с целью выбираем ближайший, где склон на 3–9 м вперёд безопасен
                let slope_at = |ang: f64| -> f64 {
                    [3.0, 6.0, 9.0].iter().map(|r| {
                        let (gx, gy) = self.terrain.gradient(b.pos.x + r * ang.cos(), b.pos.y + r * ang.sin());
                        (gx * gx + gy * gy).sqrt()
                    }).fold(0.0, f64::max)
                };
                let mut best = want;
                let mut best_cost = f64::MAX;
                for k in 0..13 {
                    let off = (k as f64 - 6.0) * 0.45;
                    let ang = want + off;
                    let cost = off.abs() + 8.0 * (slope_at(ang) - 0.35).max(0.0);
                    if cost < best_cost { best_cost = cost; best = ang; }
                }
                let mut err = best - self.heading;
                while err > PI { err -= 2.0 * PI }
                while err < -PI { err += 2.0 * PI }
                let fwd = (1.0 - err.abs() / (0.5 * PI)).clamp(0.0, 1.0);
                // поворот ограничен запасом устойчивости: v·ω ≤ 0.25·g, на Луне это заметно
                let yaw_lim = 0.25 * g / self.speed.abs().max(0.3);
                let yaw_des = (1.5 * err).clamp(-yaw_lim, yaw_lim);
                let turn = (2.0 * (yaw_des - self.yaw_rate)).clamp(-1.0, 1.0);
                // предел скорости по запасу устойчивости: v·ω ≤ 0.3·g
                let (hx, hy) = self.terrain.gradient(b.pos.x, b.pos.y);
                let v_base = 1.5 * (1.0 - (hx * hx + hy * hy).sqrt() / 0.9).clamp(0.3, 1.0);
                let v_max = if self.yaw_rate.abs() > 1e-3 { (0.3 * g / self.yaw_rate.abs()).min(v_base) } else { v_base };
                let spd = if self.speed > v_max { 0.0 } else { fwd };
                vec![(spd - turn).clamp(-1.0, 1.0), (spd + turn).clamp(-1.0, 1.0)]
            }
        }
    }

    /// Поза для визуализации: позиция, кватернион, курс ровера.
    pub fn pose(&self) -> (Vec3, Quat) {
        match self.cfg.task {
            Task::Rover => (self.body.pos, Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), self.heading)),
            _ => (self.body.pos, self.body.q),
        }
    }
    pub fn mass(&self) -> f64 { self.body.mass }
    pub fn air_density(&self) -> f64 { self.air(self.body.pos.z).0 }

    /// Для проверок: подменить рельеф и состояние.
    pub fn set_terrain(&mut self, t: Terrain) { self.terrain = Arc::new(t); }
    pub fn set_state(&mut self, pos: Vec3, vel: Vec3) {
        self.body.pos = pos;
        self.body.vel = vel;
        self.body.q = Quat::ID;
        self.body.w = Vec3::ZERO;
        self.speed = 0.0;
        self.yaw_rate = 0.0;
        self.heading = 0.0;
        self.done = false;
        self.in_contact = false;
        self.touched = false;
        self.settle_t = 0.0;
        self.ground_t = 0.0;
        self.prev_shaping = self.shaping();
    }
    pub fn velocity(&self) -> Vec3 { if self.cfg.task == Task::Rover { self.body.vel } else { self.body.vel } }
    pub fn battery_frac(&self) -> f64 { match self.cfg.task { Task::Drone => self.battery / self.p.battery_j, Task::Rover => self.battery / (0.6 * 3.6e6), _ => (self.body.mass - self.p.dry_mass) / self.p.fuel } }
    pub fn isp(&self) -> f64 { self.p.isp }
    pub fn dry_mass(&self) -> f64 { self.p.dry_mass }
    pub fn wind(&self) -> Vec3 { self.wind }
    pub fn rover_speed(&self) -> f64 { self.speed }
    pub fn gust_velocity(&self) -> Vec3 { self.gust_v }
    pub fn applied_action(&self) -> [f64; 4] { self.act_state }
    pub fn in_contact(&self) -> bool { self.in_contact }
    /// Истина для сравнения с наблюдением GPS: относительная позиция и скорость без шума.
    pub fn true_rel_pos(&self) -> Vec3 { if self.cfg.task == Task::Lander { self.body.pos - self.goal } else { self.goal - self.body.pos } }
    pub fn angular_velocity(&self) -> Vec3 { self.body.w }

    /// Имена истинного состояния (см. `state`): без шума, то, чего агент напрямую не видит.
    pub fn state_names(&self) -> Vec<&'static str> {
        match self.cfg.task {
            Task::Rover => vec!["x", "y", "z", "heading", "speed", "yaw_rate", "battery_j", "sun_elev", "rho", "g"],
            _ => vec!["x", "y", "z", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz", "mass", "battery_j",
                "air_vx", "air_vy", "air_vz", "rho", "g", "contact", "act_0", "act_1", "act_2", "act_3"],
        }
    }

    /// Истинное состояние в этот момент.
    pub fn state(&self) -> Vec<f64> {
        let b = &self.body;
        let (rho, _, _) = self.air(b.pos.z);
        match self.cfg.task {
            Task::Rover => vec![b.pos.x, b.pos.y, b.pos.z, self.heading, self.speed, self.yaw_rate, self.battery, self.sun_elev, rho, self.world.g_at(0.0)],
            _ => {
                let air = self.wind + self.gust_v;
                let e = self.effective_action();
                vec![b.pos.x, b.pos.y, b.pos.z, b.vel.x, b.vel.y, b.vel.z, b.q.w, b.q.x, b.q.y, b.q.z, b.w.x, b.w.y, b.w.z, b.mass, self.battery,
                    air.x, air.y, air.z, rho, self.g(), if self.in_contact { 1.0 } else { 0.0 }, e[0], e[1], e[2], e[3]]
            }
        }
    }

    /// Скрытые параметры эпизода, порядок как в `PARAM_NAMES`.
    pub fn params(&self) -> Vec<f64> {
        let p = &self.p;
        let (rho, pr, t) = self.air(0.0);
        let (fa, fg) = self.fault.map(|(i, g)| (i as f64, g)).unwrap_or((-1.0, 1.0));
        vec![self.world.g_at(0.0), rho, pr / 1e3, t, self.terrain.friction, p.dry_mass, p.fuel, p.isp, p.t_max, p.torque_max, p.rotor_area, p.rover_mass,
            self.wind.x, self.wind.y, self.cfg.gust, self.cfg.actuator_lag, fa, fg, self.world_index as f64, self.goal.x, self.goal.y, self.goal.z, self.control_dt()]
    }
}
