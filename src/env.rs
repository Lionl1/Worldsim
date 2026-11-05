//! Environment for training AI agents and robots (in the style of Gymnasium).
//!
//! The agent receives **sensor data only** (noisy, like a real robot) and outputs
//! continuous actuator commands. The physics is the same as in the rest of worldsim: world gravity,
//! air density from hydrostatics, wind, drag, Archimedes force, the Tsiolkovsky
//! equation for the engine, momentum theory for the drone rotors, Coulomb traction for the wheels.
//!
//! Tasks:
//!   * `lander` — soft landing on a pad (thrust + attitude, fuel use by Isp)
//!   * `drone`  — quadcopter flies to a point; thrust ∝ ρ, power ∝ T^{3/2}/√(2ρA)
//!   * `rover`  — rover drives to a goal over terrain; solar panel, slopes, lidar
//!
//! The same agent can be trained on Earth, the Moon, Mars and mixed worlds (domain randomization).

use crate::rng::Rng;
use crate::vec3::Vec3;
use crate::world::{World, WorldSpec};
use std::f64::consts::PI;
use std::sync::Arc;

pub const G0: f64 = 9.80665;
const LEG_K: f64 = 6.0e4;   // leg stiffness, N/m
const LEG_C: f64 = 6.0e3;   // leg damping, N·s/m
const LEG_CT: f64 = 2.0e3;  // viscous approximation of static friction, N·s/m

// ================================================================ terrain

#[derive(Clone, Debug)]
pub struct Terrain {
    pub n: usize,     // nodes per side
    pub size: f64,    // m
    pub h: Vec<f64>,  // n×n, row = y
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
    /// Procedural terrain for the world surface type: craters on airless bodies, hills on Earth.
    pub fn generate(w: &World, seed: u64, size: f64, n: usize, pad_radius: f64) -> Terrain { Self::generate_tuned(w, seed, size, n, pad_radius, 1.0, 1.0, -1.0) }

    /// `amp_scale` and `crater_scale` scale the terrain, `friction` ≥ 0 replaces surface friction.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_tuned(w: &World, seed: u64, size: f64, n: usize, pad_radius: f64, amp_scale: f64, crater_scale: f64, friction_override: f64) -> Terrain {
        let surf = w.surface.name.as_str();
        let airless = w.atmosphere.is_none();
        let (amp, craters, friction) = match surf {
            "moon" | "moon_mare" | "mercury" => (18.0, 30, 0.6),
            "mars" => (25.0, 14, 0.55),
            "venus" => (12.0, 4, 0.6),
            _ => (15.0, if airless { 20 } else { 0 }, 0.8),
        };
        let (amp, craters) = (amp * amp_scale, (craters as f64 * crater_scale) as usize);
        let friction = if friction_override >= 0.0 { friction_override } else { friction };
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
            let depth = 0.2 * r * amp_scale.min(1.0);
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
        // landing pad / start: a flat circle in the center
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

    /// Inclined plane (for checks): height = grade · x.
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

    /// Ray-terrain intersection: distance to the surface or max_range.
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

// ================================================================ attitude

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
    /// body → world
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let p = self.mul(Quat { w: 0.0, x: v.x, y: v.y, z: v.z }).mul(self.conj());
        Vec3::new(p.x, p.y, p.z)
    }
    pub fn inv_rotate(self, v: Vec3) -> Vec3 { self.conj().rotate(v) }
    pub fn normalize(self) -> Quat {
        let n = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        Quat { w: self.w / n, x: self.x / n, y: self.y / n, z: self.z / n }
    }
    /// rotation by angular velocity ω (in body axes) over dt
    pub fn integrate(self, w_body: Vec3, dt: f64) -> Quat {
        let ang = w_body.norm() * dt;
        if ang < 1e-12 { return self; }
        self.mul(Quat::from_axis_angle(w_body, ang)).normalize()
    }
}

// ================================================================ task descriptions

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Task { Lander, Drone, Rover }

pub const TASKS: &[(&str, &str)] = &[
    ("lander", "Lander: land softly on a pad. Actions: thrust 0…1, attitude torques on 3 axes. Fuel is used according to the Tsiolkovsky equation."),
    ("drone", "Quadcopter: fly to a point and hover. Actions: 4 rotors 0…1. Thrust ∝ air density: Mars needs huge rotors, flight on the Moon is impossible."),
    ("rover", "Rover: drive to a goal over terrain. Actions: left and right wheel torques −1…1. Coulomb traction, slopes, solar panel."),
];

#[derive(Clone, Debug)]
pub struct EnvConfig {
    pub task: Task,
    pub world: WorldSpec,
    pub randomize: bool,   // parameter randomization (mass, engine, wind, world)
    pub noise: f64,        // sensor noise factor (0 means ideal sensors)
    pub frame_skip: usize, // physics steps per action
    pub max_steps: usize,
    pub rotor_scale: f64,  // for the drone: rotor area factor (Ingenuity ≈ 60)
    pub worlds: Vec<WorldSpec>, // if not empty, the world is chosen anew in each episode
    pub gust: f64,         // wind gust standard deviation, m/s (Ornstein–Uhlenbeck process, τ = 2 s)
    pub actuator_lag: f64, // actuator time constant, s (0 means instant)
    pub fault_prob: f64,   // probability that one actuator is weakened in an episode (hidden parameter)
    pub vehicle: Vec<(String, f64)>, // registry parameters, see PARAMS
    pub custom_terrain: Option<Arc<Terrain>>, // terrain of a real site instead of the procedural one
}

const L: u8 = 1;
const D: u8 = 2;
const R: u8 = 4;
const ALL: u8 = 7;

/// One configurable parameter. The interface, API, CLI, missions and Python all read from here,
/// so a new parameter only needs to be added to `PARAMS` and used in `Tune`.
#[derive(Clone, Copy, Debug)]
pub struct ParamDef {
    pub name: &'static str, pub group: &'static str, pub label: &'static str, pub unit: &'static str,
    pub min: f64, pub max: f64, pub tasks: u8, pub defaults: [f64; 3], pub help: &'static str,
}
const fn p(name: &'static str, group: &'static str, label: &'static str, unit: &'static str, min: f64, max: f64, tasks: u8, d: f64, help: &'static str) -> ParamDef {
    ParamDef { name, group, label, unit, min, max, tasks, defaults: [d, d, d], help }
}
const fn p3(name: &'static str, group: &'static str, label: &'static str, unit: &'static str, min: f64, max: f64, tasks: u8, defaults: [f64; 3], help: &'static str) -> ParamDef {
    ParamDef { name, group, label, unit, min, max, tasks, defaults, help }
}

pub const PARAMS: &[ParamDef] = &[
    // Vehicle
    p("dry_mass", "Vehicle", "Dry mass", "kg", 50.0, 50_000.0, L, 1000.0, "Mass without fuel."),
    p("fuel", "Vehicle", "Fuel load", "kg", 1.0, 50_000.0, L, 450.0, "Fuel mass at episode start."),
    p("isp", "Vehicle", "Specific impulse", "s", 50.0, 1000.0, L, 300.0, "Higher means less propellant for the same thrust."),
    p("t_max", "Vehicle", "Maximum thrust", "N", 100.0, 1e6, L, 22_000.0, "Main engine thrust at full throttle."),
    p("throttle_min", "Vehicle", "Minimum throttle", "fraction", 0.0, 0.9, L, 0.0, "The engine cannot throttle below this: it is either at this fraction or above, or off."),
    p("thrust_misalign", "Vehicle", "Thrust misalignment", "°", 0.0, 10.0, L, 0.0, "Angle between the engine axis and the vehicle axis."),
    p("cg_offset", "Vehicle", "Center of mass offset", "m", 0.0, 0.5, L, 0.0, "Thrust creates a parasitic torque that the attitude thrusters must counter."),
    p("torque_max", "Vehicle", "Attitude torque", "N·m", 10.0, 1e5, L, 3000.0, "Maximum attitude thruster torque."),
    p("drag_cd", "Vehicle", "Drag coefficient", "", 0.1, 3.0, L, 1.0, ""),
    p("drag_area", "Vehicle", "Drag area", "m²", 0.1, 100.0, L, 4.0, ""),
    p("bus_power", "Vehicle", "Bus power", "W", 0.0, 100_000.0, ALL, 0.0, "Constant load: avionics, communications, heaters. 0 means no bus load."),
    p("lander_battery_kwh", "Vehicle", "Lander battery", "kWh", 0.01, 1000.0, L, 5.0, "Used only with a bus load. When it is empty, the engine and thrusters stop responding."),
    p("drone_mass", "Vehicle", "Drone mass", "kg", 0.1, 50.0, D, 1.5, ""),
    p("thrust_to_weight", "Vehicle", "Thrust-to-weight on Earth", "", 1.1, 10.0, D, 2.5, "Rotor thrust margin at sea level with standard area."),
    p("battery_kj", "Vehicle", "Battery", "kJ", 1.0, 1e5, D, 150.0, ""),
    p("rotor_scale", "Vehicle", "Rotor area", "×", 0.1, 500.0, D, 1.0, "A thin atmosphere needs more: about this on Mars 16."),
    p("rover_mass", "Vehicle", "Rover mass", "kg", 10.0, 5000.0, R, 180.0, ""),
    p("motor_force", "Vehicle", "Motor force", "N", 10.0, 50_000.0, R, 900.0, "Total wheel force when starting."),
    p("rover_vmax", "Vehicle", "No-load speed", "m/s", 0.1, 20.0, R, 2.5, "Motor force falls to zero at this speed."),
    p("cg_height", "Vehicle", "Center of mass height", "m", 0.1, 3.0, R, 0.6, "Higher means easier to tip over."),
    p("rolling_resistance", "Vehicle", "Rolling resistance", "", 0.0, 0.5, R, 0.05, ""),
    p("panel_area", "Vehicle", "Solar panel area", "m²", 0.0, 50.0, R, 1.5, ""),
    p("battery_kwh", "Vehicle", "Battery", "kWh", 0.01, 100.0, R, 0.6, ""),
    // Scenario
    p3("start_alt_min", "Scenario", "Start altitude, from", "m", 1.0, 20_000.0, L | D, [300.0, 8.0, 0.0], ""),
    p3("start_alt_max", "Scenario", "Start altitude, to", "m", 1.0, 20_000.0, L | D, [700.0, 30.0, 0.0], ""),
    p("start_offset", "Scenario", "Offset from the pad, up to", "m", 0.0, 450.0, L, 200.0, "On each horizontal axis."),
    p("start_vh", "Scenario", "Horizontal speed, up to", "m/s", 0.0, 200.0, L, 15.0, "On each axis."),
    p("start_vz_min", "Scenario", "Descent speed, from", "m/s", 0.0, 500.0, L, 10.0, ""),
    p("start_vz_max", "Scenario", "Descent speed, to", "m/s", 0.0, 500.0, L, 35.0, ""),
    p("start_tilt", "Scenario", "Initial tilt, up to", "°", 0.0, 60.0, L, 8.6, ""),
    p3("goal_dist_min", "Scenario", "Distance to goal, from", "m", 3.0, 280.0, D | R, [0.0, 20.0, 80.0], ""),
    p3("goal_dist_max", "Scenario", "Distance to goal, to", "m", 3.0, 280.0, D | R, [0.0, 60.0, 220.0], ""),
    p3("pad_radius", "Scenario", "Flat pad radius", "m", 2.0, 200.0, ALL, [25.0, 15.0, 8.0], "Terrain is leveled here."),
    p("terrain_scale", "Scenario", "Terrain height", "×", 0.0, 5.0, ALL, 1.0, "0 — flat surface."),
    p("crater_scale", "Scenario", "Number of craters", "×", 0.0, 5.0, ALL, 1.0, ""),
    p("friction", "Scenario", "Soil friction", "μ", -1.0, 2.0, ALL, -1.0, "−1 — value from the surface type."),
    p("sun_elev", "Scenario", "Sun elevation", "°", -1.0, 90.0, R, -1.0, "−1 — random in each episode."),
    // Criteria
    p("touch_speed_limit", "Criteria", "Allowed touchdown speed", "m/s", 0.1, 50.0, L, 2.5, ""),
    p("touch_tilt_limit", "Criteria", "Allowed touchdown tilt", "°", 1.0, 60.0, L, 15.0, ""),
    p("touch_slope_limit", "Criteria", "Allowed pad slope", "°", 1.0, 60.0, L, 12.0, ""),
    p("impact_limit", "Criteria", "Allowed impact speed", "m/s", 0.1, 50.0, D, 3.0, ""),
    p3("reward_success", "Criteria", "Reward for success", "", 0.0, 1e4, ALL, [100.0, 100.0, 100.0], ""),
    p3("reward_crash", "Criteria", "Penalty for a crash", "", 0.0, 1e4, ALL, [100.0, 100.0, 50.0], ""),
    // Environment
    p("gravity", "Environment", "Gravity, custom", "m/s²", 0.0, 50.0, ALL, 0.0, "0 — same as the selected world."),
    p("air_scale", "Environment", "Atmosphere density", "×", 0.0, 100.0, ALL, 1.0, "Density and pressure factor of the selected atmosphere."),
    p("wind_x", "Environment", "Steady wind to the east", "m/s", -100.0, 100.0, L | D, 0.0, ""),
    p("wind_y", "Environment", "Steady wind to the north", "m/s", -100.0, 100.0, L | D, 0.0, ""),
    p("gust", "Environment", "Wind gusts, standard deviation", "m/s", 0.0, 50.0, L | D, 0.0, "Random process with a 2 s time constant."),
    p("solar_scale", "Environment", "Illumination", "×", 0.0, 10.0, R, 1.0, "Insolation factor for the solar panel."),
    // Sensors
    p("noise", "Sensors", "Overall noise", "×", 0.0, 100.0, ALL, 1.0, "0 — ideal sensors."),
    p("noise_pos", "Sensors", "Position noise", "×", 0.0, 100.0, ALL, 1.0, "GPS or relative navigation."),
    p("noise_vel", "Sensors", "Velocity noise", "×", 0.0, 100.0, ALL, 1.0, ""),
    p("noise_att", "Sensors", "Attitude noise", "×", 0.0, 100.0, ALL, 1.0, ""),
    p("noise_gyro", "Sensors", "Gyroscope noise", "×", 0.0, 100.0, ALL, 1.0, ""),
    p("noise_accel", "Sensors", "Accelerometer noise", "×", 0.0, 100.0, L | D, 1.0, ""),
    p("noise_range", "Sensors", "Altimeter and lidar noise", "×", 0.0, 100.0, ALL, 1.0, ""),
    p("gyro_bias", "Sensors", "Gyroscope bias, standard deviation", "rad/s", 0.0, 1.0, ALL, 0.0, "Constant during an episode and hidden from the agent."),
    p("accel_bias", "Sensors", "Accelerometer bias, standard deviation", "m/s²", 0.0, 5.0, L | D, 0.0, "Constant during an episode and hidden from the agent."),
    p("sensor_dropout", "Sensors", "Reading dropouts", "fraction", 0.0, 0.9, ALL, 0.0, "With this probability a channel repeats its previous value."),
    p("obs_delay", "Sensors", "Observation delay", "steps", 0.0, 200.0, ALL, 0.0, "Communication or processing: the agent sees the past."),
    // Control
    p("frame_skip", "Control", "Physics steps per action", "", 1.0, 100.0, ALL, 4.0, "Sets the control rate."),
    p3("physics_dt", "Control", "Physics step", "s", 0.0005, 0.05, ALL, [0.005, 0.005, 0.025], "Smaller is more accurate and slower."),
    p3("max_steps", "Control", "Step limit", "", 1.0, 1e7, ALL, [5000.0, 2500.0, 4000.0], ""),
    p("actuator_lag", "Control", "Actuator lag", "s", 0.0, 10.0, ALL, 0.0, "First-order time constant."),
    p("action_noise", "Control", "Execution noise", "fraction", 0.0, 1.0, ALL, 0.0, "Random command error relative to the range."),
    p("fault_prob", "Control", "Actuator fault probability", "", 0.0, 1.0, ALL, 0.0, "One actuator is weakened in an episode, and the agent cannot see it."),
];

pub fn param_def(name: &str) -> Option<&'static ParamDef> { PARAMS.iter().find(|d| d.name == name) }

impl Task {
    fn bit(self) -> u8 { match self { Task::Lander => L, Task::Drone => D, Task::Rover => R } }
    fn index(self) -> usize { match self { Task::Lander => 0, Task::Drone => 1, Task::Rover => 2 } }
    pub fn name(self) -> &'static str { match self { Task::Lander => "lander", Task::Drone => "drone", Task::Rover => "rover" } }
}

/// Parameters already resolved at episode start: the physics loop must not search strings.
#[derive(Clone, Debug)]
struct Tune {
    throttle_min: f64, misalign: f64, cg_offset: f64, drone_mass: f64, t2w: f64, rover_vmax: f64, tip_lat: f64, tip_long: f64, battery_rover: f64, bus_power: f64, battery_lander: f64,
    alt: (f64, f64), offset: f64, vh: f64, vz: (f64, f64), tilt: f64, goal: (f64, f64), pad: f64, terrain: f64, craters: f64, friction: f64, sun: f64,
    lim_v: f64, lim_tilt: f64, lim_slope: f64, impact: f64, rew_ok: f64, rew_bad: f64, air: f64, wind: Vec3, solar: f64,
    n_pos: f64, n_vel: f64, n_att: f64, n_gyro: f64, n_acc: f64, n_rng: f64, gyro_bias: f64, accel_bias: f64, dropout: f64, delay: usize, dt: f64, action_noise: f64,
}

impl Tune {
    fn zero() -> Tune {
        Tune { throttle_min: 0.0, misalign: 0.0, cg_offset: 0.0, drone_mass: 1.5, t2w: 2.5, rover_vmax: 2.5, tip_lat: 1.0, tip_long: 1.33, battery_rover: 2.16e6, bus_power: 0.0, battery_lander: 0.0,
            alt: (0.0, 0.0), offset: 0.0, vh: 0.0, vz: (0.0, 0.0), tilt: 0.0, goal: (0.0, 0.0), pad: 25.0, terrain: 1.0, craters: 1.0, friction: -1.0, sun: -1.0,
            lim_v: 2.5, lim_tilt: 15.0, lim_slope: 12.0, impact: 3.0, rew_ok: 100.0, rew_bad: 100.0, air: 1.0, wind: Vec3::ZERO, solar: 1.0,
            n_pos: 1.0, n_vel: 1.0, n_att: 1.0, n_gyro: 1.0, n_acc: 1.0, n_rng: 1.0, gyro_bias: 0.0, accel_bias: 0.0, dropout: 0.0, delay: 0, dt: 0.005, action_noise: 0.0 }
    }
    fn resolve(cfg: &EnvConfig) -> Tune {
        let i = cfg.task.index();
        let v = |k: &str| -> f64 {
            cfg.vehicle.iter().find(|(n, _)| n == k).map(|x| x.1).unwrap_or_else(|| param_def(k).map(|d| d.defaults[i]).unwrap_or(0.0))
        };
        let h = v("cg_height");
        let ordered = |a: f64, b: f64| (a.min(b), a.max(b));
        Tune { throttle_min: v("throttle_min"), misalign: v("thrust_misalign").to_radians(), cg_offset: v("cg_offset"), drone_mass: v("drone_mass"), t2w: v("thrust_to_weight"),
            rover_vmax: v("rover_vmax"), tip_lat: 0.6 / h, tip_long: 0.8 / h, battery_rover: v("battery_kwh") * 3.6e6, bus_power: v("bus_power"), battery_lander: v("lander_battery_kwh") * 3.6e6,
            alt: ordered(v("start_alt_min"), v("start_alt_max")), offset: v("start_offset"), vh: v("start_vh"), vz: ordered(v("start_vz_min"), v("start_vz_max")), tilt: v("start_tilt").to_radians(),
            goal: ordered(v("goal_dist_min"), v("goal_dist_max")), pad: v("pad_radius"), terrain: v("terrain_scale"), craters: v("crater_scale"), friction: v("friction"), sun: v("sun_elev"),
            lim_v: v("touch_speed_limit"), lim_tilt: v("touch_tilt_limit"), lim_slope: v("touch_slope_limit"), impact: v("impact_limit"), rew_ok: v("reward_success"), rew_bad: v("reward_crash"),
            air: v("air_scale"), wind: Vec3::new(v("wind_x"), v("wind_y"), 0.0), solar: v("solar_scale"),
            n_pos: v("noise_pos"), n_vel: v("noise_vel"), n_att: v("noise_att"), n_gyro: v("noise_gyro"), n_acc: v("noise_accel"), n_rng: v("noise_range"),
            gyro_bias: v("gyro_bias"), accel_bias: v("accel_bias"), dropout: v("sensor_dropout"), delay: v("obs_delay") as usize, dt: v("physics_dt"), action_noise: v("action_noise") }
    }
}

impl EnvConfig {
    pub fn new(task: &str, physics: &str) -> Result<EnvConfig, String> {
        let task = match task {
            "lander" => Task::Lander,
            "drone" => Task::Drone,
            "rover" => Task::Rover,
            _ => return Err(format!("unknown task '{task}': lander, drone, rover")),
        };
        Ok(EnvConfig { task, world: WorldSpec::preset(physics), randomize: false, noise: 1.0, frame_skip: 4,
            max_steps: match task { Task::Lander => 5000, Task::Drone => 2500, Task::Rover => 4000 }, rotor_scale: 1.0,
            worlds: vec![], gust: 0.0, actuator_lag: 0.0, fault_prob: 0.0, vehicle: vec![], custom_terrain: None })
    }

    /// Set a registry parameter, checking its range and applicability to the task.
    pub fn set(&mut self, name: &str, value: f64) -> Result<(), String> {
        let d = param_def(name).ok_or_else(|| {
            let mut near: Vec<&str> = PARAMS.iter().filter(|d| d.tasks & self.task.bit() != 0).map(|d| d.name).collect();
            near.sort();
            format!("unknown parameter '{name}'. For {}: {}", self.task.name(), near.join(", "))
        })?;
        if d.tasks & self.task.bit() == 0 { return Err(format!("parameter '{name}' does not apply to task {}", self.task.name())); }
        if !value.is_finite() || value < d.min || value > d.max { return Err(format!("{name} = {value} out of range [{}, {}] {}", d.min, d.max, d.unit)); }
        match name {
            "noise" => self.noise = value,
            "rotor_scale" => self.rotor_scale = value,
            "gust" => self.gust = value,
            "actuator_lag" => self.actuator_lag = value,
            "fault_prob" => self.fault_prob = value,
            "frame_skip" => self.frame_skip = value as usize,
            "max_steps" => self.max_steps = value as usize,
            _ => { self.vehicle.retain(|(n, _)| n != name); self.vehicle.push((name.to_string(), value)); }
        }
        Ok(())
    }

    /// A list like `dry_mass=900,gust=2` from the CLI, API and Python.
    pub fn set_list(&mut self, list: &str) -> Result<(), String> {
        for item in list.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()) {
            let (k, v) = item.split_once('=').or_else(|| item.split_once(':')).ok_or(format!("'{item}': must look like name=value"))?;
            let x: f64 = v.trim().parse().map_err(|_| format!("{}: expected a number, got '{v}'", k.trim()))?;
            self.set(k.trim(), x)?;
        }
        Ok(())
    }

    /// Current parameter value, using the default for the task.
    pub fn get(&self, name: &str) -> Option<f64> {
        let d = param_def(name)?;
        Some(match name {
            "noise" => self.noise, "rotor_scale" => self.rotor_scale, "gust" => self.gust, "actuator_lag" => self.actuator_lag,
            "fault_prob" => self.fault_prob, "frame_skip" => self.frame_skip as f64, "max_steps" => self.max_steps as f64,
            _ => self.vehicle.iter().find(|(n, _)| n == name).map(|x| x.1).unwrap_or(d.defaults[self.task.index()]),
        })
    }
}

/// Default parameter value for the task.
pub fn param_default(task: Task, name: &str) -> Option<f64> { param_def(name).map(|d| d.defaults[task.index()]) }

impl EnvConfig {
    /// A custom map must fit the scenario: otherwise a flight ends at the map edge and the report blames the vehicle.
    fn check_terrain(&self) -> Result<(), String> {
        let Some(t) = &self.custom_terrain else { return Ok(()) };
        let half = t.size / 2.0;
        let need = match self.task {
            Task::Lander => self.get("start_offset").unwrap_or(0.0),
            _ => self.get("goal_dist_max").unwrap_or(0.0),
        };
        if need >= half {
            let k = if self.task == Task::Lander { "start_offset" } else { "goal_dist_max" };
            return Err(format!("map {:.0} m is too small for the scenario: {k} = {need} m, but the edge is at {half:.0} m. Reduce {k}", t.size));
        }
        Ok(())
    }

    fn apply_gravity(&mut self) -> Result<(), String> {
        // custom gravity is absolute, while the world stores a scale
        let g = self.vehicle.iter().find(|(n, _)| n == "gravity").map(|x| x.1).unwrap_or(0.0);
        if g > 0.0 {
            let scale = |spec: &mut WorldSpec| -> Result<(), String> { let now = World::build(spec)?.g_surface(); spec.gravity_scale *= g / now; Ok(()) };
            scale(&mut self.world)?;
            for w in &mut self.worlds { scale(w)?; }
            self.vehicle.retain(|(n, _)| n != "gravity");
        }
        Ok(())
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
    // drone
    arm: f64, rotor_area: f64, k_yaw: f64, battery_j: f64,
    // rover
    track: f64, crr: f64, motor_force: f64, panel_area: f64, rover_mass: f64,
}

#[derive(Clone, Debug, Default)]
pub struct StepInfo {
    pub success: bool,
    pub crashed: bool,
    pub reason: String,
    pub fuel_used: f64,
    pub energy_used: f64,
    pub touch_speed: f64,  // speed at first touch, m/s
    pub touch_tilt: f64,   // tilt at first touch, °
    pub touch_offset: f64, // miss distance from the pad or goal, m
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
    pub last_lidar: Vec<(Vec3, Vec3)>, // (start, hit point) for visualization
    pub info: StepInfo,
    pub seed: u64,
    body: Body,
    p: Params,
    rng: Rng,
    battery: f64,
    heading: f64, // rover
    speed: f64,
    yaw_rate: f64,
    sun_elev: f64,
    prev_shaping: f64,
    base_world: World,
    hover_time: f64,
    pub world_index: usize,
    act_state: [f64; 4],  // actuator position (including lag)
    act_gain: [f64; 4],   // hidden actuator degradation
    fault: Option<(usize, f64)>,
    gust_v: Vec3,
    dyn_rng: Rng,         // a separate stream for disturbances: sensor noise does not affect them
    in_contact: bool,
    settle_t: f64,
    ground_t: f64,
    touch_msg: String,
    touch_bonus: f64,
    touched: bool,
    tune: Tune,
    obs_queue: std::collections::VecDeque<Vec<f64>>,
    held: Vec<f64>,
    gyro_b: Vec3,
    accel_b: Vec3,
}

/// Names of hidden episode parameters (see `Env::params`): what the world model must infer on its own.
pub const PARAM_NAMES: &[&str] = &["g", "rho0", "p0_kpa", "t0_k", "friction", "dry_mass", "fuel_mass", "isp", "t_max", "torque_max", "rotor_area", "rover_mass",
    "wind_x", "wind_y", "gust", "actuator_lag", "fault_actuator", "fault_gain", "world_index", "goal_x", "goal_y", "goal_z", "control_dt", "pad_radius", "gyro_bias_x", "gyro_bias_y", "gyro_bias_z"];

impl Env {
    pub fn new(mut cfg: EnvConfig) -> Result<Env, String> {
        cfg.apply_gravity()?;
        cfg.check_terrain()?;
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
            tune: Tune::zero(), obs_queue: Default::default(), held: vec![], gyro_b: Vec3::ZERO, accel_b: Vec3::ZERO,
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
            Some(a) => { let s = a.at(z.max(0.0)); (s.density * self.tune.air, s.pressure * self.tune.air, s.temperature) }
            None => (0.0, 0.0, self.world.mean_surface_temperature()),
        }
    }

    pub fn reset(&mut self, seed: u64) -> Vec<f64> {
        self.seed = seed;
        self.tune = Tune::resolve(&self.cfg);
        self.rng = Rng::new(seed);
        // world randomization: g factor ±20%, distance to the star
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
        self.dt = self.tune.dt;
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
        let pad = self.tune.pad;
        let size = match self.cfg.task { Task::Rover => 600.0, _ => 1200.0 };
        self.terrain = match &self.cfg.custom_terrain {
            Some(t) => t.clone(),
            None => Arc::new(Terrain::generate_tuned(&self.world, seed, size, 161, pad, self.tune.terrain, self.tune.craters, self.tune.friction)),
        };
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
        if rho > 0.0 { self.wind += self.tune.wind; }
        let h0 = self.terrain.height(0.0, 0.0);
        match self.cfg.task {
            Task::Lander => {
                let mut rng = Rng::new(seed ^ 77);
                let t = self.tune.clone();
                self.battery = if t.bus_power > 0.0 { t.battery_lander } else { 0.0 };
                let alt = rng.uniform(t.alt.0, t.alt.1);
                let off = Vec3::new(rng.uniform(-t.offset, t.offset), rng.uniform(-t.offset, t.offset), 0.0);
                let tilt = Quat::from_axis_angle(Vec3::new(rng.normal(), rng.normal(), 0.0), rng.uniform(0.0, t.tilt));
                let m = self.p.dry_mass + self.p.fuel;
                self.body = Body { pos: Vec3::new(off.x, off.y, h0 + alt), vel: Vec3::new(rng.uniform(-t.vh, t.vh), rng.uniform(-t.vh, t.vh), -rng.uniform(t.vz.0, t.vz.1)),
                    q: tilt, w: Vec3::new(rng.normal(), rng.normal(), rng.normal()) * 0.02, mass: m,
                    inertia: Vec3::new(m * (3.0 * 1.5f64.powi(2) + 3.0f64.powi(2)) / 12.0, m * (3.0 * 1.5f64.powi(2) + 3.0f64.powi(2)) / 12.0, m * 1.5f64.powi(2) / 2.0) };
                self.goal = Vec3::new(0.0, 0.0, h0);
            }
            Task::Drone => {
                let mut rng = Rng::new(seed ^ 99);
                let m = self.tune.drone_mass;
                let ang = rng.uniform(0.0, 2.0 * PI);
                let d = rng.uniform(self.tune.goal.0, self.tune.goal.1);
                let (gx, gy) = (d * ang.cos(), d * ang.sin());
                self.goal = Vec3::new(gx, gy, self.terrain.height(gx, gy) + rng.uniform(5.0, 20.0));
                // start in the air: otherwise without an autopilot an episode ends in a fraction of a second and yields no data
                self.body = Body { pos: Vec3::new(0.0, 0.0, h0 + rng.uniform(self.tune.alt.0, self.tune.alt.1)), vel: Vec3::ZERO, q: Quat::ID, w: Vec3::ZERO, mass: m, inertia: Vec3::new(0.02, 0.02, 0.04) * (m / 1.5) };
                self.battery = self.p.battery_j;
            }
            Task::Rover => {
                let mut rng = Rng::new(seed ^ 55);
                let ang = rng.uniform(0.0, 2.0 * PI);
                let d = rng.uniform(self.tune.goal.0, self.tune.goal.1);
                let (gx, gy) = (d * ang.cos(), d * ang.sin());
                self.goal = Vec3::new(gx, gy, self.terrain.height(gx, gy));
                self.heading = rng.uniform(-PI, PI);
                self.speed = 0.0;
                self.yaw_rate = 0.0;
                self.sun_elev = rng.uniform(0.15, 1.2);
                if self.tune.sun >= 0.0 { self.sun_elev = self.tune.sun.to_radians(); }
                self.battery = self.tune.battery_rover;
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
        let (gb, ab) = (self.tune.gyro_bias, self.tune.accel_bias);
        self.gyro_b = Vec3::new(self.dyn_rng.normal(), self.dyn_rng.normal(), self.dyn_rng.normal()) * gb;
        self.accel_b = Vec3::new(self.dyn_rng.normal(), self.dyn_rng.normal(), self.dyn_rng.normal()) * ab;
        self.prev_shaping = self.shaping();
        self.last_action = vec![0.0; n_act];
        self.held.clear();
        self.obs_queue.clear();
        self.last_obs = self.observe();
        self.last_obs.clone()
    }

    // ------------------------------------------------------------ sensors

    fn n(&mut self, sigma: f64) -> f64 { sigma * self.cfg.noise * self.rng.normal() }

    fn observe(&mut self) -> Vec<f64> {
        let b = self.body.clone();
        let t = self.tune.clone();
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
                    rel.x + self.n(1.0 * t.n_pos), rel.y + self.n(1.0 * t.n_pos), rel.z + self.n(1.0 * t.n_pos),
                    b.vel.x + self.n(0.1 * t.n_vel), b.vel.y + self.n(0.1 * t.n_vel), b.vel.z + self.n(0.1 * t.n_vel),
                    up_b.x + self.n(0.005 * t.n_att), up_b.y + self.n(0.005 * t.n_att), up_b.z + self.n(0.005 * t.n_att),
                    b.w.x + self.gyro_b.x + self.n(0.005 * t.n_gyro), b.w.y + self.gyro_b.y + self.n(0.005 * t.n_gyro), b.w.z + self.gyro_b.z + self.n(0.005 * t.n_gyro),
                    radar + self.n(0.3 * t.n_rng),
                    acc.x + self.accel_b.x + self.n(0.05 * t.n_acc), acc.y + self.accel_b.y + self.n(0.05 * t.n_acc), acc.z + self.accel_b.z + self.n(0.05 * t.n_acc),
                ];
                if self.cfg.task == Task::Lander {
                    o.push((b.mass - self.p.dry_mass) / self.p.fuel);
                    o.push(p / 1e3 * (1.0 + self.n(0.002)));
                    for k in 0..8 {
                        // cone of rays down and outward, 30° from the body vertical
                        let a = 2.0 * PI * k as f64 / 8.0;
                        let dir_b = Vec3::new(0.5 * a.cos(), 0.5 * a.sin(), -0.866);
                        let d = b.q.rotate(dir_b);
                        let r = self.terrain.raycast(b.pos, d, 1000.0);
                        self.last_lidar.push((b.pos, b.pos + d.unit() * r));
                        o.push(r + self.n(0.2 * t.n_rng));
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
                    rel.x + self.n(0.5 * t.n_pos), rel.y + self.n(0.5 * t.n_pos), c + self.n(0.01 * t.n_att), s + self.n(0.01 * t.n_att),
                    self.speed + self.n(0.02 * t.n_vel), self.yaw_rate + self.gyro_b.z + self.n(0.005 * t.n_gyro), pitch + self.n(0.005 * t.n_att), roll + self.n(0.005 * t.n_att),
                    self.battery / t.battery_rover, self.sun_elev, temp + self.n(0.5), p / 1e3,
                ];
                let eye = b.pos + Vec3::new(0.0, 0.0, 1.0);
                let mut hits = Vec::with_capacity(16);
                for k in 0..16 {
                    let a = self.heading + 2.0 * PI * k as f64 / 16.0;
                    let d = Vec3::new(a.cos(), a.sin(), -0.06);
                    let r = self.terrain.raycast(eye, d, 40.0);
                    self.last_lidar.push((eye, eye + d.unit() * r));
                    o.push(r + self.n(0.05 * t.n_rng));
                    hits.push(if r < 40.0 { 1.0 } else { 0.0 });
                }
                o.extend(hits);
                o
            }
        }
    }

    /// The accelerometer measures specific force (everything except gravity), in body axes.
    fn specific_force(&self) -> Vec3 {
        let eff = self.effective_action();
        let f = self.forces(&self.body, &eff).0;
        self.body.q.inv_rotate(f / self.body.mass + Vec3::new(0.0, 0.0, self.g()))
    }

    /// Depth camera w×h, looking along the body axis (lander and drone: down and forward, rover: forward).
    pub fn depth_camera(&self, w: usize, h: usize, fov_deg: f64) -> Vec<f64> {
        let (origin, rays) = self.camera_rays(w, h, fov_deg);
        rays.iter().map(|d| self.terrain.raycast(origin, *d, 400.0)).collect()
    }

    /// Simple RGB render (w×h×3, by rows): Lambertian terrain, soil and sky color from the world, haze ∝ air density.
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

    // ------------------------------------------------------------ physics

    /// (force in the world frame, torque in body axes, mass flow, power)
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
                let cmd = a.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
                // the engine is either off or runs at or above minimum throttle
                let cmd = if cmd > 1e-3 { cmd.max(self.tune.throttle_min) } else { 0.0 };
                let thr = if fuel_left > 0.0 { cmd * self.p.t_max } else { 0.0 };
                let axis = Vec3::new(self.tune.misalign.sin(), 0.0, self.tune.misalign.cos());
                f += b.q.rotate(axis * thr);
                mdot = thr / (self.p.isp * G0); // Tsiolkovsky: ṁ = T / (Isp·g₀)
                let rcs = |k: usize| a.get(k).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
                tq = Vec3::new(rcs(1), rcs(2), rcs(3)) * self.p.torque_max;
                // the center of mass is off the engine axis: thrust creates a constant torque
                tq += Vec3::new(self.tune.cg_offset, 0.0, 0.0).cross(axis * thr);
                f += v_rel * (-0.5 * rho * self.p.cd * self.p.area * v_rel.norm());
                f += Vec3::new(0.0, 0.0, rho * self.p.volume * g); // Archimedes
            }
            Task::Drone => {
                // momentum theory: T_max ∝ ρ·A (at fixed blade tip speed), P_ind = T^{3/2}/√(2ρA)
                let a_rot = self.p.rotor_area / 4.0;
                let t_max_each = self.tune.t2w * self.tune.drone_mass * G0 / 4.0 * (rho / 1.225) * (a_rot / (PI * 0.12f64.powi(2)));
                let powered = self.battery > 0.0;
                let mut ts = [0.0; 4];
                for i in 0..4 {
                    ts[i] = if powered { a.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0) * t_max_each } else { 0.0 };
                    if rho > 0.0 && ts[i] > 0.0 { power += ts[i].powf(1.5) / (2.0 * rho * a_rot).sqrt() / 0.7; }
                }
                let total: f64 = ts.iter().sum();
                f += b.q.rotate(Vec3::new(0.0, 0.0, total));
                // rotors: 1 (+x), 2 (+y), 3 (−x), 4 (−y); 1,3 clockwise, 2,4 counterclockwise
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

    /// Legs: four legs as spring-dampers with Coulomb friction. Without this the model never sees contact.
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

    /// Parameters from the vehicle description; episode randomization does not apply to them.
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
                "rolling_resistance" => p.crr = *v,
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

    /// Command including actuator degradation.
    fn effective_action(&self) -> [f64; 4] {
        let mut e = [0.0; 4];
        for i in 0..4 { e[i] = self.act_state[i] * self.act_gain[i]; }
        e
    }

    /// Actuators: first-order lag (τ = actuator_lag) per physics step.
    fn update_actuators(&mut self, cmd: &[f64]) {
        let k = if self.cfg.actuator_lag > 0.0 { 1.0 - (-self.dt / self.cfg.actuator_lag).exp() } else { 1.0 };
        for (i, c) in cmd.iter().enumerate().take(4) { self.act_state[i] += (c - self.act_state[i]) * k; }
    }

    /// Gusts: Ornstein–Uhlenbeck process with stationary standard deviation `gust` and τ = 2 s.
    fn update_gust(&mut self, dt: f64) {
        if self.cfg.gust <= 0.0 || self.air(self.body.pos.z).0 <= 0.0 { return; }
        let tau = 2.0;
        let s = self.cfg.gust * (2.0 * dt / tau).sqrt();
        let n = Vec3::new(self.dyn_rng.normal(), self.dyn_rng.normal(), 0.3 * self.dyn_rng.normal());
        self.gust_v = self.gust_v * (1.0 - dt / tau) + n * s;
    }

    fn physics_step(&mut self, cmd: &[f64]) {
        self.update_actuators(cmd);
        let mut eff = self.effective_action();
        // an empty lander battery leaves the engine and thrusters without control
        if self.cfg.task == Task::Lander && self.tune.bus_power > 0.0 && self.battery <= 0.0 { for x in eff.iter_mut() { *x = 0.0; } }
        let a: &[f64] = &eff[..cmd.len().min(4)];
        let dt = self.dt;
        match self.cfg.task {
            Task::Lander | Task::Drone => {
                let (f, tq, mdot, power) = self.forces(&self.body, a);
                let b = &mut self.body;
                // semi-implicit Euler (symplectic for translation)
                b.vel += f * (dt / b.mass);
                b.pos += b.vel * dt;
                // Euler equations for rotation: I ω̇ = τ − ω × Iω
                let iw = Vec3::new(b.inertia.x * b.w.x, b.inertia.y * b.w.y, b.inertia.z * b.w.z);
                let wdot = tq - b.w.cross(iw);
                b.w += Vec3::new(wdot.x / b.inertia.x, wdot.y / b.inertia.y, wdot.z / b.inertia.z) * dt;
                b.q = b.q.integrate(b.w, dt);
                if self.cfg.task == Task::Lander {
                    let dm = (mdot * dt).min(b.mass - self.p.dry_mass);
                    b.mass -= dm;
                    self.info.fuel_used += dm;
                    self.draw_energy(self.tune.bus_power * dt);
                } else {
                    self.draw_energy((power + self.tune.bus_power) * dt);
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
                    // motor characteristic: force falls with speed to zero at v_max (back EMF)
                    let along = if cmd * self.speed > 0.0 { (1.0 - self.speed.abs() / self.tune.rover_vmax).max(0.0) } else { 1.0 };
                    // traction: |F| ≤ μ·N/2 (Coulomb law), otherwise the wheels slip
                    side[k] = (cmd * along * self.p.motor_force / 2.0).clamp(-self.terrain.friction * n_force / 2.0, self.terrain.friction * n_force / 2.0);
                }
                let f_grav = -m * g * slope_along * cos_t; // gravity component along the slope
                let f_rr = -self.p.crr * n_force * self.speed.signum() * (self.speed.abs() / 0.05).min(1.0);
                let f_drag = -0.5 * rho * 1.0 * 1.2 * self.speed * self.speed.abs();
                // on a slope steeper than the friction angle the rover slides even with locked wheels
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
                let bus = if self.battery > 0.0 { self.tune.bus_power } else { 0.0 };
                let p_solar = self.solar_power();
                self.battery = (self.battery + (p_solar - p_mech - bus) * dt).clamp(0.0, self.tune.battery_rover);
                self.info.energy_used += (p_mech + bus) * dt;
            }
        }
        self.t += dt;
    }

    /// Takes energy from the battery; an empty battery gives nothing, so used + left stays equal to the initial charge.
    fn draw_energy(&mut self, e: f64) {
        let used = e.min(self.battery);
        self.battery -= used;
        self.info.energy_used += used;
    }

    /// Solar panel power on the rover, W.
    fn solar_power(&self) -> f64 {
        let (rho, _, _) = self.air(self.body.pos.z);
        self.p.panel_area * 0.25 * self.tune.solar * self.world.insolation() * self.sun_elev.sin() * if rho > 0.0 { 0.75 } else { 1.0 }
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

    /// Environment step: action → (observation, reward, terminated, truncated).
    pub fn step(&mut self, action: &[f64]) -> (Vec<f64>, f64, bool, bool) {
        if self.done { return (self.last_obs.clone(), 0.0, true, false); }
        let (lo, hi) = self.act_bounds();
        let mut a: Vec<f64> = action.iter().enumerate().map(|(i, x)| if x.is_finite() { x.clamp(lo[i], hi[i]) } else { 0.0 }).collect();
        if self.tune.action_noise > 0.0 {
            for i in 0..a.len() { a[i] = (a[i] + self.tune.action_noise * (hi[i] - lo[i]) * self.dyn_rng.normal()).clamp(lo[i], hi[i]); }
        }
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
            Task::Lander => {
                reward -= 0.02 * (self.info.fuel_used - fuel0);
                if self.info.crashed && self.tune.bus_power > 0.0 && self.battery <= 0.0 { self.info.reason = "battery depleted, then crash".into(); }
            }
            Task::Drone => {
                reward -= 1e-4 * (self.info.energy_used - en0);
                self.info.touch_offset = (self.goal - self.body.pos).norm();
                if (self.goal - self.body.pos).norm() < 2.0 && self.body.vel.norm() < 1.0 {
                    reward += 0.5;
                    self.hover_time += self.control_dt();
                    if self.hover_time > 3.0 && !terminated {
                        reward += self.tune.rew_ok; terminated = true; self.info.success = true;
                        self.info.reason = format!("hovered at the goal for 3 s, used {:.1} kJ", self.info.energy_used / 1e3);
                    }
                } else { self.hover_time = 0.0; }
                if self.battery <= 0.0 && !terminated { self.info.reason = "battery depleted".into(); }
                if self.in_contact { self.ground_t += self.control_dt(); } else { self.ground_t = 0.0; }
                if self.ground_t > 3.0 && !terminated {
                    reward -= 0.2 * self.tune.rew_bad; terminated = true; self.info.crashed = true;
                    self.info.reason = "did not take off: 3 s on the ground".into();
                }
            }
            Task::Rover => {
                reward -= 2e-5 * (self.info.energy_used - en0);
                let d = Vec3::new(self.goal.x - self.body.pos.x, self.goal.y - self.body.pos.y, 0.0).norm();
                self.info.touch_offset = d;
                if d < 3.0 { reward += self.tune.rew_ok; terminated = true; self.info.success = true; self.info.reason = "goal reached".into(); }
                let (gx, gy) = self.terrain.gradient(self.body.pos.x, self.body.pos.y);
                let (c, s) = (self.heading.cos(), self.heading.sin());
                let cos_t = 1.0 / (1.0 + gx * gx + gy * gy).sqrt();
                let g_n = (self.world.g_at(0.0) * cos_t).max(1e-6);
                // tip-over: slope tangent + turning inertia against the stability margin (track/2h, wheelbase/2h)
                let lateral = -gx * s + gy * c + self.speed * self.yaw_rate / g_n;
                let longitudinal = gx * c + gy * s + (self.speed - speed0) / self.control_dt() / g_n;
                if !terminated && (lateral.abs() > self.tune.tip_lat || longitudinal.abs() > self.tune.tip_long) {
                    reward -= self.tune.rew_bad; terminated = true; self.info.crashed = true; self.info.reason = "tipped over on a slope".into();
                }
                if self.battery <= 0.0 && (self.sun_elev < 0.05 || self.tune.bus_power > self.solar_power()) { terminated = true; self.info.reason = "battery depleted".into(); }
            }
        }
        let half = self.terrain.size / 2.0;
        if !terminated && (self.body.pos.x.abs() > half || self.body.pos.y.abs() > half || self.body.pos.z > self.terrain.height(0.0, 0.0) + 50_000.0) {
            terminated = true;
            reward -= 50.0;
            self.info.reason = "left the map".into();
        }
        let truncated = !terminated && self.steps >= self.cfg.max_steps;
        if truncated { self.info.reason = "time limit".into(); }
        self.done = terminated || truncated;
        self.ep_return += reward;
        let raw = self.observe();
        (self.delayed(raw), reward, terminated, truncated)
    }

    /// Reading dropouts and delay: the agent may see stale data.
    fn delayed(&mut self, mut raw: Vec<f64>) -> Vec<f64> {
        if self.tune.dropout > 0.0 && self.held.len() == raw.len() {
            for i in 0..raw.len() { if self.dyn_rng.f64() < self.tune.dropout { raw[i] = self.held[i]; } }
        }
        self.held = raw.clone();
        let out = if self.tune.delay > 0 {
            self.obs_queue.push_back(raw);
            if self.obs_queue.len() > self.tune.delay + 1 { self.obs_queue.pop_front(); }
            self.obs_queue.front().cloned().unwrap_or_default()
        } else { raw };
        self.last_obs = out.clone();
        out
    }

    /// Surface contact: (reward, message, success) if the episode ended here.
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
                    if !(v < self.tune.lim_v && tilt < self.tune.lim_tilt && slope < self.tune.lim_slope) {
                        if !self.touched {
                            self.info.touch_speed = v;
                            self.info.touch_tilt = tilt;
                            self.info.touch_offset = ((b.pos.x - self.goal.x).powi(2) + (b.pos.y - self.goal.y).powi(2)).sqrt();
                        }
                        return Some((-self.tune.rew_bad, format!("crash: {v:.1} m/s, tilt {tilt:.0}°, slope {slope:.0}°"), false));
                    }
                    if !self.touched {
                        let off = ((b.pos.x - self.goal.x).powi(2) + (b.pos.y - self.goal.y).powi(2)).sqrt();
                        self.touch_bonus = self.tune.rew_ok * if off < self.tune.pad { 1.0 } else { 0.4 };
                        self.touch_msg = format!("soft landing: {v:.2} m/s, tilt {tilt:.1}°, {off:.0} m from the pad center");
                        self.info.touch_speed = v;
                        self.info.touch_tilt = tilt;
                        self.info.touch_offset = off;
                    }
                    self.touched = true;
                }
                self.in_contact = touching;
                if !touching { self.settle_t = 0.0; return None; }
                if tilt > 35.0 { return Some((-self.tune.rew_bad, format!("tipped over after touchdown: tilt {tilt:.0}°"), false)); }
                if v < 0.2 && self.body.w.norm() < 0.15 { self.settle_t += self.dt; } else { self.settle_t = 0.0; }
                if self.settle_t > 0.5 { return Some((self.touch_bonus, self.touch_msg.clone(), true)); }
                None
            }
            Task::Drone => {
                let b = &self.body;
                let ground = self.terrain.height(b.pos.x, b.pos.y) + 0.05;
                if b.pos.z > ground { self.in_contact = false; return None; }
                let v = b.vel.norm();
                if !self.in_contact && v > self.tune.impact { return Some((-self.tune.rew_bad, format!("ground impact at {v:.1} m/s"), false)); }
                let tilt = b.q.rotate(Vec3::new(0.0, 0.0, 1.0)).z.clamp(-1.0, 1.0).acos().to_degrees();
                if tilt > 60.0 { return Some((-self.tune.rew_bad, format!("tipped over on the ground: tilt {tilt:.0}°"), false)); }
                // inelastic contact: the drone rests on the ground and can take off again
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

    // ------------------------------------------------------------ for people and language models

    /// Text description of the sensors: input for an LLM or VLM agent.
    pub fn describe(&self) -> String {
        let b = &self.body;
        let (rho, p, t) = self.air(b.pos.z);
        let wname = &self.world.physics.name;
        match self.cfg.task {
            Task::Lander => {
                let d = b.pos - self.goal;
                let tilt = b.q.rotate(Vec3::new(0.0, 0.0, 1.0)).z.clamp(-1.0, 1.0).acos().to_degrees();
                format!("World: {wname}, g = {:.2} m/s², pressure {:.0} Pa. Lander at altitude {:.0} m above the pad, offset {:.0} m east and {:.0} m north. \
Speed: down {:.1} m/s, horizontal {:.1} m/s. Tilt from vertical {:.1}°. Fuel {:.0}%. Radar altimeter: {:.0} m to the ground. \
Actions: throttle 0…1 (max thrust {:.0} kN), torque_x/y/z −1…1.",
                    self.g(), p, d.z, d.x, d.y, -b.vel.z, (b.vel.x.powi(2) + b.vel.y.powi(2)).sqrt(), tilt,
                    100.0 * (b.mass - self.p.dry_mass) / self.p.fuel, self.last_obs.get(12).copied().unwrap_or(0.0), self.p.t_max / 1e3)
            }
            Task::Drone => {
                let d = self.goal - b.pos;
                format!("World: {wname}, air density {rho:.4} kg/m³, g = {:.2} m/s². Goal: {:.1} m east, {:.1} m north, {:.1} m up. \
Speed {:.1} m/s, battery {:.0}%. Actions: 4 rotors 0…1.",
                    self.g(), d.x, d.y, d.z, b.vel.norm(), 100.0 * self.battery / self.p.battery_j)
            }
            Task::Rover => {
                let d = self.goal - b.pos;
                let ahead = self.last_obs.get(12).copied().unwrap_or(40.0);
                format!("World: {wname}, T = {t:.0} K. Goal at {:.0} m (east {:.0}, north {:.0}), heading {:.0}°. Speed {:.2} m/s, battery {:.0}%, \
Sun elevation {:.0}°. Lidar ahead: {:.1} m. Actions: left, right −1…1.",
                    (d.x * d.x + d.y * d.y).sqrt(), d.x, d.y, self.heading.to_degrees().rem_euclid(360.0), self.speed,
                    100.0 * self.battery / self.tune.battery_rover, self.sun_elev.to_degrees(), ahead)
            }
        }
    }

    /// Built-in autopilot (PD controller): a baseline for comparison with trained agents.
    pub fn autopilot(&self) -> Vec<f64> { self.autopilot_nav(Vec3::ZERO, Vec3::ZERO) }

    /// Autopilot with navigation error: shows how sensor quality affects the landing.
    /// `pos_err` and `vel_err` — how far the estimate is from the truth (attitude is assumed known).
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
                // height above terrain, not above the pad: otherwise the lander flies into a crater wall
                let alt = (b.pos.z - self.terrain.height(b.pos.x, b.pos.y)).min(d.z) - self.p.leg;
                let off = (d.x * d.x + d.y * d.y).sqrt();
                // desired horizontal speed toward the pad and vertical speed from altitude
                let vh_target = Vec3::new(-d.x, -d.y, 0.0) * 0.12;
                let vh_target = if vh_target.norm() > 15.0 { vh_target.unit() * 15.0 } else { vh_target };
                // the thrust margin sets how fast it can descend and still brake in time
                // the autopilot does not know about actuator weakening, like the agent: it uses rated thrust
                let brake = (self.p.t_max / b.mass - g).max(0.3);
                let vz_stop = (1.4 * brake * alt.max(0.0)).sqrt();
                // until over the pad do not descend below 40 m: only level ground is safe to land on
                let vz_target = if off > self.tune.pad * 0.48 && alt < 40.0 { (40.0 - alt) * 0.3 } else { -(1.0 + 0.12 * alt).min(vz_stop + 1.0).min(50.0) };
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
                let tmax = self.tune.t2w * self.tune.drone_mass * G0 / 4.0 * (rho / 1.225) * (a_rot / (PI * 0.12f64.powi(2)));
                if tmax <= 0.0 { return vec![0.0; 4]; }
                let base = acc_des.norm() * b.mass / 4.0 / tmax;
                let tx = 0.6 * err_b.x - 0.15 * b.w.x;
                let ty = 0.6 * err_b.y - 0.15 * b.w.y;
                let tz = -0.05 * b.w.z;
                let l = self.p.arm;
                // torque allocation to rotors (inverse of the mixing matrix)
                let dx = tx / (2.0 * l) / tmax;
                let dy = ty / (2.0 * l) / tmax;
                let dz = tz / (4.0 * self.p.k_yaw) / tmax;
                vec![(base - dy - dz).clamp(0.0, 1.0), (base + dx + dz).clamp(0.0, 1.0), (base + dy - dz).clamp(0.0, 1.0), (base - dx + dz).clamp(0.0, 1.0)]
            }
            Task::Rover => {
                let d = self.goal - b.pos;
                let want = d.y.atan2(d.x);
                // hazard map: among headings near the goal pick the closest one where the slope 3–9 m ahead is safe
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
                // turning is limited by the stability margin: v·ω ≤ 0.25·g, which matters on the Moon
                let yaw_lim = 0.25 * g / self.speed.abs().max(0.3);
                let yaw_des = (1.5 * err).clamp(-yaw_lim, yaw_lim);
                let turn = (2.0 * (yaw_des - self.yaw_rate)).clamp(-1.0, 1.0);
                // speed limit from the stability margin: v·ω ≤ 0.3·g
                let (hx, hy) = self.terrain.gradient(b.pos.x, b.pos.y);
                let v_base = 1.5 * (1.0 - (hx * hx + hy * hy).sqrt() / 0.9).clamp(0.3, 1.0);
                let v_max = if self.yaw_rate.abs() > 1e-3 { (0.3 * g / self.yaw_rate.abs()).min(v_base) } else { v_base };
                let spd = if self.speed > v_max { 0.0 } else { fwd };
                vec![(spd - turn).clamp(-1.0, 1.0), (spd + turn).clamp(-1.0, 1.0)]
            }
        }
    }

    /// Pose for visualization: position, quaternion, rover heading.
    pub fn pose(&self) -> (Vec3, Quat) {
        match self.cfg.task {
            Task::Rover => (self.body.pos, Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), self.heading)),
            _ => (self.body.pos, self.body.q),
        }
    }
    pub fn mass(&self) -> f64 { self.body.mass }
    pub fn air_density(&self) -> f64 { self.air(self.body.pos.z).0 }

    /// For checks: override terrain and state.
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
    pub fn battery_frac(&self) -> f64 { match self.cfg.task { Task::Drone => self.battery / self.p.battery_j, Task::Rover => self.battery / self.tune.battery_rover, _ => (self.body.mass - self.p.dry_mass) / self.p.fuel } }
    pub fn isp(&self) -> f64 { self.p.isp }
    pub fn dry_mass(&self) -> f64 { self.p.dry_mass }
    pub fn wind(&self) -> Vec3 { self.wind }
    pub fn rover_speed(&self) -> f64 { self.speed }
    pub fn gust_velocity(&self) -> Vec3 { self.gust_v }
    pub fn applied_action(&self) -> [f64; 4] { self.act_state }
    pub fn in_contact(&self) -> bool { self.in_contact }
    /// Truth for comparison with the GPS observation: relative position and velocity without noise.
    pub fn true_rel_pos(&self) -> Vec3 { if self.cfg.task == Task::Lander { self.body.pos - self.goal } else { self.goal - self.body.pos } }
    pub fn angular_velocity(&self) -> Vec3 { self.body.w }

    /// Names of the true state (see `state`): no noise, what the agent cannot see directly.
    pub fn state_names(&self) -> Vec<&'static str> {
        match self.cfg.task {
            Task::Rover => vec!["x", "y", "z", "heading", "speed", "yaw_rate", "battery_j", "sun_elev", "rho", "g"],
            _ => vec!["x", "y", "z", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz", "mass", "battery_j",
                "air_vx", "air_vy", "air_vz", "rho", "g", "contact", "act_0", "act_1", "act_2", "act_3"],
        }
    }

    /// True state at this moment.
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

    /// Hidden episode parameters, in the same order as `PARAM_NAMES`.
    pub fn params(&self) -> Vec<f64> {
        let p = &self.p;
        let (rho, pr, t) = self.air(0.0);
        let (fa, fg) = self.fault.map(|(i, g)| (i as f64, g)).unwrap_or((-1.0, 1.0));
        vec![self.world.g_at(0.0), rho, pr / 1e3, t, self.terrain.friction, p.dry_mass, p.fuel, p.isp, p.t_max, p.torque_max, p.rotor_area, p.rover_mass,
            self.wind.x, self.wind.y, self.cfg.gust, self.cfg.actuator_lag, fa, fg, self.world_index as f64, self.goal.x, self.goal.y, self.goal.z, self.control_dt(), self.tune.pad, self.gyro_b.x, self.gyro_b.y, self.gyro_b.z]
    }
}
