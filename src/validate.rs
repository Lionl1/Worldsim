//! Набор проверок: каждая сравнивает симуляцию с аналитическим решением,
//! законом сохранения или наблюдаемым значением. `worldsim validate` печатает таблицу.

use crate::ballistics::{self, Launch, Options, Projectile};
use crate::constants::{C, DAY, YEAR};
use crate::magneto::{self, ParticleLaunch};
use crate::nbody::{Body, NBody};
use crate::orbit::{elements_to_state, state_to_elements};
use crate::presets;
use crate::thermal::{self, ThermalOptions};
use crate::vec3::Vec3;
use crate::world::World;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Check {
    pub law: &'static str,
    pub test: String,
    pub value: String,
    pub expected: String,
    pub pass: bool,
}

fn rel(a: f64, b: f64) -> f64 { ((a - b) / b).abs() }

fn chk(law: &'static str, test: &str, value: f64, expected: f64, tol_rel: f64, unit: &str) -> Check {
    Check { law, test: test.into(), value: format!("{value:.6} {unit}"), expected: format!("{expected:.6} {unit} ±{:.2}%", tol_rel * 100.0), pass: rel(value, expected) <= tol_rel }
}
fn chk_max(law: &'static str, test: &str, value: f64, max: f64) -> Check {
    Check { law, test: test.into(), value: format!("{value:.3e}"), expected: format!("< {max:.0e}"), pass: value.abs() < max }
}

pub fn conservation_solar_system(years: f64) -> Vec<Check> {
    let mut s = NBody::from_presets(&["sun", "mercury", "venus", "earth", "moon", "mars", "jupiter"]).unwrap();
    let i0 = s.invariants();
    let dt = 0.05 * DAY;
    let steps = (years * YEAR / dt) as usize;
    let (mut de, mut dl): (f64, f64) = (0.0, 0.0);
    for _ in 0..steps {
        s.step(dt);
        let i = s.invariants();
        de = de.max(rel(i.energy, i0.energy));
        dl = dl.max((i.angular_momentum - i0.angular_momentum).norm() / i0.angular_momentum.norm());
    }
    let i = s.invariants();
    let dp = (i.momentum - i0.momentum).norm() / s.momentum_scale();
    // Луна должна остаться на орбите Земли
    let (e, m) = (s.index("earth").unwrap(), s.index("moon").unwrap());
    let d_em = (s.bodies[m].r - s.bodies[e].r).norm();
    vec![
        chk_max("Сохранение энергии", &format!("Солнечная система, {years} г, max |ΔE/E|"), de, 1e-9),
        chk_max("Сохранение импульса", "|ΔP| / Σ|p_i|", dp, 1e-12),
        chk_max("Сохранение момента импульса", "max |ΔL|/|L|", dl, 1e-12),
        Check { law: "Тяготение (иерархия)", test: "Луна осталась на орбите Земли".into(), value: format!("{:.0} км", d_em / 1e3), expected: "356 000–407 000 км".into(), pass: (3.5e8..4.1e8).contains(&d_em) },
    ]
}

pub fn kepler() -> Vec<Check> {
    let earth = presets::physics("earth").unwrap();
    let a = earth.radius + 400e3;
    let mu = earth.gm;
    let (r, v) = elements_to_state(mu, a, 0.1, 51.6, 30.0, 40.0, 0.0);
    let mut s = NBody::new(vec![
        Body { name: "earth".into(), gm: mu, r: Vec3::ZERO, v: Vec3::ZERO },
        Body { name: "sat".into(), gm: 1e-6, r, v },
    ]);
    let t_theory = 2.0 * PI * (a.powi(3) / mu).sqrt();
    let n = 2000;
    for _ in 0..n { s.step(t_theory / n as f64); }
    let back = (s.bodies[1].r - s.bodies[0].r - r).norm();
    vec![
        Check { law: "3-й закон Кеплера", test: "орбита e=0.1, h=400 км: возврат через T=2π√(a³/μ)".into(), value: format!("{back:.2e} м"), expected: "< 1 м".into(), pass: back < 1.0 },
    ]
}

/// Прецессия перигелия Меркурия от 1PN-поправки ОТО (≈ 43″ за столетие).
pub fn mercury_precession() -> Vec<Check> {
    let run = |gr: bool| -> f64 {
        let mut s = NBody::from_presets(&["sun", "mercury"]).unwrap();
        if gr { s.enable_relativity("sun").unwrap(); }
        let p0 = state_to_elements(s.bodies[0].gm + s.bodies[1].gm, s.bodies[1].r - s.bodies[0].r, s.bodies[1].v - s.bodies[0].v).periapsis_dir;
        // целое число оборотов — короткопериодические члены 1PN сокращаются
        let mu = s.bodies[0].gm + s.bodies[1].gm;
        let a = state_to_elements(mu, s.bodies[1].r - s.bodies[0].r, s.bodies[1].v - s.bodies[0].v).a;
        let period = 2.0 * PI * (a.powi(3) / mu).sqrt();
        let orbits = 10usize;
        let per = 8000usize;
        let dt = period / per as f64;
        for _ in 0..orbits * per { s.step(dt); }
        let years = orbits as f64 * period / YEAR;
        let el = state_to_elements(s.bodies[0].gm + s.bodies[1].gm, s.bodies[1].r - s.bodies[0].r, s.bodies[1].v - s.bodies[0].v);
        let ang = p0.cross(el.periapsis_dir).z.atan2(p0.dot(el.periapsis_dir));
        ang / years * 100.0 // рад/столетие
    };
    let d = (run(true) - run(false)).to_degrees() * 3600.0;
    let m = presets::physics("mercury").unwrap();
    let sun = presets::physics("sun").unwrap();
    let o = m.orbit.unwrap();
    let per_orbit = 6.0 * PI * sun.gm / (C * C * o.a * (1.0 - o.e * o.e));
    let period = 2.0 * PI * (o.a.powi(3) / sun.gm).sqrt();
    let theory = per_orbit * (100.0 * YEAR / period) * 180.0 / PI * 3600.0;
    vec![chk("ОТО (1PN)", "прецессия перигелия Меркурия, ″/век", d, theory, 0.02, "″")]
}

pub fn atmosphere() -> Vec<Check> {
    let e = World::preset("earth").unwrap();
    let ea = e.atmosphere.as_ref().unwrap();
    let m = World::preset("mars").unwrap();
    let ma = m.atmosphere.as_ref().unwrap();
    let mut v = vec![
        chk("Гидростатика", "Земля: P0 = ∫ρg dz", ea.column_weight(e.physics.gm, e.physics.radius), 101_325.0, 0.005, "Па"),
        chk("Уравнение состояния ид. газа", "Земля: ρ у поверхности", ea.at(0.0).density, 1.225, 0.005, "кг/м³"),
        chk("Гидростатика + профиль T", "Земля: P на 11 км (US Std 1976)", ea.at(11_000.0).pressure, 22_632.0, 0.01, "Па"),
        chk("Скорость звука √(γRT/M)", "Земля, 0 м", ea.at(0.0).sound_speed, 340.29, 0.005, "м/с"),
        chk("Гидростатика", "Марс: шкала высот H=RT/Mg", ma.scale_height(m.g_surface()) / 1e3, 11.1, 0.05, "км"),
        chk("Гидростатика", "Марс: P0 = ∫ρg dz", ma.column_weight(m.physics.gm, m.physics.radius), 636.0, 0.005, "Па"),
    ];
    let comp = presets::oxides_to_elements(&presets::surface("earth").unwrap().oxides_wt).unwrap();
    let sum: f64 = comp.iter().map(|x| x.1).sum();
    let o = comp.iter().find(|x| x.0 == "O").unwrap().1;
    v.push(chk_max("Сохранение массы (стехиометрия)", "Σ элементов коры − 1", sum - 1.0, 1e-12));
    v.push(chk("Стехиометрия", "доля O в коре Земли", o * 100.0, 45.5, 0.04, "%"));
    v
}

pub fn temperatures() -> Vec<Check> {
    let e = World::preset("earth").unwrap();
    let m = World::preset("mars").unwrap();
    let mut v = vec![
        chk("Стефан–Больцман", "Земля: T_eq", e.equilibrium_temperature(), 254.0, 0.01, "К"),
        chk("Парниковый эффект (Эддингтон)", "Земля: средняя T поверхности", e.mean_surface_temperature(), 288.0, 0.01, "К"),
        chk("Стефан–Больцман", "Марс: T_eq", m.equilibrium_temperature(), 210.0, 0.02, "К"),
        chk("Парниковый эффект (Эддингтон)", "Венера: средняя T поверхности", World::preset("venus").unwrap().mean_surface_temperature(), 737.0, 0.02, "К"),
    ];
    let moon = World::preset("moon").unwrap();
    let r = thermal::simulate(&moon, ThermalOptions { spinup_days: 6, ..Default::default() });
    v.push(chk("Фурье + Стефан–Больцман", "Луна, экватор: T max (Diviner ≈ 390 К)", r.t_max, 390.0, 0.04, "К"));
    v.push(chk("Фурье + Стефан–Больцман", "Луна, экватор: T min (Diviner ≈ 95 К)", r.t_min, 95.0, 0.15, "К"));
    v.push(chk_max("1-е начало термодинамики", "Луна: невязка энергобаланса за сутки", r.energy_residual, 1e-6));
    let rm = thermal::simulate(&m, ThermalOptions { spinup_days: 10, ..Default::default() });
    v.push(Check { law: "Фурье + Стефан–Больцман", test: "Марс, экватор: суточный ход (Curiosity ~ 190–280 К)".into(),
        value: format!("{:.0}–{:.0} К", rm.t_min, rm.t_max), expected: "min 170–215, max 260–300".into(),
        pass: (170.0..215.0).contains(&rm.t_min) && (260.0..300.0).contains(&rm.t_max) });
    v.push(chk_max("1-е начало термодинамики", "Марс: невязка энергобаланса", rm.energy_residual, 1e-6));
    v
}

pub fn ballistics_checks() -> Vec<Check> {
    let mut v = Vec::new();
    // 1) вакуум, без вращения: дальность v²/g
    let moon = World::preset("moon").unwrap();
    let o = Options { rotating: false, j2: false, drag: false, buoyancy: false, sample_dt: 0.05, ..Default::default() };
    let mut moon_static = moon.clone();
    moon_static.physics.rotation_period = 1e30;
    let r = ballistics::simulate(&moon_static, Projectile::sphere(0.05, 2000.0),
        Launch { lat: 0.0, lon: 0.0, altitude: 0.0, speed: 10.0, elevation: 45.0, azimuth: 90.0 }, o);
    v.push(chk("2-й закон Ньютона (вакуум)", "Луна: дальность броска 10 м/с под 45° = v²/g", r.downrange, 100.0 / moon.g_surface(), 1e-4, "м"));
    // 2) Кориолис: отклонение к востоку при падении с 100 м на экваторе
    let earth = World::preset("earth").unwrap();
    let h = 100.0;
    let o2 = Options { j2: false, drag: false, buoyancy: false, sample_dt: 0.01, ..Default::default() };
    let r = ballistics::simulate(&earth, Projectile::sphere(0.05, 7800.0),
        Launch { lat: 0.0, lon: 0.0, altitude: h, speed: 0.0, elevation: -90.0, azimuth: 0.0 }, o2);
    let last = r.samples.last().unwrap();
    let g = earth.g_surface() - earth.omega().powi(2) * earth.physics.radius;
    let t = (2.0 * h / g).sqrt();
    let theory = earth.omega() * g * t.powi(3) / 3.0;
    let east = last.lon.to_radians().sin() * earth.physics.radius;
    v.push(chk("Сила Кориолиса", "Земля: отклонение к востоку при падении со 100 м", east * 1000.0, theory * 1000.0, 0.02, "мм"));
    // 3) баланс энергии с сопротивлением, J2 и вращением
    let r = ballistics::simulate(&earth, Projectile::sphere(0.1, 1000.0),
        Launch { lat: 55.75, lon: 37.6, altitude: 0.0, speed: 600.0, elevation: 50.0, azimuth: 30.0 }, Options::default());
    v.push(chk_max("1-е начало (работа сопротивления → тепло)", "Земля: |E−E0−W_nc| / масштаб", r.energy_error, 1e-8));
    v.push(Check { law: "Сопротивление среды", test: "в воздухе дальность меньше, чем в вакууме".into(),
        value: format!("{:.1} км", r.downrange / 1e3), expected: format!("< {:.1} км", 600f64.powi(2) / 9.81 / 1e3), pass: r.downrange < 600f64.powi(2) / 9.81 && r.dissipated_energy > 0.0 });
    // 4) Архимед: гелиевый шар всплывает
    let balloon = Projectile { mass: 0.5, area: PI, cd: 0.47, volume: 4.0 / 3.0 * PI };
    let r = ballistics::simulate(&earth, balloon, Launch { lat: 0.0, lon: 0.0, altitude: 1.0, speed: 0.0, elevation: 90.0, azimuth: 0.0 },
        Options { max_time: 120.0, ..Default::default() });
    v.push(Check { law: "Закон Архимеда", test: "шар 0.5 кг, V=4.19 м³ всплывает в атмосфере Земли".into(),
        value: format!("+{:.0} м за 120 с", r.max_altitude), expected: "> 100 м".into(), pass: r.max_altitude > 100.0 });
    v
}

pub fn magnetosphere() -> Vec<Check> {
    let earth = World::preset("earth").unwrap();
    let sp = magneto::species("proton").unwrap();
    let r = magneto::simulate(&earth, sp, ParticleLaunch { l_shell: 3.0, energy_mev: 1.0, pitch_angle: 40.0 }, 20.0, 100);
    vec![
        chk_max("Сила Лоренца + СТО (Борис)", "протон 1 МэВ, L=3: |ΔE_k|/E_k", r.energy_drift, 1e-10),
        Check { law: "Адиабатический инвариант μ", test: "зеркальная широта, α_eq=40°".into(),
            value: format!("{:.2}°", r.max_abs_lat), expected: format!("{:.2}° ±1.5°", r.mirror_lat_theory),
            pass: (r.max_abs_lat - r.mirror_lat_theory).abs() < 1.5 },
        Check { law: "Градиентно-кривизненный дрейф", test: "протон дрейфует на запад".into(),
            value: format!("{:.2}°", r.drift_longitude), expected: "< 0°".into(), pass: r.drift_longitude < 0.0 && r.trapped },
    ]
}

pub fn all() -> Vec<Check> {
    let mut v = Vec::new();
    v.extend(kepler());
    v.extend(conservation_solar_system(1.0));
    v.extend(mercury_precession());
    v.extend(atmosphere());
    v.extend(temperatures());
    v.extend(ballistics_checks());
    v.extend(magnetosphere());
    v
}
