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

pub fn stars_checks() -> Vec<Check> {
    use crate::stars;
    let sun = stars::star("sun").unwrap();
    let hz = sun.habitable_zone();
    let ms = stars::Star::main_sequence(1.0);
    vec![
        chk("Стефан–Больцман из спектра Планка", "π∫B_λdλ / σT⁴ при 5772 К", stars::planck_integrated(5772.0) / (crate::constants::SIGMA_SB * 5772f64.powi(4)), 1.0, 1e-6, ""),
        chk("Закон смещения Вина", "λ_max·T численно из Планка", stars::wien_peak(5772.0) * 5772.0 * 1e3, 2.897_771_955, 1e-4, "мм·К"),
        chk("Масса–светимость + Стефан–Больцман", "T_eff звезды 1 M☉ главной последовательности", ms.teff, 5772.0, 0.001, "К"),
        chk("Обитаемая зона (Kopparapu 2014)", "внутренняя граница у Солнца", hz.0, 0.95, 0.01, "а.е."),
        chk("Обитаемая зона (Kopparapu 2014)", "внешняя граница у Солнца", hz.1, 1.67, 0.01, "а.е."),
        Check { law: "Излучение чёрного тела → цвет", test: "Проксима красная, Сириус голубоватый".into(),
            value: format!("R/B: {:.2} / {:.2}", stars::star("proxima").unwrap().rgb()[0] / stars::star("proxima").unwrap().rgb()[2], stars::star("sirius_a").unwrap().rgb()[0] / stars::star("sirius_a").unwrap().rgb()[2]),
            expected: "> 1.5 / < 1".into(),
            pass: { let p = stars::star("proxima").unwrap().rgb(); let s = stars::star("sirius_a").unwrap().rgb(); p[0] / p[2] > 1.5 && s[0] / s[2] < 1.0 } },
    ]
}

pub fn starsystem_checks() -> Vec<Check> {
    use crate::starsystem::StarSystem;
    let mut worst: f64 = 0.0;
    let mut stable = 0;
    let n = 12;
    for seed in 0..n {
        let sys = StarSystem::generate(seed, None).unwrap();
        let mut nb = sys.nbody();
        let i0 = nb.invariants();
        let p_in = sys.planets[0].a.powf(1.5) / sys.star.mass.sqrt() * YEAR;
        let dt = p_in / 200.0;
        let a0: Vec<f64> = nb.bodies.iter().skip(1).map(|b| (b.r - nb.bodies[0].r).norm()).collect();
        for _ in 0..4000 { nb.step(dt); }
        let i1 = nb.invariants();
        worst = worst.max(rel(i1.energy, i0.energy));
        let ok = nb.bodies.iter().skip(1).zip(&a0).all(|(b, r0)| { let r = (b.r - nb.bodies[0].r).norm(); r < 3.0 * r0 && r > r0 / 3.0 });
        if ok { stable += 1; }
    }
    vec![
        chk_max("Сохранение энергии", "12 сгенерированных систем, 20 оборотов внутренней планеты", worst, 1e-6),
        Check { law: "Устойчивость по Хиллу", test: "разнос 10–25 R_H: планеты остаются на орбитах".into(), value: format!("{stable}/{n}"), expected: format!("{n}/{n}"), pass: stable == n },
    ]
}

pub fn galaxy_checks() -> Vec<Check> {
    use crate::galaxy;
    let mut g = galaxy::preset("plummer", 3000, 1).unwrap();
    let (k, w) = g.energy();
    let acc = g.tree_acc();
    let mut errs: Vec<f64> = (0..150).map(|i| { let d = g.direct_acc(i * 20); (acc[i * 20] - d).norm() / d.norm() }).collect();
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let e0 = k + w;
    let p_scale: f64 = (0..g.len()).map(|i| g.mass[i] * g.vel[i].norm()).sum();
    let tc = galaxy::crossing_time(1.0, 1.0);
    for _ in 0..100 { g.step(tc / 100.0); }
    let (k1, w1) = g.energy();
    let mw = galaxy::preset("milky_way", 20000, 2).unwrap();
    let rc = mw.rotation_curve(20.0, 10);
    let v8 = rc.iter().find(|(r, _)| *r > 7.0).map(|x| x.1).unwrap_or(0.0);
    vec![
        chk("Теорема вириала", "сфера Пламмера: 2K/|W|", 2.0 * k / w.abs(), 1.0, 0.03, ""),
        chk_max("Барнс–Хат (θ=0.6) vs прямая сумма", "медианная относительная ошибка силы", errs[errs.len() / 2], 0.01),
        chk_max("Сохранение энергии", "Пламмер, 1 время пересечения, |ΔE/E|", ((k1 + w1) - e0) / e0.abs(), 1e-3),
        chk_max("Сохранение импульса", "|ΔP| / Σm|v|", g.momentum().norm() / p_scale, 1e-3),
        chk("Кривая вращения (тёмное гало)", "модель Млечного Пути: v_c на 8 кпк", v8, 220.0, 0.1, "км/с"),
    ]
}

pub fn cosmology_checks() -> Vec<Check> {
    use crate::cosmology::{Cosmology, PmSim, GYR_S};
    let c = Cosmology::planck2018();
    let eds = Cosmology::preset("eds").unwrap();
    let a0 = 0.02;
    let a1 = 0.1;
    let mut pm = PmSim::new(c, 32, 128.0, a0, 3);
    let p0 = pm.measure_power(8);
    let ps = pm.momentum_scale();
    for _ in 0..40 { pm.step((a1 - a0) / 40.0); }
    let p1 = pm.measure_power(8);
    let ratio: f64 = p0.iter().zip(&p1).take(3).map(|(x, y)| y.1 / x.1).sum::<f64>() / 3.0;
    vec![
        chk("Уравнение Фридмана (Planck 2018)", "возраст Вселенной", c.age_gyr(1.0), 13.787, 0.002, "млрд лет"),
        chk("Уравнение Фридмана (аналитика)", "Эйнштейн–де Ситтер: t₀ = 2/(3H₀)", eds.age_gyr(1.0), 2.0 / 3.0 / eds.h0_si() / GYR_S, 1e-4, "млрд лет"),
        chk("Линейный рост", "Эйнштейн–де Ситтер: D(a) = a", eds.growth(0.5), 0.5, 1e-4, ""),
        chk("Линейный рост", "ΛCDM: f(z=0) ≈ Ω_m^0.55", c.growth_rate(1.0), c.omega_m.powf(0.55), 0.01, ""),
        chk("Нормировка спектра", "σ₈ из P(k)", c.sigma_r(8.0), 0.8102, 1e-6, ""),
        chk("Фотометрическое расстояние", "d_L(z=1), Planck 2018 (≈6.78 Гпк)", c.luminosity_distance(1.0) / 1e3, 6.78, 0.01, "Гпк"),
        chk("Гравитационная неустойчивость (PM)", "рост P(k) на больших масштабах a=0.02→0.1 vs D²", ratio, (c.growth(a1) / c.growth(a0)).powi(2), 0.08, ""),
        chk_max("Сохранение импульса (PM, CIC)", "|ΣP| / Σ|p|", pm.total_momentum() / ps, 1e-10),
    ]
}

pub fn climate_checks() -> Vec<Check> {
    use crate::climate::{self, ClimateOptions};
    let e = World::preset("earth").unwrap();
    let r = climate::simulate(&e, &ClimateOptions::default());
    let eq = r.annual_mean[r.lat.len() / 2];
    let snow = climate::simulate(&e, &ClimateOptions { initial_t: Some(220.0), years: 60, ..Default::default() });
    let m = World::preset("mars").unwrap();
    let rm = climate::simulate(&m, &ClimateOptions::default());
    vec![
        chk("Энергобаланс + перенос (EBM)", "Земля: среднегодовая глобальная T", r.global_mean, 288.0, 0.01, "К"),
        chk("Энергобаланс + перенос (EBM)", "Земля: экватор, среднегодовая (≈299 К)", eq, 299.0, 0.02, "К"),
        chk("Ледово-альбедная обратная связь", "Земля: кромка льда (наблюдается ≈70°)", r.ice_edge, 70.0, 0.12, "°"),
        chk_max("1-е начало термодинамики", "Земля: невязка энергии за год", r.energy_residual, 1e-9),
        chk_max("Радиационное равновесие", "Земля: |⟨ASR−OLR⟩| на верхней границе", r.toa_imbalance, 0.05),
        Check { law: "Бистабильность (Будыко)", test: "холодный старт при тех же параметрах → «Земля-снежок»".into(),
            value: format!("{:.0} К, лёд до {:.0}°", snow.global_mean, snow.ice_edge), expected: "лёд до экватора".into(), pass: snow.ice_edge < 5.0 },
        Check { law: "Конденсация CO₂ (148 К)", test: "Марс: полярные шапки CO₂".into(),
            value: format!("кромка {:.0}°, полюс {:.0} К", rm.ice_edge, rm.annual_mean[rm.lat.len() - 1]), expected: "шапки выше 70°".into(), pass: rm.ice_edge > 70.0 && rm.ice_edge < 90.0 },
    ]
}

pub fn env_checks() -> Vec<Check> {
    use crate::env::{Env, EnvConfig, Terrain, G0};
    let mut out = Vec::new();
    // 1) Циолковский: полный расход топлива в невесомости и вакууме
    let mut cfg = EnvConfig::new("lander", "moon").unwrap();
    cfg.world.gravity_scale = 1e-12;
    cfg.noise = 0.0;
    let mut e = Env::new(cfg).unwrap();
    e.reset(1);
    let h0 = e.terrain.height(0.0, 0.0);
    e.set_state(Vec3::new(0.0, 0.0, h0 + 100.0), Vec3::ZERO);
    let m0 = e.mass();
    for _ in 0..4000 { e.step(&[1.0, 0.0, 0.0, 0.0]); }
    let dv = e.velocity().norm();
    let theory = e.isp() * G0 * (m0 / e.dry_mass()).ln();
    out.push(chk("Уравнение Циолковского", "посадочный модуль: Δv при полном расходе топлива", dv, theory, 0.002, "м/с"));
    // 2) Дрон: зависание на Земле при тяге = весу; на Луне нет подъёмной силы; на Марсе обычных винтов мало
    let hover = |planet: &str, thr: Option<f64>, rotor: f64| -> f64 {
        let mut c = EnvConfig::new("drone", planet).unwrap();
        c.noise = 0.0;
        c.rotor_scale = rotor;
        let mut e = Env::new(c).unwrap();
        e.reset(3);
        let h = e.terrain.height(0.0, 0.0);
        e.set_state(Vec3::new(0.0, 0.0, h + 30.0), Vec3::ZERO);
        let rho = e.air_density();
        let u = thr.unwrap_or(if rho > 0.0 { 0.4 * 1.225 / rho / rotor } else { 1.0 });
        for _ in 0..50 { e.step(&[u; 4]); }
        e.velocity().z
    };
    out.push(chk_max("Импульсная теория винта", "Земля: тяга = весу → вертикальная скорость за 1 с, м/с", hover("earth", None, 1.0), 0.05));
    out.push(Check { law: "Подъёмная сила ∝ ρ", test: "Луна (вакуум): дрон на полной тяге падает".into(),
        value: format!("v_z = {:.2} м/с", hover("moon", Some(1.0), 1.0)), expected: "< 0".into(), pass: hover("moon", Some(1.0), 1.0) < -1.0 });
    out.push(Check { law: "Подъёмная сила ∝ ρ", test: "Марс: обычные винты не поднимают, ×40 (как Ingenuity) поднимают".into(),
        value: format!("{:.2} / {:+.2} м/с", hover("mars", Some(1.0), 1.0), hover("mars", Some(1.0), 40.0)), expected: "< 0 / > 0".into(),
        pass: hover("mars", Some(1.0), 1.0) < 0.0 && hover("mars", Some(1.0), 40.0) > 0.0 });
    // 3) Ровер: сцепление по Кулону — въезд на склон tanθ < μ − C_rr, сползание при tanθ > μ
    let climb = |grade: f64| -> f64 {
        let mut c = EnvConfig::new("rover", "earth").unwrap();
        c.noise = 0.0;
        let mut e = Env::new(c).unwrap();
        e.reset(1);
        e.set_terrain(Terrain::plane(grade, 0.6));
        e.set_state(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO);
        for _ in 0..200 { e.step(&[1.0, 1.0]); }
        e.pose().0.x
    };
    let (x1, x2) = (climb(0.3), climb(0.75));
    out.push(Check { law: "Трение Кулона (сцепление колёс)", test: "μ=0.6: склон 17° — въезжает, 37° — сползает".into(),
        value: format!("{x1:+.1} м / {x2:+.1} м"), expected: "> 0 / < 0".into(), pass: x1 > 0.0 && x2 < 0.0 });
    // 4) Детерминизм
    let run = || { let mut e = Env::new(EnvConfig::new("lander", "mars").unwrap()).unwrap(); e.reset(42); let mut o = vec![]; for k in 0..200 { o = e.step(&[0.6, (k as f64 * 0.1).sin(), 0.0, 0.1]).0; } o };
    let (a, b) = (run(), run());
    out.push(chk_max("Воспроизводимость", "одинаковый seed и действия → одинаковые наблюдения", a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max), 1e-12));
    // 5) Шум сенсоров
    let mut e = Env::new(EnvConfig::new("rover", "moon").unwrap()).unwrap();
    e.reset(5);
    let xs: Vec<f64> = (0..2000).map(|_| e.step(&[0.0, 0.0]).0[0]).collect();
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    let sd = (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt();
    out.push(chk("Модель сенсоров", "GPS ровера: СКО шума", sd, 0.5, 0.06, "м"));
    // Контакт: на плоскости модуль стоит на опорах, значит реакция равна весу, а акселерометр показывает g
    let mut c = EnvConfig::new("lander", "mars").unwrap();
    c.noise = 0.0;
    let mut e = Env::new(c).unwrap();
    e.reset(1);
    e.set_terrain(Terrain::plane(0.0, 0.8));
    e.set_state(Vec3::new(0.0, 0.0, 1.7), Vec3::ZERO);
    let g_mars = e.world.g_at(0.0);
    let mut az = 0.0;
    for _ in 0..40 { let (o, _, t, _) = e.step(&[0.0, 0.0, 0.0, 0.0]); az = o[15]; if t { break; } }
    out.push(chk("Третий закон Ньютона (опоры)", "модуль стоит на Марсе: акселерометр показывает g", az, g_mars, 0.03, "м/с²"));
    // Опрокидывание: запас устойчивости зависит от g, поэтому тот же манёвр опасен на Луне и безопасен на Земле
    let turn = |planet: &str| -> bool {
        let mut c = EnvConfig::new("rover", planet).unwrap();
        c.noise = 0.0;
        let mut e = Env::new(c).unwrap();
        e.reset(1);
        e.set_terrain(Terrain::plane(0.0, 0.8));
        e.set_state(Vec3::ZERO, Vec3::ZERO);
        for _ in 0..30 { e.step(&[1.0, 1.0]); }
        for _ in 0..200 { let (_, _, t, _) = e.step(&[1.0, -1.0]); if t { return e.info.crashed; } }
        false
    };
    let (moon_tip, earth_tip) = (turn("moon"), turn("earth"));
    out.push(Check { law: "Статическая устойчивость (v·ω / g)", test: "тот же поворот: на Луне опрокидывает, на Земле нет".into(),
        value: format!("{moon_tip} / {earth_tip}"), expected: "true / false".into(), pass: moon_tip && !earth_tip });
    // Порывы: стационарное СКО процесса Орнштейна–Уленбека
    let mut c = EnvConfig::new("drone", "earth").unwrap();
    c.gust = 3.0;
    c.max_steps = usize::MAX;
    let mut e = Env::new(c).unwrap();
    e.reset(4);
    let h = e.terrain.height(0.0, 0.0);
    let mut sum = 0.0;
    let mut k = 0.0;
    for t in 0..60_000 {
        e.set_state(Vec3::new(0.0, 0.0, h + 100.0), Vec3::ZERO);
        e.step(&[0.4; 4]);
        if t > 1000 { let v = e.gust_velocity().x; sum += v * v; k += 1.0; }
    }
    out.push(chk("Процесс Орнштейна–Уленбека (порывы)", "СКО порывов равно заданному", (sum / k).sqrt(), 3.0, 0.1, "м/с"));
    // Приводы: апериодическое звено, за время τ выход доходит до 1 − 1/e
    let mut c = EnvConfig::new("lander", "moon").unwrap();
    c.noise = 0.0;
    c.actuator_lag = 0.2;
    let mut e = Env::new(c).unwrap();
    e.reset(1);
    let h = e.terrain.height(0.0, 0.0);
    e.set_state(Vec3::new(0.0, 0.0, h + 400.0), Vec3::ZERO);
    for _ in 0..10 { e.step(&[1.0, 0.0, 0.0, 0.0]); } // 10 шагов по 20 мс = τ
    out.push(chk("Звено 1-го порядка (приводы)", "за время τ команда отрабатывается на 1 − 1/e", e.applied_action()[0], 1.0 - (-1f64).exp(), 0.01, ""));
    // Двигатель ровера: без нагрузки скорость упирается в холостой ход
    let mut c = EnvConfig::new("rover", "earth").unwrap();
    c.noise = 0.0;
    c.world.atmosphere = Some("none".into());
    let mut e = Env::new(c).unwrap();
    e.reset(1);
    e.set_terrain(Terrain::plane(0.0, 0.8));
    e.set_state(Vec3::ZERO, Vec3::ZERO);
    for _ in 0..300 { e.step(&[1.0, 1.0]); }
    out.push(Check { law: "Характеристика мотора", test: "ровер на ровном: скорость ниже холостого хода".into(), value: format!("{:.2} м/с", e.rover_speed()), expected: "1…2.5 м/с".into(), pass: e.rover_speed() > 1.0 && e.rover_speed() < 2.5 });
    // Ветвление: копия среды идёт точно так же, как оригинал, иначе контрфактические данные неверны
    let mut a = Env::new(EnvConfig::new("lander", "earth").unwrap()).unwrap();
    a.reset(9);
    for _ in 0..50 { a.step(&[0.6, 0.0, 0.1, 0.0]); }
    let mut b = a.clone();
    let mut diff: f64 = 0.0;
    for _ in 0..100 {
        let (x, y) = (a.step(&[0.5, 0.1, 0.0, 0.0]).0, b.step(&[0.5, 0.1, 0.0, 0.0]).0);
        diff = x.iter().zip(&y).map(|(p, q)| (p - q).abs()).fold(diff, f64::max);
    }
    out.push(chk_max("Точность ветвления", "копия среды повторяет оригинал точно", diff, 1e-12));
    // 6) Решаемость: автопилот сажает модуль на Луну
    let mut cfg = EnvConfig::new("lander", "moon").unwrap();
    cfg.randomize = true;
    let mut e = Env::new(cfg).unwrap();
    let mut ok = 0;
    for ep in 0..10 {
        e.reset(ep + 1);
        loop { let a = e.autopilot(); let (_, _, t, tr) = e.step(&a); if t || tr { break; } }
        if e.info.success { ok += 1; }
    }
    out.push(Check { law: "Решаемость задачи", test: "ПД-автопилот: посадка на Луну, 10 эпизодов".into(), value: format!("{ok}/10"), expected: "≥ 7/10".into(), pass: ok >= 7 });
    out
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
    v.extend(stars_checks());
    v.extend(starsystem_checks());
    v.extend(galaxy_checks());
    v.extend(cosmology_checks());
    v.extend(climate_checks());
    v.extend(env_checks());
    v
}
