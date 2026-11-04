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

pub fn quantum_checks() -> Vec<Check> {
    use crate::quantum as q;
    let mut v = vec![
        chk("Постоянная тонкой структуры", "1/α из e, ħ, c, ε₀", 1.0 / q::alpha(), 137.035999, 1e-6, ""),
        chk("Спектр водорода", "Лайман-α, вакуум (NIST 121.567 нм)", q::hydrogen_line_nm(1.0, 2.0), 121.567, 1e-3, "нм"),
        chk("Спектр водорода", "Бальмер-α, вакуум (NIST 656.46 нм)", q::hydrogen_line_nm(2.0, 3.0), 656.46, 1e-3, "нм"),
        chk("Уравнение Шрёдингера", "атом водорода численно: E₁ = −1/2 а.е.", q::hydrogen_s_level(0, 40.0, 4000), -0.5, 0.002, "а.е."),
        chk("Уравнение Шрёдингера", "атом водорода численно: E₂ = −1/8 а.е.", q::hydrogen_s_level(1, 60.0, 6000), -0.125, 0.002, "а.е."),
        chk("Квантовый осциллятор", "E₀ = ½ħω численно", q::bound_level_1d(|x| 0.5 * x * x, 8.0, 2000, 0), 0.5, 1e-3, "а.е."),
        chk("Квантовый осциллятор", "E₃ = 7/2 ħω численно", q::bound_level_1d(|x| 0.5 * x * x, 8.0, 2000, 3), 3.5, 1e-3, "а.е."),
    ];
    // Пакет: норма, неопределённость, расплывание
    let mut w = q::Wave::gaussian(-60.0, 60.0, 2400, 0.01, 0.0, 1.0, 0.0, |_| 0.0);
    v.push(chk("Соотношение неопределённостей", "гауссов пакет: Δx·Δp = ħ/2", w.sigma_x() * w.sigma_p(), 0.5, 0.005, "ħ"));
    let n0 = w.norm();
    for _ in 0..500 { w.step(); }
    v.push(chk_max("Сохранение вероятности", "норма после 500 шагов Кранка–Николсон", (w.norm() - n0).abs(), 1e-9));
    v.push(chk("Расплывание пакета", "σ(t) = σ₀√(1 + (t/2σ₀²)²), t = 5", w.sigma_x(), (1.0f64 + 2.5 * 2.5).sqrt(), 0.005, ""));
    // Туннелирование: пакет с узким спектром против точной формулы
    let (e0, v0, a) = (0.5f64, 0.8f64, 1.0f64);
    let k0 = (2.0 * e0).sqrt();
    let mut t = q::Wave::gaussian(-150.0, 150.0, 6000, 0.02, -60.0, 12.0, k0, move |x| if (0.0..a).contains(&x) { v0 } else { 0.0 });
    for _ in 0..6500 { t.step(); }
    v.push(chk("Туннельный эффект", "прохождение барьера при E < V₀", t.right_of(a), q::barrier_transmission(e0, v0, a), 0.03, ""));
    let (eg, peak) = q::gamow(1.0, 1.0, crate::constants::M_P, crate::constants::M_P, 1.57e7);
    v.push(chk("Туннелирование в звёздах", "энергия Гамова p–p", eg, 493.0, 0.01, "кэВ"));
    v.push(chk("Туннелирование в звёздах", "пик Гамова в центре Солнца", peak, 6.0, 0.05, "кэВ"));
    let (x1, w3) = q::lane_emden(3.0);
    v.push(chk("Уравнение Лейна–Эмдена", "n = 3: первый ноль ξ₁", x1, 6.89685, 1e-4, ""));
    v.push(chk("Уравнение Лейна–Эмдена", "n = 3: −ξ²θ'", w3, 2.01824, 1e-4, ""));
    v.push(chk("Предел Чандрасекара", "белый карлик, μₑ = 2", q::chandrasekhar_mass(2.0), 1.44, 0.015, "M☉"));
    v
}

pub fn blackhole_checks() -> Vec<Check> {
    use crate::blackhole as b;
    use crate::constants::G;
    use crate::quantum::M_SUN;
    let c = crate::constants::C;
    let sec = 180.0 / std::f64::consts::PI * 3600.0;
    let rg_sun = G * M_SUN / (c * c);
    let r_sun = 6.957e8;
    let mut v = vec![
        chk("Метрика Шварцшильда", "ISCO из устойчивости орбиты", b::isco_numeric(), 6.0, 1e-3, "M"),
        chk("Метрика Шварцшильда", "фотонная сфера", b::photon_sphere_numeric(), 3.0, 1e-3, "M"),
        chk("Тень чёрной дыры", "критический прицельный параметр лучей", b::shadow_radius_numeric(), 27f64.sqrt(), 2e-3, "M"),
        chk("Отклонение света", "геодезическая у края Солнца (Эддингтон 1.75″)", b::deflection(r_sun / rg_sun).unwrap_or(f64::NAN) * sec, 4.0 * rg_sun / r_sun * sec, 0.02, "″"),
        chk("Отклонение света", "сильное поле, b = 20: 4/b + 15π/4b²", b::deflection(20.0).unwrap_or(f64::NAN), 4.0 / 20.0 + 15.0 * std::f64::consts::PI / 4.0 / 400.0 + 128.0 / 3.0 / 8000.0, 0.02, "рад"),
        chk("Прецессия орбит", "численно против 6π/p при p = 1000", b::precession_per_orbit(1000.0 / (1.0 - 0.04), 0.2), 6.0 * std::f64::consts::PI / 1000.0, 0.01, "рад"),
        chk("Метрика Керра", "ISCO a = 0 по минимуму энергии", b::kerr_isco_numeric(1e-9), 6.0, 1e-3, "M"),
        chk("Метрика Керра", "ISCO a = 0.9: численно против формулы", b::kerr_isco_numeric(0.9), b::kerr_isco(0.9), 1e-3, "M"),
        chk("Метрика Керра", "ISCO a = 0.998 (теория 1.237)", b::kerr_isco(0.998), 1.237, 2e-3, "M"),
        chk("Эффективность аккреции", "a = 0: 1 − √(8/9)", b::radiative_efficiency(1e-9), 1.0 - (8f64 / 9.0).sqrt(), 1e-3, ""),
        chk("Эффективность аккреции", "a = 0.998 (теория ≈ 0.32)", b::radiative_efficiency(0.998), 0.321, 0.01, ""),
        chk("Излучение Хокинга", "T для 1 M☉ (6.17·10⁻⁸ К)", b::hawking_temperature(M_SUN) * 1e8, 6.17, 2e-3, "·10⁻⁸ К"),
    ];
    // первый закон термодинамики чёрных дыр проверяет T и S независимо друг от друга
    let (m, dm) = (10.0 * M_SUN, 1e-6 * M_SUN);
    let t_ds = b::hawking_temperature(m) * (b::bh_entropy(m + dm) - b::bh_entropy(m - dm)) / (2.0 * dm * c * c);
    v.push(chk("Термодинамика чёрных дыр", "dE = T·dS", t_ds, 1.0, 1e-6, ""));
    v.push(chk("Излучение Хокинга", "время испарения 1 M☉ (2.1·10⁶⁷ лет)", b::evaporation_years(M_SUN) / 1e67, 2.1, 0.02, "·10⁶⁷ лет"));
    // PSR B1913+16: наблюдённая скорость убывания согласуется с ОТО
    v.push(chk("Гравитационные волны", "PSR B1913+16: dP/dt (ОТО −2.40·10⁻¹²)", b::period_decay(1.4398 * M_SUN, 1.3886 * M_SUN, 27906.98, 0.6171) * 1e12, -2.402, 0.005, "·10⁻¹²"));
    let (m1, m2, a) = (30.0 * M_SUN, 30.0 * M_SUN, 1e9);
    v.push(chk("Гравитационные волны", "время слияния: численно против Петерса", b::merger_time_numeric(m1, m2, a), b::merger_time(m1, m2, a), 1e-3, "с"));
    // GPS: суммарное расхождение часов
    let (re, rs) = (6.371e6, 2.6571e7);
    let gm = 3.986004418e14;
    let grav = gm * (1.0 / re - 1.0 / rs) / (c * c);
    let vel = gm / rs / (2.0 * c * c);
    v.push(chk("Замедление времени", "часы GPS за сутки (+38.6 мкс)", (grav - vel) * 86400.0 * 1e6, 38.6, 0.02, "мкс"));
    v
}

pub fn fluid_checks() -> Vec<Check> {
    use crate::fluid as f;
    let mut s = f::Euler1D::sod(800);
    let (m0, e0) = (s.rho.iter().sum::<f64>(), s.en.iter().sum::<f64>());
    s.run_to(0.2);
    let n = s.rho.len();
    let l1 = (0..n).map(|i| { let x = (i as f64 + 0.5) * s.dx; (s.rho[i] - f::sod_exact(x, s.t).0).abs() * s.dx }).sum::<f64>();
    let mut v = vec![
        chk_max("Задача Сода", "ошибка L1 плотности против точного решения", l1, 0.01),
        chk_max("Сохранение массы", "уравнения Эйлера, относительная ошибка", (s.rho.iter().sum::<f64>() - m0).abs() / m0, 1e-12),
        chk_max("Сохранение энергии", "уравнения Эйлера, относительная ошибка", (s.en.iter().sum::<f64>() - e0).abs() / e0, 1e-12),
    ];
    let (g, h0, l) = (9.81, 1.0, 100.0);
    let mut d = f::Shallow1D::dam(1000, l, g, h0);
    let v0: f64 = d.h.iter().sum();
    d.run_to(4.0);
    let err = (0..d.h.len()).map(|i| { let x = (i as f64 + 0.5) * d.dx; (d.h[i] - f::ritter_depth(x, l / 2.0, d.t, g, h0)).abs() * d.dx }).sum::<f64>() / (h0 * l / 2.0);
    v.push(chk_max("Разрушение плотины (Риттер)", "ошибка глубины против точного решения", err, 0.02));
    v.push(chk_max("Сохранение объёма", "мелкая вода, относительная ошибка", (d.h.iter().sum::<f64>() - v0).abs() / v0, 1e-10));
    v
}

pub fn chemistry_checks() -> Vec<Check> {
    use crate::chemistry as c;
    let (kf, kr) = (2.0, 0.5);
    let (a, b) = c::reversible(1.0, 0.0, kf, kr, 0.7, 2000);
    let exact = kr / (kf + kr) + kf / (kf + kr) * (-(kf + kr) * 0.7f64).exp();
    let (ae, be) = c::reversible(1.0, 0.0, kf, kr, 50.0, 20000);
    let mut v = vec![
        chk("Кинетика 1-го порядка", "A ⇌ B против точного решения", a, exact, 1e-8, ""),
        chk_max("Сохранение вещества", "A + B = const", (a + b - 1.0).abs(), 1e-12),
        chk("Химическое равновесие", "B/A → K = k₊/k₋", be / ae, kf / kr, 1e-6, ""),
        chk("Закон Аррениуса", "Eₐ = 50 кДж/моль: рост k при 298 → 308 К", c::arrhenius(1.0, 5e4, 308.0) / c::arrhenius(1.0, 5e4, 298.0), 1.927, 0.01, "раз"),
    ];
    let ch = c::Chapman::stratosphere();
    let mut y = [1e7, 0.21 * ch.m, 1e12];
    let atoms0 = y[0] + 2.0 * y[1] + 3.0 * y[2];
    for _ in 0..4000 { y = ch.implicit_step(y, 3600.0); }
    v.push(chk("Цикл Чепмена", "стационарное O₃/O₂ против аналитики", y[2] / y[1], ch.steady_ratio(), 0.02, ""));
    v.push(chk_max("Сохранение атомов", "цикл Чепмена, относительная ошибка", (y[0] + 2.0 * y[1] + 3.0 * y[2] - atoms0).abs() / atoms0, 1e-8));
    // RS-25: водород–кислород, ~3600 К, M ≈ 13.5 г/моль, γ ≈ 1.2, степень расширения 69:1. Паспортный вакуумный Isp 452 с
    let (g, tc, mol): (f64, f64, f64) = (1.2, 3600.0, 0.0135);
    let (mut lo, mut hi): (f64, f64) = (1e-6, 0.5);
    for _ in 0..80 { let m = (lo * hi).sqrt(); if c::area_ratio(g, m) > 69.0 { lo = m } else { hi = m } }
    let pr = (lo * hi).sqrt();
    let ve = c::exhaust_velocity(g, tc, mol, pr);
    let area_e = c::area_ratio(g, pr);
    // вакуумный импульс добавляет к скорости член давления на срезе; оценка через характеристическую величину
    let cstar = (g * crate::constants::R_GAS / mol * tc).sqrt() / (g * (2.0 / (g + 1.0)).powf((g + 1.0) / (2.0 * (g - 1.0))).sqrt());
    let isp_vac = (ve + pr * area_e * cstar) / crate::env::G0;
    v.push(chk("Идеальное сопло", "водород–кислород, вакуум (RS-25 452 с, потери не учтены)", isp_vac, 452.0, 0.1, "с"));
    v
}

pub fn collision_checks() -> Vec<Check> {
    use crate::collisions as k;
    let (m, u, de) = k::merge(3.0, Vec3::new(2.0, 0.0, 1.0), 1.0, Vec3::new(-6.0, 4.0, 1.0));
    let p0 = Vec3::new(2.0, 0.0, 1.0) * 3.0 + Vec3::new(-6.0, 4.0, 1.0);
    let mut v = vec![
        chk_max("Сохранение импульса", "неупругое слияние", (u * m - p0).norm(), 1e-12),
        Check { law: "Второе начало", test: "кинетическая энергия при слиянии только теряется".into(), value: format!("{de:.2} Дж"), expected: "> 0".into(), pass: de > 0.0 },
        chk("Закон разрушения (Лейнхардт–Стюарт)", "при Q = Q* остаётся половина массы", k::largest_remnant_fraction(1.0, 1.0), 0.5, 1e-9, ""),
    ];
    // Гипотеза гигантского удара: 0.1 M⊕ со скоростью около 2-й космической не разрушает Землю
    let (me, mt, vesc) = (5.972e24, 6.4e23, 1.12e4);
    let q = k::impact_energy(me, mt, vesc * 1.1);
    let qs = k::disruption_energy(me + mt, 5500.0, 1.9);
    v.push(Check { law: "Закон разрушения (Лейнхардт–Стюарт)", test: "удар Тейи не разрушает Землю".into(), value: k::outcome(q, qs).into(), expected: "не катастрофа".into(), pass: q < qs });
    // Кратер Бэрринджер (Метеоритный): около 1.2 км; тело около 50 м, железо, 12.8 км/с
    v.push(chk("Масштабирование кратеров", "кратер Бэрринджер (1.19 км), порядок", k::crater_diameter(7800.0, 2500.0, 50.0, 12800.0, 9.81, 45.0).1 / 1e3, 1.19, 0.35, "км"));
    // Чиксулуб: около 180 км; тело около 12 км, 20 км/с
    v.push(chk("Масштабирование кратеров", "Чиксулуб (≈180 км), порядок", k::crater_diameter(2700.0, 2600.0, 12000.0, 20000.0, 9.81, 45.0).1 / 1e3, 180.0, 0.35, "км"));
    let (d1, d2) = (k::crater_diameter(3000.0, 2500.0, 10.0, 1e4, 9.81, 45.0).0, k::crater_diameter(3000.0, 2500.0, 10.0, 1e4, 1.62, 45.0).0);
    v.push(chk("Масштабирование кратеров", "тот же удар на Луне: D ∝ g^(−0.22)", d2 / d1, (9.81f64 / 1.62).powf(0.22), 1e-9, "раз"));
    v.push(chk("Предел Роша", "Земля и лёд: кольца Сатурна в масштабе (Сатурн ≈ 2.2R при льде)", k::roche_limit(1.0, 687.0, 934.0), 2.2, 0.03, "R"));
    v
}

pub fn tectonics_checks() -> Vec<Check> {
    use crate::tectonics as t;
    let l = t::Lithosphere::earth();
    let myr = YEAR * 1e6;
    let mut v = vec![
        chk("Теплопроводность", "остывание плиты: численно против erf", l.temperature_numeric(5e4, 80.0 * myr), l.temperature_exact(5e4, 80.0 * myr), 0.005, "К"),
        chk("Остывание литосферы", "глубина океана: ~350 м·√(млн лет) (Парсонс–Склейтер)", l.subsidence(myr), 350.0, 0.12, "м"),
        chk("Остывание литосферы", "тепловой поток в плите 80 млн лет (наблюдается ≈ 55 мВт/м²)", l.heat_flow(80.0 * myr) * 1e3, 55.0, 0.2, "мВт/м²"),
    ];
    let ra = t::rayleigh(4000.0, 9.8, 3e-5, 3000.0, 2.9e6, 1e-6, 1e22);
    v.push(Check { law: "Конвекция мантии", test: "число Рэлея выше критического (≈ 10³)".into(), value: format!("{ra:.2e}"), expected: "> 1e3".into(), pass: ra > 1e3 });
    let s = t::convection_speed(1e-6, 2.9e6, ra) * YEAR * 100.0;
    v.push(Check { law: "Конвекция мантии", test: "скорость плит по порядку величины (см/год)".into(), value: format!("{s:.1} см/год"), expected: "1…30 см/год".into(), pass: (1.0..=30.0).contains(&s) });
    v
}

pub fn all() -> Vec<Check> {
    let mut v = Vec::new();
    v.extend(quantum_checks());
    v.extend(blackhole_checks());
    v.extend(fluid_checks());
    v.extend(chemistry_checks());
    v.extend(collision_checks());
    v.extend(tectonics_checks());
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
