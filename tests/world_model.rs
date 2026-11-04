//! Инструменты world model: данные, Mission Lab, разбор JSON, поведение CLI.

use std::process::Command;
use worldsim::env::{Env, EnvConfig};
use worldsim::json::J;
use worldsim::mission::{self, wilson, MissionSpec};

const BIN: &str = env!("CARGO_BIN_EXE_worldsim");

fn cli(args: &[&str]) -> (bool, String) {
    let o = Command::new(BIN).args(args).output().unwrap();
    (o.status.success(), String::from_utf8_lossy(&o.stderr).to_string() + &String::from_utf8_lossy(&o.stdout))
}

fn spec(extra: &str) -> String {
    format!(r#"{{"task":"lander","world":"moon","runs":24,"seed":3,"vehicle":{{"t_max":6000,"fuel":100}},
      "dispersions":{{"t_max":0.05,"nav_noise":1}},"requirements":[{{"metric":"success","op":">=","value":0.5}}],
      "technologies":[{{"name":"Сильнее","set":{{"t_max":8000}},"dry_mass_delta":10}}]{extra}}}"#)
}

#[test]
fn json_round_trip_and_errors() {
    let j = J::parse(r#"{"a":[1,2.5,-3e2],"b":{"c":"т\"е\nкстЖ"},"d":true,"e":null}"#).unwrap();
    assert_eq!(j.get("a").unwrap().as_arr().unwrap()[2].as_f64(), Some(-300.0));
    assert_eq!(j.get("b").unwrap().get("c").unwrap().as_str(), Some("т\"е\nкстЖ"));
    assert!(J::parse(&j.render()).is_ok());
    for bad in ["", "{", "[1,", "{\"a\"}", "{\"a\":1,}", "tru", "\"abc", "1 2", "{a:1}"] { assert!(J::parse(bad).is_err(), "{bad}"); }
    // глубокая вложенность не должна ронять процесс
    assert!(J::parse(&"[".repeat(100_000)).is_err());
}

#[test]
fn mission_spec_rejects_mistakes() {
    assert!(MissionSpec::parse(&spec("")).is_ok());
    for bad in [
        r#"{"task":"nope","requirements":[{"metric":"success","value":1}]}"#,
        r#"{"task":"lander","world":"nosuch","requirements":[{"metric":"success","value":1}]}"#,
        r#"{"task":"lander","requirements":[]}"#,
        r#"{"task":"lander","requirements":[{"metric":"zzz","value":1,"percentile":50}]}"#,
        r#"{"task":"lander","requirements":[{"metric":"fuel_used","value":1}]}"#,
        r#"{"task":"lander","runs":0,"requirements":[{"metric":"success","value":1}]}"#,
        r#"{"task":"lander","vehicle":{"warp":1},"requirements":[{"metric":"success","value":1}]}"#,
        r#"{"task":"lander","vehicle":{"isp":-1},"requirements":[{"metric":"success","value":1}]}"#,
        r#"{"task":"lander","dispersions":{"isp":-1},"requirements":[{"metric":"success","value":1}]}"#,
    ] { assert!(MissionSpec::parse(bad).is_err(), "{bad}"); }
}

#[test]
fn mission_result_does_not_depend_on_threads() {
    let s = MissionSpec::parse(&spec("")).unwrap();
    let strip = |j: &J| format!("{}{}{}", j.get("scorecard").unwrap().render(), j.get("drivers").unwrap().render(), j.get("technologies").unwrap().render());
    let (a, b) = (mission::run(&s, 1).unwrap(), mission::run(&s, 8).unwrap());
    assert_eq!(strip(&a), strip(&b));
    assert!(mission::markdown(&a).contains("Ценность технологий"));
}

#[test]
fn wilson_interval_is_sane() {
    let (l, h) = wilson(50.0, 100.0);
    assert!((l - 0.404).abs() < 0.01 && (h - 0.596).abs() < 0.01);
    let (l, h) = wilson(0.0, 10.0);
    assert!(l == 0.0 && h > 0.2 && h < 0.35); // ноль отказов из десяти ещё ничего не доказывает
    let (l, h) = wilson(10.0, 10.0);
    assert!(h == 1.0 && l < 0.8);
}

#[test]
fn technology_saving_is_paired_on_same_conditions() {
    // при отсутствии разницы парное сравнение обязано дать ровно ноль
    let s = MissionSpec::parse(&spec("").replace("\"t_max\":8000", "\"t_max\":6000").replace(",\"dry_mass_delta\":10", "")).unwrap();
    let r = mission::run(&s, 4).unwrap();
    let t = &r.get("technologies").unwrap().as_arr().unwrap()[0];
    assert_eq!(t.get("saved").unwrap().as_f64(), Some(0.0));
    assert_eq!(t.get("lost").unwrap().as_f64(), Some(0.0));
}

#[test]
fn cli_rejects_typos_and_bad_ports() {
    let (ok, msg) = cli(&["throw", "earth", "--spede", "5"]);
    assert!(!ok && msg.contains("--spede"), "{msg}");
    assert!(!cli(&["serve", "--port", "99999", "--no-open"]).0);
    assert!(!cli(&["serve", "--port", "0", "--no-open"]).0);
    assert!(!cli(&["throw", "earth", "--speed", "-5"]).0);
    assert!(!cli(&["throw", "earth", "--radius", "0"]).0);
    assert!(cli(&["throw", "earth", "--speed", "5"]).0);
    assert!(cli(&["info", "--physics", "mars", "--atmosphere", "earth"]).0);
}

#[test]
fn failed_dataset_leaves_no_files() {
    let d = std::env::temp_dir().join(format!("ws_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    for args in [vec!["--task", "nope"], vec!["--task", "particle", "--worlds", "earth,moon"], vec!["--task", "orbit", "--n", "0"], vec!["--task", "orbit", "--worlds", "nosuch"]] {
        let mut a = vec!["dataset", "--out", d.to_str().unwrap()];
        a.extend(args);
        assert!(!cli(&a).0);
        assert!(!d.exists(), "{a:?}");
    }
}

#[test]
fn vehicle_settings_change_physics() {
    let accel = |t: f64| {
        let mut c = EnvConfig::new("lander", "moon").unwrap();
        c.noise = 0.0;
        c.vehicle = vec![("t_max".into(), t), ("dry_mass".into(), 500.0)];
        let mut e = Env::new(c).unwrap();
        e.reset(1);
        let h = e.terrain.height(0.0, 0.0);
        e.set_state(worldsim::vec3::Vec3::new(0.0, 0.0, h + 300.0), worldsim::vec3::Vec3::ZERO);
        e.step(&[1.0, 0.0, 0.0, 0.0]).0[15]
    };
    assert!(accel(20000.0) > 1.8 * accel(10000.0));
}

#[test]
fn every_world_list_entry_is_used_and_deterministic() {
    let mut c = EnvConfig::new("rover", "earth").unwrap();
    c.worlds = ["moon", "mars", "mars:earth:moon"].iter().map(|w| worldsim::world::WorldSpec::parse(w)).collect();
    let mut e = Env::new(c).unwrap();
    let picks: Vec<usize> = (0..60).map(|s| { e.reset(s); e.world_index }).collect();
    for i in 0..3 { assert!(picks.contains(&i)); }
    let again: Vec<usize> = (0..60).map(|s| { e.reset(s); e.world_index }).collect();
    assert_eq!(picks, again);
    let mut c = EnvConfig::new("rover", "earth").unwrap();
    c.worlds = vec![worldsim::world::WorldSpec::preset("nosuch")];
    assert!(Env::new(c).is_err());
}

#[test]
fn drone_can_rest_on_ground_and_take_off_again() {
    let mut c = EnvConfig::new("drone", "earth").unwrap();
    c.noise = 0.0;
    let mut e = Env::new(c).unwrap();
    e.reset(2);
    e.set_terrain(worldsim::env::Terrain::plane(0.0, 0.8));
    e.set_state(worldsim::vec3::Vec3::new(0.0, 0.0, 0.3), worldsim::vec3::Vec3::ZERO);
    for _ in 0..40 { let (_, _, t, _) = e.step(&[0.3; 4]); assert!(!t); }
    assert!(e.in_contact());
    for _ in 0..50 { e.step(&[0.8; 4]); }
    assert!(!e.in_contact() && e.pose().0.z > 1.0);
}

#[test]
fn lander_contact_is_observed_before_episode_ends() {
    let mut c = EnvConfig::new("lander", "moon").unwrap();
    c.noise = 0.0;
    let mut e = Env::new(c).unwrap();
    e.reset(1);
    e.set_terrain(worldsim::env::Terrain::plane(0.0, 0.8));
    e.set_state(worldsim::vec3::Vec3::new(0.0, 0.0, 2.2), worldsim::vec3::Vec3::new(0.0, 0.0, -0.5));
    let mut in_contact_steps = 0;
    loop {
        let (_, _, t, c) = e.step(&[0.0; 4]);
        if e.in_contact() { in_contact_steps += 1; }
        if t || c { break; }
    }
    assert!(e.info.success, "{}", e.info.reason);
    assert!(in_contact_steps > 10); // модель видит сам контакт, а не только конец эпизода
}

#[test]
fn every_parameter_works_at_both_limits() {
    use worldsim::env::PARAMS;
    for task in ["lander", "drone", "rover"] {
        for d in PARAMS {
            let probe = EnvConfig::new(task, "mars").unwrap();
            if probe.clone().set(d.name, d.min).is_err() { continue; } // не применим к этой задаче
            for v in [d.min, d.max] {
                let mut c = EnvConfig::new(task, "mars").unwrap();
                c.set(d.name, v).unwrap();
                if d.name == "max_steps" { c.max_steps = c.max_steps.min(300); }
                let mut e = Env::new(c).unwrap_or_else(|x| panic!("{task} {} = {v}: {x}", d.name));
                for seed in 0..2 {
                    let o = e.reset(seed);
                    assert!(o.iter().all(|x| x.is_finite()), "{task} {} = {v}: reset {o:?}", d.name);
                    for _ in 0..150 {
                        let a = e.autopilot();
                        assert!(a.iter().all(|x| x.is_finite()), "{task} {} = {v}: действие {a:?}", d.name);
                        let (o, r, t, c) = e.step(&a);
                        assert!(o.iter().all(|x| x.is_finite()) && r.is_finite(), "{task} {} = {v}: шаг {o:?} {r}", d.name);
                        if t || c { break; }
                    }
                }
            }
        }
    }
}

#[test]
fn parameters_reject_bad_values_and_names() {
    let mut c = EnvConfig::new("lander", "moon").unwrap();
    assert!(c.set("isp", -1.0).is_err());
    assert!(c.set("isp", f64::NAN).is_err());
    assert!(c.set("warp_drive", 1.0).is_err());
    assert!(c.set("rover_mass", 100.0).is_err()); // не та задача
    assert!(c.set_list("isp=320,throttle_min=0.2").is_ok());
    assert_eq!(c.get("isp"), Some(320.0));
    assert!(c.set_list("isp").is_err());
    assert!(c.set_list("isp=abc").is_err());
}

#[test]
fn new_parameters_change_behaviour() {
    let run = |list: &str| {
        let mut c = EnvConfig::new("lander", "moon").unwrap();
        c.noise = 0.0;
        c.set_list(list).unwrap();
        let mut e = Env::new(c).unwrap();
        e.reset(3);
        let h = e.terrain.height(0.0, 0.0);
        e.set_state(worldsim::vec3::Vec3::new(0.0, 0.0, h + 300.0), worldsim::vec3::Vec3::ZERO);
        let mut o = vec![];
        for _ in 0..30 { o = e.step(&[0.5, 0.0, 0.0, 0.0]).0; }
        o
    };
    let base = run("");
    assert_eq!(base, run(""));
    // смещение центра масс закручивает, перекос тяги сдвигает, минимальный газ не даёт работать ниже порога
    assert!((run("cg_offset=0.2")[10] - base[10]).abs() > 1e-3);
    assert!((run("thrust_misalign=5")[3] - base[3]).abs() > 1e-3);
    assert!((run("throttle_min=0.8")[5] - base[5]).abs() > 1e-3);
    assert!((run("gravity=5")[5] - base[5]).abs() > 1e-3);
    // смещение гироскопа видно в наблюдении, а задержка отдаёт прошлое значение
    assert!((run("gyro_bias=0.2")[9] - base[9]).abs() > 1e-3);
    assert!((run("obs_delay=10")[5] - base[5]).abs() > 1e-3);
}
