//! World model tools: data, Mission Lab, JSON parsing, behavior CLI.

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
      "technologies":[{{"name":"Stronger","set":{{"t_max":8000}},"dry_mass_delta":10}}]{extra}}}"#)
}

#[test]
fn json_round_trip_and_errors() {
    let j = J::parse(r#"{"a":[1,2.5,-3e2],"b":{"c":"t\"e\nxtΩ"},"d":true,"e":null}"#).unwrap();
    assert_eq!(j.get("a").unwrap().as_arr().unwrap()[2].as_f64(), Some(-300.0));
    assert_eq!(j.get("b").unwrap().get("c").unwrap().as_str(), Some("t\"e\nxtΩ"));
    assert!(J::parse(&j.render()).is_ok());
    for bad in ["", "{", "[1,", "{\"a\"}", "{\"a\":1,}", "tru", "\"abc", "1 2", "{a:1}"] { assert!(J::parse(bad).is_err(), "{bad}"); }
    // deep nesting must not crash the process
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
    assert!(mission::markdown(&a).contains("Value of technologies"));
}

#[test]
fn job_free_mission_rejects_nonnumeric_terrain() {
    let bad = spec(r#","terrain":{"n":3,"size":900,"heights":[0,0,0,0,"x",0,0,0,0]}"#);
    assert!(MissionSpec::parse(&bad).is_err());
    let hole = spec(r#","terrain":{"n":3,"size":900,"heights":[0,0,0,0,null,0,0,0,0]}"#);
    assert!(MissionSpec::parse(&hole).is_ok());
    let file = spec(r#","terrain":{"file":"/etc/hosts","size":900}"#);
    assert!(MissionSpec::parse(&file).is_err());
}

#[test]
fn rover_navigation_noise_changes_the_outcome_only_through_sensors() {
    let text = |noise: f64| format!(r#"{{"task":"rover","world":"moon","runs":30,"seed":4,"dispersions":{{"nav_noise":{noise}}},"requirements":[{{"metric":"touch_offset","percentile":50,"op":"<=","value":1000}}]}}"#);
    let mean = |n: f64| { let r = mission::run(&MissionSpec::parse(&text(n)).unwrap(), 4).unwrap(); r.get("scorecard").unwrap().as_arr().unwrap()[0].get("value").unwrap().as_f64().unwrap() };
    assert_ne!(mean(0.0), mean(40.0));
}

#[test]
fn wilson_interval_is_sane() {
    let (l, h) = wilson(50.0, 100.0);
    assert!((l - 0.404).abs() < 0.01 && (h - 0.596).abs() < 0.01);
    let (l, h) = wilson(0.0, 10.0);
    assert!(l == 0.0 && h > 0.2 && h < 0.35); // zero failures out of ten still proves nothing
    let (l, h) = wilson(10.0, 10.0);
    assert!(h == 1.0 && l < 0.8);
}

#[test]
fn technology_saving_is_paired_on_same_conditions() {
    // with no difference a paired comparison must give exactly zero
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
    assert!(in_contact_steps > 10); // the model sees the contact itself, not only the episode end
}

#[test]
fn every_parameter_works_at_both_limits() {
    use worldsim::env::PARAMS;
    for task in ["lander", "drone", "rover"] {
        for d in PARAMS {
            let probe = EnvConfig::new(task, "mars").unwrap();
            if probe.clone().set(d.name, d.min).is_err() { continue; } // does not apply to this task
            for v in [d.min, d.max] {
                let mut c = EnvConfig::new(task, "mars").unwrap();
                // an outage needs a link cycle to happen in
                if d.name == "link_outage" { c.set("link_period", 1e6).unwrap(); }
                c.set(d.name, v).unwrap();
                if d.name == "max_steps" { c.max_steps = c.max_steps.min(300); }
                let mut e = Env::new(c).unwrap_or_else(|x| panic!("{task} {} = {v}: {x}", d.name));
                for seed in 0..2 {
                    let o = e.reset(seed);
                    assert!(o.iter().all(|x| x.is_finite()), "{task} {} = {v}: reset {o:?}", d.name);
                    for _ in 0..150 {
                        let a = e.autopilot();
                        assert!(a.iter().all(|x| x.is_finite()), "{task} {} = {v}: action {a:?}", d.name);
                        let (o, r, t, c) = e.step(&a);
                        assert!(o.iter().all(|x| x.is_finite()) && r.is_finite(), "{task} {} = {v}: step {o:?} {r}", d.name);
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
    assert!(c.set("rover_mass", 100.0).is_err()); // wrong task
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
    // center of mass offset spins, thrust misalignment shifts, minimum throttle prevents operating below the threshold
    assert!((run("cg_offset=0.2")[10] - base[10]).abs() > 1e-3);
    assert!((run("thrust_misalign=5")[3] - base[3]).abs() > 1e-3);
    assert!((run("throttle_min=0.8")[5] - base[5]).abs() > 1e-3);
    assert!((run("gravity=5")[5] - base[5]).abs() > 1e-3);
    // gyro bias shows in the observation, and delay returns a past value
    assert!((run("gyro_bias=0.2")[9] - base[9]).abs() > 1e-3);
    assert!((run("obs_delay=10")[5] - base[5]).abs() > 1e-3);
}

#[test]
fn sha256_matches_known_vectors() {
    use worldsim::store::sha256;
    assert_eq!(sha256(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert_eq!(sha256(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert_eq!(sha256(&[b'a'; 1000]), "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
}

#[test]
fn run_store_saves_lists_and_rejects_bad_ids() {
    use worldsim::store;
    let d = std::env::temp_dir().join(format!("ws_runs_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let text = spec("");
    let r = mission::run(&MissionSpec::parse(&text).unwrap(), 2).unwrap();
    let a = store::save_run(&d, &text, &r).unwrap();
    let b = store::save_run(&d, &text, &r).unwrap();
    assert_ne!(a, b); // the same report does not overwrite a previous run
    assert_eq!(store::list_runs(&d).as_arr().unwrap().len(), 2);
    assert!(store::load_report(&d, &a).is_ok());
    for bad in ["../x", "..", "a/b", "", "ZZZ"] { assert!(store::load_report(&d, bad).is_err(), "{bad}"); }
    let c = store::compare(&d, &a, &b).unwrap();
    assert_eq!(c.get("delta").unwrap().as_f64(), Some(0.0));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn openapi_lists_every_route() {
    let api = J::parse(include_str!("../docs/openapi.json")).unwrap();
    let paths: Vec<String> = api.get("paths").unwrap().as_obj().unwrap().iter().map(|x| x.0.clone()).collect();
    let src = include_str!("../src/server.rs");
    let mut seen = 0;
    for (i, _) in src.match_indices("\"/api/") {
        let rest = &src[i + 1..];
        let lit = &rest[..rest.find('"').unwrap()];
        if lit.ends_with('/') || lit.contains(' ') { continue; } // prefixes and text are written through {id}
        seen += 1;
        assert!(paths.iter().any(|p| p == lit), "{lit} missing in openapi.json");
    }
    assert!(seen > 20);
}

/// A controller that repeats the autopilot must get the same analysis as `mission::run`.
#[test]
fn controller_analysis_matches_the_autopilot_run() {
    let spec = MissionSpec::parse(r#"{"name":"t","task":"lander","world":"moon","runs":12,"seed":3,"vehicle":{"t_max":8000,"fuel":120,"start_offset":5,"start_vh":1},
      "dispersions":{"t_max":0.05,"dry_mass":0.03},"requirements":[{"metric":"success","op":">=","value":0.5},{"metric":"touch_speed","percentile":95,"op":"<=","value":2.5}],
      "technologies":[{"name":"Stronger","set":{"t_max":12000},"dry_mass_delta":10},{"name":"Weak","set":{"t_max":3000}}]}"#).unwrap();
    let auto = mission::run(&spec, 2).unwrap();
    let mut r = mission::Runner::new(spec, 2).unwrap();
    let series = r.analysis_plan().as_arr().unwrap().len();
    assert_eq!(series, 2 + 2 + 2);
    for i in 0..series {
        r.select(i).unwrap();
        loop {
            let a: Vec<Vec<f64>> = r.envs.iter().map(|e| e.autopilot()).collect();
            if r.step(&a) == 0 { break; }
        }
        r.record();
    }
    let own = r.analysis_report("autopilot-copy").unwrap();
    let f = |j: &J, k: &str| j.get(k).unwrap().as_f64().unwrap();
    assert_eq!(f(&own, "success"), f(&auto, "success"));
    assert_eq!(f(&own, "baseline_no_dispersions"), f(&auto, "baseline_no_dispersions"));
    assert_eq!(f(&own, "delta_vs_autopilot"), 0.0);
    for key in ["drivers", "technologies"] {
        let (a, b) = (own.get(key).unwrap().as_arr().unwrap(), auto.get(key).unwrap().as_arr().unwrap());
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert_eq!(x.get("name").unwrap().as_str(), y.get("name").unwrap().as_str());
            assert_eq!(f(x, "success"), f(y, "success"), "{key}");
        }
    }
    assert!(own.get("markdown").unwrap().as_str().unwrap().contains("Value of technologies"));
    // after the analysis a plain report still describes the mission itself
    assert!(r.report("again").is_ok());
}

/// Winds of Earth from its temperature contrast: the mean flow must be in gradient balance and westerly in midlatitudes.
#[test]
fn earth_winds_are_balanced_westerlies() {
    use worldsim::{climate::{simulate, ClimateOptions}, shallow::world_winds, world::World};
    let w = World::preset("earth").unwrap();
    let c = simulate(&w, &ClimateOptions::default());
    let r = world_winds(&w, &c, 21, 60.0).unwrap();
    assert!(r.peak_u > 10.0 && r.peak_u < 80.0, "peak {} m/s", r.peak_u);
    assert!(r.peak_lat.abs() > 15.0 && r.peak_lat.abs() < 65.0, "jet at {}°", r.peak_lat);
    assert!(r.balance_error < 0.2, "balance {}", r.balance_error);
    // air rises near the warm equator: the cloudiest zone sits there
    let at = |lat: f64| r.lat.iter().position(|x| (x - lat).abs() < 3.0).map(|i| r.cloud[i]).unwrap();
    assert!(at(2.8) > at(25.0), "cloud {} at the equator, {} at 25°", at(2.8), at(25.0));
    // worlds where the model would give a number with nothing behind it are refused
    for name in ["moon", "venus", "jupiter"] { assert!(world_winds(&World::preset(name).unwrap(), &c, 21, 1.0).is_err(), "{name}"); }
    assert!(world_winds(&World::preset("mars").unwrap(), &simulate(&World::preset("mars").unwrap(), &ClimateOptions::default()), 21, 5.0).is_ok());
}

/// The pair sum must not disturb the linear growth of large scales.
#[test]
fn p3m_keeps_linear_growth() {
    use worldsim::cosmology::{Cosmology, PmSim};
    let c = Cosmology::planck2018();
    let growth = |p3m: bool| {
        let mut pm = PmSim::new(c, 16, 200.0, 0.1, 3);
        pm.p3m = p3m;
        let before = pm.measure_power(4)[0].1;
        for _ in 0..8 { pm.step(0.025); }
        pm.measure_power(4)[0].1 / before
    };
    let expected = (c.growth(0.3) / c.growth(0.1)).powi(2);
    let got = growth(true);
    assert!((got / expected - 1.0).abs() < 0.2, "P(k) grew ×{got:.2}, linear theory ×{expected:.2}");
}

/// The fast ray test must give exactly the distance of a plain march over the same steps.
#[test]
fn fast_raycast_matches_a_plain_march() {
    use worldsim::env::Terrain;
    use worldsim::rng::Rng;
    use worldsim::vec3::Vec3;
    use worldsim::world::World;
    let plain = |t: &Terrain, o: Vec3, d: Vec3, max: f64| -> f64 {
        let d = d.unit();
        let (mut s, mut prev) = (0.0, 0.0);
        let step = (t.size / (t.n - 1) as f64 * 0.5).max(0.25);
        while s < max {
            let p = o + d * s;
            if p.z <= t.height(p.x, p.y) {
                let (mut lo, mut hi) = (prev, s);
                for _ in 0..30 {
                    let m = 0.5 * (lo + hi);
                    let q = o + d * m;
                    if q.z <= t.height(q.x, q.y) { hi = m } else { lo = m }
                }
                return hi;
            }
            prev = s;
            s += step;
        }
        max
    };
    let moon = World::preset("moon").unwrap();
    let flat = Terrain::new(33, 800.0, vec![5.0; 33 * 33], 0.6);
    let maps = [Terrain::generate(&moon, 7, 1200.0, 161, 25.0), Terrain::plane(0.3, 0.6), flat];
    let mut r = Rng::new(11);
    for t in &maps {
        for _ in 0..3000 {
            // from outside the map as well: the height is clamped at the edge there
            let o = Vec3::new(r.uniform(-900.0, 900.0), r.uniform(-900.0, 900.0), r.uniform(-50.0, 900.0));
            let d = Vec3::new(r.uniform(-1.0, 1.0), r.uniform(-1.0, 1.0), r.uniform(-1.0, 0.2));
            let (a, b) = (t.raycast(o, d, 1000.0), plain(t, o, d, 1000.0));
            assert_eq!(a.to_bits(), b.to_bits(), "{o:?} {d:?}: {a} against {b}");
        }
    }
}

/// Skipping the ray sensors must not change the flight: the noise stream stays the same and only those channels differ.
#[test]
fn skipping_rays_keeps_the_flight() {
    for task in ["lander", "drone", "rover"] {
        let fly = |skip: bool| {
            let mut c = EnvConfig::new(task, "moon").unwrap();
            c.skip_rays = skip;
            let mut e = Env::new(c).unwrap();
            e.reset(5);
            let mut first = vec![];
            for _ in 0..400 {
                let a = e.autopilot();
                let (o, _, t, tr) = e.step(&a);
                first = o[..6].to_vec();
                if t || tr { break; }
            }
            (e.state(), first, e.steps)
        };
        assert_eq!(fly(true), fly(false), "{task}");
    }
}
