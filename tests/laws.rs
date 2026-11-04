use worldsim::validate::{self, Check};
use worldsim::world::{World, WorldSpec};

fn assert_all(v: Vec<Check>) {
    let failed: Vec<_> = v.iter().filter(|c| !c.pass).map(|c| format!("{} / {}: {} (ожидалось {})", c.law, c.test, c.value, c.expected)).collect();
    assert!(failed.is_empty(), "не выполнены законы:\n{}", failed.join("\n"));
}

#[test] fn kepler() { assert_all(validate::kepler()); }
#[test] fn conservation() { assert_all(validate::conservation_solar_system(1.0)); }
#[test] fn general_relativity() { assert_all(validate::mercury_precession()); }
#[test] fn atmosphere_and_composition() { assert_all(validate::atmosphere()); }
#[test] fn thermodynamics() { assert_all(validate::temperatures()); }
#[test] fn mechanics() { assert_all(validate::ballistics_checks()); }
#[test] fn stars() { assert_all(validate::stars_checks()); }
#[test] fn star_systems() { assert_all(validate::starsystem_checks()); }
#[test] fn galaxies() { assert_all(validate::galaxy_checks()); }
#[test] fn cosmology() { assert_all(validate::cosmology_checks()); }
#[test] fn climate() { assert_all(validate::climate_checks()); }
#[test] fn quantum() { assert_all(validate::quantum_checks()); }
#[test] fn black_holes() { assert_all(validate::blackhole_checks()); }
#[test] fn fluids() { assert_all(validate::fluid_checks()); }
#[test] fn chemistry() { assert_all(validate::chemistry_checks()); }
#[test] fn collisions() { assert_all(validate::collision_checks()); }
#[test] fn tectonics() { assert_all(validate::tectonics_checks()); }
#[test] fn robot_env() { assert_all(validate::env_checks()); }
#[test] fn electromagnetism() { assert_all(validate::magnetosphere()); }

#[test]
fn mixed_world_is_self_consistent() {
    // Марс с земной атмосферой: давление у поверхности то же, но столб газа тяжелее (g меньше),
    // значит τ больше и поверхность теплее, чем T_eq Марса.
    let w = World::build(&WorldSpec { physics: "mars".into(), atmosphere: Some("earth".into()), surface: Some("moon".into()), gravity_scale: 1.0, distance_au: None }).unwrap();
    let a = w.atmosphere.as_ref().unwrap();
    assert!((a.column_weight(w.physics.gm, w.physics.radius) / 101_325.0 - 1.0).abs() < 0.005);
    assert!(w.greenhouse_tau() > 0.875 * 2.0);
    assert!(w.mean_surface_temperature() > w.equilibrium_temperature());
    assert!(a.scale_height(w.g_surface()) > 20_000.0); // ниже g → атмосфера «пухлее»
}

#[test]
fn moon_has_no_air() {
    let w = World::preset("moon").unwrap();
    assert!(w.atmosphere.is_none());
    assert_eq!(w.density(0.0), 0.0);
}
