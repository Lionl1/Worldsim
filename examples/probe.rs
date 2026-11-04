use worldsim::*;
use std::time::Instant;

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_default();
    if arg == "stars" || arg.is_empty() {
        let s = stars::star("sun").unwrap();
        println!("sun HZ {:?} rgb {:?}", s.habitable_zone(), s.rgb());
        println!("MS 1.0 {:?}", stars::Star::main_sequence(1.0));
        println!("wien 5772: {:.4e} m·K (2.8978e-3)", stars::wien_peak(5772.0) * 5772.0);
        println!("SB ratio {:.8}", stars::planck_integrated(5772.0) / (constants::SIGMA_SB * 5772f64.powi(4)));
        for n in ["proxima", "sirius_a", "trappist1", "betelgeuse"] { let s = stars::star(n).unwrap(); println!("{n} rgb {:?} hz {:?} {}", s.rgb(), s.habitable_zone(), s.spectral_class()); }
        let ss = starsystem::StarSystem::generate(7, None).unwrap();
        println!("star {:?}", ss.star);
        for p in &ss.planets { println!("  {} m={:.2} r={:.2} a={:.3} T={:.0} {} hz={}", p.name, p.mass, p.radius, p.a, p.t_eq, p.kind, p.in_hz); }
    }
    if arg == "galaxy" || arg.is_empty() {
        let t = Instant::now();
        let mut g = galaxy::preset("plummer", 4000, 1).unwrap();
        let (k, w) = g.energy();
        println!("plummer virial 2K/|W| = {:.3}  build {:?}", 2.0 * k / w.abs(), t.elapsed());
        let acc = g.tree_acc();
        let mut errs: Vec<f64> = (0..200).map(|i| { let d = g.direct_acc(i * 20); (acc[i * 20] - d).norm() / d.norm() }).collect();
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("BH rel err median {:.2e} p90 {:.2e}", errs[100], errs[180]);
        let tc = galaxy::crossing_time(1.0, 1.0);
        let e0 = k + w;
        let t = Instant::now();
        for _ in 0..200 { g.step(tc / 100.0); }
        let (k1, w1) = g.energy();
        println!("energy drift over 2 t_cr: {:.2e}, virial {:.3}, {:?}", ((k1 + w1) - e0) / e0.abs(), 2.0 * k1 / w1.abs(), t.elapsed());
        let t = Instant::now();
        let mut mw = galaxy::preset("milky_way", 30000, 2).unwrap();
        println!("mw build {:?}", t.elapsed());
        let t = Instant::now();
        mw.step(0.001);
        println!("mw step 30k {:?}", t.elapsed());
        println!("rot curve {:?}", mw.rotation_curve(20.0, 10).iter().map(|(r, v)| format!("{r:.0}:{v:.0}")).collect::<Vec<_>>());
    }
    if arg == "cosmo" || arg.is_empty() {
        let c = cosmology::Cosmology::planck2018();
        println!("age {:.4} Gyr (13.787)", c.age_gyr(1.0));
        println!("D(0.5)={:.4} f(1)={:.4} (Ωm^0.55={:.4})", c.growth(0.5), c.growth_rate(1.0), c.omega_m.powf(0.55));
        println!("sigma8 check {:.4}", c.sigma_r(8.0));
        println!("dL(z=1) {:.1} Mpc", c.luminosity_distance(1.0));
        let eds = cosmology::Cosmology::preset("eds").unwrap();
        println!("EdS age {:.4} vs 2/(3H0)={:.4}  D(0.5)={:.4}", eds.age_gyr(1.0), 2.0 / 3.0 / eds.h0_si() / cosmology::GYR_S, eds.growth(0.5));
        let t = Instant::now();
        let a0 = 0.02;
        let mut pm = cosmology::PmSim::new(c, 32, 128.0, a0, 3);
        println!("pm init {:?}", t.elapsed());
        let p0 = pm.measure_power(8);
        let t = Instant::now();
        let steps = 40;
        let a1 = 0.1;
        let p_scale = pm.momentum_scale();
        for _ in 0..steps { pm.step((a1 - a0) / steps as f64); }
        println!("pm 40 steps {:?}, |P|/scale {:.2e}", t.elapsed(), pm.total_momentum() / pm.momentum_scale().max(p_scale));
        let p1 = pm.measure_power(8);
        let expect = (c.growth(a1) / c.growth(a0)).powi(2);
        for (x, y) in p0.iter().zip(&p1).take(4) { println!("  k={:.3} ratio {:.2} expect {:.2}  P0={:.3e} Plin(a0)={:.3e}", x.0, y.1 / x.1, expect, x.1, c.power(x.0) * c.growth(a0).powi(2)); }
        let t = Instant::now();
        let mut pm = cosmology::PmSim::new(c, 64, 200.0, 0.02, 3);
        for _ in 0..10 { pm.step(0.01); }
        println!("pm64 10 steps {:?}", t.elapsed());
    }
    if arg == "climate" || arg.is_empty() {
        for (n, opt) in [("earth", climate::ClimateOptions::default()), ("mars", climate::ClimateOptions::default()), ("venus", climate::ClimateOptions::default())] {
            let w = world::World::preset(n).unwrap();
            let t = Instant::now();
            let r = climate::simulate(&w, &opt);
            let eq = r.annual_mean[r.lat.len() / 2];
            let np = r.annual_mean[r.lat.len() - 1];
            println!("{n}: global {:.1} eq {:.1} pole {:.1} ice {:.1} resid {:.2e} imb {:.3} D {:.3} C {:.2e} {:?}", r.global_mean, eq, np, r.ice_edge, r.energy_residual, r.toa_imbalance, r.d, r.heat_capacity, t.elapsed());
            // seasonal amplitude at 45N
            let j = r.lat.iter().position(|l| *l > 45.0).unwrap();
            let s: Vec<f64> = r.temp.iter().map(|row| row[j]).collect();
            println!("   45N min {:.1} max {:.1}", s.iter().cloned().fold(1e9, f64::min), s.iter().cloned().fold(-1e9, f64::max));
        }
        let w = world::World::preset("earth").unwrap();
        let r = climate::simulate(&w, &climate::ClimateOptions { initial_t: Some(220.0), years: 60, ..Default::default() });
        println!("earth cold start: global {:.1} ice {:.1}", r.global_mean, r.ice_edge);
        let r = climate::simulate(&w, &climate::ClimateOptions { solar_scale: 0.9, years: 60, ..Default::default() });
        println!("earth S*0.9: global {:.1} ice {:.1}", r.global_mean, r.ice_edge);
    }
}
