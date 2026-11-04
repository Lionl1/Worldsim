//! Experiments for the Laboratory tab: a thin wrapper over the physics modules, all ranges are checked here.

use crate::{chemistry as ch, collisions as co, fluid as fl, quantum as qm, tectonics as te};
use crate::constants::YEAR;
use crate::json::J;
use std::collections::HashMap;

type Q = HashMap<String, String>;

pub fn num(q: &Q, k: &str, d: f64, lo: f64, hi: f64) -> Result<f64, String> {
    let x = match q.get(k).filter(|v| !v.is_empty()) { Some(v) => v.parse::<f64>().map_err(|_| format!("{k}: expected a number, got '{v}'"))?, None => d };
    if !x.is_finite() || x < lo || x > hi { return Err(format!("{k} = {x} out of range [{lo}, {hi}]")); }
    Ok(x)
}

fn thin(v: &[f64], every: usize) -> Vec<f64> { v.iter().step_by(every).copied().collect() }

pub fn run(topic: &str, q: &Q) -> Result<J, String> {
    match topic {
        "tunnel" => {
            let (e, v0, a) = (num(q, "e", 0.5, 0.01, 5.0)?, num(q, "v0", 0.8, 0.0, 10.0)?, num(q, "a", 1.0, 0.1, 10.0)?);
            let k0 = (2.0 * e).sqrt();
            let mut w = qm::Wave::gaussian(-150.0, 150.0, 6000, 0.02, -60.0, 12.0, k0, move |x| if (0.0..a).contains(&x) { v0 } else { 0.0 });
            let travel = (60.0 + 90.0) / k0 / 0.02;
            let steps = travel as usize;
            let mut frames = vec![];
            for i in 0..steps {
                w.step();
                if i % (steps / 40).max(1) == 0 { frames.push(J::from(thin(&w.density(), 12))); }
            }
            Ok(J::obj().set("x", thin(&w.x, 12)).set("v", thin(&w.v, 12)).set("frames", J::Arr(frames)).set("t_exact", qm::barrier_transmission(e, v0, a)).set("t_numeric", w.right_of(a)).set("norm", w.norm()))
        }
        "levels" => {
            let osc: Vec<f64> = (0..6).map(|k| qm::bound_level_1d(|x| 0.5 * x * x, 8.0, 2000, k)).collect();
            let hyd: Vec<f64> = (0..3).map(|k| qm::hydrogen_s_level(k, 90.0, 6000)).collect();
            let mut rows: Vec<J> = osc.iter().enumerate().map(|(k, e)| J::obj().set("name", format!("Oscillator, n = {k}")).set("value", *e).set("exact", k as f64 + 0.5)).collect();
            rows.extend(hyd.iter().enumerate().map(|(k, e)| J::obj().set("name", format!("Hydrogen, n = {}", k + 1)).set("value", *e).set("exact", -0.5 / ((k + 1) as f64).powi(2))));
            Ok(J::obj().set("rows", J::Arr(rows)).set("series", J::Arr(vec![
                J::obj().set("name", "Oscillator").set("n", (0..6).map(|k| k as f64).collect::<Vec<_>>()).set("e", osc),
                J::obj().set("name", "Hydrogen").set("n", (0..3).map(|k| k as f64).collect::<Vec<_>>()).set("e", hyd),
            ])))
        }
        "hydrogen" => {
            let lines = [("Lyman-α", 1.0, 2.0), ("Lyman-β", 1.0, 3.0), ("Balmer-α", 2.0, 3.0), ("Balmer-β", 2.0, 4.0), ("Paschen-α", 3.0, 4.0)];
            Ok(J::obj().set("lines", J::Arr(lines.iter().map(|(n, a, b)| J::obj().set("name", *n).set("nm", qm::hydrogen_line_nm(*a, *b))).collect()))
                .set("levels", J::Arr((1..=8).map(|n| J::obj().set("n", n as f64).set("ev", qm::hydrogen_level_ev(n as f64))).collect())))
        }
        "sod" => {
            let t = num(q, "t", 0.2, 0.001, 0.3)?;
            let mut s = fl::Euler1D::sod(400);
            let m0: f64 = s.rho.iter().sum();
            s.run_to(t);
            let x: Vec<f64> = (0..400).map(|i| (i as f64 + 0.5) * s.dx).collect();
            let exact: Vec<f64> = x.iter().map(|x| fl::sod_exact(*x, s.t).0).collect();
            let l1 = s.rho.iter().zip(&exact).map(|(a, b)| (a - b).abs() * s.dx).sum::<f64>();
            let me = (s.rho.iter().sum::<f64>() - m0).abs() / m0;
            Ok(J::obj().set("x", x).set("rho", s.rho).set("exact", exact).set("l1", l1).set("mass_error", me))
        }
        "dam" => {
            let (t, h0) = (num(q, "t", 4.0, 0.1, 6.0)?, num(q, "h0", 1.0, 0.1, 10.0)?);
            let (g, l) = (9.81, 200.0);
            let mut d = fl::Shallow1D::dam(800, l, g, h0);
            d.run_to(t);
            let x: Vec<f64> = (0..800).map(|i| (i as f64 + 0.5) * d.dx).collect();
            let exact: Vec<f64> = x.iter().map(|x| fl::ritter_depth(*x, l / 2.0, d.t, g, h0)).collect();
            let err = d.h.iter().zip(&exact).map(|(a, b)| (a - b).abs() * d.dx).sum::<f64>() / (h0 * l / 2.0);
            Ok(J::obj().set("x", x).set("h", d.h).set("exact", exact).set("err", err).set("front", 2.0 * (g * h0).sqrt()))
        }
        "chapman" => {
            let hours = num(q, "hours", 2000.0, 1.0, 20000.0)?;
            let c = ch::Chapman::stratosphere();
            let mut y = [1e7, 0.21 * c.m, 1e12];
            let a0 = y[0] + 2.0 * y[1] + 3.0 * y[2];
            let steps = 200;
            let (mut t, mut o3) = (vec![], vec![]);
            for i in 0..steps {
                y = c.implicit_step(y, hours / steps as f64 * 3600.0);
                t.push((i + 1) as f64 * hours / steps as f64);
                o3.push(y[2]);
            }
            Ok(J::obj().set("t", t).set("o3", o3).set("ratio", y[2] / y[1]).set("steady", c.steady_ratio()).set("atoms", (y[0] + 2.0 * y[1] + 3.0 * y[2] - a0).abs() / a0))
        }
        "nozzle" => {
            let (g, tc, m) = (num(q, "gamma", 1.2, 1.05, 1.67)?, num(q, "tc", 3600.0, 300.0, 6000.0)?, num(q, "molar", 13.5, 1.0, 100.0)? / 1e3);
            let (mut ratio, mut isp) = (vec![], vec![]);
            for i in 0..80 {
                let pr = 10f64.powf(-0.5 - 4.0 * i as f64 / 79.0);
                ratio.push(ch::area_ratio(g, pr));
                isp.push(ch::specific_impulse(g, tc, m, pr));
            }
            Ok(J::obj().set("ratio", ratio).set("isp", isp).set("isp_sl", ch::specific_impulse(g, tc, m, 1.0 / 200.0)).set("ve_vac", ch::exhaust_velocity(g, tc, m, 1e-12)))
        }
        "crater" => {
            let (d, v, ang, g) = (num(q, "d", 50.0, 0.1, 1e5)?, num(q, "v", 13.0, 0.1, 100.0)? * 1e3, num(q, "angle", 45.0, 1.0, 90.0)?, num(q, "g", 9.81, 0.01, 50.0)?);
            let (tc, fin) = co::crater_diameter(7800.0, 2500.0, d, v, g, ang);
            let m = 7800.0 * std::f64::consts::PI / 6.0 * d.powi(3);
            let ds: Vec<f64> = (0..50).map(|i| 10f64.powf(i as f64 / 49.0 * 4.3)).collect();
            let fs: Vec<f64> = ds.iter().map(|x| co::crater_diameter(7800.0, 2500.0, *x, v, g, ang).1 / 1e3).collect();
            Ok(J::obj().set("transient", tc).set("final", fin).set("energy_mt", 0.5 * m * v * v / 4.184e15).set("ds", ds).set("fs", fs))
        }
        "collide" => {
            let me = 5.972e24;
            let (m1, m2, v) = (num(q, "m1", 1.0, 1e-4, 1000.0)? * me, num(q, "m2", 0.1, 1e-4, 1000.0)? * me, num(q, "v", 12.0, 0.1, 200.0)? * 1e3);
            let qs = co::disruption_energy(m1 + m2, 5500.0, 1.9);
            let qr = co::impact_energy(m1, m2, v);
            let vs: Vec<f64> = (1..80).map(|i| i as f64).collect();
            let fr: Vec<f64> = vs.iter().map(|x| co::largest_remnant_fraction(co::impact_energy(m1, m2, x * 1e3), qs)).collect();
            Ok(J::obj().set("q", qr).set("q_star", qs).set("remnant", co::largest_remnant_fraction(qr, qs)).set("outcome", co::outcome(qr, qs)).set("vs", vs).set("fr", fr))
        }
        "ocean" => {
            let l = te::Lithosphere::earth();
            let myr = YEAR * 1e6;
            let age: Vec<f64> = (1..=120).map(|a| a as f64).collect();
            let depth: Vec<f64> = age.iter().map(|a| 2500.0 + l.subsidence(a * myr)).collect();
            let ra = te::rayleigh(4000.0, 9.8, 3e-5, 3000.0, 2.9e6, 1e-6, 1e22);
            Ok(J::obj().set("age", age).set("depth", depth).set("slope", l.subsidence(myr)).set("q80", l.heat_flow(80.0 * myr) * 1e3).set("ra", ra).set("speed", te::convection_speed(1e-6, 2.9e6, ra) * YEAR * 100.0))
        }
        _ => Err(format!("no such experiment: {topic}")),
    }
}
