//! Live collisions in N-body: bodies that touch merge or break up by the Leinhardt–Stewart scaling law.
//!
//! The outcome and the remnant mass come from the law; fragments are a schematic set of six equal bodies, not a hydrodynamic
//! result: there is no SPH here, no melting and no ejecta shape. Mass, momentum and the center of mass stay exact.

use crate::collisions;
use crate::constants::G;
use crate::nbody::{Body, Integrator, NBody};
use crate::vec3::Vec3;
use std::f64::consts::PI;

/// Below this share of the total mass, fragments are not worth tracking: the whole mass stays in the remnant.
const MIN_FRAGMENT_SHARE: f64 = 1e-4;

#[derive(Clone, Debug)]
pub struct Event { pub t: f64, pub a: String, pub b: String, pub outcome: String, pub speed: f64, pub remnant_share: f64, pub heat: f64 }

#[derive(Clone, Debug)]
pub struct ImpactSim { pub sim: NBody, pub radius: Vec<f64>, pub events: Vec<Event> }

fn radius_of(mass: f64, density: f64) -> f64 { (3.0 * mass / (4.0 * PI * density)).cbrt() }

impl ImpactSim {
    /// `radius` matches `sim.bodies`.
    pub fn new(mut sim: NBody, radius: Vec<f64>) -> Result<ImpactSim, String> {
        if radius.len() != sim.bodies.len() { return Err("one radius per body is required".into()); }
        if sim.relativity_center.is_some() { return Err("live collisions do not support the GR correction".into()); }
        sim.integrator = Integrator::Ias15;
        Ok(ImpactSim { sim, radius, events: vec![] })
    }

    /// A target and an impactor on a collision course, in the center-of-mass frame.
    pub fn two_bodies(target_mass: f64, target_radius: f64, mass_ratio: f64, speed_over_escape: f64, impact_parameter: f64, density: f64) -> Result<ImpactSim, String> {
        let m2 = target_mass * mass_ratio;
        let r2 = radius_of(m2, density);
        let reach = target_radius + r2;
        let v_esc = (2.0 * G * (target_mass + m2) / reach).sqrt();
        let start = 3.0 * reach;
        let (mu1, mu2) = (target_mass / (target_mass + m2), m2 / (target_mass + m2));
        let rel = Vec3::new(start, impact_parameter * reach, 0.0);
        let vel = Vec3::new(-speed_over_escape * v_esc, 0.0, 0.0);
        let body = |name: &str, m: f64, r: Vec3, v: Vec3| Body { name: name.into(), gm: G * m, r, v };
        Self::new(NBody::new(vec![body("target", target_mass, rel * -mu2, vel * -mu2), body("impactor", m2, rel * mu1, vel * mu1)]), vec![target_radius, r2])
    }

    pub fn step(&mut self, dt: f64) {
        let before: Vec<Vec3> = self.sim.bodies.iter().map(|b| b.r).collect();
        self.sim.step(dt);
        // bodies that touched during the step, not only at its end: relative motion is taken as straight within a step
        let mut hit = None;
        for i in 0..self.sim.bodies.len() {
            for j in (i + 1)..self.sim.bodies.len() {
                let (d0, d1) = (before[j] - before[i], self.sim.bodies[j].r - self.sim.bodies[i].r);
                let dd = d1 - d0;
                let s = if dd.norm2() > 0.0 { (-d0.dot(dd) / dd.norm2()).clamp(0.0, 1.0) } else { 1.0 };
                if (d0 + dd * s).norm() <= self.radius[i] + self.radius[j] { hit = Some((i, j)); break; }
            }
            if hit.is_some() { break; }
        }
        if let Some((i, j)) = hit { self.resolve(i, j); }
    }

    fn resolve(&mut self, i: usize, j: usize) {
        let (a, b) = (self.sim.bodies[i].clone(), self.sim.bodies[j].clone());
        let (m1, m2) = (a.gm / G, b.gm / G);
        let total = m1 + m2;
        let volume = 4.0 / 3.0 * PI * (self.radius[i].powi(3) + self.radius[j].powi(3));
        let density = total / volume;
        let speed = (a.v - b.v).norm();
        let q = collisions::impact_energy(m1, m2, speed);
        let q_star = collisions::disruption_energy(total, density, 1.9);
        let share = collisions::largest_remnant_fraction(q, q_star).clamp(0.0, 1.0);
        let center = (a.r * m1 + b.r * m2) / total;
        let velocity = (a.v * m1 + b.v * m2) / total;
        let kinetic = 0.5 * m1 * m2 / total * speed * speed;
        let keep = if collisions::outcome(q, q_star) == "merger" || (1.0 - share) * total / 6.0 < MIN_FRAGMENT_SHARE * total { 1.0 } else { share.max(1e-3) };
        let mut names = format!("{}+{}", a.name, b.name);
        if names.len() > 40 { names = "remnant".into(); }
        let remnant_r = radius_of(total * keep, density);
        let mut born = vec![(Body { name: names, gm: G * total * keep, r: center, v: velocity }, remnant_r)];
        let mut frag_kinetic = 0.0;
        if keep < 1.0 {
            let piece = total * (1.0 - keep) / 6.0;
            let r_piece = radius_of(piece, density);
            let v_esc = (2.0 * G * total / radius_of(total, density)).sqrt();
            // half of the impact energy can go into the motion of fragments; the rest is heat
            let v = (1.1 * v_esc).min((0.5 * kinetic / (6.0 * 0.5 * piece)).sqrt());
            let reach = (remnant_r + r_piece) * 1.3;
            for (k, dir) in [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0)].into_iter().enumerate() {
                born.push((Body { name: format!("fragment {}", self.events.len() * 6 + k + 1), gm: G * piece, r: center + dir * reach, v: velocity + dir * v }, r_piece));
            }
            frag_kinetic = 6.0 * 0.5 * piece * v * v;
        }
        let outcome = if keep >= 1.0 { "merger" } else { collisions::outcome(q, q_star) };
        self.events.push(Event { t: self.sim.t, a: a.name, b: b.name, outcome: outcome.to_string(), speed, remnant_share: keep, heat: kinetic - frag_kinetic });
        self.sim.bodies.remove(j);
        self.radius.remove(j);
        self.sim.bodies.remove(i);
        self.radius.remove(i);
        for (body, r) in born { self.sim.bodies.push(body); self.radius.push(r); }
        self.sim.reset_step();
    }
}
