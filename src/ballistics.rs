//! Flight near a world surface in the rotating planetary frame.
//! Laws: Newtonian gravity + J2 oblateness, Coriolis force, centrifugal force,
//! aerodynamic drag (½ρC_dAv²), Archimedes force, atmosphere from hydrostatics.
//! Integration: Dormand–Prince 5(4) with an adaptive step.
//! Energy accounting: E(t) − E(0) = work of non-conservative forces (first law), which is checked.

use crate::vec3::Vec3;
use crate::world::World;

#[derive(Clone, Copy, Debug)]
pub struct Projectile {
    pub mass: f64,   // kg
    pub area: f64,   // m^2 (frontal area)
    pub cd: f64,     // drag coefficient
    pub volume: f64, // m^3 (for the Archimedes force)
}

impl Projectile {
    /// Sphere of radius r and density rho.
    pub fn sphere(radius: f64, density: f64) -> Self {
        let vol = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
        Projectile { mass: vol * density, area: std::f64::consts::PI * radius * radius, cd: 0.47, volume: vol }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Launch {
    pub lat: f64,       // degrees
    pub lon: f64,       // degrees
    pub altitude: f64,  // m
    pub speed: f64,     // m/s relative to the surface
    pub elevation: f64, // degrees above the horizon
    pub azimuth: f64,   // degrees clockwise from north
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub rotating: bool,
    pub j2: bool,
    pub drag: bool,
    pub buoyancy: bool,
    pub max_time: f64,
    pub rtol: f64,
    pub sample_dt: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options { rotating: true, j2: true, drag: true, buoyancy: true, max_time: 1.0e5, rtol: 1e-10, sample_dt: 0.1 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub t: f64,
    pub lat: f64,
    pub lon: f64,
    pub altitude: f64,
    pub speed: f64,
    pub mach: f64,
    pub downrange: f64,
    pub pos: Vec3,
    pub vel: Vec3,
}

#[derive(Clone, Debug)]
pub struct Result {
    pub samples: Vec<Sample>,
    pub flight_time: f64,
    pub downrange: f64,
    pub max_altitude: f64,
    pub impact_speed: f64,
    pub landed: bool,
    pub energy_error: f64,      // max |E − E0 − W_nc| / scale
    pub dissipated_energy: f64, // J, lost to heat (minus the work of drag)
}

type State = [f64; 7]; // r, v, W_nc (per unit mass)

struct Model<'a> {
    w: &'a World,
    p: Projectile,
    o: Options,
    omega: Vec3,
}

impl<'a> Model<'a> {
    fn grav(&self, r: Vec3) -> Vec3 {
        let mu = self.w.physics.gm;
        let rn = r.norm();
        let mut a = r * (-mu / rn.powi(3));
        if self.o.j2 {
            let rr = self.w.physics.radius;
            let k = -1.5 * self.w.physics.j2 * mu * rr * rr / rn.powi(5);
            let z2 = r.z * r.z / (rn * rn);
            a += Vec3::new(r.x * (1.0 - 5.0 * z2), r.y * (1.0 - 5.0 * z2), r.z * (3.0 - 5.0 * z2)) * k;
        }
        a
    }
    fn potential(&self, r: Vec3) -> f64 {
        let mu = self.w.physics.gm;
        let rn = r.norm();
        let mut phi = -mu / rn;
        if self.o.j2 {
            let rr = self.w.physics.radius;
            phi += mu * self.w.physics.j2 * rr * rr * (3.0 * r.z * r.z - rn * rn) / (2.0 * rn.powi(5));
        }
        if self.o.rotating { phi -= 0.5 * self.omega.cross(r).norm2(); }
        phi
    }
    fn centrifugal(&self, r: Vec3) -> Vec3 { -self.omega.cross(self.omega.cross(r)) }
    fn altitude(&self, r: Vec3) -> f64 { r.norm() - self.w.physics.radius }

    fn deriv(&self, s: &State) -> State {
        let r = Vec3::new(s[0], s[1], s[2]);
        let v = Vec3::new(s[3], s[4], s[5]);
        let g = self.grav(r);
        let mut a = g;
        let v_air = if self.o.rotating { v } else { v - self.omega.cross(r) };
        if self.o.rotating {
            a += self.centrifugal(r) - self.omega.cross(v) * 2.0;
        }
        let mut a_nc = Vec3::ZERO;
        let rho = self.w.density(self.altitude(r));
        if rho > 0.0 {
            if self.o.drag {
                a_nc += v_air * (-0.5 * rho * self.p.cd * self.p.area * v_air.norm() / self.p.mass);
            }
            if self.o.buoyancy && self.p.volume > 0.0 {
                let g_eff = g + self.centrifugal(r);
                a_nc += g_eff * (-rho * self.p.volume / self.p.mass);
            }
        }
        a += a_nc;
        [v.x, v.y, v.z, a.x, a.y, a.z, a_nc.dot(v)]
    }

    fn energy(&self, s: &State) -> f64 {
        let r = Vec3::new(s[0], s[1], s[2]);
        let v = Vec3::new(s[3], s[4], s[5]);
        0.5 * v.norm2() + self.potential(r)
    }
}

fn axpy(y: &State, h: f64, ks: &[(&State, f64)]) -> State {
    let mut o = *y;
    for (k, c) in ks {
        for i in 0..7 { o[i] += h * c * k[i]; }
    }
    o
}

/// One Dormand–Prince step: (y5, error estimate)
fn dopri_step(m: &Model, y: &State, h: f64) -> (State, f64) {
    let k1 = m.deriv(y);
    let k2 = m.deriv(&axpy(y, h, &[(&k1, 1.0 / 5.0)]));
    let k3 = m.deriv(&axpy(y, h, &[(&k1, 3.0 / 40.0), (&k2, 9.0 / 40.0)]));
    let k4 = m.deriv(&axpy(y, h, &[(&k1, 44.0 / 45.0), (&k2, -56.0 / 15.0), (&k3, 32.0 / 9.0)]));
    let k5 = m.deriv(&axpy(y, h, &[(&k1, 19372.0 / 6561.0), (&k2, -25360.0 / 2187.0), (&k3, 64448.0 / 6561.0), (&k4, -212.0 / 729.0)]));
    let k6 = m.deriv(&axpy(y, h, &[(&k1, 9017.0 / 3168.0), (&k2, -355.0 / 33.0), (&k3, 46732.0 / 5247.0), (&k4, 49.0 / 176.0), (&k5, -5103.0 / 18656.0)]));
    let y5 = axpy(y, h, &[(&k1, 35.0 / 384.0), (&k3, 500.0 / 1113.0), (&k4, 125.0 / 192.0), (&k5, -2187.0 / 6784.0), (&k6, 11.0 / 84.0)]);
    let k7 = m.deriv(&y5);
    let e = [71.0 / 57600.0, 0.0, -71.0 / 16695.0, 71.0 / 1920.0, -17253.0 / 339200.0, 22.0 / 525.0, -1.0 / 40.0];
    let ks = [&k1, &k2, &k3, &k4, &k5, &k6, &k7];
    let rtol = m.o.rtol;
    let mut err: f64 = 0.0;
    for i in 0..6 {
        let ei: f64 = h * (0..7).map(|j| e[j] * ks[j][i]).sum::<f64>();
        let scale_r = m.w.physics.radius * rtol + 1e-6;
        let scale_v = y[3..6].iter().map(|x| x.abs()).fold(0.0, f64::max) * rtol + 1e-9;
        let sc = if i < 3 { scale_r.max(y[i].abs() * rtol) } else { scale_v.max(y[i].abs() * rtol) };
        err = err.max((ei / sc).abs());
    }
    (y5, err)
}

fn hermite(y0: &State, d0: &State, y1: &State, d1: &State, h: f64, th: f64) -> (Vec3, Vec3) {
    let (t2, t3) = (th * th, th * th * th);
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + th;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    let p = |i: usize| h00 * y0[i] + h10 * h * d0[i] + h01 * y1[i] + h11 * h * d1[i];
    let dp = |i: usize| ((6.0 * t2 - 6.0 * th) * y0[i] + (3.0 * t2 - 4.0 * th + 1.0) * h * d0[i]
        + (-6.0 * t2 + 6.0 * th) * y1[i] + (3.0 * t2 - 2.0 * th) * h * d1[i]) / h;
    (Vec3::new(p(0), p(1), p(2)), Vec3::new(dp(0), dp(1), dp(2)))
}

pub fn simulate(world: &World, proj: Projectile, launch: Launch, opts: Options) -> Result {
    let omega = Vec3::new(0.0, 0.0, world.omega());
    let m = Model { w: world, p: proj, o: opts, omega };
    let rr = world.physics.radius;
    let (phi, lam) = (launch.lat.to_radians(), launch.lon.to_radians());
    let up = Vec3::new(phi.cos() * lam.cos(), phi.cos() * lam.sin(), phi.sin());
    let east = Vec3::new(-lam.sin(), lam.cos(), 0.0);
    let north = up.cross(east);
    let (el, az) = (launch.elevation.to_radians(), launch.azimuth.to_radians());
    let r0 = up * (rr + launch.altitude);
    let mut v0 = (east * az.sin() + north * az.cos()) * (launch.speed * el.cos()) + up * (launch.speed * el.sin());
    if !opts.rotating { v0 += omega.cross(r0); } // in the inertial frame add the surface velocity
    let mut y: State = [r0.x, r0.y, r0.z, v0.x, v0.y, v0.z, 0.0];
    let e0 = m.energy(&y);
    let ke0 = 0.5 * v0.norm2();
    let mut max_dphi: f64 = 0.0;
    let mut max_res: f64 = 0.0;
    let mut t = 0.0;
    let mut h = 1e-3_f64.max(opts.sample_dt * 0.1);
    let mut samples = Vec::new();
    let mut next_sample = 0.0;
    let mut max_alt = launch.altitude;
    let mut landed = false;

    let to_sample = |t: f64, r: Vec3, v: Vec3, m: &Model| -> Sample {
        let rn = r.norm();
        let mut lon = r.y.atan2(r.x);
        let v_ground = if opts.rotating { v } else { v - omega.cross(r) };
        if !opts.rotating { lon -= omega.z * t; }
        let ground_dir = Vec3::new((r.z / rn).asin().cos() * lon.cos(), (r.z / rn).asin().cos() * lon.sin(), r.z / rn);
        let alt = rn - rr;
        let cs = m.w.atmosphere.as_ref().map(|a| a.at(alt).sound_speed).unwrap_or(f64::NAN);
        Sample {
            t, lat: (r.z / rn).asin().to_degrees(), lon: lon.to_degrees().rem_euclid(360.0), altitude: alt,
            speed: v_ground.norm(), mach: v_ground.norm() / cs,
            downrange: rr * up.dot(ground_dir).clamp(-1.0, 1.0).acos(), pos: r, vel: v,
        }
    };

    loop {
        let d0 = m.deriv(&y);
        if t >= next_sample - 1e-12 && samples.is_empty() {
            samples.push(to_sample(t, Vec3::new(y[0], y[1], y[2]), Vec3::new(y[3], y[4], y[5]), &m));
            next_sample += opts.sample_dt;
        }
        h = h.min(opts.max_time - t);
        let (mut y1, err) = dopri_step(&m, &y, h);
        if err > 1.0 {
            h *= (0.9 * err.powf(-0.2)).max(0.2);
            continue;
        }
        let mut h_used = h;
        // surface contact event: bisection within the step
        if m.altitude(Vec3::new(y1[0], y1[1], y1[2])) < 0.0 {
            let (mut lo, mut hi) = (0.0, h);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                let (ym, _) = dopri_step(&m, &y, mid);
                if m.altitude(Vec3::new(ym[0], ym[1], ym[2])) < 0.0 { hi = mid; } else { lo = mid; }
            }
            h_used = hi;
            y1 = dopri_step(&m, &y, hi).0;
            landed = true;
        }
        let d1 = m.deriv(&y1);
        while next_sample <= t + h_used && !samples.is_empty() {
            let th = (next_sample - t) / h_used;
            let (r, v) = hermite(&y, &d0, &y1, &d1, h_used, th);
            samples.push(to_sample(next_sample, r, v, &m));
            next_sample += opts.sample_dt;
        }
        t += h_used;
        y = y1;
        let r = Vec3::new(y[0], y[1], y[2]);
        max_alt = max_alt.max(m.altitude(r));
        let dphi = (m.potential(r) - (e0 - ke0)).abs();
        max_dphi = max_dphi.max(dphi);
        max_res = max_res.max((m.energy(&y) - e0 - y[6]).abs());
        if landed || t >= opts.max_time || r.norm() > 1e3 * rr { break; }
        if err > 0.0 { h *= (0.9 * err.powf(-0.2)).clamp(0.2, 5.0); } else { h *= 5.0; }
        h = h.min(opts.sample_dt.max(1e-3) * 50.0);
    }
    let last = to_sample(t, Vec3::new(y[0], y[1], y[2]), Vec3::new(y[3], y[4], y[5]), &m);
    if samples.last().map(|s| s.t < t).unwrap_or(true) { samples.push(last); }
    Result {
        flight_time: t,
        downrange: last.downrange,
        max_altitude: max_alt,
        impact_speed: last.speed,
        landed,
        energy_error: max_res / ke0.max(max_dphi).max(1e-30),
        dissipated_energy: -y[6] * proj.mass,
        samples,
    }
}
