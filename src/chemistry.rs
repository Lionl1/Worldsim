//! Chemistry: kinetics and equilibrium, the Chapman ozone cycle, a rocket nozzle and specific impulse.
//! Chapman rate coefficients are typical textbook values for the stratosphere, not an inventory for forecasting.

use crate::constants::R_GAS;

/// Arrhenius law: k = A·exp(−Ea/RT).
pub fn arrhenius(a: f64, ea: f64, t: f64) -> f64 { a * (-ea / (R_GAS * t)).exp() }

/// Reversible reaction A ⇌ B, RK4: (A, B) at time t. The exact solution is checked in the laws.
pub fn reversible(a0: f64, b0: f64, kf: f64, kr: f64, t: f64, steps: usize) -> (f64, f64) {
    let h = t / steps as f64;
    let (mut a, mut b) = (a0, b0);
    let f = |a: f64, b: f64| -kf * a + kr * b;
    for _ in 0..steps {
        let k1 = f(a, b);
        let k2 = f(a + h / 2.0 * k1, b - h / 2.0 * k1);
        let k3 = f(a + h / 2.0 * k2, b - h / 2.0 * k2);
        let k4 = f(a + h * k3, b - h * k3);
        let d = h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        a += d;
        b -= d;
    }
    (a, b)
}

/// Chapman cycle at fixed air density M.
/// O₂ + hν → 2O (j2); O + O₂ + M → O₃ + M (k2); O₃ + hν → O + O₂ (j3); O + O₃ → 2O₂ (k4).
#[derive(Clone, Copy)]
pub struct Chapman { pub j2: f64, pub k2: f64, pub j3: f64, pub k4: f64, pub m: f64 }

impl Chapman {
    /// Typical stratosphere around 30 km, CGS (cm⁻³, s⁻¹).
    pub fn stratosphere() -> Chapman { Chapman { j2: 1e-11, k2: 1e-33, j3: 1e-3, k4: 1e-15, m: 3.6e17 } }

    fn rates(&self, y: [f64; 3]) -> [f64; 3] {
        let [o, o2, o3] = y;
        let (r1, r2, r3, r4) = (self.j2 * o2, self.k2 * o * o2 * self.m, self.j3 * o3, self.k4 * o * o3);
        [2.0 * r1 - r2 + r3 - r4, -r1 - r2 + r3 + 2.0 * r4, r2 - r3 - r4]
    }

    /// Implicit Euler step with a numerical Jacobian: the system is very stiff (O lives seconds, O₃ lives hours).
    pub fn implicit_step(&self, y: [f64; 3], dt: f64) -> [f64; 3] {
        let mut x = y;
        for _ in 0..12 {
            let f = self.rates(x);
            let g = [x[0] - y[0] - dt * f[0], x[1] - y[1] - dt * f[1], x[2] - y[2] - dt * f[2]];
            let mut jm = [[0.0; 3]; 3];
            for c in 0..3 {
                let e = x[c].abs().max(1.0) * 1e-6;
                let mut xp = x;
                xp[c] += e;
                let fp = self.rates(xp);
                for r in 0..3 { jm[r][c] = (if r == c { 1.0 } else { 0.0 }) - dt * (fp[r] - f[r]) / e; }
            }
            let d = solve3(jm, [-g[0], -g[1], -g[2]]);
            for k in 0..3 { x[k] += d[k]; }
        }
        x
    }

    /// Steady O₃/O₂ ratio in the Chapman approximation: √(j₂ k₂ M / (j₃ k₄)).
    pub fn steady_ratio(&self) -> f64 { (self.j2 * self.k2 * self.m / (self.j3 * self.k4)).sqrt() }
}

fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> [f64; 3] {
    for i in 0..3 {
        let p = (i..3).max_by(|x, y| a[*x][i].abs().total_cmp(&a[*y][i].abs())).unwrap();
        a.swap(i, p);
        b.swap(i, p);
        for r in i + 1..3 {
            let f = a[r][i] / a[i][i];
            for c in i..3 { a[r][c] -= f * a[i][c]; }
            b[r] -= f * b[i];
        }
    }
    let mut x = [0.0; 3];
    for i in (0..3).rev() {
        let mut v = b[i];
        for c in i + 1..3 { v -= a[i][c] * x[c]; }
        x[i] = v / a[i][i];
    }
    x
}

/// Ideal nozzle: exhaust velocity, m/s. tc is chamber temperature, m is molar mass, kg/mol, pr = pe/pc.
pub fn exhaust_velocity(gamma: f64, tc: f64, molar: f64, pr: f64) -> f64 {
    (2.0 * gamma / (gamma - 1.0) * R_GAS * tc / molar * (1.0 - pr.powf((gamma - 1.0) / gamma))).sqrt()
}

/// Nozzle expansion ratio for a pressure ratio pr.
pub fn area_ratio(gamma: f64, pr: f64) -> f64 {
    let g = gamma;
    let a = ((g + 1.0) / 2.0).powf(1.0 / (g - 1.0)) * pr.powf(1.0 / g);
    1.0 / (a * ((g + 1.0) / (g - 1.0) * (1.0 - pr.powf((g - 1.0) / g))).sqrt())
}

/// Specific impulse from propellant chemistry: links the reaction to the parameter `isp` of the lander.
pub fn specific_impulse(gamma: f64, tc: f64, molar: f64, pr: f64) -> f64 { exhaust_velocity(gamma, tc, molar, pr) / crate::env::G0 }
