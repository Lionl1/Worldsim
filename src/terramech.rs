//! Wheel on soft soil: Bekker pressure-sinkage, compaction resistance, Mohr–Coulomb shear with Janosi–Hanamoto slip.
//!
//! Rigid wheel, steady motion. The standard semi-empirical model for early rover studies, not a test-validated
//! prediction for a specific wheel: grousers, dynamic sinkage and slip sinkage are not modeled.

/// Soil and wheel. SI units: kc in N/m^(n+1), kphi in N/m^(n+2), cohesion in Pa, phi in radians.
#[derive(Clone, Debug)]
pub struct Soil { pub kc: f64, pub kphi: f64, pub n: f64, pub cohesion: f64, pub phi: f64, pub shear_k: f64, pub radius: f64, pub width: f64 }

/// One wheel under load.
#[derive(Clone, Debug)]
pub struct Wheel { pub sinkage: f64, pub contact: f64, pub compaction: f64, pub h_max: f64, pub shear_k: f64 }

impl Soil {
    pub fn wheel(&self, load: f64) -> Wheel {
        let (k, n, d) = (self.kc + self.width * self.kphi, self.n, 2.0 * self.radius);
        let sinkage = (3.0 * load / ((3.0 - n) * k * d.sqrt())).powf(2.0 / (2.0 * n + 1.0));
        Wheel { sinkage, contact: (d * sinkage).sqrt(), compaction: k * sinkage.powf(n + 1.0) / (n + 1.0),
            h_max: self.width * (d * sinkage).sqrt() * self.cohesion + load * self.phi.tan(), shear_k: self.shear_k }
    }
}

impl Wheel {
    /// Share of the maximum shear force developed at slip s: the mean of 1 − exp(−j/K) over the contact, j = s·x.
    pub fn shear_factor(&self, s: f64) -> f64 {
        if s <= 0.0 { return 0.0; }
        let x = s * self.contact / self.shear_k;
        1.0 - (1.0 - (-x).exp()) / x
    }
    /// Traction the wheel gives for a demanded force, and the slip it takes. Slip 1 means the demand exceeds what the soil holds.
    pub fn traction(&self, demand: f64) -> (f64, f64) {
        let top = self.h_max * self.shear_factor(1.0);
        if demand >= top { return (top, 1.0); }
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..40 {
            let s = 0.5 * (lo + hi);
            if self.h_max * self.shear_factor(s) < demand { lo = s; } else { hi = s; }
        }
        (demand, 0.5 * (lo + hi))
    }
}
