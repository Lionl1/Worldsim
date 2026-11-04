//! Мир = физика тела + атмосфера + поверхность (можно брать от разных тел).
//! Производные величины (температура, τ, g) пересчитываются из законов, а не копируются.

use crate::atmosphere::Atmosphere;
use crate::constants::{AU, L_SUN, SIGMA_SB};
use crate::presets::{self, PhysicsPreset, SurfacePreset};
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct World {
    pub physics: PhysicsPreset,
    pub atmosphere: Option<Atmosphere>,
    pub surface: SurfacePreset,
    pub star_luminosity: f64, // Вт
    pub star_distance: f64,   // м
}

#[derive(Clone, Debug)]
pub struct WorldSpec {
    pub physics: String,
    pub atmosphere: Option<String>,
    pub surface: Option<String>,
    pub gravity_scale: f64,
    pub distance_au: Option<f64>,
}

impl WorldSpec {
    pub fn preset(name: &str) -> Self {
        WorldSpec { physics: name.into(), atmosphere: None, surface: None, gravity_scale: 1.0, distance_au: None }
    }
}

impl World {
    /// Мир целиком из одного пресета: `World::preset("mars")`.
    pub fn preset(name: &str) -> Result<World, String> { World::build(&WorldSpec::preset(name)) }

    pub fn build(spec: &WorldSpec) -> Result<World, String> {
        let mut physics = presets::physics(&spec.physics)?;
        physics.gm *= spec.gravity_scale;
        let atmo_name = spec.atmosphere.clone().unwrap_or_else(|| spec.physics.clone());
        let atmo_preset = presets::atmosphere(&atmo_name).or_else(|e| {
            if spec.atmosphere.is_none() { Ok(None) } else { Err(e) }
        })?;
        let surf_name = spec.surface.clone().unwrap_or_else(|| spec.physics.clone());
        let surface = presets::surface(&surf_name).or_else(|e| {
            if spec.surface.is_none() { presets::surface("earth") } else { Err(e) }
        })?;
        let star_distance = match spec.distance_au {
            Some(d) => d * AU,
            None => heliocentric_distance(&physics)?,
        };
        let mut w = World { physics, atmosphere: None, surface, star_luminosity: L_SUN, star_distance };
        if let Some(ap) = atmo_preset {
            let t0 = if ap.fixed_temperature { ap.reference_temperature } else { w.surface_temperature_with(Some(&ap)) };
            w.atmosphere = Some(Atmosphere::build(&ap, w.physics.gm, w.physics.radius, t0)?);
        }
        Ok(w)
    }

    pub fn name(&self) -> String {
        let a = self.atmosphere.as_ref().map(|a| a.preset.name.as_str()).unwrap_or("none");
        format!("physics={} atmosphere={} surface={}", self.physics.name, a, self.surface.name)
    }

    pub fn g_surface(&self) -> f64 { self.physics.gm / self.physics.radius.powi(2) }
    pub fn g_at(&self, h: f64) -> f64 { self.physics.gm / (self.physics.radius + h).powi(2) }
    pub fn escape_velocity(&self) -> f64 { (2.0 * self.physics.gm / self.physics.radius).sqrt() }
    pub fn circular_velocity(&self, h: f64) -> f64 { (self.physics.gm / (self.physics.radius + h)).sqrt() }
    pub fn omega(&self) -> f64 { 2.0 * PI / self.physics.rotation_period }

    /// Солнечная постоянная на орбите (закон обратных квадратов).
    pub fn insolation(&self) -> f64 { self.star_luminosity / (4.0 * PI * self.star_distance.powi(2)) }

    /// Равновесная температура (Стефан–Больцман, глобальное среднее).
    pub fn equilibrium_temperature(&self) -> f64 {
        (self.insolation() * (1.0 - self.physics.bond_albedo) / (4.0 * SIGMA_SB)).powf(0.25)
    }

    /// τ масштабируется пропорционально массе столба газа P0/g.
    pub fn greenhouse_tau_with(&self, ap: Option<&presets::AtmospherePreset>) -> f64 {
        match ap {
            Some(a) => a.greenhouse_tau * a.reference_g / self.g_surface(),
            None => 0.0,
        }
    }

    fn surface_temperature_with(&self, ap: Option<&presets::AtmospherePreset>) -> f64 {
        // серая атмосфера Эддингтона: T_s^4 = T_eq^4 (1 + 3τ/4)
        self.equilibrium_temperature() * (1.0 + 0.75 * self.greenhouse_tau_with(ap)).powf(0.25)
    }

    pub fn greenhouse_tau(&self) -> f64 { self.greenhouse_tau_with(self.atmosphere.as_ref().map(|a| &a.preset)) }

    pub fn mean_surface_temperature(&self) -> f64 {
        match &self.atmosphere {
            Some(a) => a.t0,
            None => self.equilibrium_temperature(),
        }
    }

    pub fn density(&self, h: f64) -> f64 { self.atmosphere.as_ref().map(|a| a.at(h).density).unwrap_or(0.0) }
}

/// Расстояние до звезды: большая полуось тела или его родителя.
pub fn heliocentric_distance(p: &PhysicsPreset) -> Result<f64, String> {
    match &p.orbit {
        None => Ok(AU),
        Some(o) if o.parent == "sun" => Ok(o.a),
        Some(o) => heliocentric_distance(&presets::physics(&o.parent)?),
    }
}
