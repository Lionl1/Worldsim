//! worldsim — физические модели миров для симуляции и генерации данных.
//!
//! ```no_run
//! use worldsim::world::{World, WorldSpec};
//! // Марс с земной атмосферой и лунным реголитом:
//! let w = World::build(&WorldSpec { physics: "mars".into(), atmosphere: Some("earth".into()),
//!     surface: Some("moon".into()), gravity_scale: 1.0, distance_au: None }).unwrap();
//! println!("{} g={:.2} T={:.1}", w.name(), w.g_surface(), w.mean_surface_temperature());
//! ```

pub mod atmosphere;
pub mod ballistics;
pub mod climate;
pub mod constants;
pub mod cosmology;
pub mod datagen;
pub mod galaxy;
pub mod magneto;
pub mod nbody;
pub mod orbit;
pub mod presets;
pub mod rng;
pub mod stars;
pub mod starsystem;
pub mod thermal;
pub mod validate;
pub mod vec3;
pub mod world;
