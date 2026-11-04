//! worldsim — physical models of worlds for simulation and data generation.
//!
//! ```no_run
//! use worldsim::world::{World, WorldSpec};
//! // Mars with Earth atmosphere and lunar regolith:
//! let w = World::build(&WorldSpec { physics: "mars".into(), atmosphere: Some("earth".into()),
//!     surface: Some("moon".into()), gravity_scale: 1.0, distance_au: None }).unwrap();
//! println!("{} g={:.2} T={:.1}", w.name(), w.g_surface(), w.mean_surface_temperature());
//! ```

pub mod atmosphere;
pub mod ballistics;
pub mod blackhole;
pub mod chemistry;
pub mod climate;
pub mod collisions;
pub mod collect;
pub mod constants;
pub mod cosmology;
pub mod datagen;
pub mod dem;
pub mod env;
pub mod ffi;
pub mod fluid;
pub mod galaxy;
pub mod json;
pub mod lab;
pub mod magneto;
pub mod mission;
pub mod nbody;
pub mod orbit;
pub mod presets;
pub mod quantum;
pub mod rng;
pub mod server;
pub mod stars;
pub mod store;
pub mod starsystem;
pub mod tectonics;
pub mod thermal;
pub mod validate;
pub mod vec3;
pub mod world;
