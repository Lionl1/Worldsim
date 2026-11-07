# worldsim compared with alternatives

An honest comparison: where worldsim is stronger, where it falls behind and why.

## In short

Large projects usually do **one** of three things: show things beautifully (SpaceEngine, OpenSpace),
compute **one** class of problems accurately (REBOUND for orbits, GADGET/SWIFT for cosmology, ExoPlaSim for climate),
or provide a game-like sandbox (Universe Sandbox). worldsim bridges the gap between them: **one open
program** in which scales from a stone throw to the cosmic web are computed from physical
laws, each law is **checked automatically** (158 tests against analytic solutions and observations),
and any simulation turns into a **dataset** for machine learning.

## Table

| | worldsim | Universe Sandbox | SpaceEngine | OpenSpace | REBOUND | GADGET-4 / SWIFT | ExoPlaSim |
|---|---|---|---|---|---|---|---|
| License | **MIT, open source** | closed, $29.99 | closed, paid | MIT | GPL-3 | GPL-3 / LGPL-3 | GPL-3 |
| What it does | simulation + visuals + data | sandbox simulator | visualization, procedural universe | scientific data visualization | N bodies | cosmological simulations | exoplanet climate |
| Scales | throw → climate → systems → galaxies → cosmology | planets → galaxies | planets → universe | spacecraft → universe | planetary systems | clusters → cosmology | planetary atmosphere |
| Automatic law checks | **158 tests in one command** | no | no | n/a | code tests | code tests | no |
| Dataset generation | **built in, with a validity flag** | no | no | no | through Python | manual | manual |
| Preset mixing (physics ≠ atmosphere ≠ crust) | **yes, recomputed from the laws** | partial (materials) | no | no | no | no | parameters |
| Real catalogs | HYG (11 thousand stars), 5.3 thousand exoplanets | own | HIPPARCOS, NGC, NASA Exoplanet Archive | Gaia and others | no | no | no |
| Real exoplanet → world model | **in one click** | manual | visual only | no | no | no | manual |
| Orbit integrators | IAS15, Wisdom–Holman, Yoshida, leapfrog | own | — | — | **IAS15, WHFast, MERCURIUS and more** | — | — |
| GR | 1PN (Mercury 42.7″/century) | no | no | no | 1PN (REBOUNDx) | no | no |
| Climate | seasonal EBM for any world, “snowball” hysteresis, one-layer winds | simplified | no | no | no | no | **full GCM** |
| Gas | SPH, thousands of particles, no cooling | yes | no | no | no | **SPH with cooling and more, billions** | **atmosphere** |
| Galaxies | Barnes–Hut, live dark halo | yes | procedural (not dynamics) | data | no | **TreePM/FMM, SPH** | no |
| Dependencies | **zero** (only std Rust) | Unity engine | — | C++, many | C | MPI, FFTW, HDF5 | Fortran, Python |
| Launch | `cargo run --release` | installer | installer | installer | `pip install` | cluster | `pip install` |

## Where worldsim is better

1. **Verifiability.** None of the visual simulators listed shows the user
   whether the laws hold. In worldsim the “Laws” tab recomputes 158 tests in about 5 seconds.
2. **End-to-end chain.** A real exoplanet from a catalog → a star with color from the Planck spectrum →
   a world with an atmosphere from hydrostatics → seasonal climate → a throw with Coriolis. In other packages this means
   several programs and manual transfer of parameters.
3. **Data for ML.** `worldsim dataset` outputs CSV and metadata where each sample has an error
   of the conservation law and a validity flag, plus seed reproducibility and measurement noise. An environment for
   robots (`lander`, `drone`, `rover`) lets one agent train on different and mixed worlds.
4. **Self-contained.** No dependencies, one binary, the interface is built in.

## Space and world models

This is a different group of tools, so it is kept out of the table above. Details and links are in [docs/PRODUCT.md](docs/PRODUCT.md).

- **Space Robotics Bench, MarsLab, MARE** offer much richer graphics and physics on Isaac Sim or Unreal Engine. They are made first of all for training and perception. worldsim is much simpler in physics, but has requirements analysis and model evaluation built in, and it runs on a laptop without GPU.
- **MiraBench, CrashTwin, WorldArena** evaluate world models on video and on manipulation or collision tasks. worldsim evaluates in sensor channels, against mission tolerances and with true counterfactual branches. It is not yet built for video models.
- **POST2 and similar landing analysis tools** are far more accurate and proven on real missions. Mission Lab borrows their approach of scorecards and factor sweeps, but is meant only for early design decisions.

## Where worldsim falls behind

- **Scale and maturity of “narrow” codes.** worldsim now has IAS15 and a Wisdom–Holman map, checked on eccentric orbits and against each other, but no close-encounter handling or high-order correctors, so REBOUND stays the choice for long or chaotic orbital work.
  GADGET-4 and SWIFT compute billions of particles with hydrodynamics on clusters. worldsim runs up to 128³ (2.1 million) mesh particles at about 0.25 s per step on 8 cores, P3M at 64³ at about 2 s per step, and its SPH gas handles thousands of particles and is not coupled to cosmology.
- **Climate.** ExoPlaSim is a full general circulation model. worldsim has a latitudinal energy balance model plus one-layer winds on a sphere and a rule-of-thumb cloud cover. The winds pass exact tests of the solver, but the jets of Earth come out too weak and too far from the equator, and clouds do not feed back on temperature. Venus and giant planets are refused, because it cannot represent them. Do not use it for climate prediction.
- **Graphics.** SpaceEngine and OpenSpace are visually richer: photorealistic surfaces, flight
  to any star. worldsim draws surfaces procedurally, with color computed from crust chemistry and climate.
  Earth and the Moon can use NASA photo maps at 2048×1024; other bodies have none.
- **Sandbox.** Universe Sandbox lets you collide planets and melt crust in real time. worldsim has live collisions in N-body and in the Laboratory, with outcomes from a scaling law and exact mass and momentum, but the fragments are schematic and nothing melts.

## What is planned to close the gaps

- A hybrid integrator for close encounters.
- A tree for the long range and gas coupled to cosmology.
- More layers and moisture in the atmosphere, with clouds feeding back on temperature.
- Photo maps for more bodies, from sources with stable addresses.
- Impacts through SPH instead of schematic fragments.

## Sources

- Universe Sandbox — [Steam](https://store.steampowered.com/app/230290/Universe%20Sandbox): $29.99, Windows and macOS, N bodies, ice and seasons, collisions, materials.
- SpaceEngine — [spaceengine.org](https://www.spaceengine.org): procedural universe on the HIPPARCOS, NGC/IC and NASA Exoplanet Archive catalogs; distributed through Steam.
- OpenSpace — [openspaceproject.com](https://openspaceproject.com/about/about.html): open (MIT) visualization of NASA and ESA data; AMNH, Linköping University, NASA Goddard.
- REBOUND — [PyPI](https://pypi.org/project/rebound/2.16.6/): C99 + Python, GPL-3, IAS15, WHFast and leapfrog integrators.
- GADGET-4 — [Springel et al. 2021](https://arxiv.org/abs/2010.03567); SWIFT — [Schaller et al. 2023](https://arxiv.org/pdf/2305.13380).
- ExoPlaSim — Paradise et al. 2022 (MNRAS), general circulation model for exoplanets.
