# 🌌 worldsim

**Open physics in Rust: from space mission analysis to checking trained world models. Each law is checked automatically.**

*Open, dependency-free physics in Rust: Mission Lab turns mission requirements into a scorecard and technology trade study, and a qualification ladder tells you when a learned world model can be trusted. Includes robot environments, counterfactual data, a 3D viewer from planets to cosmology, and 129 automated physics checks.*

**For mission engineers:** [Mission Lab](#mission-lab) answers whether the requirements are met, what breaks the mission and which technology pays off. **For model developers:** [data collection and qualification](#world-models-data-and-qualification) show how many seconds a model can be trusted and whether it sees crashes in advance. What it can do, what it cannot and which research it builds on is described in [docs/PRODUCT.md](docs/PRODUCT.md).

```bash
git clone https://github.com/Lionl1/worldsim && cd worldsim
./tools/fetch_data.sh        # once: download the star and exoplanet catalogs (requires git and python3)
cargo run --release          # builds and opens the 3D interface in the browser (http://localhost:8080)
```

Only Rust ≥ 1.75 is required, there are no external crates. The catalogs (the “Map” tab) are downloaded by a script from open repositories. The browser needs internet only to load three.js from CDN.

## Mission Lab

```bash
cargo run --release -- mission missions/lunar_lander.json --out report
```

A mission description sets the vehicle, dispersions, requirements and technology options. Ready examples are in [missions/](missions/). The result is also available on the “Mission Lab” tab in the interface.

| Question | What you get |
|---|---|
| Is the mission feasible? | Requirements scorecard with a 95% interval: met, not met or not enough data |
| How does it fail? | Failure modes: high touchdown speed, tip-over and others |
| What breaks it? | The effect of each dispersion on its own |
| What to buy? | Paired comparison of technologies under the same conditions: what was saved, what was lost, how much it weighs |

Example answer for the lunar lander: success 89.3%, 0 of 4 requirements met, while a 7 kN engine gives +6.0 ± 3.2 pp for 25 kg and closes two requirements.

The physics is simplified: this is a tool for early design decisions, not a replacement for high-fidelity analysis and testing.

In a mission description, `vehicle` holds registry parameters (see `worldsim params`), `dispersions` holds relative spreads and `technologies` lists the options to compare. `dry_mass_delta` in a technology exists only for the lander; on other vehicles set the mass through `set`.

## What is inside

| Tab | Model | Laws |
|---|---|---|
| 🪐 **Worlds** | Earth, Moon, Mars, Venus, Mercury, Jupiter, Sun + any mixes (physics ≠ atmosphere ≠ crust) | gravity, J2, Coriolis, hydrostatics, ideal gas, Archimedes, drag, stoichiometry |
| 🌡️ **Climate** | seasonal latitudinal energy balance model for any world | insolation from the orbit, Stefan–Boltzmann, Eddington greenhouse atmosphere, ice-albedo, 1st law |
| ☀️ **Solar system** | N bodies from J2000 elements | Newton, Yoshida symplectic integrator, GR (1PN) |
| ✨ **Star systems** | generator: star from the Kroupa IMF, planets from Hill stability | mass–luminosity, Planck → color, Kopparapu habitable zone, Chen–Kipping mass–radius |
| 🔭 **Map** | 11,270 HYG stars + 5,283 exoplanets | any exoplanet into a world model in one click |
| 🌀 **Galaxies** | Milky Way, galaxy collision, clusters | Barnes–Hut O(N log N), live dark halo, virial, rotation curve |
| 🕸️ **Cosmology** | ΛCDM, structure growth from the Big Bang | Friedmann, Zeldovich initial conditions, BBKS/σ₈ spectrum, particle-mesh through FFT |
| 🚀 **Mission Lab** | mission analysis against requirements | Monte Carlo, Wilson intervals, paired comparisons |
| 🤖 **Robots** | lander, drone, rover in real time | Tsiolkovsky, rotor momentum theory, Coulomb friction, leg contact, stability |
| ⚫ **Black holes** | ray rendering along geodesics, Schwarzschild and Kerr | ISCO, shadow, Hawking, Peters waves |
| 🔬 **Laboratory** | tunneling, shock wave, dam, ozone, nozzle, craters, tectonics | exact reference solutions |
| ✅ **Laws** | 129 tests | see below |

## Law checks

```bash
cargo run --release -- validate     # table, ~2–3 s
cargo test --release                # same, as tests
```

Examples of what is checked and what comes out:

| Law | Test | Result |
|---|---|---|
| Energy conservation | Solar system, 1 year | ΔE/E ≈ 2·10⁻¹³ |
| GR | Mercury perihelion precession | 42.7″/century (theory 43.0″) |
| Coriolis | fall from 100 m at the equator | 21.977 mm (theory 21.977) |
| Hydrostatics | Earth pressure at 11 km | 22,646 Pa (US Std 22,632) |
| Heat + radiation | Moon, equator | 98–386 K (Diviner ≈ 95–390) |
| Climate | Earth: mean T / ice edge | 289.6 K / 75° |
| Bistability | cold start of Earth | “snowball Earth” |
| Virial | Plummer sphere | 2K/|W| = 1.006 |
| Rotation curve | Milky Way at 8 kpc | 212 km/s |
| Friedmann | age of the Universe (Planck 2018) | 13.786 Gyr |
| Structure growth | P(k) in PM against D(a)² | within 4% |
| Planck → Stefan–Boltzmann, Wien | numerical integration of the spectrum | 10⁻⁸ / 10⁻⁶ |
| Tsiolkovsky equation | lander, full fuel burn | Δv = 1093.18 m/s (theory 1093.14) |
| Rotor momentum theory | drone on Mars: ordinary rotors / area ×40 | −3.41 / +8.88 m/s (falls / takes off) |
| Leg contact | lander stands on Mars | accelerometer shows g, ±3% |
| Stability v·ω/g | same rover turn | tips over on the Moon, not on Earth |
| Branching accuracy | environment copy against the original | match to within 10⁻¹² |
| Tunnel effect | wave packet through a barrier | matches the exact formula, ±3% |
| Chandrasekhar limit | white dwarf, μₑ = 2 | 1.43 M☉ |
| Black hole shadow | critical ray | 3√3 M |
| Gravitational waves | PSR B1913+16 | dP/dt agrees with GR (−2.40·10⁻¹²) |
| Sod problem | shock wave against the exact solution | L1 error < 1% |
| Chapman cycle | steady ozone | matches the analytic value, ±2% |

## Command line

```bash
worldsim info --physics mars --atmosphere earth --surface moon   # mixed world
worldsim throw mars --speed 100 --elev 45 --out traj.csv
worldsim thermal moon --lat 0
worldsim particle earth --species proton --mev 1 --L 3 --pitch 40
worldsim system --bodies sun,earth,moon,mars,jupiter --years 10 --gr
worldsim dataset --task projectile --n 10000 --worlds earth,moon,mars,mars:earth:moon --noise 0.01 --out data/proj
worldsim mission missions/lunar_lander.json --out report
worldsim collect --task lander --worlds moon,mars:earth:moon --episodes 1000 --randomize --counterfactuals 3 --out runs/data
worldsim serve --port 8080
```

Dataset tasks: `projectile`, `orbit`, `thermal`, `particle`, `system`. Each sample has
`validation_error` and `valid`, and `meta.json` stores worlds, seed and constants.

## Environment for training robots

Three tasks in the Gymnasium style. The agent sees only noisy sensor readings and outputs continuous actuator commands,
and the physics is the same as in the rest of worldsim, so one agent can train on different and mixed worlds.

| Task | Target | Actions | Observations |
|---|---|---|---|
| `lander` | land softly on a pad | `throttle` 0…1, `torque_x/y/z` −1…1 | 35: GPS, speed, attitude, gyroscope, radar altimeter, accelerometer, fuel, pressure, lidar, ray hit flags |
| `drone` | fly to a point and hover | `rotor_1…4` 0…1 | 20: goal, speed, attitude, gyroscope, radar altimeter, accelerometer, battery, pressure, air density, hit flag |
| `rover` | drive to a goal over terrain | `left`, `right` −1…1 | 44: goal, heading, speed, tilt, battery, Sun elevation, temperature, pressure, lidar, hit flags |

The lander stands on four spring legs, so the model sees the contact itself. The drone starts in the air and can land and take off again. The rover tips over if the slope and turning inertia exceed the stability margin.

Difficulty is added by wind gusts (`gust`), actuator lag (`lag`), hidden actuator fault (`fault`) and changing the world between episodes (`worlds`).

### HTTP API

```bash
worldsim serve --port 8080 --no-open

curl 'http://localhost:8080/api/env/make?task=lander&physics=moon&seed=42'   # returns the session id, names and bounds
curl 'http://localhost:8080/api/env/step?id=1&action=0.5,0,0,0'              # one action
curl 'http://localhost:8080/api/env/step?id=1&mode=auto&repeat=100&text=1'   # 100 autopilot steps + text description
```

`step` returns `obs`, `reward`, `terminated`, `truncated` and `info`. The server keeps the connection open (keep-alive), so
an agent can send many requests in a row over one connection. Numbers out of the allowed range give an error 400.

The server holds up to 1024 environments and 64 heavy simulations. When there is no room, the least recently used session of the same kind is removed. When you finish, call `close`.

| Request `/api/env/…` | Parameters |
|---|---|
| `tasks`, `list` | — |
| `make` | `task`, `physics`, `worlds`, `atmosphere`, `surface`, `gravity_scale`, `randomize=1`, `noise` (0 — ideal sensors), `rotor_scale`, `max_steps`, `frame_skip`, `gust`, `lag`, `fault`, `seed` |
| `reset` | `id`, `seed` |
| `step` | `id`, `action=a,b,…` or `mode=auto\|random`, `repeat` (1–100), `text=1`, `traj=1` (all intermediate steps) |
| `state` | `id`: also the true state and hidden parameters |
| `fork` | `id`: environment copy, to play another action from the same point |
| `close` | `id` |
| `terrain` | `id`, `n` (grid 17–257) |
| `camera` | `id`, `w`, `h` (4–256), `fov`, `mode=depth\|rgb` |

### From Rust

```rust
use worldsim::env::{Env, EnvConfig};
let mut cfg = EnvConfig::new("lander", "moon")?;
cfg.randomize = true;                  // random parameters in each episode
let mut env = Env::new(cfg)?;
env.reset(42);
loop {
    let action = env.autopilot();      // replace with your policy: observations → action
    let (_obs, _reward, terminated, truncated) = env.step(&action);
    if terminated || truncated { break; }
}
println!("{}: {:.1}", env.info.reason, env.ep_return);
```

### Baseline

A trained agent can be compared with these numbers. The built-in PD autopilot, successful episodes out of 10, with random
parameters and the standard rotor size:

| Task | Earth | Moon | Mars |
|---|---|---|---|
| `lander` | 10 | 10 | 10 |
| `drone` | 10 | 0 | 0 |
| `rover` | 8 | 6 | 4 |

On the Moon a drone has nothing to push against: there is no air. On Mars ordinary rotors lack atmospheric density. To repeat the measurement:
`cargo run --release --example probe_env`.

## World models: data and qualification

```bash
cargo build --release
worldsim collect --task lander --worlds moon --episodes 500 --randomize --noise 0 --counterfactuals 3 --out runs/moon
worldsim collect --task lander --worlds earth --episodes 500 --randomize --noise 0 --counterfactuals 3 --out runs/earth
worldsim collect --task lander --worlds mars:earth:moon --episodes 300 --randomize --noise 0 --counterfactuals 3 --out runs/mix

cd python && pip install -e '.[baseline]'
python -m worldsim.baseline train --data ../runs/moon ../runs/earth --out ../runs/base.pt
python -m worldsim.baseline qualify --model ../runs/base.pt --world moon=../runs/moon --world mix=../runs/mix --out ../runs/report
```

`collect` writes `.npy` and `meta.json`: observations, actions, true state, hidden parameters and episode end reasons. The result does not depend on thread count. With `--counterfactuals N` branches are added at N points per 1000 steps: different actions from one point with the true response.

You can connect your own model of any architecture:

```python
from worldsim import Dataset, WorldModel, qualify, to_markdown

class MyModel(WorldModel):
    name = "my-rssm"
    def init(self, obs, params):      # initial state from observations
        return encode(obs)
    def step(self, state, action):    # a step in imagination
        state = dynamics(state, action)
        return state, decode(state)   # prediction in sensor channels

report = qualify(MyModel(), {"moon": Dataset("runs/moon"), "mix": Dataset("runs/mix")})
print(to_markdown(report))
```

| Level | What it checks |
|---|---|
| L1 Accuracy | trust horizon under mission tolerances |
| L2 Physics | law violations in imagination relative to the simulator |
| L3 Response | action effect against true branches |
| L4 Optimism | missed crashes and false alarms |
| L5 Boundaries | same checks in worlds outside training |

For training without files there is a vector environment that works without HTTP:

```python
from worldsim import VectorEnv
env = VectorEnv("lander", "moon,mars,mars:earth:moon", n=64, randomize=True, gust=2.0, threads=8)
obs = env.reset()
obs, reward, terminated, truncated, info = env.step(env.autopilot())
```

## To work with your system

**Custom controller.** Same flights and requirements, but your system controls. The report also shows a paired comparison with the built-in autopilot:

```python
import json
from worldsim import MissionRunner

runner = MissionRunner(json.load(open("missions/lunar_lander.json")))
report = runner.evaluate(lambda obs: my_policy(obs), name="my-gnc")   # obs: N×O, response: N×A
print(report["success"], report["delta_vs_autopilot"], report["scorecard"])
```

**Telemetry calibration.** Parameters are fitted so the model reproduces a recorded flight. The result contains uncertainties and a flag for whether the data determines the parameters:

```bash
cd python && python -m worldsim.calibrate --task lander --world moon --csv flight.csv \
  --fit isp:200:400 --fit t_max:1000:50000 --set terrain_scale=0,crater_scale=0 --out params.json
```

The CSV needs columns `act_0…` and state columns (`x y z vx vy vz`, optional `mass`). The table step equals the control step. Verified only on synthetic recordings: not yet run on real telemetry. A good residual does not prove that the simplified physics is right for your vehicle. From code, use `worldsim.fit_telemetry`.

**Terrain of a real site.** Formats `.asc` (ESRI ASCII Grid), `.csv`, `.npy`; the map center coincides with the pad:

```bash
worldsim collect --task rover --worlds moon --terrain-file site.asc --episodes 100 --out runs/site
```

In the mission description: `"terrain": {"file": "site.asc"}` for a local run or `"terrain": {"n": 65, "size": 800, "heights": [...]}` through the API. The server does not read files by a path from the request.

**Export for training.** LeRobotDataset v3 format, loading verified on lerobot 0.6.1. Requires `pyarrow` and `pandas`:

```bash
cd python && python -m worldsim.export --data ../runs/moon --out ../runs/moon_lerobot
```

**Interface settings.** Parameters, requirements and technology options are edited in the forms of the “Robots” and “Mission Lab” tabs. Presets can be saved and imported there too, and the “Settings” tab has theme, text size, panel width and graphics quality. All of this is stored in the browser.

**Parameters.** A list of all configurable quantities with units and ranges: `worldsim params --task lander`. They work in `collect --set`, in a mission description, in API (`params=isp=320,gust=2`) and in `VectorEnv(params={...})`.

## As a library

```rust
use worldsim::{world::World, climate::{simulate, ClimateOptions}};
let mars = World::preset("mars")?;
let c = simulate(&mars, &ClimateOptions::default());
println!("Mars: {:.0} K, CO₂ caps from {:.0}°", c.global_mean, c.ice_edge);
```

## Layout

```
src/
  constants  presets  world  atmosphere      — CODATA constants, presets, world assembly
  ballistics  thermal  magneto  nbody  orbit  — mechanics, heat, particles, N bodies
  stars  starsystem                           — stars, system generator
  climate                                     — seasonal energy balance model
  galaxy                                      — Barnes–Hut, galaxy initial conditions
  cosmology                                   — Friedmann, FFT, particle-mesh
  env                                         — environment for training robots: lander, drone, rover
  quantum  blackhole  fluid  chemistry         — quantum, black holes, fluid dynamics, chemistry
  collisions  tectonics  lab                  — impacts, tectonics, experiments for the interface
  validate                                    — 129 law checks
  mission  collect  ffi                       — Mission Lab, transition collection, vector environment for Python
  datagen  server  json  main                 — datasets, HTTP server, CLI
web/index.html                                — 3D interface (three.js), built into the binary
missions/                                     — example missions
python/worldsim/                              — client, vector environment, data, qualification, baseline model
docs/PRODUCT.md                               — product, research, limitations, plans
data/                                         — HYG and OEC catalogs (downloaded by tools/fetch_data.sh, have their own licenses)
tools/fetch_data.sh, build_data.py           — downloading and building data from catalogs
```

## Running for a team

```bash
docker build -t worldsim .
docker run -p 8080:8080 -e WORLDSIM_TOKEN=choose-a-long-token -v worldsim-data:/data worldsim
```

The star and exoplanet catalogs are not in the image. For the “Map” tab, run `./tools/fetch_data.sh` and add `-v "$PWD/data:/opt/worldsim/data:ro"`. No release has been published yet, so until then use the image or build from source.

Or a prebuilt binary from the releases page:

```bash
WORLDSIM_TOKEN=choose-a-long-token worldsim serve --host 0.0.0.0 --data-dir /var/lib/worldsim --log
```

| What | How |
|---|---|
| Access | header `Authorization: Bearer <token>`; on an external address the server will not start without a token |
| Liveness check | `GET /healthz` or `worldsim health` |
| API description | `GET /api/openapi.json` |
| Long calculations | `POST /api/jobs/mission`, then `GET /api/jobs/{id}`; cancel — `POST /api/jobs/{id}/cancel` |
| History | `GET /api/runs`, comparison `GET /api/runs/compare?a=…&b=…`; files are in `--data-dir` |
| Requests from other sites | are rejected; can be allowed with the `--cors` flag |

The server does not support TLS: on a network put it behind a reverse proxy. Details in [SECURITY.md](SECURITY.md), change history in [CHANGELOG.md](CHANGELOG.md).

## Comparison with alternatives

Universe Sandbox, SpaceEngine, OpenSpace, REBOUND, GADGET-4/SWIFT, ExoPlaSim — in [COMPARISON.md](COMPARISON.md):
where worldsim is stronger, where it falls behind and what is planned. A comparison with space mission tools and a gap table: [docs/COMPETITORS.md](docs/COMPETITORS.md).

## Model boundaries

Classical mechanics, thermodynamics, electromagnetism, SR, GR (1PN, plus Schwarzschild geodesics and Kerr formulas), nonrelativistic single-particle quantum mechanics, 1D fluid dynamics, chemical kinetics, scaling laws of collisions and tectonics.

| Section | What is there | What is not |
|---|---|---|
| Quantum mechanics | 1D Schrödinger equation, tunneling, hydrogen, electron degeneracy | many-particle systems, spin, fields, relativistic quantum theory |
| Black holes | ISCO, shadow, light deflection, Hawking, Peters waves, render | numerical GR, mergers, render with hole rotation, polarization |
| Fluid dynamics | 1D Euler and shallow water with exact references | 3D, turbulence, global circulation |
| Chemistry | kinetics, Chapman cycle, ideal nozzle | chemical equilibria of real propellants, combustion |
| Collisions | scaling laws of disruption and craters, merging | calculating the disruption of a specific body (SPH) |
| Tectonics | plate cooling, seafloor depth, order of convection speed | plate motion, volcanism |

Presets are mean values from open sources (NASA Fact Sheets, JPL, US Std Atmosphere 1976, Planck 2018 and others).

## License

The code is MIT. Data in `data/` — under their own licenses: HYG — CC BY-SA 4.0, Open Exoplanet Catalogue — MIT.
