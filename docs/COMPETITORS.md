# Comparison with alternatives and gaps

Status as of November 2026. Information comes from public pages and papers, linked below. Prices of closed products could not be found, so they are not given. We did not run these tools; the table reflects their descriptions, not our own measurements.

## In short

worldsim does not replace these tools and should not try to. In orbit accuracy, flight heritage and photorealism it is far behind, and part of that gap cannot be closed with code. Its niche is narrower: **early comparison of mission options and checking trained models against exact physics on a single mission description**.

## Who does what

| Tool | What it is | Strength | License |
|---|---|---|---|
| [STK](https://www.refontelearning.com/blog/astrodynamics-software-tools-gmat-stk-poliastro) | commercial mission scenario platform | access and coverage, sensors, communications, terrain | closed |
| [GMAT](https://wikipedia.com/wiki/General_Mission_Analysis_Tool) | NASA tool for trajectory analysis | trajectory optimization, used on LCROSS, OSIRIS-REx, TESS | Apache 2.0 |
| [FreeFlyer](https://au.mathworks.com/products/connections/product_detail/freeflyer.html) | commercial system for satellites and constellations | coverage and contacts, maneuvers, orbit determination | closed |
| [Basilisk](https://hanspeterschaub.info/basilisk/) | modular vehicle dynamics framework | bit-for-bit repeatable Monte Carlo, guidance development, hardware synchronization | open, check the repository |
| [POST2](https://ntrs.nasa.gov/citations/20240006746) | NASA trajectory tool for landing | Monte Carlo statistics against requirements, real mission heritage | on NASA terms |
| [Sedaro](https://satnews.com/2026/06/24/digital-twins-for-space-command-sedaro-tapped-by-space-force-to-build-federated-engineering-platform/) | cloud digital twins of spacecraft | coupled subsystems, Space Force contracts | closed |
| [OmniLRS](https://github.com/OmniLRS/OmniLRS) | lunar simulator on Isaac Sim | photorealism, procedural environments, ROS 1 and 2 | open |
| [Space Robotics Bench](https://arxiv.org/abs/2509.23328) | environments for training robots in space | massive GPU parallelism, sim-to-real transfer | open |
| worldsim | mission analysis and model qualification | requirements, value of technologies, true branches, runs without GPU | MIT |

## Capability comparison

“Yes” and “no” mean: that is what the descriptions we found say. A dash means we found no information, which is not a claim that the capability is missing.

| Capability | worldsim | GMAT, STK, FreeFlyer | Basilisk | POST2 | Sedaro | OmniLRS, SRB |
|---|---|---|---|---|---|---|
| Exact orbits and ephemerides | **weak**: N bodies, J2 | yes | yes | yes | — | no |
| Proven on real missions | **no** | yes (GMAT) | — | yes | — | **partial** (transfer to a robot, SRB) |
| Monte Carlo requirements scorecard | yes | — | has Monte Carlo | yes | — | no |
| Value of technologies, paired comparison | yes | — | — | — | — | no |
| Coupled subsystems: power, communications, thermal | **weak**: battery and bus load only; no communications or thermal | partial | yes | — | yes | planned (OmniLRS) |
| Working with real flight software and hardware | **no** | — | yes | — | yes | ROS (OmniLRS) |
| Photorealism | **no** | STK visualization | Vizard | no | CesiumJS | yes |
| Deformable soil | **no** | no | no | no | — | in development |
| Massive parallelism | CPU | — | multiprocess Monte Carlo | — | cloud | GPU |
| Qualification of trained models | yes | no | no | no | — | no |
| True counterfactual branches | yes | no | — | no | — | no |
| Mixed worlds | yes | no | — | no | — | no |
| Automatic checks of physical laws | yes, 133 | code tests | code tests | — | — | — |
| Running without installing dependencies | yes | installer | Python and a build | limited, NASA | cloud | GPU and Isaac Sim |

## Gaps and what we do about them

| # | Gap | Importance for companies | Solution | Status |
|---|---|---|---|---|
| 1 | No check against real data | **critical** | Telemetry calibration | tool exists, verified on synthetic data; **no check against a real flight** |
| 2 | Our autopilot is evaluated, not the customer system | **critical** | `MissionRunner.analyze`: custom controller, scorecard, main drivers, value of technologies, comparison with the autopilot | closed; a controller in Python only, no flight software |
| 3 | Terrain is procedural only | high | Loading `.asc`, `.csv`, `.npy` | closed |
| 4 | No industrial delivery: CI, image, versions, API | high | CI, Docker, releases, OpenAPI | done, see the repository for CI results; no releases published yet |
| 5 | No token access or limits; server open to requests from other sites | high | Token, limits, origin check | closed; TLS and roles not done, no independent audit |
| 6 | Long calculations block a request, runs are not saved | high | Background jobs, run storage and comparison | closed |
| 7 | Configuration only through JSON | medium | Forms from the parameter registry, presets | closed |
| 8 | Data exchange with the training ecosystem | medium | LeRobot export, verified on lerobot 0.6.1 | closed, without camera |
| 9 | Power, communications, thermal | medium | Bus load and battery on all vehicles; the rest is separate work, also pointed out by the [simulator survey](https://arxiv.org/html/2609.34743v1) | power closed; **communications and thermal not closed** |
| 10 | No work with flight software and hardware | medium, for later stages | Real-time input, a ROS 2 bridge | **not closed** |
| 11 | Exact orbits | low for the niche | Do not compete; import a trajectory from GMAT or STK | **not closed** |
| 12 | Photorealism | low for the niche | Use OmniLRS where vision is needed | **not closed** |
| 13 | Flight heritage, certification, support | **critical for trust** | Cannot be closed with code: needs pilot projects with companies | **not closed** |

## What this means for sales

Honest position: “Use us before bringing in high-fidelity tools: to discard weak options and learn whether a model can be trusted. Do the final check with flight-proven tools”. Promising a replacement for STK or POST2 is not possible.
