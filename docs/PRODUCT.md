# Mission Lab: a product for space companies

Working document: what data the product rests on, what it does and what it cannot do yet. Status as of November 2026.

## Research findings

Three areas develop separately, with a gap between them.

**1. World models have learned to be checked for physics, but not in space.**
[MiraBench](https://arxiv.org/abs/2605.29360) (May 2026) splits reliability into three levels: physical plausibility, action following and optimism detection. The authors report that visual quality predicts action following poorly and that optimism bias is common. [CrashTwin](https://arxiv.org/abs/2606.28757) (June 2026) checks momentum and energy conservation and shows that high image quality can hide physical violations. A [survey](https://arxiv.org/abs/2606.00133) (May 2026) names error accumulation, sim-to-real transfer and fragmented evaluation among the open problems.

**2. Model structure matters more for stability than its size.**
[A paper on conservation and factorization](https://www.alphaxiv.org/abs/2609.19674) (September 2026) reports that long-horizon stability and transfer to changed laws come from different structural choices, which can be set independently. [A definition and roadmap](https://arxiv.org/abs/2607.06401) (July 2026) stresses that the most important events in physical systems, such as crashes and rare failures, may be absent from training data altogether. So a simulator that can generate them is needed.

**3. World models are appearing in space, while environments are tailored to training.**
[DreamLander](https://arc.aiaa.org/doi/10.2514/6.2026-1035) (AIAA 2026) applies DreamerV3 to six-degree-of-freedom landing, so far in simulation. [Reward-Free Continual Adaptation](https://arxiv.org/abs/2608.23452) (August 2026) uses latent world models to adapt space robots after failures. [Space Robotics Bench](https://arxiv.org/abs/2509.23328) offers procedural environments on Isaac Lab. A [simulator survey](https://arxiv.org/html/2609.34743v1) (September 2026) asks for shared task definitions and clear evaluation procedures, as well as explicit modeling of communications, computing and power.

**4. Space engineering lives by requirements.**
[In NASA landing analysis](https://ntrs.nasa.gov/citations/20240006746) Monte Carlo statistics are shown as scorecards, and main drivers are found by one-at-a-time sweeps (OVAT). The [ECSS-E-HB-40-02A handbook](https://ecss.nl/home/ecss-e-hb-40-02a-15-november-2024/) (2024) describes how to build and verify machine learning functions, and an [ESA project](https://vnv-web-1b010b.io.esa.int/) with KP Labs and Airbus is developing a plan and tools for such verification.

**The gap:** a mission engineer asks “are the requirements met and what should we buy”, a researcher answers with a loss curve. We found no tool in this search that performs both the analysis and the model evaluation on one mission description. This is an observation from public sources, not proof that no competitors exist.

## What the product does

Mission description (`missions/*.json`) contains the vehicle, dispersions, requirements and technology options. Then three steps.

### 1. Is the mission feasible

Requirements scorecard with a 95% interval: **met**, **not met** or **not enough data**. The last verdict matters: with few runs the tool gives no false confidence.

### 2. What breaks it and what to buy

- **Failure modes:** not just “failure”, but high touchdown speed, tip-over and so on.
- **Main drivers:** each dispersion is enabled separately.
- **Value of technologies:** a paired comparison under the same flight conditions shows how many runs a technology saved and how many it spoiled, and what that costs in mass.

### 3. Can a trained model be trusted

Qualification ladder `python/worldsim/qualify.py`:

| Level | Question | How it is measured |
|---|---|---|
| L1 Accuracy | How many seconds can a prediction be trusted? | Trust horizon under mission tolerances |
| L2 Physics | Does the model violate laws in imagination? | Comparison with the level of the simulator itself |
| L3 Response | Does it respond to actions correctly? | R² of the action effect against true branches |
| L4 Optimism | Does it see a crash in advance? | Share of missed crashes and false alarms |
| L5 Boundaries | Where does the model apply? | The same levels in worlds absent from training |

### What sets it apart

1. **True counterfactual data.** The environment can be copied at any moment and different actions played from that point. Real video has no such answer.
2. **Mixed worlds as a transfer check.** Mars gravity with Earth air occurs nowhere, so a model cannot memorize it.
3. **One language with the engineer.** Model metrics are expressed in mission tolerances.
4. **Lightweight.** Rust with no dependencies, calculations on a laptop in seconds.

## Formats

| What | Format | Why |
|---|---|---|
| Mission description | JSON | Can be stored in git and versions compared |
| Transitions | `.npy` + `meta.json` | Readable by numpy with no dependencies, large files open through memmap |
| Branches | `cf_*.npy` | True response to different actions from one point |
| Environment in Python | `VectorEnv` through ctypes | No HTTP and no extra packages |
| Model interface | `init(obs, params)`, `step(state, action)` | The state is opaque: RSSM, transformer, diffusion or physical model all fit |
| Reports | `.md` and `.json` | For project review and for automation |

Models without a decoder, such as JEPA, do not output observations. They need a probe trained on top of the latent into sensor channels. This case is described in the interface, but has not been checked on a real JEPA.

Export to [LeRobotDataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3): Parquet and metadata, no camera. Loading verified on lerobot 0.6.1.

## Verified figures

Measured on a MacBook (8 cores), one seed. The numbers show the order of magnitude and depend on the task and machine.

| What | Result |
|---|---|
| Transition collection, `collect`, 8 threads | lander 271 thousand steps/s (219–252 thousand with branches), rover 297 thousand, drone 860 thousand |
| Vector environment from Python, 64 environments | 78 thousand steps/s (1 thread), 184 thousand (8 threads) |
| HTTP API, Python client | about 17 thousand steps/s per connection |
| Baseline MLP, Moon landing | trust horizon 0.46 s, 94% of crashes missed, 14% false alarms |
| Example `lunar_lander`, 300 runs | 89.3% success, 0 of 4 requirements; best technology: 7 kN engine, +6.0 ± 3.2 pp. |

The low result of the baseline MLP is expected: it is the simplest model for comparison. The tool shows where it breaks, despite a median one-step R² above 0.99 across channels.

## Limitations

These are worth telling a customer directly.

- **This is not a flight qualification tool.** The physics is simplified: no deformable soil, no propellant slosh, the contact is a model, the rover moves on a simplified model. The results are suitable for early design decisions and comparing options.
- **Not verified on real data.** The calibration tool has been checked only on synthetic recordings. There is no comparison with telemetry of real vehicles. Absolute numbers cannot be trusted, relative comparisons only with care.
- **The autopilot affects the result.** Technology analysis evaluates the vehicle together with the built-in autopilot. A custom controller can be connected through `MissionRunner`, it gets a requirements scorecard and a comparison with the autopilot, but driver and technology analysis still runs only on the autopilot.
- **No compliance with standards is claimed.** The report is not evidence of ECSS compliance. We know the content of the handbook from reviews and have not yet matched it line by line.
- **Demand is not verified.** There have been no conversations with space companies. The described value is a hypothesis.

## Parameters to add next

The registry has 64 parameters today. These are not in it yet, and each needs a physics model, not only a new number:

| Parameter group | What it would add | Why it matters |
|---|---|---|
| Power budget for the lander and drone | Bus load, battery state, power margin | Only the rover and drone track energy now, and neither has a bus load |
| Thermal limits | Component temperature range, engine duty limits, survival heating | A mission can pass today while overheating |
| Communications | Link windows, data rate, signal blackout, command latency | Only a fixed observation delay exists |
| Compute budget | Control loop overrun, inference time for a learned model | Decides whether a trained model fits flight hardware |
| Terramechanics | Bekker soil constants, wheel sinkage, slip | Rover traction is Coulomb only |
| Propellant slosh | Slosh mass, frequency and damping | Affects attitude control at low fuel |
| Engine behavior | Ignition delay, restart count, minimum impulse bit | Throttle limits exist, but start and pulse behavior does not |
| Structural limits | Maximum acceleration and leg load | The current limits only cover touchdown speed, tilt and slope |
| Plume and dust | Surface erosion, lidar and camera occlusion near touchdown | Sensors stay clean at touchdown today |

## What is next

| Priority | Step | Why |
|---|---|---|
| 1 | Talk to 5–10 GNC and autonomy engineers | Check that this is the right pain before investing further |
| 2 | Run calibration on real telemetry | Customer trust cannot be earned without it; the tool exists, the data does not |
| 3 | L6: planning in imagination and checking in the simulator | Closest to real use |
| 4 | Driver and technology analysis for a customer controller | Currently it runs only on the autopilot |
| 5 | Power, communications and computing budgets | The simulator survey points to this gap |
| 6 | Mapping to ECSS-E-HB-40-02A | Entry into company processes |
| 7 | Deformable soil ([example approach](https://arxiv.org/html/2601.04547)) | The main physical simplification for rovers |
