# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow [SemVer](https://semver.org/). The API may change before 1.0.

## 0.3.3 — 2026-11-08

Reports, law checks and collected data are bit-for-bit the same as in 0.3.2, except P3M.

### Changed
- Mission Lab is about five times faster: autopilot flights skip the radar and lidar rays they never read, rays skip height lookups high above the ground, and flight series of one analysis share terrain maps.
- Transition collection for the lander is about 1.8 times faster, SPH about 3 times, the winds experiment about 10 times, cosmology about 2.3 times, P3M about 3 times.
- The winds experiment solves only the zonal mode, because its flow does not depend on longitude. Earth and Mars gave the same numbers as the full solve to the printed digits.

### Fixed
- P3M missed pairs in neighboring cells on small grids.

## 0.3.2 — 2026-11-07

### Fixed
- `tools/fetch_textures.sh` no longer stops at the first refused download: it fetches the rest, names what failed with a page to get it from by hand, and exits with an error. Files already in place are not downloaded again.

## 0.3.1 — 2026-11-06

### Added
- Release archives include `tools/`, so catalogs and photo maps can be downloaded after unpacking.
- Compensated summation in IAS15, with a check over 1000 orbits.

### Fixed
- The server finds `data/` next to the binary as well as in the current folder.
- Live collisions keep equal axis scales, so bodies are drawn as circles.
- The live collision animation no longer stops on its first frame.
- The README described a release page that had no releases.

## 0.3.0 — 2026-11-05

### Added
- Mission Lab: requirements, failure modes, main drivers, value of technologies; background jobs with progress and cancellation, run storage and comparison.
- World model qualification: accuracy, physical laws, action effect, optimism, applicability boundaries.
- Transition collection `collect`, counterfactual branching, vector environment for Python.
- Parameter registry: 89 parameters available in the API, CLI, missions and Python.
- Physics: quantum mechanics, black holes, fluid dynamics, chemistry, collisions, tectonics; 157 checks at that version.
- Custom controller in Mission Lab with the full analysis (`MissionRunner.analyze`: main drivers and value of technologies), telemetry calibration, terrain of a real site, export to LeRobot.
- Budgets, all off by default: power (`bus_power`, `lander_battery_kwh`), thermal node, command latency and link outages, decision time, Bekker soil for the rover.
- L6 Planning: `planning_check` flies missions by planning in the model's imagination.
- `worldsim loop`: external software in the loop over TCP, lockstep or soft real time.
- Read-only token, Docker Compose with Caddy for TLS.
- IAS15 and Wisdom–Holman integrators, live collisions, SPH gas, one-layer winds on a sphere with rule-of-thumb clouds, P3M, 128³ particles, NASA photo maps of Earth and the Moon.
- Settings forms, presets, import and export, theme, text size, panel width, graphics quality.
- Access token, limits, request origin check, `/healthz`, `/api/version`, OpenAPI, Docker, CI, releases.

### Changed
- The project is now in English: code, interface, reports and documentation.
- New interface: a single layout without floating windows.
- Environment observation sizes grew: ray hit flags were added.
- Requests from other sites are now rejected. The server used to allow them for everyone; for the old mode use `--cors`.
- Unknown CLI keys and out-of-range values now give an error instead of being silently replaced.
