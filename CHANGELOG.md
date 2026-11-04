# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow [SemVer](https://semver.org/). The API may change before 1.0.

## 0.3.0 — unreleased

### Added
- Mission Lab: requirements, failure modes, main drivers, value of technologies; background jobs with progress and cancellation, run storage and comparison.
- World model qualification: accuracy, physical laws, action effect, optimism, applicability boundaries.
- Transition collection `collect`, counterfactual branching, vector environment for Python.
- Parameter registry: 64 parameters available in the API, CLI, missions and Python.
- Physics: quantum mechanics, black holes, fluid dynamics, chemistry, collisions, tectonics; 129 checks.
- Custom controller in Mission Lab, telemetry calibration, terrain of a real site, export to LeRobot.
- Settings forms, presets, import and export, theme, text size, panel width, graphics quality.
- Access token, limits, request origin check, `/healthz`, `/api/version`, OpenAPI, Docker, CI, releases.

### Changed
- New interface: a single layout without floating windows.
- Environment observation sizes grew: ray hit flags were added.
- Requests from other sites are now rejected. The server used to allow them for everyone; for the old mode use `--cors`.
- Unknown CLI keys and out-of-range values now give an error instead of being silently replaced.
