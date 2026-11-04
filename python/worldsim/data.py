"""Reading data `worldsim collect`: large arrays are not loaded into memory whole."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


class Dataset:
    def __init__(self, path):
        self.path = Path(path)
        meta = self.path / "meta.json"
        if not meta.exists():
            raise FileNotFoundError(f"{meta}: this is not a worldsim collect folder")
        self.meta = json.loads(meta.read_text(encoding="utf8"))
        load = lambda n: np.load(self.path / f"{n}.npy", mmap_mode="r")
        self.obs, self.next_obs, self.act = load("obs"), load("next_obs"), load("act")
        self.state, self.next_state = load("state"), load("next_state")
        self.reward, self.terminated, self.truncated = load("reward"), load("terminated"), load("truncated")
        self.episode, self.step, self.params = load("episode"), load("step"), load("params")
        self.obs_names, self.param_names = self.meta["obs_names"], self.meta["param_names"]
        self.dt = self.meta["control_dt"]
        self.task = self.meta["task"]
        self.starts = np.flatnonzero(np.asarray(self.step) == 0)
        self.lengths = np.diff(np.append(self.starts, len(self.obs)))
        self.has_counterfactuals = (self.path / "cf_obs.npy").exists()
        if self.has_counterfactuals:
            self.cf_obs, self.cf_act, self.cf_next, self.cf_alive, self.cf_params = load("cf_obs"), load("cf_act"), load("cf_next"), load("cf_alive"), load("cf_params")

    def __len__(self):
        return len(self.obs)

    def channel(self, name):
        return self.obs_names.index(name)

    def param(self, name):
        return self.param_names.index(name)

    def episodes_table(self):
        with open(self.path / "episodes.csv", encoding="utf8", newline="") as f:
            return list(csv.DictReader(f))

    def fingerprint(self):
        """Data fingerprint for the report: reproducibility and traceability."""
        h = hashlib.sha256()
        for n in ("meta.json", "episodes.csv"):
            h.update((self.path / n).read_bytes())
        h.update(np.asarray(self.obs[:: max(1, len(self.obs) // 2048)]).tobytes())
        return h.hexdigest()[:16]
