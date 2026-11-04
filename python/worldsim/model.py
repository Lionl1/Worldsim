"""What a model must support to be qualified.

Any architecture fits: the model state is opaque to the evaluator (observation vector, RSSM, latent JEPA…).
The only requirement is producing a prediction in sensor channels; for purely latent models
this is usually a linear “probe” trained on top of the latent.
"""
import numpy as np


class WorldModel:
    """Batch interface: all arrays have first dimension N (the number of parallel trajectories)."""

    name = "model"

    def init(self, obs, params):
        """Initial model state from an observation (N×O) and known world parameters (N×P)."""
        return np.asarray(obs, dtype=np.float32).copy()

    def step(self, state, action):
        """One step in imagination: return (new state, predicted observation N×O)."""
        raise NotImplementedError


class FunctionModel(WorldModel):
    """Memoryless model: next_obs = fn(obs, action, params)."""

    def __init__(self, fn, name="function"):
        self.fn, self.name, self._params = fn, name, None

    def init(self, obs, params):
        self._params = params
        return np.asarray(obs, dtype=np.float32).copy()

    def step(self, state, action):
        n = np.asarray(self.fn(state, action, self._params), dtype=np.float32)
        return n, n


class Persistence(WorldModel):
    """Null model: “nothing changes”. Any real model must do better."""

    name = "persistence"

    def step(self, state, action):
        return state, state
