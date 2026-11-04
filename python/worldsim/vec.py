"""Векторная среда без HTTP: вызовы напрямую в библиотеку Rust через ctypes.

Сборка: `cargo build --release`; путь задаёт WORLDSIM_LIB или ищется в target/release.
Завершившиеся среды сами начинают новый эпизод; конечное наблюдение — в `info["final_obs"]`.
"""
import ctypes as C
import json
import os
import sys
from pathlib import Path

import numpy as np


def _find_lib():
    name = {"darwin": "libworldsim.dylib", "win32": "worldsim.dll"}.get(sys.platform, "libworldsim.so")
    env = os.environ.get("WORLDSIM_LIB")
    if env:
        return env
    p = Path(__file__).resolve().parents[2] / "target" / "release" / name
    if p.exists():
        return str(p)
    raise FileNotFoundError(f"не найден {name}: выполните cargo build --release или задайте WORLDSIM_LIB")


def _load():
    lib = C.CDLL(_find_lib())
    F, P, B = C.POINTER(C.c_float), C.c_void_p, C.POINTER(C.c_ubyte)
    lib.ws_vec_new.restype = P
    lib.ws_vec_new.argtypes = [C.c_char_p, C.c_char_p, C.c_size_t, C.c_uint64, C.c_double, C.c_bool, C.c_size_t, C.c_size_t, C.c_double, C.c_double, C.c_double, C.c_double, C.c_size_t]
    lib.ws_vec_free.argtypes = [P]
    lib.ws_vec_spec.restype = C.c_size_t
    lib.ws_vec_spec.argtypes = [P, C.c_char_p, C.c_size_t]
    lib.ws_last_error.restype = C.c_size_t
    lib.ws_last_error.argtypes = [C.c_char_p, C.c_size_t]
    lib.ws_vec_reset.argtypes = [P, F]
    lib.ws_vec_step.argtypes = [P, F, F, F, B, B, F]
    for n in ("ws_vec_state", "ws_vec_params", "ws_vec_autopilot"):
        getattr(lib, n).argtypes = [P, F]
    return lib


_F = C.POINTER(C.c_float)


class VectorEnv:
    def __init__(self, task="lander", worlds="moon", n=16, seed=1, noise=1.0, randomize=False, frame_skip=0, max_steps=0, rotor_scale=1.0, gust=0.0, lag=0.0, fault=0.0, threads=1):
        self.lib = _load()
        self._p = self.lib.ws_vec_new(task.encode(), worlds.encode(), n, seed, noise, randomize, frame_skip, max_steps, rotor_scale, gust, lag, fault, threads)
        if not self._p:
            buf = C.create_string_buffer(2048)
            k = self.lib.ws_last_error(buf, 2048)
            raise ValueError(buf.raw[:k].decode("utf8", "replace"))
        buf = C.create_string_buffer(65536)
        k = self.lib.ws_vec_spec(self._p, buf, 65536)
        self.spec = json.loads(buf.raw[:k].decode("utf8"))
        self.n = n
        self.obs_dim, self.act_dim = len(self.spec["obs_names"]), len(self.spec["act_names"])
        self.action_low = np.array(self.spec["act_low"], dtype=np.float32)
        self.action_high = np.array(self.spec["act_high"], dtype=np.float32)
        self._obs = np.zeros((n, self.obs_dim), np.float32)
        self._final = np.zeros((n, self.obs_dim), np.float32)
        self._rew = np.zeros(n, np.float32)
        self._term = np.zeros(n, np.uint8)
        self._trunc = np.zeros(n, np.uint8)

    def reset(self):
        self.lib.ws_vec_reset(self._p, self._obs.ctypes.data_as(_F))
        return self._obs.copy()

    def step(self, actions):
        a = np.asarray(actions, dtype=np.float32)
        if a.shape != (self.n, self.act_dim):
            raise ValueError(f"действия: нужна форма {(self.n, self.act_dim)}, получено {a.shape}")
        U = C.POINTER(C.c_ubyte)
        self.lib.ws_vec_step(self._p, a.ctypes.data_as(_F), self._obs.ctypes.data_as(_F), self._rew.ctypes.data_as(_F),
                             self._term.ctypes.data_as(U), self._trunc.ctypes.data_as(U), self._final.ctypes.data_as(_F))
        return self._obs.copy(), self._rew.copy(), self._term.astype(bool), self._trunc.astype(bool), {"final_obs": self._final.copy()}

    def _export(self, fn, width):
        out = np.zeros((self.n, width), np.float32)
        fn(self._p, out.ctypes.data_as(_F))
        return out

    def state(self):
        return self._export(self.lib.ws_vec_state, len(self.spec["state_names"]))

    def params(self):
        return self._export(self.lib.ws_vec_params, len(self.spec["param_names"]))

    def autopilot(self):
        return self._export(self.lib.ws_vec_autopilot, self.act_dim)

    def close(self):
        if self._p:
            self.lib.ws_vec_free(self._p)
            self._p = None

    def __del__(self):
        self.close()
