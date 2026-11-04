"""Custom controller in Mission Lab: same flights and requirements, but your system controls instead of the built-in autopilot.

    runner = MissionRunner(json.load(open("missions/lunar_lander.json")))
    report = runner.evaluate(lambda obs: my_policy(obs))
    print(report["success"], report["delta_vs_autopilot"])
"""
import ctypes as C
import json

import numpy as np

from .vec import _F, _load


class MissionRunner:
    def __init__(self, spec, threads=8):
        self.lib = _load()
        lib, P = self.lib, C.c_void_p
        lib.ws_mission_new.restype = P
        lib.ws_mission_new.argtypes = [C.c_char_p, C.c_size_t]
        lib.ws_mission_free.argtypes = [P]
        lib.ws_mission_spec.restype = C.c_size_t
        lib.ws_mission_spec.argtypes = [P, C.c_char_p, C.c_size_t]
        lib.ws_mission_reset.argtypes = [P, _F]
        lib.ws_mission_step.restype = C.c_size_t
        lib.ws_mission_step.argtypes = [P, _F, _F, C.POINTER(C.c_ubyte)]
        lib.ws_mission_report.restype = C.c_size_t
        lib.ws_mission_report.argtypes = [P, C.c_char_p, C.c_size_t, C.c_char_p]
        lib.ws_last_error.restype = C.c_size_t
        lib.ws_last_error.argtypes = [C.c_char_p, C.c_size_t]
        text = spec if isinstance(spec, str) else json.dumps(spec, ensure_ascii=False)
        self._p = lib.ws_mission_new(text.encode(), threads)
        if not self._p:
            buf = C.create_string_buffer(4096)
            k = lib.ws_last_error(buf, 4096)
            raise ValueError(buf.raw[:k].decode("utf8", "replace"))
        buf = C.create_string_buffer(65536)
        k = lib.ws_mission_spec(self._p, buf, 65536)
        self.spec = json.loads(buf.raw[:k].decode())
        self.n, self.obs_dim, self.act_dim = self.spec["runs"], len(self.spec["obs_names"]), len(self.spec["act_names"])
        self.action_low, self.action_high = np.array(self.spec["act_low"], np.float32), np.array(self.spec["act_high"], np.float32)

    def evaluate(self, controller, name="controller", max_steps=None):
        """`controller(obs N×O) → actions N×A`. Finished flights are no longer counted, so their actions are ignored."""
        obs = np.zeros((self.n, self.obs_dim), np.float32)
        done = np.zeros(self.n, np.uint8)
        self.lib.ws_mission_reset(self._p, obs.ctypes.data_as(_F))
        U = C.POINTER(C.c_ubyte)
        left, step = self.n, 0
        while left > 0 and (max_steps is None or step < max_steps):
            a = np.asarray(controller(obs), dtype=np.float32)
            if a.shape != (self.n, self.act_dim):
                raise ValueError(f"the controller returned shape {a.shape}, required {(self.n, self.act_dim)}")
            left = self.lib.ws_mission_step(self._p, a.ctypes.data_as(_F), obs.ctypes.data_as(_F), done.ctypes.data_as(U))
            step += 1
        buf = C.create_string_buffer(1 << 22)
        k = self.lib.ws_mission_report(self._p, buf, 1 << 22, name.encode())
        return json.loads(buf.raw[:k].decode("utf8"))

    def close(self):
        if getattr(self, "_p", None):
            self.lib.ws_mission_free(self._p)
            self._p = None

    def __del__(self):
        self.close()
