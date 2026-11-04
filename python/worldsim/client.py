"""Клиент HTTP API среды (`worldsim serve`). Интерфейс как у Gymnasium, но без зависимости от него."""
import http.client
import json
import os
import time
from urllib.parse import urlencode

import numpy as np


class Client:
    def __init__(self, host="localhost", port=8080, timeout=60, token=None):
        self.conn = http.client.HTTPConnection(host, port, timeout=timeout)
        token = token or os.environ.get("WORLDSIM_TOKEN")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def get(self, path, **params):
        q = urlencode({k: v for k, v in params.items() if v is not None})
        self.conn.request("GET", f"{path}?{q}" if q else path, headers=self.headers)
        return self._read()

    def post(self, path, body=b""):
        self.conn.request("POST", path, body=body, headers=self.headers)
        return self._read()

    def run_mission(self, spec, on_progress=None, poll_s=0.3):
        """Миссия в фоне: ждём результат и возвращаем отчёт. `spec` — словарь или текст JSON."""
        body = spec if isinstance(spec, str) else json.dumps(spec, ensure_ascii=False)
        job = self.post("/api/jobs/mission", body.encode())["id"]
        while True:
            j = self.get(f"/api/jobs/{job}")
            if on_progress:
                on_progress(j["fraction"])
            if j["status"] == "done":
                return self.get(f"/api/runs/{j['run']}")
            if j["status"] != "running":
                raise RuntimeError(j.get("error") or "расчёт отменён")
            time.sleep(poll_s)

    def _read(self):
        r = self.conn.getresponse()
        body = r.read()
        if r.status != 200:
            raise RuntimeError(json.loads(body).get("error", body[:200]))
        return json.loads(body) if r.getheader("Content-Type", "").startswith("application/json") else body

    def make(self, task="lander", physics="moon", **kw):
        return RemoteEnv(self, self.get("/api/env/make", task=task, physics=physics, **kw))


class RemoteEnv:
    """Сессия среды на сервере: reset, step, fork, state, camera, close."""

    def __init__(self, client, meta):
        self.client, self.meta, self.id = client, meta, meta["id"]
        self.obs_names, self.act_names = meta.get("obs_names"), meta.get("act_names")

    def reset(self, seed=1):
        r = self.client.get("/api/env/reset", id=self.id, seed=seed)
        return np.array(r["obs"], dtype=np.float32), r["info"]

    def step(self, action):
        a = ",".join(str(float(x)) for x in action)
        r = self.client.get("/api/env/step", id=self.id, action=a)
        return np.array(r["obs"], dtype=np.float32), r["reward"], r["terminated"], r["truncated"], r["info"]

    def autopilot_step(self, repeat=1):
        return self.client.get("/api/env/step", id=self.id, mode="auto", repeat=repeat, traj=1)

    def state(self):
        return self.client.get("/api/env/state", id=self.id)

    def fork(self):
        """Копия с тем же состоянием: проиграть другое действие и сравнить с предсказанием модели."""
        r = self.client.get("/api/env/fork", id=self.id)
        return RemoteEnv(self.client, {**self.meta, "id": r["id"]})

    def camera(self, w=64, h=64, mode="depth", fov=70):
        r = self.client.get("/api/env/camera", id=self.id, w=w, h=h, mode=mode, fov=fov)
        if mode == "rgb":
            return np.frombuffer(r, dtype=np.uint8).reshape(h, w, 3)
        return np.array(r["depth"], dtype=np.float32).reshape(r["h"], r["w"])

    def close(self):
        return self.client.get("/api/env/close", id=self.id)
