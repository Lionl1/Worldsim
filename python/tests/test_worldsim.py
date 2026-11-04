"""Проверки Python-инструментов. Запуск: cd python && python -m unittest discover tests"""
import http.client
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import worldsim
from worldsim.model import FunctionModel, Persistence
from worldsim.qualify import qualify, to_markdown

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target" / "release" / "worldsim"


def collect(out, task="lander", worlds="moon", episodes=24, extra=()):
    subprocess.run([str(BIN), "collect", "--task", task, "--worlds", worlds, "--episodes", str(episodes), "--randomize", "--noise", "0", "--counterfactuals", "2", "--threads", "4", "--out", str(out), *extra], check=True, capture_output=True)
    return worldsim.Dataset(out)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)


class Data(Base):
    def test_shapes_and_episode_boundaries(self):
        d = collect(self.tmp / "a")
        self.assertEqual(d.obs.shape, d.next_obs.shape)
        self.assertEqual(len(d.starts), 24)
        self.assertEqual(d.params.shape, (24, len(d.param_names)))
        # next_obs шага t = obs шага t+1 внутри эпизода
        same = np.asarray(d.episode[1:]) == np.asarray(d.episode[:-1])
        self.assertTrue(np.array_equal(np.asarray(d.next_obs[:-1])[same], np.asarray(d.obs[1:])[same]))
        # в конце каждого эпизода ровно один флаг завершения
        ends = np.asarray(d.terminated) | np.asarray(d.truncated)
        self.assertEqual(int(ends.sum()), 24)

    def test_independent_of_thread_count(self):
        def run(n, name):
            o = self.tmp / name
            subprocess.run([str(BIN), "collect", "--task", "rover", "--worlds", "moon,mars", "--episodes", "16", "--threads", str(n), "--out", str(o)], check=True, capture_output=True)
            return (o / "obs.npy").read_bytes(), (o / "episodes.csv").read_bytes()
        self.assertEqual(run(1, "t1"), run(8, "t8"))

    def test_counterfactuals_share_a_start(self):
        d = collect(self.tmp / "cf", episodes=12)
        self.assertGreater(len(d.cf_obs), 0)
        self.assertEqual(d.cf_next.shape[1:3], (d.meta["cf_branches"], d.meta["cf_horizon"]))
        # разные действия дают разный исход, если эпизод ещё идёт
        alive = np.asarray(d.cf_alive)[:, :, -1].all(axis=1)
        self.assertGreater(alive.sum(), 0)
        v = d.channel("vel_z")
        spread = np.asarray(d.cf_next)[alive, :, -1, v].std(axis=1)
        self.assertGreater(float(spread.mean()), 0.01)

    def test_bad_arguments_leave_nothing(self):
        o = self.tmp / "bad"
        r = subprocess.run([str(BIN), "collect", "--task", "nope", "--out", str(o)], capture_output=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(o.exists())


class Vector(unittest.TestCase):
    def test_step_shapes_and_autoreset(self):
        v = worldsim.VectorEnv("drone", "moon", n=8, seed=3, noise=0.0)
        obs = v.reset()
        self.assertEqual(obs.shape, (8, v.obs_dim))
        done = 0
        for _ in range(600):
            obs, r, te, tr, info = v.step(np.zeros((8, v.act_dim), np.float32))
            done += int((te | tr).sum())
        self.assertGreater(done, 0)
        self.assertTrue(np.isfinite(obs).all())
        self.assertEqual(v.state().shape[0], 8)

    def test_threads_do_not_change_results(self):
        def run(threads):
            v = worldsim.VectorEnv("lander", "moon,mars", n=12, seed=9, noise=1.0, randomize=True, threads=threads)
            v.reset()
            for _ in range(300):
                o, *_ = v.step(v.autopilot())
            return o
        self.assertTrue(np.array_equal(run(1), run(4)))

    def test_bad_config_raises(self):
        with self.assertRaises(ValueError):
            worldsim.VectorEnv("nope", "moon")
        with self.assertRaises(ValueError):
            worldsim.VectorEnv("lander", "nosuchworld")

    def test_wrong_action_shape(self):
        v = worldsim.VectorEnv("rover", "moon", n=2)
        v.reset()
        with self.assertRaises(ValueError):
            v.step(np.zeros((3, 2)))


class Qualification(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.ds = collect(cls.tmp / "q", episodes=60)

    def test_persistence_fails_and_report_renders(self):
        r = qualify(Persistence(), {"moon": self.ds}, n_starts=60)
        w = r["worlds"]["moon"]
        self.assertFalse(w["gates"]["L3"])  # не реагирует на действия
        self.assertFalse(w["gates"]["L1"])
        self.assertLess(w["action"]["worst"], 0.1)
        self.assertIn("Квалификация", to_markdown(r))
        json.dumps(r)

    def test_kinematic_model_beats_persistence(self):
        names, dt = self.ds.obs_names, self.ds.dt
        def fn(o, a, p):
            n = o.copy()
            for x, v in ((0, 3), (1, 4), (2, 5)):
                n[:, x] += o[:, v] * dt
            return n
        a = qualify(FunctionModel(fn, "kin"), {"m": self.ds}, n_starts=60)["worlds"]["m"]
        b = qualify(Persistence(), {"m": self.ds}, n_starts=60)["worlds"]["m"]
        self.assertGreaterEqual(a["accuracy"]["trust_horizon_s"], b["accuracy"]["trust_horizon_s"])

    def test_nan_model_is_flagged_not_crashing(self):
        bad = FunctionModel(lambda o, a, p: np.full_like(o, np.nan), "nan")
        r = qualify(bad, {"m": self.ds}, n_starts=20)
        self.assertEqual(r["worlds"]["m"]["accuracy"]["trust_horizon_s"], 0)
        self.assertFalse(r["worlds"]["m"]["gates"]["L1"])


class Server(unittest.TestCase):
    PORT = 18765

    @classmethod
    def setUpClass(cls):
        cls.data = tempfile.mkdtemp()
        cls.proc = subprocess.Popen([str(BIN), "serve", "--port", str(cls.PORT), "--no-open", "--data-dir", cls.data], stdout=subprocess.DEVNULL)
        for _ in range(50):
            try:
                cls.c = worldsim.Client(port=cls.PORT)
                cls.c.get("/api/presets")
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("сервер не запустился")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait()
        shutil.rmtree(cls.data)

    def test_fork_reproduces_and_branches(self):
        e = self.c.make("lander", "earth", seed=3)
        for _ in range(20):
            e.step([0.5, 0, 0, 0])
        f = e.fork()
        same = [e.step([0.6, 0, 0, 0])[0] for _ in range(10)], [f.step([0.6, 0, 0, 0])[0] for _ in range(10)]
        self.assertTrue(np.array_equal(same[0], same[1]))
        g = e.fork()
        self.assertFalse(np.array_equal(e.step([1, 0, 0, 0])[0], g.step([0, 0, 0, 0])[0]))
        self.assertEqual(len(e.state()["state"]), len(e.state()["state_names"]))
        self.assertTrue(e.close()["closed"])
        with self.assertRaises(RuntimeError):
            e.state()

    def test_invalid_numbers_are_rejected(self):
        for path, q in [("/api/world", {"gravity_scale": "abc"}), ("/api/world", {"gravity_scale": 0}), ("/api/throw", {"speed": "NaN"}), ("/api/throw", {"speed": -5}), ("/api/throw", {"radius": 0}),
                        ("/api/climate", {"solar": -1}), ("/api/climate", {"ecc": 5}), ("/api/climate", {"ocean": -3}), ("/api/galaxy/step", {"id": 1, "dt": "NaN"}), ("/api/nbody/step", {"id": 1, "days": "NaN"}),
                        ("/api/system", {"star": "ms:-1"}), ("/api/nbody/new", {"bodies": "sun,sun"}), ("/api/env/make", {"frame_skip": 0}), ("/api/env/make", {"gust": -1})]:
            with self.assertRaises(RuntimeError, msg=str((path, q))):
                self.c.get(path, **q)

    def test_active_session_survives_many_new_ones(self):
        keep = self.c.make("rover", "moon")
        for i in range(70):
            self.c.get("/api/nbody/new")
            if i % 10 == 0:
                keep.state()
        keep.state()
        first = self.c.get("/api/nbody/new")["id"]
        for _ in range(70):
            self.c.get("/api/nbody/new")
        with self.assertRaises(RuntimeError):
            self.c.get("/api/nbody/step", id=first)

    def test_trajectory_returns_all_steps(self):
        e = self.c.make("lander", "moon", seed=2, noise=0)
        r = self.c.get("/api/env/step", id=e.id, mode="auto", repeat=5, traj=1)
        self.assertEqual(len(r["traj_obs"]), 5)
        self.assertEqual(r["traj_obs"][-1], r["obs"])

    def test_schema_and_params(self):
        schema = self.c.get("/api/env/schema")
        names = {d["name"] for d in schema}
        self.assertIn("throttle_min", names)
        e = self.c.make("lander", "moon", params="isp=320,gust=0")
        self.assertEqual(e.state()["params"][e.state()["param_names"].index("isp")], 320)
        with self.assertRaises(RuntimeError):
            self.c.make("lander", "moon", params="warp=1")
        v = worldsim.VectorEnv("lander", "moon", n=2, params={"isp": 333})
        self.assertAlmostEqual(float(v.params()[0, v.spec["param_names"].index("isp")]), 333, places=2)
        with self.assertRaises(ValueError):
            worldsim.VectorEnv("lander", "moon", params={"warp": 1})

    def test_physics_lab_routes(self):
        for topic in ("tunnel", "levels", "hydrogen", "sod", "dam", "chapman", "nozzle", "crater", "collide", "ocean"):
            self.assertIsInstance(self.c.get(f"/api/lab/{topic}"), dict, topic)
        with self.assertRaises(RuntimeError):
            self.c.get("/api/lab/nosuch")
        with self.assertRaises(RuntimeError):
            self.c.get("/api/lab/crater", d=-1)
        t = self.c.get("/api/lab/tunnel", e=0.5, v0=0.8, a=1)
        self.assertLess(abs(t["t_numeric"] - t["t_exact"]) / t["t_exact"], 0.05)
        self.assertLess(abs(t["norm"] - 1), 1e-6)
        p = self.c.get("/api/blackhole/props", m_sun=10, a=0)
        self.assertLess(abs(p["isco"] - 6), 1e-3)
        with self.assertRaises(RuntimeError):
            self.c.get("/api/blackhole/props", a=1.5)
        b = self.c.get("/api/blackhole/render", w=64, h=40)
        self.assertEqual(len(b), 64 * 40 * 3)
        # в центре кадра тень, сбоку свет диска или фон
        px = np.frombuffer(b, np.uint8).reshape(40, 64, 3)
        self.assertLess(int(px[20, 32].sum()), 30)
        self.assertGreater(int(px.sum()), 0)

    def test_jobs_progress_cancel_and_runs(self):
        spec = self.c.get("/api/mission/example", name="mars_helicopter")
        spec["runs"] = 16
        seen = []
        r = self.c.run_mission(spec, seen.append)
        self.assertIn("scorecard", r)
        self.assertTrue(r["spec_sha256"])
        self.assertTrue(seen)
        spec2 = dict(spec, rotor_scale=25)
        self.c.run_mission(spec2)
        runs = self.c.get("/api/runs")
        self.assertGreaterEqual(len(runs), 2)
        cmp = self.c.get("/api/runs/compare", a=runs[1]["id"], b=runs[0]["id"])
        self.assertTrue(any(x["key"] == "rotor_scale" for x in cmp["spec_changes"]))
        # отмена долгой задачи
        big = dict(spec, runs=20000)
        job = self.c.post("/api/jobs/mission", json.dumps(big).encode())["id"]
        self.c.post(f"/api/jobs/{job}/cancel")
        for _ in range(100):
            j = self.c.get(f"/api/jobs/{job}")
            if j["status"] != "running":
                break
            time.sleep(0.1)
        self.assertEqual(j["status"], "cancelled")
        # ошибки доступа и пути
        for bad in ("../../etc", "..%2f", "abc!"):
            with self.assertRaises(RuntimeError):
                self.c.get(f"/api/runs/{bad}")
        with self.assertRaises(RuntimeError):
            self.c.post("/api/jobs/mission", b"{bad")
        with self.assertRaises(RuntimeError):
            self.c.get("/api/mission/run")

    def test_cameras(self):
        e = self.c.make("rover", "mars")
        self.assertEqual(e.camera(8, 6, "rgb").shape, (6, 8, 3))
        self.assertEqual(e.camera(8, 6).shape, (6, 8))

    def test_unit_fixes(self):
        w = self.c.get("/api/world", m=0.77, r=0.91, a=0.02, ms=0.09, rs=0.12, ts=2516, star_name="T", name="e")
        self.assertTrue(w["tidally_locked"])
        hot = self.c.get("/api/world", seed=42, star="random", planet=0)
        self.assertGreaterEqual(hot["t_mean"], hot["t_eq"])
        for p in ("plummer", "globular"):
            g = self.c.get("/api/galaxy/new", preset=p, n=2000)
            self.assertGreater(len(self.c.get("/api/galaxy/rotation", id=g["id"])["r"]), 3)

    def test_mission_api(self):
        spec = self.c.get("/api/mission/example", name="mars_helicopter")
        spec["runs"] = 12
        r = self.c.post("/api/mission/run", json.dumps(spec).encode())
        self.assertIn("scorecard", r)
        self.assertIn("markdown", r)
        self.c.conn.request("POST", "/api/mission/run", body="{bad")
        resp = self.c.conn.getresponse()
        self.assertEqual(resp.status, 400)
        resp.read()


class Security(unittest.TestCase):
    PORT = 18766

    def raw(self, method="GET", path="/api/version", headers=None, body=None):
        c = http.client.HTTPConnection("localhost", self.PORT, timeout=10)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        return r.status, dict(r.getheaders()), r.read()

    @classmethod
    def setUpClass(cls):
        cls.data = tempfile.mkdtemp()
        env = dict(os.environ, WORLDSIM_TOKEN="secret-token")
        cls.proc = subprocess.Popen([str(BIN), "serve", "--port", str(cls.PORT), "--no-open", "--data-dir", cls.data], stdout=subprocess.DEVNULL, env=env)
        for _ in range(50):
            try:
                http.client.HTTPConnection("localhost", cls.PORT, timeout=1).connect()
                return
            except OSError:
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait()
        shutil.rmtree(cls.data)

    def test_token(self):
        self.assertEqual(self.raw()[0], 401)
        self.assertEqual(self.raw(headers={"Authorization": "Bearer wrong-token-x"})[0], 401)
        self.assertEqual(self.raw(headers={"Authorization": "Bearer secret-token"})[0], 200)
        self.assertEqual(self.raw(path="/healthz")[0], 200)
        self.assertEqual(self.raw(path="/")[0], 200)
        c = worldsim.Client(port=self.PORT, token="secret-token")
        self.assertEqual(c.get("/api/version")["name"], "worldsim")

    def test_foreign_site_rejected_without_wildcard(self):
        ok = {"Authorization": "Bearer secret-token"}
        self.assertEqual(self.raw(headers={**ok, "Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.raw(headers={**ok, "Origin": f"http://localhost:{self.PORT}"})[0], 200)
        self.assertNotIn("Access-Control-Allow-Origin", self.raw(headers=ok)[1])

    def test_limits(self):
        ok = {"Authorization": "Bearer secret-token"}
        self.assertEqual(self.raw("POST", "/api/mission/run", ok, b"x" * (1 << 21))[0], 413)
        self.assertEqual(self.raw(path="/api/version?" + "a" * 20000, headers=ok)[0], 431)
        self.assertEqual(self.raw("DELETE", "/api/version", ok)[0], 405)

    def test_external_bind_requires_token(self):
        env = {k: v for k, v in os.environ.items() if k != "WORLDSIM_TOKEN"}
        r = subprocess.run([str(BIN), "serve", "--host", "0.0.0.0", "--port", "18767", "--no-open"], capture_output=True, env=env, timeout=10)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("токен", r.stderr.decode())

    def test_health_command(self):
        self.assertEqual(subprocess.run([str(BIN), "health", "--port", str(self.PORT)], capture_output=True).returncode, 0)
        self.assertNotEqual(subprocess.run([str(BIN), "health", "--port", "18799"], capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
