"""Проверки Python-инструментов. Запуск: cd python && python -m unittest discover tests"""
import json
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
        cls.proc = subprocess.Popen([str(BIN), "serve", "--port", str(cls.PORT), "--no-open"], stdout=subprocess.DEVNULL)
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
        self.c.conn.request("POST", "/api/mission/run", body=json.dumps(spec))
        r = json.loads(self.c.conn.getresponse().read())
        self.assertIn("scorecard", r)
        self.assertIn("markdown", r)
        self.c.conn.request("POST", "/api/mission/run", body="{bad")
        resp = self.c.conn.getresponse()
        self.assertEqual(resp.status, 400)
        resp.read()


if __name__ == "__main__":
    unittest.main()
