"""Python tool checks. Run: cd python && python -m unittest discover tests"""
import http.client
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
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
        # next_obs steps t = obs of step t+1 within the episode
        same = np.asarray(d.episode[1:]) == np.asarray(d.episode[:-1])
        self.assertTrue(np.array_equal(np.asarray(d.next_obs[:-1])[same], np.asarray(d.obs[1:])[same]))
        # exactly one termination flag at the end of each episode
        ends = np.asarray(d.terminated) | np.asarray(d.truncated)
        self.assertEqual(int(ends.sum()), 24)

    def test_independent_of_thread_count(self):
        def run(n, name):
            o = self.tmp / name
            subprocess.run([str(BIN), "collect", "--task", "rover", "--worlds", "moon,mars", "--episodes", "16", "--threads", str(n), "--out", str(o)], check=True, capture_output=True)
            return (o / "obs.npy").read_bytes(), (o / "episodes.csv").read_bytes()
        self.assertEqual(run(1, "t1"), run(8, "t8"))

    def test_counterfactuals_cover_whole_episode(self):
        d = collect(self.tmp / "cf2", episodes=12, extra=("--counterfactuals", "20"))
        # the radar altimeter at branch points shows both high altitude and approach to the surface
        alt = np.asarray(d.cf_obs)[:, d.channel("radar_alt")]
        self.assertLess(float(np.percentile(alt, 5)), 50.0)

    def test_counterfactuals_share_a_start(self):
        d = collect(self.tmp / "cf", episodes=12)
        self.assertGreater(len(d.cf_obs), 0)
        self.assertEqual(d.cf_next.shape[1:3], (d.meta["cf_branches"], d.meta["cf_horizon"]))
        # different actions give different outcomes while the episode is still running
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
        self.assertFalse(w["gates"]["L3"])  # does not respond to actions
        self.assertFalse(w["gates"]["L1"])
        self.assertLess(w["action"]["worst"], 0.1)
        self.assertIn("qualification", to_markdown(r))
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

    def test_gates_are_reachable_and_not_gameable(self):
        # L2 must be reachable: true trajectories pass, even though the simulator has |up| − 1 almost zero
        from worldsim.qualify import level_physics
        n = 40
        true = np.asarray(self.ds.next_obs[:n], np.float32)[:, None, :]
        mask = np.ones((n, 1), bool)
        same = np.asarray(self.ds.episode[:n]) == np.asarray(self.ds.episode[0])
        self.assertTrue(same.all())
        phys = level_physics(true, true, mask, self.ds)
        self.assertTrue(all(v["passed"] for v in phys.values()), phys)
        # a model that always predicts a crash: no misses, but L4 fails on false alarms
        idx = self.ds.channel("radar_alt"), self.ds.channel("vel_z")

        def alarm(o, a, p):
            n = o.copy()
            n[:, idx[0]] = 0.0
            n[:, idx[1]] = -50.0
            return n
        w = qualify(FunctionModel(alarm, "alarm"), {"m": self.ds}, n_starts=20)["worlds"]["m"]
        self.assertEqual(w["optimism"]["missed_crash_rate"], 0.0)
        self.assertFalse(w["gates"]["L4"])

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
        raise RuntimeError("server did not start")

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

    def test_api_does_not_read_server_files(self):
        spec = {"task": "lander", "runs": 2, "requirements": [{"metric": "success", "op": ">=", "value": 0.5}], "terrain": {"file": "/etc/hosts", "size": 100}}
        with self.assertRaisesRegex(RuntimeError, "terrain.file"):
            self.c.post("/api/mission/run", json.dumps(spec).encode())
        # but terrain directly in the description works
        spec["terrain"] = {"n": 9, "size": 900, "heights": [0.0] * 81}
        self.assertIn("scorecard", self.c.post("/api/mission/run", json.dumps(spec).encode()))

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
        # shadow in the center of the frame, disk light or background at the side
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
        # cancelling a long job
        big = dict(spec, runs=20000)
        job = self.c.post("/api/jobs/mission", json.dumps(big).encode())["id"]
        self.c.post(f"/api/jobs/{job}/cancel")
        for _ in range(100):
            j = self.c.get(f"/api/jobs/{job}")
            if j["status"] != "running":
                break
            time.sleep(0.1)
        self.assertEqual(j["status"], "cancelled")
        # access and path errors
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


class ProductGaps(Base):
    SPEC = {"name": "test", "task": "lander", "world": "moon", "runs": 24, "seed": 2, "vehicle": {"t_max": 8000, "fuel": 120, "start_offset": 5, "start_vh": 1},
            "dispersions": {"t_max": 0.03}, "requirements": [{"metric": "success", "op": ">=", "value": 0.5}]}

    def test_full_analysis_for_own_controller(self):
        spec = dict(self.SPEC, technologies=[{"name": "Weak", "set": {"t_max": 2500}}, {"name": "Strong", "set": {"t_max": 12000}, "dry_mass_delta": 5}])
        r = worldsim.MissionRunner(spec)
        names = r.spec["obs_names"]
        vz, gz = names.index("vel_z"), names.index("gps_dz")

        def pd(obs):
            a = np.zeros((len(obs), r.act_dim), np.float32)
            target = -np.minimum(1.0 + 0.12 * obs[:, gz], 40.0)
            a[:, 0] = np.clip(0.2 + 0.5 * (target - obs[:, vz]), 0, 1)
            return a

        seen = []
        rep = r.analyze(pd, "my-pd", on_progress=lambda done, total, s: seen.append((done, total, s["kind"])))
        self.assertEqual(seen[0], (1, 5, "base"))
        self.assertEqual([x[2] for x in seen], ["base", "clean", "driver", "technology", "technology"])
        self.assertEqual({d["name"] for d in rep["drivers"]}, {"t_max"})
        self.assertEqual({t["name"] for t in rep["technologies"]}, {"Weak", "Strong"})
        self.assertEqual(rep["controller"], "my-pd")
        self.assertIn("Controller: my-pd", rep["markdown"])
        by = {t["name"]: t for t in rep["technologies"]}
        self.assertLess(by["Weak"]["success"], rep["success"] + 1e-9)
        # the same seeds give the same answer
        again = r.analyze(pd, "my-pd")
        self.assertEqual(again["success"], rep["success"])
        self.assertEqual(again["technologies"][0]["success"], rep["technologies"][0]["success"])
        # a plain evaluate after the analysis still flies the mission itself
        self.assertEqual(r.evaluate(pd, "my-pd")["success"], rep["success"])
        off = r.analyze(lambda obs: np.zeros((len(obs), r.act_dim), np.float32), "off")
        self.assertEqual(off["success"], 0.0)
        self.assertEqual(off["baseline_no_dispersions"], 0.0)

    def test_bus_power_drains_the_lander_battery(self):
        env = worldsim.VectorEnv("lander", "moon", n=2, params={"bus_power": 1000, "lander_battery_kwh": 0.01, "start_alt_min": 3000, "start_alt_max": 3000})
        env.reset()
        charge = env.spec["state_names"].index("battery_j")
        self.assertEqual(env.state()[0, charge], 36000.0)
        for _ in range(100):
            env.step(env.autopilot())
        self.assertAlmostEqual(env.state()[0, charge], 36000.0 - 1000.0 * 100 * env.spec["control_dt"], delta=1.0)

    def test_own_controller_in_mission(self):
        r = worldsim.MissionRunner(self.SPEC)
        names = r.spec["obs_names"]
        vz, gz, up = names.index("vel_z"), names.index("gps_dz"), names.index("up_bz")

        def pd(obs):
            # external controller: vertical channel from sensors, zero torque
            a = np.zeros((r.n, r.act_dim), np.float32)
            target = -np.minimum(1.0 + 0.12 * obs[:, gz], 40.0)
            a[:, 0] = np.clip(0.2 + 0.5 * (target - obs[:, vz]), 0, 1)
            return a

        good = r.evaluate(pd, "my-pd")
        self.assertEqual(good["controller"], "my-pd")
        self.assertIn("scorecard", good)
        self.assertEqual(good["unfinished"], 0)
        self.assertGreater(good["autopilot_success"], 0)
        off = r.evaluate(lambda obs: np.zeros((r.n, r.act_dim), np.float32), "off")
        self.assertEqual(off["success"], 0.0)  # nobody lands without an engine
        self.assertGreater(good["success"], off["success"])
        with self.assertRaises(ValueError):
            r.evaluate(lambda obs: np.zeros((1, 1), np.float32))
        with self.assertRaises(ValueError):
            worldsim.MissionRunner({"task": "nope", "requirements": []})

    def test_calibration_recovers_hidden_parameters(self):
        from worldsim.calibrate import replay
        rng = np.random.default_rng(0)
        T = 400
        act = np.zeros((T, 4), np.float32)
        act[:, 0] = np.clip(0.3 + 0.2 * np.sin(np.arange(T) / 30), 0, 1)
        act[::7, 1] = 0.05
        init = [0, 0, 900, 0, 0, -20]
        flat = {"terrain_scale": 0, "crater_scale": 0, "fuel": 400}
        truth = {"t_max": 15000.0, "dry_mass": 800.0, "isp": 310.0}
        s, names = replay("lander", "moon", {**flat, **truth}, init, act)
        self.assertEqual(len(s), T)
        meas = {k: s[:, names.index(k)] + rng.normal(0, e, T) for k, e in (("vz", 0.02), ("mass", 0.05))}
        r = worldsim.fit_telemetry("lander", "moon", act, meas, init, {"t_max": (5000, 40000), "dry_mass": (300, 2000), "isp": (150, 600)}, flat, restarts=2)
        for k, v in truth.items():
            self.assertLess(abs(r["params"][k] - v) / v, 0.05, (k, r["params"][k]))
        self.assertTrue(r["identifiable"])
        with self.assertRaises(ValueError):
            worldsim.fit_telemetry("lander", "moon", act, {"vz": meas["vz"][:5]}, init, {"t_max": (5000, 40000)}, flat)
        with self.assertRaises(ValueError):
            worldsim.fit_telemetry("lander", "moon", act, meas, init, {"warp": (1, 2)}, flat, restarts=1)

    def test_calibration_csv_is_aligned(self):
        from worldsim.calibrate import align_table, load_csv, replay
        T = 200
        act = np.zeros((T, 4), np.float32)
        act[:, 0] = 0.4
        init = [0, 0, 500, 0, 0, -10]
        flat = {"terrain_scale": 0, "crater_scale": 0, "t_max": 12000}
        s, names = replay("lander", "moon", flat, init, act)
        # table like a real recording: the first row is the initial state, the command of this row has not acted yet
        p = self.tmp / "flight.csv"
        with open(p, "w") as f:
            f.write("act_0,act_1,act_2,act_3,z,vz\n")
            f.write(f"0.4,0,0,0,{init[2]},{init[5]}\n")
            for i in range(T - 1):
                f.write(f"0.4,0,0,0,{s[i, names.index('z')]},{s[i, names.index('vz')]}\n")
        a, st = align_table(*load_csv(p))
        sim, _ = replay("lander", "moon", flat, init, a)
        self.assertLess(np.abs(sim[:, names.index("vz")] - st["vz"]).max(), 1e-3)

    def test_form_values_for_engine_are_normalized(self):
        # a parameter in the vehicle section must not silently override a technology
        spec = {"task": "drone", "world": "mars", "runs": 12, "vehicle": {"rotor_scale": 8}, "requirements": [{"metric": "success", "op": ">=", "value": 0.5}],
                "technologies": [{"name": "large rotors", "set_top": {"rotor_scale": 30}}]}
        c = self.tmp / "s.json"
        c.write_text(json.dumps(spec), encoding="utf8")
        out = self.tmp / "norm"
        subprocess.run([str(BIN), "mission", str(c), "--out", str(out)], check=True, capture_output=True)
        t = json.loads((out / "report.json").read_text(encoding="utf8"))["technologies"][0]
        self.assertGreater(t["delta"], 0.3)
        for bad in ({"task": "rover", "technologies": [{"set": {"motor_force": 500}, "dry_mass_delta": 5}]}, {"task": "lander", "dispersions": {"rover_mass": 0.1}}):
            c.write_text(json.dumps({"requirements": [{"metric": "success", "op": ">=", "value": 0.5}], **bad}), encoding="utf8")
            r = subprocess.run([str(BIN), "mission", str(c)], capture_output=True)
            self.assertNotEqual(r.returncode, 0)

    def test_world_auto_and_small_terrain(self):
        out = self.tmp / "auto"
        subprocess.run([str(BIN), "collect", "--task", "rover", "--worlds", "moon:auto:mars", "--episodes", "1", "--max-steps", "5", "--out", str(out)], check=True, capture_output=True)
        p = self.tmp / "tiny.npy"
        np.save(p, np.zeros((9, 9)))
        spec = {"task": "lander", "runs": 1, "requirements": [{"metric": "success", "op": ">=", "value": 0.5}], "terrain": {"file": str(p), "size": 100}}
        with self.assertRaisesRegex(ValueError, "start_offset"):
            worldsim.MissionRunner(spec)

    def test_real_terrain_from_file(self):
        n = 81
        size = 800.0
        x = np.linspace(-size / 2, size / 2, n)
        h = 100 + 0.02 * x[None, :] + 0 * x[:, None]
        # a step 40 m to the east: landing on it gives a large height above the pad
        h[:, x > 150] += 40
        for ext, write in {".npy": lambda p: np.save(p, h), ".csv": lambda p: np.savetxt(p, h, delimiter=",")}.items():
            p = self.tmp / f"site{ext}"
            write(p)
            spec = {**self.SPEC, "runs": 4, "terrain": {"file": str(p), "size": size}}
            rep = worldsim.MissionRunner(spec).evaluate(lambda o: np.zeros((4, 4), np.float32))
            self.assertEqual(rep["unfinished"], 0)
        asc = self.tmp / "site.asc"
        with open(asc, "w") as f:
            f.write(f"ncols {n}\nnrows {n}\nxllcorner 0\nyllcorner 0\ncellsize 10\nNODATA_value -9999\n")
            for row in h[::-1]:
                f.write(" ".join(f"{v:.3f}" for v in row) + "\n")
        out = self.tmp / "dem_run"
        subprocess.run([str(BIN), "collect", "--task", "rover", "--worlds", "mars", "--episodes", "4", "--terrain-file", str(asc), "--terrain", "9", "--out", str(out)], check=True, capture_output=True)
        d = np.load(out / "terrain.npy")
        self.assertLess(abs(float(d[0].max()) - float(h.max())), 2.0)
        bad = self.tmp / "bad.asc"
        bad.write_text("ncols 3\nnrows 3\ncellsize 1\n1 2 3\n")
        r = subprocess.run([str(BIN), "collect", "--task", "rover", "--terrain-file", str(bad), "--out", str(self.tmp / "x")], capture_output=True)
        self.assertNotEqual(r.returncode, 0)

    def test_lerobot_export(self):
        try:
            import pandas  # noqa: F401
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("no pyarrow or pandas")
        from worldsim.export import to_lerobot
        d = collect(self.tmp / "lr", episodes=6)
        out = to_lerobot(d, self.tmp / "lr_out")
        t = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
        self.assertEqual(t.num_rows, len(d))
        info = json.loads((out / "meta" / "info.json").read_text())
        self.assertEqual(info["total_episodes"], 6)
        self.assertEqual(info["features"]["action"]["shape"], [d.act.shape[1]])
        e = pq.read_table(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        self.assertEqual(int(e["dataset_to_index"][5].as_py()), len(d))
        import pandas as pd
        tasks = pd.read_parquet(out / "meta" / "tasks.parquet")
        self.assertIn("lander", tasks.index[0])  # LeRobot takes the task text from the index


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
        # an image on another site: no Origin, but the browser sends Sec-Fetch-Site
        self.assertEqual(self.raw(headers={**ok, "Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(self.raw(headers={**ok, "Sec-Fetch-Site": "same-origin"})[0], 200)
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
        self.assertIn("token", r.stderr.decode())

    def test_health_command(self):
        self.assertEqual(subprocess.run([str(BIN), "health", "--port", str(self.PORT)], capture_output=True).returncode, 0)
        self.assertNotEqual(subprocess.run([str(BIN), "health", "--port", "18799"], capture_output=True).returncode, 0)


class RigidBodyModel(worldsim.WorldModel):
    """Exact thrust, gravity and rotation for the test lander; horizontal drift is left out. Enough to plan a straight landing."""
    name = "rigid-body"

    def __init__(self, spec):
        self.dt, self.n = spec["control_dt"], {k: i for i, k in enumerate(spec["obs_names"])}
        mass = 1000.0 + 120.0
        self.accel = 8000.0 / mass
        self.spin = 3000.0 / np.array([mass * (3 * 1.5 ** 2 + 9) / 12, mass * (3 * 1.5 ** 2 + 9) / 12, mass * 1.5 ** 2 / 2])

    def init(self, obs, params):
        return np.array(obs, np.float32)

    def step(self, state, action):
        s, n, dt = state.copy(), self.n, self.dt
        wx, wy, wz = (s[:, n["gyro_" + k]] + action[:, i + 1] * self.spin[i] * dt for i, k in enumerate("xyz"))
        ux, uy, uz = s[:, n["up_bx"]], s[:, n["up_by"]], s[:, n["up_bz"]]
        s[:, n["gyro_x"]], s[:, n["gyro_y"]], s[:, n["gyro_z"]] = wx, wy, wz
        s[:, n["up_bx"]], s[:, n["up_by"]], s[:, n["up_bz"]] = ux - (wy * uz - wz * uy) * dt, uy - (wz * ux - wx * uz) * dt, uz - (wx * uy - wy * ux) * dt
        s[:, n["vel_z"]] += (self.accel * action[:, 0] * uz - 1.625) * dt
        s[:, n["gps_dz"]] += s[:, n["vel_z"]] * dt
        return s, s


class NullModel(worldsim.WorldModel):
    name = "nothing-changes"

    def init(self, obs, params):
        return np.array(obs, np.float32)

    def step(self, state, action):
        return state, state


class PlanningLevel(unittest.TestCase):
    SPEC = {"name": "plan", "task": "lander", "world": "moon", "runs": 6, "seed": 2, "vehicle": {"t_max": 8000, "fuel": 120, "start_offset": 0, "start_vh": 0, "start_tilt": 0, "start_alt_min": 60, "start_alt_max": 80, "start_vz_min": 2, "start_vz_max": 5, "terrain_scale": 0, "crater_scale": 0},
            "requirements": [{"metric": "success", "op": ">=", "value": 0.5}]}

    def test_planning_tells_a_useful_model_from_a_useless_one(self):
        r = worldsim.MissionRunner(self.SPEC)
        good = worldsim.planning_check(RigidBodyModel(r.spec), self.SPEC, plans=64, horizon=30)
        bad = worldsim.planning_check(NullModel(), self.SPEC, plans=64, horizon=30)
        self.assertGreaterEqual(good["success"], 0.8, good["failure_modes"])
        self.assertEqual(bad["success"], 0.0)
        self.assertFalse(bad["passed"])
        self.assertLess(good["plan_error"], 1.0)
        self.assertIn("L6 Planning", worldsim.planning_markdown(good))


class Loop(unittest.TestCase):
    def _start(self, port, *extra):
        p = subprocess.Popen([str(BIN), "loop", "moon", "--task", "lander", "--port", str(port), "--noise", "0", "--once", *extra], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.addCleanup(p.kill)
        p.stdout.readline()
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.addCleanup(s.close)
        return s, s.makefile("r"), None

    def test_lockstep_matches_the_vector_environment(self):
        s, f, _ = self._start(18731)
        hello = json.loads(f.readline())
        self.assertEqual(hello["task"], "lander")
        s.sendall(b'{"action":[1,2]}\n')
        self.assertIn("error", json.loads(f.readline()))
        for _ in range(5):
            s.sendall(b'{"action":[0.5,0,0,0]}\n')
            r = json.loads(f.readline())
        self.assertEqual(r["misses"], 0)
        self.assertAlmostEqual(r["t"], 5 * hello["control_dt"], places=6)
        s.sendall(b'{"reset":7}\n')
        self.assertEqual(json.loads(f.readline())["t"], 0)

    def test_real_time_counts_missed_commands(self):
        s, f, _ = self._start(18732, "--realtime")
        hello = json.loads(f.readline())
        r = json.loads(f.readline())  # no command sent: the tick passes without one
        self.assertTrue(r["late"])
        self.assertEqual(r["misses"], 1)
        s.sendall(b'{"action":[0.5,0,0,0]}\n')
        while True:
            r = json.loads(f.readline())
            if not r["late"]:
                break
        s.sendall(b'{"bye":true}\n')
        self.assertGreater(hello["control_dt"], 0)


class Roles(unittest.TestCase):
    def _get(self, port, path, token, method="GET"):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method, data=b"{}" if method == "POST" else None, headers={"Authorization": f"Bearer {token}"})
        try:
            return urllib.request.urlopen(req, timeout=10).status
        except urllib.error.HTTPError as e:
            return e.code

    def test_viewer_token_reads_but_does_not_run(self):
        env = dict(os.environ, WORLDSIM_TOKEN="main-token", WORLDSIM_VIEWER_TOKEN="view-token")
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.Popen([str(BIN), "serve", "--port", "18733", "--no-open", "--data-dir", d], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.addCleanup(p.kill)
            for _ in range(50):
                try:
                    urllib.request.urlopen("http://127.0.0.1:18733/healthz", timeout=1)
                    break
                except OSError:
                    time.sleep(0.1)
            self.assertEqual(self._get(18733, "/api/runs", "view-token"), 200)
            self.assertEqual(self._get(18733, "/api/env/schema", "view-token"), 200)
            self.assertEqual(self._get(18733, "/api/mission/run", "view-token", "POST"), 403)
            self.assertEqual(self._get(18733, "/api/jobs/mission", "view-token", "POST"), 403)
            self.assertEqual(self._get(18733, "/api/env/make", "view-token"), 403)
            self.assertEqual(self._get(18733, "/api/runs", "wrong"), 401)
            self.assertEqual(self._get(18733, "/api/env/make", "main-token"), 200)

    def test_viewer_token_alone_is_refused(self):
        env = {k: v for k, v in os.environ.items() if k != "WORLDSIM_TOKEN"}
        env["WORLDSIM_VIEWER_TOKEN"] = "v"
        p = subprocess.run([str(BIN), "serve", "--port", "18734", "--no-open"], env=env, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("main token", p.stderr)


if __name__ == "__main__":
    unittest.main()
