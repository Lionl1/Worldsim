"""L6 Planning: fly real missions by planning in the model's imagination, then judge what actually happened.

The earlier levels ask whether predictions are right. This one asks whether they are good enough to act on: the model
plans by the cross-entropy method, the simulator executes, and the result goes through the mission requirements.
The same flights with the built-in autopilot serve as the reference.
"""
import numpy as np

from .mission import MissionRunner
from .qualify import TOLERANCES


def _lander_cost(o, n):
    h = np.maximum(o[:, n["gps_dz"]], 0.0)
    target = -np.minimum(1.0 + 0.12 * h, 40.0)
    cost = (o[:, n["vel_z"]] - target) ** 2 + 0.5 * (o[:, n["vel_x"]] ** 2 + o[:, n["vel_y"]] ** 2)
    cost += 0.01 * (o[:, n["gps_dx"]] ** 2 + o[:, n["gps_dy"]] ** 2) + 20.0 * (o[:, n["up_bx"]] ** 2 + o[:, n["up_by"]] ** 2)
    return cost + 2.0 * (o[:, n["gyro_x"]] ** 2 + o[:, n["gyro_y"]] ** 2 + o[:, n["gyro_z"]] ** 2)


def _drone_cost(o, n):
    return o[:, n["goal_dx"]] ** 2 + o[:, n["goal_dy"]] ** 2 + o[:, n["goal_dz"]] ** 2 + 0.5 * (o[:, n["vel_x"]] ** 2 + o[:, n["vel_y"]] ** 2 + o[:, n["vel_z"]] ** 2) + 5.0 * (o[:, n["up_bx"]] ** 2 + o[:, n["up_by"]] ** 2)


def _rover_cost(o, n):
    return o[:, n["goal_dx"]] ** 2 + o[:, n["goal_dy"]] ** 2


COSTS = {"lander": _lander_cost, "drone": _drone_cost, "rover": _rover_cost}


class _Planner:
    def __init__(self, model, runner, task, cost, plans, horizon, replan, iterations, spread, action_scale, seed):
        self.model, self.r, self.task, self.cost = model, runner, task, cost
        self.plans, self.horizon, self.replan, self.iterations = plans, horizon, replan, iterations
        self.low, self.high = runner.action_low, runner.action_high
        scale = np.ones(runner.act_dim) if action_scale is None else np.asarray(action_scale, np.float32)
        self.sigma0 = spread * (self.high - self.low) * scale
        self.names = {k: i for i, k in enumerate(runner.spec["obs_names"])}
        self.tol = [(self.names[k], v) for k, v in TOLERANCES[task].items() if k in self.names]
        self.rng = np.random.default_rng(seed)
        self.errors = []
        self.reset(runner.n)

    def reset(self, n):
        self.step = 0
        # the plan starts from the midpoint of a bounded command, zero for a symmetric one
        mid = np.where(self.low + self.high == 0, 0.0, 0.5 * (self.low + self.high))
        self.mean = np.tile(mid, (n, self.horizon, 1)).astype(np.float32)
        self.expected, self.action = None, None

    def __call__(self, obs):
        if self.step % self.replan == 0:
            self._check_expected(obs)
            self._plan(np.asarray(obs, np.float32))
        k = self.step % self.replan
        self.step += 1
        return self.current[:, k].copy()

    def _check_expected(self, obs):
        if self.expected is None:
            return
        live = ~self.r.finished.astype(bool)
        if live.any():
            e = [np.mean(((obs[live, i] - self.expected[live, i]) / t) ** 2) for i, t in self.tol]
            self.errors.append(float(np.sqrt(np.max(e))))

    def _plan(self, obs):
        n, a_dim = len(obs), self.r.act_dim
        params = self.r.params()
        o0, p0 = np.repeat(obs, self.plans, 0), np.repeat(params, self.plans, 0)
        sigma = np.tile(self.sigma0, (n, self.horizon, 1))
        elites = max(2, self.plans // 10)
        for _ in range(self.iterations):
            noise = self.rng.normal(size=(n, self.plans, self.horizon, a_dim)).astype(np.float32)
            noise[:, 0] = 0.0  # the current plan stays among the candidates
            acts = np.clip(self.mean[:, None] + noise * sigma[:, None], self.low, self.high)
            state = self.model.init(o0, p0)
            total = np.zeros(n * self.plans)
            flat = acts.reshape(n * self.plans, self.horizon, a_dim)
            at_replan = None
            for t in range(self.horizon):
                state, pred = self.model.step(state, flat[:, t])
                total += np.nan_to_num(self.cost(pred, self.names), nan=1e12, posinf=1e12)
                if t == self.replan - 1:
                    at_replan = pred
            best = np.argsort(total.reshape(n, self.plans), axis=1)[:, :elites]
            rows = np.arange(n)[:, None]
            chosen = acts[rows, best]
            self.mean, sigma = chosen.mean(1), np.maximum(chosen.std(1), 0.08 * self.sigma0)
            first = best[:, 0]
        self.current = self.mean
        self.expected = None if at_replan is None else at_replan.reshape(n, self.plans, -1)[np.arange(n), first]
        # the next plan starts from the rest of this one
        shift = np.concatenate([self.mean[:, self.replan:], np.repeat(self.mean[:, -1:], self.replan, 1)], 1) if self.replan < self.horizon else self.mean
        self.mean = shift


def planning_check(model, mission, runs=32, plans=64, horizon=25, replan=5, iterations=2, spread=0.25, cost=None, action_scale=None, seed=0, threads=8):
    """Returns the mission report for the planning controller, with `plan_error` and `passed`.

    `plan_error`: how far reality went from the plan by the next replan, in mission tolerances; 1 is exactly at tolerance.
    `passed`: the flights are no worse than the autopilot within the paired interval, and the plan error is within tolerance.
    `action_scale` narrows the search per action, for example zero for channels the model does not know.
    """
    spec = dict(mission, runs=min(runs, mission.get("runs", runs)))
    r = MissionRunner(spec, threads=threads)
    task = spec.get("task", "lander")
    fn = cost or COSTS[task]
    planner = _Planner(model, r, task, fn, plans, horizon, replan, iterations, spread, action_scale, seed)
    report = r.evaluate(planner, name=getattr(model, "name", "model"))
    report["plan_error"] = float(np.median(planner.errors)) if planner.errors else float("nan")
    report["passed"] = bool(report["delta_vs_autopilot"] >= -report["delta_ci"] and report["plan_error"] <= 1.0)
    return report


def planning_markdown(r):
    verdict = "passed" if r["passed"] else "not passed"
    return (f"## L6 Planning\n\nPlanning in the model's imagination, flown in the simulator: **{r['success'] * 100:.1f}%** successful flights of {r['runs']}, "
            f"autopilot under the same conditions {r['autopilot_success'] * 100:.1f}% ({r['delta_vs_autopilot'] * 100:+.1f} ± {r['delta_ci'] * 100:.1f} pp.). "
            f"Requirements met: {r['requirements_met']} of {r['requirements_total']}. Plan error at the next replan: {r['plan_error']:.2f} of the tolerance. Result: {verdict}.\n")
