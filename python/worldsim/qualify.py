"""World model qualification ladder. The truth is always known, so each check yields a number, not an opinion.

L1 Accuracy      long-range prediction error and the “trust horizon” under mission tolerances.
L2 Physical laws violations in imagination relative to the level of the simulator itself.
L3 Action effect different actions from one point against the true physical response (branching).
L4 Optimism      does the model predict well-being where there is really a crash.
L5 Boundaries    the same checks in worlds absent from training: where the model can be trusted.

Builds on the ideas of MiraBench and CrashTwin, but unlike them works with exact physics and mission tolerances.
The report is engineering evidence for a review, not certification.
"""
import datetime
import json
import platform

import numpy as np

# default tolerances: what error is still good enough for planning. Custom ones can be set.
TOLERANCES = {
    "lander": {"gps_dx": 10.0, "gps_dy": 10.0, "gps_dz": 5.0, "vel_x": 2.0, "vel_y": 2.0, "vel_z": 1.0, "up_bx": 0.1, "up_by": 0.1},
    "drone": {"goal_dx": 3.0, "goal_dy": 3.0, "goal_dz": 2.0, "vel_x": 1.0, "vel_y": 1.0, "vel_z": 0.5, "up_bx": 0.1, "up_by": 0.1},
    "rover": {"goal_dx": 3.0, "goal_dy": 3.0, "head_cos": 0.1, "head_sin": 0.1, "speed": 0.3},
}
HORIZONS_S = (0.2, 1.0, 2.0, 5.0, 10.0, 20.0)
UNSAFE_SPEED = {"lander": 2.5, "drone": 3.0}
DIVERGED = 1e6


def _rollout(model, ds, starts, horizon):
    """Open loop on logged actions. Returns predictions, truth and an “episode still running” mask."""
    obs0 = np.asarray(ds.obs[starts], dtype=np.float32)
    ep = np.asarray(ds.episode[starts]).astype(int)
    params = np.asarray(ds.params[ep], dtype=np.float32)
    state = model.init(obs0, params)
    n, o = len(starts), obs0.shape[1]
    pred, true = np.zeros((horizon, n, o), np.float32), np.zeros((horizon, n, o), np.float32)
    mask = np.zeros((horizon, n), bool)
    total = len(ds.obs)
    for t in range(horizon):
        idx = starts + t
        valid = idx < total
        valid[valid] = np.asarray(ds.episode[idx[valid]]).astype(int) == ep[valid]
        mask[t] = valid
        safe = np.where(valid, idx, starts)
        action = np.asarray(ds.act[safe], dtype=np.float32)
        state, p = model.step(state, action)
        # a diverged model counts as a large error, it does not crash the calculation
        pred[t], true[t] = np.clip(np.nan_to_num(p, nan=DIVERGED, posinf=DIVERGED, neginf=-DIVERGED), -DIVERGED, DIVERGED), np.asarray(ds.next_obs[safe], dtype=np.float32)
    return pred, true, mask


def _choose_starts(ds, n, rng, min_left):
    good = np.flatnonzero(ds.lengths > min_left)
    if len(good) == 0:
        good = np.arange(len(ds.starts))
    ep = rng.choice(good, size=n, replace=len(good) < n)
    left = np.maximum(ds.lengths[ep] - min_left, 1)
    return ds.starts[ep] + (rng.random(n) * left).astype(int)


def level_accuracy(pred, true, mask, ds, tol):
    dt = ds.dt
    idx = {k: ds.channel(k) for k in tol if k in ds.obs_names}
    per = {}
    for s in HORIZONS_S:
        t = int(round(s / dt)) - 1
        if t >= len(pred) or mask[t].sum() < 5:
            continue
        m = mask[t]
        per[f"{s:g}s"] = {k: float(np.sqrt(np.mean((pred[t, m, i].astype(np.float64) - true[t, m, i]) ** 2))) for k, i in idx.items()}
    # trust horizon: until what time ALL channels stay within tolerances (by RMSE). Truth is plentiful, so the threshold is stable.
    horizon = 0
    limiting = None
    for t in range(len(pred)):
        m = mask[t]
        if m.sum() < 5:
            break
        bad = None
        for k, i in idx.items():
            # through not <=: NaN and infinity must count as out of tolerance
            if not np.sqrt(np.mean((pred[t, m, i].astype(np.float64) - true[t, m, i]) ** 2)) <= tol[k]:
                bad = k
                break
        if bad:
            limiting = bad
            break
        horizon = t + 1
    far = np.abs(pred.astype(np.float64)).max(axis=2) >= DIVERGED * 0.99
    return {"rmse": per, "trust_horizon_s": horizon * dt, "limited_by": limiting, "diverged_share": float(far[-1][mask[-1]].mean()) if mask[-1].any() else None}


def _law_series(x, ds):
    """Law violations in trajectories x (T×N×O). Returns a dictionary law → array of deviations."""
    names, dt = ds.obs_names, ds.dt
    laws = {}
    if "up_bx" in names:
        i, j, k = [names.index(c) for c in ("up_bx", "up_by", "up_bz")]
        laws["|up| = 1 (rotation preserves length)"] = np.abs(np.sqrt(x[..., i] ** 2 + x[..., j] ** 2 + x[..., k] ** 2) - 1)
    if "head_cos" in names:
        i, j = names.index("head_cos"), names.index("head_sin")
        laws["cos² + sin² = 1 (heading)"] = np.abs(np.sqrt(x[..., i] ** 2 + x[..., j] ** 2) - 1)
    if ds.task == "rover":
        p, c, s, v = names.index("goal_dx"), names.index("head_cos"), names.index("head_sin"), names.index("speed")
        laws["dx/dt = v·cos (kinematics)"] = np.abs(np.diff(x[..., p], axis=0) + x[1:, :, v] * x[1:, :, c] * dt)
    else:
        sign = 1.0 if ds.task == "lander" else -1.0
        for a, b, name in ((0, 3, "x"), (1, 4, "y"), (2, 5, "z")):
            laws[f"d{name}/dt = v{name} (kinematics)"] = np.abs(np.diff(x[..., a], axis=0) - sign * x[1:, :, b] * dt)
    for ch, label in (("fuel_frac", "fuel does not increase"), ("battery_frac", "charge does not grow without recharging")):
        if ch in names and not (ds.task == "rover"):
            laws[label] = np.maximum(np.diff(x[..., names.index(ch)], axis=0), 0)
    return laws


def _law_tolerance(law, ds):
    """What violation is still allowed per step or in norm."""
    if law.startswith("|up|") or law.startswith("cos²"):
        return 0.02
    if "kinematics" in law:
        return 0.02
    return 1e-4


def level_physics(pred, true, mask, ds):
    # float64: squares of diverged predictions must not overflow the calculation
    pl, tl = _law_series(pred.astype(np.float64), ds), _law_series(true.astype(np.float64), ds)
    out = {}
    for k in pl:
        m = mask[-pl[k].shape[0]:]
        # median: one diverged trajectory must not hide typical behavior; the share of divergence is shown separately
        mp, tp = float(np.median(pl[k][m])), float(np.median(tl[k][m]))
        # a tolerance is needed because the simulator satisfies some laws almost perfectly: comparing with zero is wrong
        limit = max(10 * tp, _law_tolerance(k, ds))
        out[k] = {"model": mp, "simulator": tp, "limit": limit, "passed": bool(mp <= limit)}
    return out


def level_action(model, ds, tol, rng, max_points=2000):
    """Predicted difference between branches against the true one: R² = 1 only if the model responds to an action correctly."""
    if not ds.has_counterfactuals or len(ds.cf_obs) == 0:
        return {"skipped": "no branches: collect data with --counterfactuals"}
    m = min(max_points, len(ds.cf_obs))
    pick = np.sort(rng.choice(len(ds.cf_obs), m, replace=False))
    obs = np.asarray(ds.cf_obs[pick], np.float32)
    act = np.asarray(ds.cf_act[pick], np.float32)
    nxt = np.asarray(ds.cf_next[pick], np.float32)
    alive = np.asarray(ds.cf_alive[pick])
    par = np.asarray(ds.cf_params[pick], np.float32)
    B, H = act.shape[1], nxt.shape[2]
    rows = {}
    channels = [k for k in tol if k in ds.obs_names and k.startswith(("vel", "up", "speed", "head"))]
    idx = [ds.channel(k) for k in channels]
    pred = np.zeros((m, B, H, obs.shape[1]), np.float32)
    for b in range(B):
        s = model.init(obs, par)
        for t in range(H):
            s, p = model.step(s, act[:, b])
            pred[:, b, t] = p
    for t, name in ((4, "0.1 s"), (H - 1, f"{H * ds.dt:g} s")):
        good = alive[:, 0, t] & alive[:, 1:, t].all(axis=1)
        if good.sum() < 10:
            continue
        # the difference relative to the first branch removes the common shift: only the action effect remains
        dt_ = nxt[good, 1:, t][:, :, idx] - nxt[good, :1, t][:, :, idx]
        dp = pred[good, 1:, t][:, :, idx] - pred[good, :1, t][:, :, idx]
        ss = ((dt_ - dt_.mean((0, 1), keepdims=True)) ** 2).sum((0, 1))
        r2 = 1 - ((dp - dt_) ** 2).sum((0, 1)) / np.maximum(ss, 1e-12)
        rows[name] = {k: float(r2[i]) for i, k in enumerate(channels)}
    worst = min((v for r in rows.values() for v in r.values()), default=float("nan"))
    return {"r2_of_action_effect": rows, "worst": worst, "points": int(m)}


def level_optimism(model, ds, rng, lead_s=1.0, max_eps=600):
    """Takes the moment lead_s before the episode end and asks the model how it ends. Compares with what really happened."""
    table = ds.episodes_table()
    lead = int(round(lead_s / ds.dt))
    eps = [i for i, r in enumerate(table) if int(r["steps"]) > lead and (r["crashed"] == "1" or r["success"] == "1")]
    if not eps:
        return {"skipped": "no finished episodes"}
    if len(eps) > max_eps:
        eps = list(rng.choice(eps, max_eps, replace=False))
    eps = np.array(sorted(eps))
    crashed = np.array([table[i]["crashed"] == "1" for i in eps])
    starts = ds.starts[eps] + ds.lengths[eps] - lead
    obs = np.asarray(ds.obs[starts], np.float32)
    state = model.init(obs, np.asarray(ds.params[eps], np.float32))
    names = ds.obs_names
    flag = np.zeros(len(eps), bool)
    last = obs
    for t in range(lead):
        state, last = model.step(state, np.asarray(ds.act[starts + t], np.float32))
        if ds.task in UNSAFE_SPEED:
            v = np.sqrt(sum(last[:, names.index(c)] ** 2 for c in ("vel_x", "vel_y", "vel_z")))
            # dangerous speed near the surface or a large tilt
            ground = last[:, names.index("radar_alt")] < (6.0 if ds.task == "lander" else 2.0)
            tilt = last[:, names.index("up_bz")] < 0.9
            flag |= ground & ((v > UNSAFE_SPEED[ds.task]) | tilt)
        else:
            pitch, roll = last[:, names.index("pitch")], last[:, names.index("roll")]
            flag |= (np.abs(np.tan(np.clip(roll, -1.5, 1.5))) > 0.8) | (np.abs(np.tan(np.clip(pitch, -1.5, 1.5))) > 1.1)
    n_crash, n_safe = int(crashed.sum()), int((~crashed).sum())
    return {
        "lead_s": lead_s, "crashes": n_crash, "safe": n_safe,
        "missed_crash_rate": float((~flag[crashed]).mean()) if n_crash else None,
        "false_alarm_rate": float(flag[~crashed].mean()) if n_safe else None,
    }


def _grade(acc, phys, action, opt, min_horizon_s):
    """Simple transparent rules: what matters is not a score but which level fails and why."""
    gates = {
        "L1": acc["trust_horizon_s"] >= min_horizon_s,
        "L2": all(v["passed"] for v in phys.values()) if phys else False,
        "L3": ("worst" in action) and action["worst"] >= 0.5,
        # both misses and false alarms: a model that always predicts a crash is useless
        "L4": opt.get("missed_crash_rate") is not None and opt["missed_crash_rate"] <= 0.2 and (opt.get("false_alarm_rate") or 0.0) <= 0.3,
    }
    return gates


def qualify(model, datasets, n_starts=400, horizon_s=20.0, min_horizon_s=2.0, tolerances=None, seed=0):
    """Main function. `datasets`: {name: Dataset}; first entry is the base world, the rest are applicability boundaries."""
    report = {
        "model": getattr(model, "name", type(model).__name__),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "n_starts": n_starts, "seed": seed, "min_horizon_s": min_horizon_s, "worlds": {},
    }
    for name, ds in datasets.items():
        rng = np.random.default_rng(seed)
        tol = (tolerances or TOLERANCES)[ds.task]
        horizon = max(2, int(round(horizon_s / ds.dt)))
        starts = _choose_starts(ds, n_starts, rng, horizon // 4)
        pred, true, mask = _rollout(model, ds, starts, horizon)
        acc = level_accuracy(pred, true, mask, ds, tol)
        phys = level_physics(pred, true, mask, ds)
        action = level_action(model, ds, tol, rng)
        opt = level_optimism(model, ds, rng)
        report["worlds"][name] = {
            "world": ds.meta["world_names"], "task": ds.task, "data": str(ds.path), "fingerprint": ds.fingerprint(), "transitions": len(ds), "noise": ds.meta.get("noise"),
            "generator": ds.meta.get("generator"), "accuracy": acc, "physics": phys, "action": action, "optimism": opt,
            "gates": _grade(acc, phys, action, opt, min_horizon_s),
        }
    return report


def _fmt(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def to_markdown(r):
    mark = lambda ok: "passed" if ok else "not passed"
    s = [f"# Model qualification: {r['model']}\n", f"Created {r['created']}. Starts per world: {r['n_starts']}, seed {r['seed']}. Minimum horizon: {r['min_horizon_s']} s.\n",
         "## Summary\n", "| World | Trust horizon | L1 Accuracy | L2 Physics | L3 Response | L4 Optimism |", "|---|---|---|---|---|---|"]
    for n, w in r["worlds"].items():
        g = w["gates"]
        s.append(f"| {n} | {w['accuracy']['trust_horizon_s']:.2f} s | {mark(g['L1'])} | {mark(g['L2'])} | {mark(g['L3'])} | {mark(g['L4'])} |")
    for n, w in r["worlds"].items():
        a, o = w["accuracy"], w["optimism"]
        s += [f"\n## {n}\n", f"World: {', '.join(w['world'])}. Data: {w['transitions']} transitions, noise {w['noise']}, fingerprint `{w['fingerprint']}`.\n",
              f"**Trust horizon {a['trust_horizon_s']:.2f} s**, limited by channel `{a['limited_by']}`. Diverged by the end of prediction: {_fmt(a['diverged_share'], 2)}.\n", "### Physical laws in imagination\n",
              "| Law | Model | Simulator | Tolerance | Result |", "|---|---|---|---|---|"]
        s += [f"| {k} | {v['model']:.5f} | {v['simulator']:.5f} | {v['limit']:.5f} | {mark(v['passed'])} |" for k, v in w["physics"].items()]
        ac = w["action"]
        s.append("\n### Action effect\n")
        if "skipped" in ac:
            s.append(ac["skipped"])
        else:
            s.append(f"R² of the action effect relative to true branching; worst channel: {ac['worst']:.3f}.\n")
            for t, row in ac["r2_of_action_effect"].items():
                s.append(f"- {t}: " + ", ".join(f"{k} {v:.2f}" for k, v in row.items()))
        s.append("\n### Optimism\n")
        s.append(o.get("skipped") or f"At {o['lead_s']:.1f} s before the end: missed crashes {_fmt(o['missed_crash_rate'])} of {o['crashes']}, false alarms {_fmt(o['false_alarm_rate'])} of {o['safe']}.")
    s.append("\n*The report describes model behavior in the simplified worldsim physics: this is engineering evidence for a review, not certification.*\n")
    return "\n".join(s)


def save(report, out):
    from pathlib import Path
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    (p / "qualification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    (p / "qualification.md").write_text(to_markdown(report), encoding="utf8")
