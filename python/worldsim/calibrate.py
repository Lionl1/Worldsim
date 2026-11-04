"""Calibration: fit model parameters so it reproduces a recorded flight.

    python -m worldsim.calibrate --task lander --world moon --csv flight.csv --fit isp:200:400 --fit t_max:1000:50000 --out params.json

The CSV needs columns `act_0…` with commands and at least one true state column (`z`, `vz`, `mass`…, cm. `state_names`).
The first row sets the initial x, y, z, vx, vy, vz. The table step must match the environment control step.
Calibration refines parameters but does not verify that worldsim physics is right for your vehicle: check the residual.
"""
import argparse
import csv
import ctypes as C
import json

import numpy as np

from .vec import _F, _load


def replay(task, world, params, init, actions):
    """Replays commands and returns the state at each step: (T×S, names). The episode may end earlier."""
    lib = _load()
    lib.ws_replay_names.restype = C.c_size_t
    lib.ws_replay_names.argtypes = [C.c_char_p, C.c_char_p, C.c_size_t]
    lib.ws_replay.restype = C.c_int64
    lib.ws_replay.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p, _F, C.c_size_t, C.c_size_t, _F, _F]
    buf = C.create_string_buffer(8192)
    k = lib.ws_replay_names(task.encode(), buf, 8192)
    names = json.loads(buf.raw[:k].decode())
    a = np.asarray(actions, dtype=np.float32)
    x0 = np.asarray(init, dtype=np.float32)
    out = np.zeros((len(a), len(names)), np.float32)
    plist = ",".join(f"{n}={v}" for n, v in params.items())
    n = lib.ws_replay(task.encode(), world.encode(), plist.encode(), x0.ctypes.data_as(_F), a.shape[0], a.shape[1], a.ctypes.data_as(_F), out.ctypes.data_as(_F))
    if n < 0:
        lib.ws_last_error.restype = C.c_size_t
        lib.ws_last_error.argtypes = [C.c_char_p, C.c_size_t]
        k = lib.ws_last_error(buf, 8192)
        raise ValueError(buf.raw[:k].decode("utf8", "replace"))
    return out[:n], names


def _nelder_mead(f, x0, step=0.15, iters=400, tol=1e-7):
    n = len(x0)
    sim = [np.array(x0, float)] + [np.array(x0, float) + step * np.eye(n)[i] for i in range(n)]
    val = [f(x) for x in sim]
    for _ in range(iters):
        order = np.argsort(val)
        sim, val = [sim[i] for i in order], [val[i] for i in order]
        if val[-1] - val[0] < tol * (abs(val[0]) + 1e-12):
            break
        c = np.mean(sim[:-1], axis=0)
        r = c + (c - sim[-1])
        fr = f(r)
        if fr < val[0]:
            e = c + 2 * (c - sim[-1])
            fe = f(e)
            sim[-1], val[-1] = (e, fe) if fe < fr else (r, fr)
        elif fr < val[-2]:
            sim[-1], val[-1] = r, fr
        else:
            k = c + 0.5 * (sim[-1] - c)
            fk = f(k)
            if fk < val[-1]:
                sim[-1], val[-1] = k, fk
            else:
                sim = [sim[0]] + [sim[0] + 0.5 * (s - sim[0]) for s in sim[1:]]
                val = [val[0]] + [f(s) for s in sim[1:]]
    i = int(np.argmin(val))
    return sim[i], val[i]


def calibrate(task, world, actions, measured, init, fit, fixed=None, restarts=3, seed=0):
    """`measured`: {state name: array}; `fit`: {parameter: (min, max)}. Returns values, uncertainties and residuals."""
    fixed = dict(fixed or {})
    names = list(fit)
    if not names:
        raise ValueError("no parameters to fit")
    lo, hi = np.array([fit[n][0] for n in names], float), np.array([fit[n][1] for n in names], float)
    if np.any(hi <= lo):
        raise ValueError("fit range: the maximum must be greater than the minimum")
    channels = list(measured)
    T = len(actions)
    for c in channels:
        if len(measured[c]) != T:
            raise ValueError(f"{c}: length {len(measured[c])} does not match the number of commands {T}")
    meas = {c: np.asarray(measured[c], float) for c in channels}
    scale = {c: max(float(np.nanstd(meas[c])), 1e-6) for c in channels}

    def residual(p):
        params = {**fixed, **{n: float(v) for n, v in zip(names, p)}}
        s, st = replay(task, world, params, init, actions)
        if len(s) < 2:
            return None
        out = []
        for c in channels:
            i = st.index(c)
            pred = np.full(T, s[-1, i], float)
            pred[:len(s)] = s[:, i]
            out.append((pred - meas[c]) / scale[c])
        return np.concatenate(out)

    def cost_u(u):
        p = lo + (hi - lo) * (0.5 + 0.5 * np.sin(u))  # a sine keeps values within bounds without penalties
        r = residual(p)
        return 1e6 if r is None or not np.all(np.isfinite(r)) else float(np.nanmean(r ** 2))

    rng = np.random.default_rng(seed)
    best = None
    for k in range(restarts):
        u0 = np.zeros(len(names)) if k == 0 else rng.uniform(-1.4, 1.4, len(names))
        u, v = _nelder_mead(cost_u, u0, step=0.4)
        if best is None or v < best[1]:
            best = (u, v)
    p = lo + (hi - lo) * (0.5 + 0.5 * np.sin(best[0]))
    r0 = residual(p)
    # uncertainties from linearization: they show what the data can determine at all
    span = hi - lo
    J = np.zeros((len(r0), len(names)))
    for i in range(len(names)):
        d = np.zeros(len(names))
        d[i] = max(abs(p[i]) * 1e-3, (hi[i] - lo[i]) * 1e-4)
        rp, rm = residual(np.minimum(p + d, hi)), residual(np.maximum(p - d, lo))
        if rp is None or rm is None:
            raise ValueError("the flight ends immediately: check the initial conditions and fit ranges")
        # derivative per fraction of the range: otherwise the condition number reflects measurement units, not the data
        J[:, i] = (rp - rm) / (np.minimum(p + d, hi)[i] - np.maximum(p - d, lo)[i]) * span[i]
    dof = max(len(r0) - len(names), 1)
    sigma2 = float(np.nansum(r0 ** 2)) / dof
    JJ = J.T @ J
    cond = float(np.linalg.cond(JJ))
    errs = np.sqrt(np.maximum(np.diag(np.linalg.pinv(JJ)) * sigma2, 0)) * span
    per = {}
    k = 0
    for c in channels:
        per[c] = float(np.sqrt(np.nanmean(r0[k:k + T] ** 2)) * scale[c])
        k += T
    return {"params": {n: float(v) for n, v in zip(names, p)}, "stderr": {n: float(v) for n, v in zip(names, errs)}, "rmse": per,
            "normalized_cost": best[1], "condition": cond, "at_bound": [n for n, v in zip(names, p) if v < lo[names.index(n)] + 1e-3 * (hi - lo)[names.index(n)] or v > hi[names.index(n)] - 1e-3 * (hi - lo)[names.index(n)]],
            "identifiable": bool(cond < 1e10)}


def load_csv(path):
    with open(path, encoding="utf8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no rows")
    acts = sorted((k for k in rows[0] if k.startswith("act_")), key=lambda x: int(x[4:]))
    if not acts:
        raise ValueError(f"{path}: no columns act_0, act_1…")
    a = np.array([[float(r[k]) for k in acts] for r in rows])
    st = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if not k.startswith("act_") and k != "t"}
    return a, st


def align_table(actions, states):
    """A table row is the state before that row command. Replay returns the state after the command, so shift by one step."""
    return actions[:-1], {k: v[1:] for k, v in states.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fit", action="append", required=True, help="name:min:max")
    ap.add_argument("--set", default="", help="fixed parameters, for example frame_skip=4,terrain_scale=0")
    ap.add_argument("--out")
    a = ap.parse_args()
    actions, st = load_csv(a.csv)
    missing = [k for k in ("x", "y", "z", "vx", "vy", "vz") if k not in st]
    if missing:
        ap.error(f"the CSV has no initial conditions: {missing}")
    init = [st[k][0] for k in ("x", "y", "z", "vx", "vy", "vz")]
    fit = {f.split(":")[0]: (float(f.split(":")[1]), float(f.split(":")[2])) for f in a.fit}
    fixed = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.set.split(",") if kv}
    actions, st = align_table(actions, st)
    r = calibrate(a.task, a.world, actions, st, init, fit, fixed)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if a.out:
        with open(a.out, "w", encoding="utf8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
