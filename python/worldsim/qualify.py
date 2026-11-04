"""Лестница квалификации world model. Истина всегда известна, поэтому каждая проверка даёт число, а не мнение.

L1 Точность      — ошибка длинного предсказания и «горизонт доверия» при допусках миссии.
L2 Законы физики — нарушения в воображении относительно уровня самого симулятора.
L3 Отклик на действие — разные действия из одной точки против истинного физического ответа (ветвление).
L4 Оптимизм      — предсказывает ли модель благополучие там, где на самом деле авария.
L5 Границы       — те же проверки в мирах, которых не было в обучении: где модели можно верить.

Опирается на идеи MiraBench и CrashTwin, но в отличие от них работает с точной физикой и миссионными допусками.
Отчёт — инженерное свидетельство для обзора, а не сертификация.
"""
import datetime
import json
import platform

import numpy as np

# допуски по умолчанию: какая ошибка ещё годится для планирования. Можно задать свои.
TOLERANCES = {
    "lander": {"gps_dx": 10.0, "gps_dy": 10.0, "gps_dz": 5.0, "vel_x": 2.0, "vel_y": 2.0, "vel_z": 1.0, "up_bx": 0.1, "up_by": 0.1},
    "drone": {"goal_dx": 3.0, "goal_dy": 3.0, "goal_dz": 2.0, "vel_x": 1.0, "vel_y": 1.0, "vel_z": 0.5, "up_bx": 0.1, "up_by": 0.1},
    "rover": {"goal_dx": 3.0, "goal_dy": 3.0, "head_cos": 0.1, "head_sin": 0.1, "speed": 0.3},
}
HORIZONS_S = (0.2, 1.0, 2.0, 5.0, 10.0, 20.0)
UNSAFE_SPEED = {"lander": 2.5, "drone": 3.0}
DIVERGED = 1e6


def _rollout(model, ds, starts, horizon):
    """Открытый цикл на логированных действиях. Возвращает предсказания, истину и маску «эпизод ещё идёт»."""
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
        # разошедшуюся модель считаем большой ошибкой, а не роняем расчёт
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
    # горизонт доверия: до какого времени ВСЕ каналы укладываются в допуски (по RMSE). Истины много — порог устойчивый.
    horizon = 0
    limiting = None
    for t in range(len(pred)):
        m = mask[t]
        if m.sum() < 5:
            break
        bad = None
        for k, i in idx.items():
            # через not <=: NaN и бесконечность должны считаться выходом за допуск
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
    """Нарушения законов в траекториях x (T×N×O). Возвращает словарь закон → массив отклонений."""
    names, dt = ds.obs_names, ds.dt
    laws = {}
    if "up_bx" in names:
        i, j, k = [names.index(c) for c in ("up_bx", "up_by", "up_bz")]
        laws["|up| = 1 (вращение сохраняет длину)"] = np.abs(np.sqrt(x[..., i] ** 2 + x[..., j] ** 2 + x[..., k] ** 2) - 1)
    if "head_cos" in names:
        i, j = names.index("head_cos"), names.index("head_sin")
        laws["cos² + sin² = 1 (курс)"] = np.abs(np.sqrt(x[..., i] ** 2 + x[..., j] ** 2) - 1)
    if ds.task == "rover":
        p, c, s, v = names.index("goal_dx"), names.index("head_cos"), names.index("head_sin"), names.index("speed")
        laws["dx/dt = v·cos (кинематика)"] = np.abs(np.diff(x[..., p], axis=0) + x[1:, :, v] * x[1:, :, c] * dt)
    else:
        sign = 1.0 if ds.task == "lander" else -1.0
        for a, b, name in ((0, 3, "x"), (1, 4, "y"), (2, 5, "z")):
            laws[f"d{name}/dt = v{name} (кинематика)"] = np.abs(np.diff(x[..., a], axis=0) - sign * x[1:, :, b] * dt)
    for ch, label in (("fuel_frac", "топливо не прибывает"), ("battery_frac", "заряд без подзарядки не растёт")):
        if ch in names and not (ds.task == "rover"):
            laws[label] = np.maximum(np.diff(x[..., names.index(ch)], axis=0), 0)
    return laws


def _law_tolerance(law, ds):
    """Какое нарушение ещё допустимо за шаг или в норме."""
    if law.startswith("|up|") or law.startswith("cos²"):
        return 0.02
    if "кинематика" in law:
        return 0.02
    return 1e-4


def level_physics(pred, true, mask, ds):
    # float64: квадраты разошедшихся предсказаний не должны переполнять расчёт
    pl, tl = _law_series(pred.astype(np.float64), ds), _law_series(true.astype(np.float64), ds)
    out = {}
    for k in pl:
        m = mask[-pl[k].shape[0]:]
        # медиана: одна разошедшаяся траектория не должна скрывать типичное поведение; долю расхождений показываем отдельно
        mp, tp = float(np.median(pl[k][m])), float(np.median(tl[k][m]))
        # допуск нужен, потому что у симулятора некоторые законы выполняются почти идеально: сравнивать с нулём нельзя
        limit = max(10 * tp, _law_tolerance(k, ds))
        out[k] = {"model": mp, "simulator": tp, "limit": limit, "passed": bool(mp <= limit)}
    return out


def level_action(model, ds, tol, rng, max_points=2000):
    """Предсказанная разница между ветвями против истинной: R² = 1 только если модель верно реагирует на действие."""
    if not ds.has_counterfactuals or len(ds.cf_obs) == 0:
        return {"skipped": "нет ветвлений: соберите данные с --counterfactuals"}
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
    for t, name in ((4, "0.1 с"), (H - 1, f"{H * ds.dt:g} с")):
        good = alive[:, 0, t] & alive[:, 1:, t].all(axis=1)
        if good.sum() < 10:
            continue
        # разница относительно первой ветви убирает общий сдвиг: остаётся только эффект действия
        dt_ = nxt[good, 1:, t][:, :, idx] - nxt[good, :1, t][:, :, idx]
        dp = pred[good, 1:, t][:, :, idx] - pred[good, :1, t][:, :, idx]
        ss = ((dt_ - dt_.mean((0, 1), keepdims=True)) ** 2).sum((0, 1))
        r2 = 1 - ((dp - dt_) ** 2).sum((0, 1)) / np.maximum(ss, 1e-12)
        rows[name] = {k: float(r2[i]) for i, k in enumerate(channels)}
    worst = min((v for r in rows.values() for v in r.values()), default=float("nan"))
    return {"r2_of_action_effect": rows, "worst": worst, "points": int(m)}


def level_optimism(model, ds, rng, lead_s=1.0, max_eps=600):
    """Берём момент за lead_s до конца эпизода и спрашиваем модель, чем кончится. Сравниваем с тем, что было на самом деле."""
    table = ds.episodes_table()
    lead = int(round(lead_s / ds.dt))
    eps = [i for i, r in enumerate(table) if int(r["steps"]) > lead and (r["crashed"] == "1" or r["success"] == "1")]
    if not eps:
        return {"skipped": "нет завершённых эпизодов"}
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
            # опасная скорость вблизи поверхности или большой наклон
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
    """Простые прозрачные правила: важен не балл, а какая ступень не пройдена и почему."""
    gates = {
        "L1": acc["trust_horizon_s"] >= min_horizon_s,
        "L2": all(v["passed"] for v in phys.values()) if phys else False,
        "L3": ("worst" in action) and action["worst"] >= 0.5,
        # и пропуски, и ложные тревоги: модель, которая всегда предсказывает аварию, бесполезна
        "L4": opt.get("missed_crash_rate") is not None and opt["missed_crash_rate"] <= 0.2 and (opt.get("false_alarm_rate") or 0.0) <= 0.3,
    }
    return gates


def qualify(model, datasets, n_starts=400, horizon_s=20.0, min_horizon_s=2.0, tolerances=None, seed=0):
    """Основная функция. `datasets`: {название: Dataset}; первая запись — базовый мир, остальные — границы применимости."""
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
    mark = lambda ok: "пройдено" if ok else "не пройдено"
    s = [f"# Квалификация модели: {r['model']}\n", f"Создано {r['created']}. Стартов на мир: {r['n_starts']}, seed {r['seed']}. Минимальный горизонт: {r['min_horizon_s']} с.\n",
         "## Сводка\n", "| Мир | Горизонт доверия | L1 Точность | L2 Физика | L3 Отклик | L4 Оптимизм |", "|---|---|---|---|---|---|"]
    for n, w in r["worlds"].items():
        g = w["gates"]
        s.append(f"| {n} | {w['accuracy']['trust_horizon_s']:.2f} с | {mark(g['L1'])} | {mark(g['L2'])} | {mark(g['L3'])} | {mark(g['L4'])} |")
    for n, w in r["worlds"].items():
        a, o = w["accuracy"], w["optimism"]
        s += [f"\n## {n}\n", f"Мир: {', '.join(w['world'])}. Данные: {w['transitions']} переходов, шум {w['noise']}, отпечаток `{w['fingerprint']}`.\n",
              f"**Горизонт доверия {a['trust_horizon_s']:.2f} с**, ограничивает канал `{a['limited_by']}`. Разошлось к концу предсказания: {_fmt(a['diverged_share'], 2)}.\n", "### Законы физики в воображении\n",
              "| Закон | Модель | Симулятор | Допуск | Итог |", "|---|---|---|---|---|"]
        s += [f"| {k} | {v['model']:.5f} | {v['simulator']:.5f} | {v['limit']:.5f} | {mark(v['passed'])} |" for k, v in w["physics"].items()]
        ac = w["action"]
        s.append("\n### Отклик на действие\n")
        if "skipped" in ac:
            s.append(ac["skipped"])
        else:
            s.append(f"R² эффекта действия относительно истинного ветвления; худший канал: {ac['worst']:.3f}.\n")
            for t, row in ac["r2_of_action_effect"].items():
                s.append(f"- {t}: " + ", ".join(f"{k} {v:.2f}" for k, v in row.items()))
        s.append("\n### Оптимизм\n")
        s.append(o.get("skipped") or f"За {o['lead_s']:.1f} с до конца: пропущено аварий {_fmt(o['missed_crash_rate'])} из {o['crashes']}, ложных тревог {_fmt(o['false_alarm_rate'])} из {o['safe']}.")
    s.append("\n*Отчёт описывает поведение модели в упрощённой физике worldsim: это инженерное свидетельство для обзора, а не сертификация.*\n")
    return "\n".join(s)


def save(report, out):
    from pathlib import Path
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    (p / "qualification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    (p / "qualification.md").write_text(to_markdown(report), encoding="utf8")
