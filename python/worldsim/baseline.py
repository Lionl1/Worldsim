"""Базовая модель для сравнения: MLP предсказывает приращение наблюдения. Каналы рельефа не предсказывает: рельеф ей неизвестен.

    python -m worldsim.baseline train --data runs/moon runs/earth --out runs/baseline.pt
    python -m worldsim.baseline qualify --model runs/baseline.pt --world moon=runs/moon --world mix=runs/mix --out runs/report
"""
import argparse

import numpy as np

from .data import Dataset
from .model import WorldModel
from .qualify import qualify, save, to_markdown


def _is_terrain(name):
    return name.startswith(("lidar", "radar"))


def _transform(o, ip):
    o = o.copy()
    if ip >= 0:
        o[..., ip] = np.log1p(np.maximum(o[..., ip], 0))
    return o


def train(paths, out, epochs=12, cap=400_000, use_context=True, unroll=1, seed=0):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dss = [Dataset(p) for p in paths]
    names = dss[0].obs_names
    keep = [i for i, n in enumerate(names) if not _is_terrain(n)]
    ip = keep.index(names.index("pressure_kpa")) if "pressure_kpa" in names else -1
    igk = [dss[0].param("g")] if use_context else []
    X, Y = [], []
    for d in dss:
        s = rng.choice(len(d), min(cap, len(d)), replace=False)
        s.sort()
        o = _transform(np.asarray(d.obs[s], np.float32)[:, keep], ip)
        n = _transform(np.asarray(d.next_obs[s], np.float32)[:, keep], ip)
        ctx = np.asarray(d.params, np.float32)[np.asarray(d.episode[s]).astype(int)][:, igk]
        X.append(np.hstack([o, np.asarray(d.act[s], np.float32), ctx]))
        Y.append(n - o)
    X, Y = np.vstack(X), np.vstack(Y)
    mx, sx = X.mean(0), X.std(0)
    sx[sx < 1e-6] = 1
    my, sy = Y.mean(0), Y.std(0)
    sy[sy < 1e-9] = 1
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, Y.shape[1]))
    tx, ty = torch.tensor((X - mx) / sx, dtype=torch.float32), torch.tensor((Y - my) / sy, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), 2e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * (len(tx) // 4096 + 1))
    for _ in range(epochs):
        p = torch.randperm(len(tx))
        for i in range(0, len(tx), 4096):
            b = p[i:i + 4096]
            loss = ((net(tx[b]) - ty[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    torch.save({"state": net.state_dict(), "keep": keep, "ip": ip, "mx": mx.tolist(), "sx": sx.tolist(), "my": my.tolist(), "sy": sy.tolist(), "ctx": igk, "names": names, "in": X.shape[1], "out": Y.shape[1]}, out)
    return out


class Baseline(WorldModel):
    def __init__(self, path, name="baseline-mlp"):
        import torch
        from torch import nn
        self.torch = torch
        # weights_only: чужой файл модели не может выполнить код при загрузке
        c = torch.load(path, weights_only=True)
        for k in ("mx", "sx", "my", "sy"):
            c[k] = np.asarray(c[k], np.float64)
        self.c, self.name = c, name
        self.net = nn.Sequential(nn.Linear(c["in"], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, c["out"]))
        self.net.load_state_dict(c["state"])
        self.net.eval()
        self.ctx = None

    def init(self, obs, params):
        self.ctx = np.asarray(params, np.float32)[:, self.c["ctx"]]
        return np.asarray(obs, np.float32).copy()

    def step(self, state, action):
        c = self.c
        o = _transform(state[:, c["keep"]], c["ip"])
        # вход держим в области обучения: вне её сеть экстраполирует и расхождение идёт лавиной
        x = np.clip((np.hstack([o, action, self.ctx]) - c["mx"]) / c["sx"], -6, 6)
        with self.torch.no_grad():
            d = self.net(self.torch.tensor(x, dtype=self.torch.float32)).numpy() * c["sy"] + c["my"]
        n = np.clip(o + d, c["mx"][:o.shape[1]] - 8 * c["sx"][:o.shape[1]], c["mx"][:o.shape[1]] + 8 * c["sx"][:o.shape[1]])
        if c["ip"] >= 0:
            n[:, c["ip"]] = np.expm1(n[:, c["ip"]])
        out = state.copy()
        out[:, c["keep"]] = n
        return out, out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=12)
    q = sub.add_parser("qualify")
    q.add_argument("--model", required=True)
    q.add_argument("--world", action="append", required=True, help="название=папка данных")
    q.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "train":
        print(train(a.data, a.out, epochs=a.epochs))
    else:
        ds = {w.split("=", 1)[0]: Dataset(w.split("=", 1)[1]) for w in a.world}
        r = qualify(Baseline(a.model), ds)
        save(r, a.out)
        print(to_markdown(r))


if __name__ == "__main__":
    main()
