"""Export to LeRobotDataset v3.0 (https://huggingface.co/docs/lerobot/lerobot-dataset-v3). Requires pyarrow and pandas.

    python -m worldsim.export --data runs/moon --out runs/moon_lerobot

Exports observations, actions, rewards and episode end flags. The true state goes into `observation.environment_state`.
The camera is not exported. Frame rate comes from the control step. Loading was verified on lerobot 0.6.1.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import Dataset


def _stats(x):
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "min": x.min(0).tolist(), "max": x.max(0).tolist(), "count": [len(x)]}


def to_lerobot(ds, out, chunk_rows=1_000_000):
    """Writes the folder `out` and returns the path. Files are Parquet, episodes run in sequence and are described in meta/episodes."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pandas  # noqa: F401
    except ImportError as e:
        raise RuntimeError("export requires pyarrow and pandas: pip install pyarrow pandas") from e
    out = Path(out)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    fps = 1.0 / ds.dt
    n, O, A, S = len(ds), ds.obs.shape[1], ds.act.shape[1], ds.state.shape[1]
    task_text = f"{ds.task}: {', '.join(ds.meta['world_names'])}"
    lists = lambda a: pa.FixedSizeListArray.from_arrays(pa.array(np.asarray(a, np.float32).ravel()), a.shape[1])
    file_index, written, per_file = 0, 0, []
    while written < n:
        end = min(written + chunk_rows, n)
        # an episode is not split between files: this makes reading simpler, both streaming and from disk
        ep_end = np.searchsorted(ds.starts, end)
        if end < n and ep_end < len(ds.starts) and ds.starts[ep_end] > written:
            end = int(ds.starts[ep_end])
        idx = np.arange(written, end)
        table = pa.table({
            "observation.state": lists(np.asarray(ds.obs[written:end])),
            "observation.environment_state": lists(np.asarray(ds.state[written:end])),
            "action": lists(np.asarray(ds.act[written:end])),
            "next.reward": pa.array(np.asarray(ds.reward[written:end], np.float32)),
            "next.done": pa.array((np.asarray(ds.terminated[written:end]) | np.asarray(ds.truncated[written:end])).astype(bool)),
            "timestamp": pa.array((np.asarray(ds.step[written:end]) * ds.dt).astype(np.float32)),
            "frame_index": pa.array(np.asarray(ds.step[written:end]).astype(np.int64)),
            "episode_index": pa.array(np.asarray(ds.episode[written:end]).astype(np.int64)),
            "index": pa.array(idx.astype(np.int64)),
            "task_index": pa.array(np.zeros(end - written, np.int64)),
        })
        pq.write_table(table, out / "data" / "chunk-000" / f"file-{file_index:03d}.parquet")
        per_file.append((written, end, file_index))
        written, file_index = end, file_index + 1
    file_of = lambda row: next(f for a, b, f in per_file if a <= row < b)
    e_starts, e_len = ds.starts, ds.lengths
    pq.write_table(pa.table({
        "episode_index": pa.array(np.arange(len(e_starts), dtype=np.int64)),
        "tasks": pa.array([[task_text]] * len(e_starts)),
        "length": pa.array(e_len.astype(np.int64)),
        "data/chunk_index": pa.array(np.zeros(len(e_starts), np.int64)),
        "data/file_index": pa.array([file_of(int(s)) for s in e_starts], type=pa.int64()),
        "dataset_from_index": pa.array(e_starts.astype(np.int64)),
        "dataset_to_index": pa.array((e_starts + e_len).astype(np.int64)),
        "meta/episodes/chunk_index": pa.array(np.zeros(len(e_starts), np.int64)),
        "meta/episodes/file_index": pa.array(np.zeros(len(e_starts), np.int64)),
    }), out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    # LeRobot reads the task text from the pandas index, so the file is written with exactly this index
    import pandas as pd
    pd.DataFrame({"task_index": [0]}, index=pd.Index([task_text], name="task")).to_parquet(out / "meta" / "tasks.parquet")
    scalar = lambda dtype: {"dtype": dtype, "shape": [1], "names": None}
    info = {
        "codebase_version": "v3.0", "robot_type": f"worldsim_{ds.task}", "total_episodes": len(e_starts), "total_frames": int(n), "total_tasks": 1,
        "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 500, "fps": fps, "splits": {"train": f"0:{len(e_starts)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet", "video_path": None,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [O], "names": ds.obs_names},
            "observation.environment_state": {"dtype": "float32", "shape": [S], "names": ds.meta["state_names"]},
            "action": {"dtype": "float32", "shape": [A], "names": ds.meta["act_names"]},
            "next.reward": scalar("float32"), "next.done": scalar("bool"), "timestamp": scalar("float32"), "frame_index": scalar("int64"), "episode_index": scalar("int64"), "index": scalar("int64"), "task_index": scalar("int64"),
        },
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf8")
    step = max(1, n // 200_000)
    stats = {"observation.state": _stats(ds.obs[::step]), "observation.environment_state": _stats(ds.state[::step]), "action": _stats(ds.act[::step]), "next.reward": _stats(ds.reward[::step])}
    (out / "meta" / "stats.json").write_text(json.dumps(stats), encoding="utf8")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(to_lerobot(Dataset(a.data), a.out))


if __name__ == "__main__":
    main()
