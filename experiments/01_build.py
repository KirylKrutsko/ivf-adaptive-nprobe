"""E0: download the dataset, compute (and for SIFT1M check) the ground truth, build and cache the IVF-PQ index.

Run from the repo root:  uv run python experiments/01_build.py [configs/sift1m.yaml]
Writes results/01_build.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from ivfadapt.config import load_config
from ivfadapt.data import compare_ground_truth, compute_ground_truth, load_dataset, query_split
from ivfadapt.index import index_paths, load_or_build_index


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    cfg = load_config(cfg_path)
    data_dir = Path(cfg["data_dir"])
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"config": cfg_path}

    t0 = time.perf_counter()
    data = load_dataset(cfg)
    xb, xq = data["base"], data["query"]
    summary["data"] = {k: list(v.shape) for k, v in data.items()}
    summary["load_seconds"] = time.perf_counter() - t0
    print("data:", summary["data"])

    t0 = time.perf_counter()
    gt = compute_ground_truth(xb, xq, k=100, cache=data_dir / "groundtruth_k100.npy")
    summary["gt_seconds"] = time.perf_counter() - t0
    if "groundtruth" in data:
        summary["gt_check_top10"] = compare_ground_truth(gt, data["groundtruth"], k=cfg["k"])
        print("ground truth vs shipped (top-10):", summary["gt_check_top10"])

    tune, test = query_split(len(xq), cfg["split"]["tune_fraction"], cfg["seed"])
    np.savez(data_dir / f"query_split_seed{cfg['seed']}.npz", tune=tune, test=test)
    summary["split"] = {"tune": int(len(tune)), "test": int(len(test)), "seed": cfg["seed"]}

    index, cell_of, list_sizes, info = load_or_build_index(xb, cfg)
    summary["index"] = {
        "paths": {k: str(v) for k, v in index_paths(cfg).items()},
        "nlist": int(index.nlist),
        "ntotal": int(index.ntotal),
        "code_size_bytes": int(index.code_size),
        **info,
    }
    q = np.percentile(list_sizes, [0, 1, 50, 99, 100]).tolist()
    summary["list_sizes"] = {
        "mean": float(list_sizes.mean()),
        "std": float(list_sizes.std()),
        "min": int(q[0]), "p1": q[1], "median": q[2], "p99": q[3], "max": int(q[4]),
        "empty_lists": int((list_sizes == 0).sum()),
    }
    print("index:", summary["index"])
    print("list sizes:", summary["list_sizes"])

    with open(results_dir / "01_build.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", results_dir / "01_build.json")


if __name__ == "__main__":
    main(*sys.argv[1:])
