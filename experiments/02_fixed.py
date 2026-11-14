"""E1: fixed-nprobe baseline on the test split.

For every nprobe in the config grid: recall@10 after exact re-ranking (D5), raw PQ
recall@10, mean scanned vectors. Per-query recall and ndis are stored for the bootstrap.
Writes results/02_fixed.csv and results/02_fixed_perquery.npz.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import recall_per_query
from ivfadapt.rules import quantizer_topk
from ivfadapt.search import fixed_search, rerank_exact, reset_stats, scanned_vectors, stats_ndis


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    env = load_env(cfg_path)
    out = Path(env.cfg["results_dir"]); out.mkdir(exist_ok=True)
    xq, gt = env.xq[env.test], env.gt[env.test]
    grid = env.cfg["search"]["nprobe_grid"]
    k, kr = env.k, env.k_rerank

    _, Iq = quantizer_topk(env.index, xq, max(grid))
    rows, rec_q, ndis_q = [], [], []
    for p in grid:
        ndis = scanned_vectors(Iq[:, :p], env.list_sizes)
        reset_stats()
        t0 = time.perf_counter()
        _, I_pq = fixed_search(env.index, xq, kr, p)
        t_search = time.perf_counter() - t0
        # Faiss's own count may differ by a cell swap on a near-tie (BLAS rounding differs
        # between code paths); require agreement to 1e-4 and record the gap.
        ndis_gap = stats_ndis() - int(ndis.sum())
        assert abs(ndis_gap) <= 1e-4 * ndis.sum(), (p, ndis_gap)
        t0 = time.perf_counter()
        _, I_rr = rerank_exact(env.xb, xq, I_pq, k)
        t_rerank = time.perf_counter() - t0
        r = recall_per_query(I_rr, gt, k)
        r_raw = recall_per_query(I_pq, gt, k)
        rows.append({
            "nprobe": p, "recall": r.mean(), "recall_raw_pq": r_raw.mean(),
            "ndis_mean": ndis.mean(), "work_mean": ndis.mean() + kr,
            "ndis_p50": np.median(ndis), "ndis_p99": np.percentile(ndis, 99), "faiss_ndis_gap": ndis_gap,
            "search_ms_per_query": 1e3 * t_search / len(xq), "rerank_ms_per_query": 1e3 * t_rerank / len(xq),
        })
        rec_q.append(r); ndis_q.append(ndis)
        print(f"nprobe={p:4d} recall={r.mean():.4f} raw={r_raw.mean():.4f} ndis={ndis.mean():9.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "02_fixed.csv", index=False)
    np.savez(out / "02_fixed_perquery.npz", nprobe=np.array(grid), recall=np.array(rec_q),
             ndis=np.array(ndis_q), query_ids=env.test)
    print("wrote", out / "02_fixed.csv")


if __name__ == "__main__":
    main(*sys.argv[1:])
