"""Wall-clock check for 06_rule_within_nprobe.py (a quick look ahead of E5, not E5 itself).

Single thread, all test queries in one batch, median of 5 runs after a warm-up, exact re-ranking
included. Three variants per (nprobe, ε):
- plain:      index.search(nprobe) + re-rank
- same path:  quantizer.search(nprobe) + search_preassigned with all nprobe cells + re-rank
- rule:       quantizer.search(nprobe) + min(rule(ε), nprobe) cells + search_preassigned + re-rank
The cap is nprobe, so the rule needs only the nprobe nearest centroids: no extra quantizer cost.
Note: the comparison is at the SAME nprobe, so the rule's recall is slightly lower.

Writes results/06_rule_within_nprobe_timing.csv.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import recall_at_k
from ivfadapt.rules import fixed_cells, ratio_cells
from ivfadapt.search import preassigned_search, rerank_exact

POINTS = [(26, 0.3287), (26, 0.286), (45, 0.3777), (122, 0.5473)]   # best-frontier ε from 06, plus 0.286


def median_time(f, reps: int = 5):
    out = f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); out = f(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), out


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    faiss.omp_set_num_threads(1)
    env = load_env(cfg_path)
    xq, gt, ix, kr, k = env.xq[env.test], env.gt[env.test], env.index, env.k_rerank, env.k

    def plain(p):
        ix.nprobe = p
        _, I = ix.search(xq, kr)
        return rerank_exact(env.xb, xq, I, k)[1]

    def path(p, eps):
        Dq, Iq = ix.quantizer.search(xq, p)
        cells = fixed_cells(Iq, p) if eps is None else ratio_cells(Dq, Iq, eps)
        _, I = preassigned_search(ix, xq, kr, cells, Dq)
        return rerank_exact(env.xb, xq, I, k)[1]

    rows = []
    for p, eps in POINTS:
        t0, I0 = median_time(lambda: plain(p))
        t1, _ = median_time(lambda: path(p, None))
        t2, I2 = median_time(lambda: path(p, eps))
        n = len(xq)
        rows.append({"nprobe": p, "eps": eps, "plain_ms_per_query": 1e3 * t0 / n,
                     "same_path_ms_per_query": 1e3 * t1 / n, "rule_ms_per_query": 1e3 * t2 / n,
                     "recall_plain": recall_at_k(I0, gt, k), "recall_rule": recall_at_k(I2, gt, k),
                     "speedup_vs_plain": t0 / t2, "speedup_vs_same_path": t1 / t2})
    df = pd.DataFrame(rows)
    out = Path(env.cfg["results_dir"]) / "06_rule_within_nprobe_timing.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
