"""E3: ε sweep of the ratio rule on both splits.

The rule keeps a prefix of each query's nearest cells, so its result at any ε is read off the
per-query curves of E2 (results/03_oracle_curves.npz): no search is needed per ε. A few ε are
then re-run through the real search path (search_preassigned + exact re-ranking) on the test
split, and must reproduce the curve values query by query.

Writes results/04_rule.csv (one row per ε and split) and results/04_rule_cells.npz
(cells per query for every ε, all 10k queries).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import recall_per_query
from ivfadapt.oracle import at_cells, load_curves, subset
from ivfadapt.rules import quantizer_topk, ratio_cells, ratio_n_cells
from ivfadapt.search import preassigned_search, rerank_exact, reset_stats, scanned_vectors, stats_ndis

EPS_GRID = np.unique(np.concatenate([[0.0], np.geomspace(1e-3, 1.0, 150)]))
CHECK_EPS = [0.02, 0.06, 0.15]


def check_against_real_search(env, curves, Dq, Iq, rows, eps) -> dict:
    """Run the rule for real on `rows` and compare with the value read off the curves."""
    xq, gt = env.xq[rows], env.gt[rows]
    cells = ratio_cells(Dq[rows], Iq[rows], eps)
    reset_stats()
    _, I_pq = preassigned_search(env.index, xq, env.k_rerank, cells, Dq[rows])
    faiss_ndis = stats_ndis()
    _, I_rr = rerank_exact(env.xb, xq, I_pq, env.k)
    r_real = recall_per_query(I_rr, gt, env.k)
    ndis_real = scanned_vectors(cells, env.list_sizes)
    from_curves = at_cells(subset(curves, rows), ratio_n_cells(Dq[rows], eps))
    return {"eps": eps, "n": len(rows),
            "recall_real": r_real.mean(), "recall_curves": from_curves["recall"].mean(),
            "recall_mismatch_queries": int((np.abs(r_real - from_curves["recall"]) > 1e-6).sum()),
            "ndis_mismatch_queries": int((ndis_real != from_curves["work"]).sum()),
            "faiss_ndis_equal": bool(faiss_ndis == ndis_real.sum())}


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    env = load_env(cfg_path)
    out = Path(env.cfg["results_dir"])
    curves, qids = load_curves(out / "03_oracle_curves.npz")
    assert len(qids) == len(env.xq) and np.array_equal(qids, np.arange(len(env.xq)))

    Dq, Iq = quantizer_topk(env.index, env.xq, curves.c_max)
    # The curves were built with the quantizer run in chunks; a fresh run on all queries can swap
    # near-tied centroids (BLAS blocking). Swapped cells have equal distances to float precision,
    # so the rule's cell counts are unaffected; require the swaps to be rare.
    mism = (Iq != curves.cells).mean()
    print(f"quantizer order vs curves: {mism:.2e} of entries differ (near-ties)")
    assert mism < 1e-3, "quantizer order differs from the one the curves were built on"

    n_cells = np.stack([ratio_n_cells(Dq, e) for e in EPS_GRID]).astype(np.int16)   # (n_eps, n_q)
    np.savez(out / "04_rule_cells.npz", eps=EPS_GRID, n_cells=n_cells, query_ids=qids)

    rows = []
    for e, n in zip(EPS_GRID, n_cells):
        res = at_cells(curves, n)
        for split, ids in [("tune", env.tune), ("test", env.test)]:
            c = n[ids]
            rows.append({"eps": e, "split": split, "recall": res["recall"][ids].mean(),
                         "recall_raw_pq": res["recall_raw"][ids].mean(), "coverage": res["coverage"][ids].mean(),
                         "ndis_mean": res["work"][ids].mean(), "work_mean": res["work"][ids].mean() + env.k_rerank,
                         "ndis_p99": np.percentile(res["work"][ids], 99),
                         "cells_mean": c.mean(), "cells_median": np.median(c), "cells_p90": np.percentile(c, 90),
                         "cells_p99": np.percentile(c, 99), "share_1_cell": (c == 1).mean(),
                         "share_at_cap": (c >= curves.c_max).mean()})
    df = pd.DataFrame(rows)
    df.to_csv(out / "04_rule.csv", index=False)

    checks = pd.DataFrame([check_against_real_search(env, curves, Dq, Iq, env.test, e) for e in CHECK_EPS])
    checks.to_csv(out / "04_rule_check.csv", index=False)
    print("real search vs curves (test split):"); print(checks.to_string(index=False))
    assert (checks["ndis_mismatch_queries"] == 0).all() and checks["faiss_ndis_equal"].all()
    assert (checks["recall_mismatch_queries"] <= 0.001 * checks["n"]).all()

    test = df[df.split == "test"]
    show = test.iloc[::8][["eps", "recall", "ndis_mean", "cells_mean", "cells_median", "cells_p99", "share_at_cap"]]
    pd.set_option("display.width", 200)
    print("\nε sweep (test split, every 8th ε):"); print(show.round(4).to_string(index=False))
    print("wrote", out / "04_rule.csv")


if __name__ == "__main__":
    main(*sys.argv[1:])
