"""E2: per-query recall/work curves for all queries, both oracles, histogram data.

Writes results/03_oracle_curves.npz (all 10k queries, C_max = max_cells),
results/03_oracle_lagrangian.csv, results/03_oracle_perquery_target.csv,
results/03_oracle_hist.csv and results/03_oracle_summary.json (test split only).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import interp_work_at_recall
from ivfadapt.oracle import (compute_curves, lagrangian_curve, load_curves, per_query_target,
                             per_query_target_curve, save_curves, subset)


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    env = load_env(cfg_path)
    out = Path(env.cfg["results_dir"]); out.mkdir(exist_ok=True)
    curves_path = out / "03_oracle_curves.npz"
    if curves_path.exists():
        curves, qids = load_curves(curves_path)
        assert curves.c_max == env.max_cells and len(qids) == len(env.xq)
        print("loaded cached curves", curves_path)
    else:
        t0 = time.perf_counter()
        curves = compute_curves(env.index, env.xb, env.xq, env.gt, env.cell_of, env.list_sizes,
                                c_max=env.max_cells, k=env.k, k_rerank=env.k_rerank)
        save_curves(curves_path, curves, np.arange(len(env.xq)))
        print(f"curves computed in {time.perf_counter() - t0:.0f}s -> {curves_path}")

    test = subset(curves, env.test)
    kr = env.k_rerank
    summary: dict = {"n_test": int(len(env.test)), "c_max": test.c_max, "k_rerank": kr}

    # fixed-nprobe curve from the same arrays (should match E1)
    fixed = pd.DataFrame({"nprobe": np.arange(1, test.c_max + 1), "recall": test.recall.mean(0),
                          "recall_raw_pq": test.recall_raw.mean(0), "coverage": test.coverage.mean(0),
                          "ndis_mean": test.work.mean(0)})
    fixed["work_mean"] = fixed["ndis_mean"] + kr
    fixed.to_csv(out / "03_fixed_from_curves.csv", index=False)

    # oracle 2: Lagrangian
    lag = lagrangian_curve(test)
    lag_df = pd.DataFrame({k: v for k, v in lag.items() if k != "cells"})
    lag_df["work_mean_total"] = lag_df["work_mean"] + kr
    lag_df.to_csv(out / "03_oracle_lagrangian.csv", index=False)
    np.save(out / "03_oracle_lagrangian_cells.npy", lag["cells"])

    # oracle 1: per-query target curve + histogram data
    targets = np.round(np.arange(0.1, 1.0001, 0.1), 1)
    pqt = pd.DataFrame(per_query_target_curve(test, targets))
    pqt["work_mean_total"] = pqt["work_mean"] + kr
    pqt.to_csv(out / "03_oracle_perquery_target.csv", index=False)

    hist_rows, hist = [], {}
    for t in env.cfg["recall_targets"] + [1.0]:
        c_star, w, reach = per_query_target(test, t)
        hist[f"c_star_{t}"] = np.where(reach, c_star, -1)
        cs = c_star[reach]
        hist_rows.append({"target": t, "unreachable": int((~reach).sum()), "median_cells": float(np.median(cs)),
                          "mean_cells": float(cs.mean()), "p90_cells": float(np.percentile(cs, 90)),
                          "p99_cells": float(np.percentile(cs, 99)), "share_1_cell": float((cs == 1).mean()),
                          "share_le_3_cells": float((cs <= 3).mean()), "share_gt_20_cells": float((cs > 20).mean())})
    pd.DataFrame(hist_rows).to_csv(out / "03_oracle_hist_summary.csv", index=False)
    np.savez(out / "03_oracle_hist.npz", query_ids=env.test, **hist)
    summary["per_query_target"] = hist_rows

    # headline: work at each recall target, fixed vs oracles
    rows = []
    for R in env.cfg["recall_targets"]:
        w_fixed = interp_work_at_recall(fixed["recall"].values, fixed["ndis_mean"].values, R)
        w_lag = interp_work_at_recall(lag["recall_mean"], lag["work_mean"], R)
        w_pqt = interp_work_at_recall(pqt["recall_mean"].values, pqt["work_mean"].values, R)
        rows.append({"target": R, "ndis_fixed": w_fixed, "ndis_oracle_lagrangian": w_lag,
                     "ndis_oracle_perquery": w_pqt,
                     "max_saving_lagrangian": (w_fixed + kr) / (w_lag + kr),
                     "max_saving_perquery": (w_fixed + kr) / (w_pqt + kr)})
    head = pd.DataFrame(rows)
    head.to_csv(out / "03_oracle_headline.csv", index=False)
    summary["headline"] = rows
    summary["coverage_at_c_max"] = float(test.coverage[:, -1].mean())
    summary["recall_at_c_max"] = float(test.recall[:, -1].mean())
    with open(out / "03_oracle_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.set_option("display.width", 200)
    print("\nper-query-target oracle (test split):"); print(pd.DataFrame(hist_rows).round(3).to_string(index=False))
    print("\nwork at recall targets (ndis, test split):"); print(head.round(3).to_string(index=False))
    print(f"\ncoverage at {test.c_max} cells: {summary['coverage_at_c_max']:.4f}, recall: {summary['recall_at_c_max']:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
