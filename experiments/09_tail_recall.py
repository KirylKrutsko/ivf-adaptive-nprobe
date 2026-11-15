"""Per-query recall distribution at EQUAL mean work: does the adaptive rule serve the worst
queries better than fixed nprobe, even though it loses on the mean?

For the fixed-nprobe work levels tuned in E4 (nprobe 13 / 26 / 45 / 122), each other method's
parameter is chosen so that its mean work on the test split matches (closest grid point).
Reported: mean recall, share of queries with recall 0, <= 0.5, p10 of per-query recall.
Methods: fixed, ratio rule, ratio + cap 32, fitted rule (08, fit on tune), Lagrangian oracle.
Writes results/09_tail_recall.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.oracle import at_cells, lagrangian_curve, load_curves, subset
from ivfadapt.rules import quantizer_topk

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
fitted = import_module("08_fitted_rule")


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    env = load_env(cfg_path)
    out = Path(env.cfg["results_dir"]); kr, C = env.k_rerank, env.max_cells
    curves, _ = load_curves(out / "03_oracle_curves.npz")
    tune, test = subset(curves, env.tune), subset(curves, env.test)
    rc = np.load(out / "04_rule_cells.npz"); rule_cells = rc["n_cells"][:, env.test]
    head = pd.read_csv(out / "05_tune_test.csv"); nprobes = head["nprobe_tuned"].astype(int).tolist()

    Dq, Iq = quantizer_topk(env.index, env.xq, C)
    X = fitted.features(Dq, env.list_sizes, Iq)
    lag_tune = lagrangian_curve(tune); j = int(np.argmin(np.abs(lag_tune["recall_mean"] - 0.9)))
    pred = fitted.ridge_fit(X[env.tune], np.log(lag_tune["cells"][j].astype(float)))
    p_te = np.exp(pred(X[env.test]))
    lag_test = lagrangian_curve(test)

    families = {
        "fixed": [np.full(len(env.test), c) for c in range(1, C + 1)],
        "ratio": list(rule_cells),
        "ratio_cap32": [np.minimum(r, 32) for r in rule_cells],
        "fitted (08)": [np.clip(np.rint(g * p_te), 1, C).astype(int) for g in np.geomspace(0.05, 20, 200)],
        "oracle_lagrangian": list(lag_test["cells"].astype(int)),
    }
    rows = []
    for p in nprobes:
        target_work = test.work[:, p - 1].mean()
        for name, allocs in families.items():
            works = np.array([at_cells(test, a)["work"].mean() for a in allocs])
            i = int(np.argmin(np.abs(works - target_work)))
            a = at_cells(test, allocs[i]); r = a["recall"]
            rows.append({"nprobe_level": p, "method": name, "work": a["work"].mean() + kr, "recall_mean": r.mean(),
                         "share_recall_0": (r == 0).mean(), "share_recall_le_0.5": (r <= 0.5).mean(),
                         "p10_recall": np.percentile(r, 10), "p25_recall": np.percentile(r, 25),
                         "cells_mean": allocs[i].mean(), "cells_p99": np.percentile(allocs[i], 99)})
    df = pd.DataFrame(rows); df.to_csv(out / "09_tail_recall.csv", index=False)
    pd.set_option("display.width", 220)
    print("per-query recall at matched mean work (test split):")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
