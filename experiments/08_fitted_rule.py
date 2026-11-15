"""Extension 6: a rule fitted to the oracle. How much of the oracle's saving is reachable by ANY
rule that only looks at the query's centroid distances?

Features per query (from the quantizer's top-256 distances): log(d_j / d_1) for a set of ranks j,
log d_1, and the log mean list size of the 8 nearest cells. Target: log of the oracle cell
count (per-query target c*(t), or the Lagrangian allocation at the λ that gives mean recall R
on tune). Ridge regression (closed form) on the tune split; on the test split the allocation
is n_q = clip(round(γ · exp(prediction)), 1, 256) and γ is swept to trace the curve. Being a
prefix rule it is read off the E2 curves. Also a shuffled control (same n_q, random queries).

Writes results/08_fitted_rule.csv and results/08_fitted_savings.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import interp_work_at_recall
from ivfadapt.oracle import at_cells, lagrangian_curve, load_curves, per_query_target, subset
from ivfadapt.rules import quantizer_topk

RANKS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def features(Dq, list_sizes, Iq):
    d = np.sqrt(np.maximum(Dq, 1e-12))
    X = [np.log(d[:, j - 1] / d[:, 0]) for j in RANKS]
    X.append(np.log(d[:, 0]))
    X.append(np.log(list_sizes[Iq[:, :8]].mean(1)))
    return np.stack(X, 1)


def ridge_fit(X, y, lam=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    A = Z.T @ Z + lam * np.eye(Z.shape[1]); A[-1, -1] -= lam
    w = np.linalg.solve(A, Z.T @ y)
    return lambda Xn: np.c_[(Xn - mu) / sd, np.ones(len(Xn))] @ w


def spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    env = load_env(cfg_path)
    out = Path(env.cfg["results_dir"]); kr, C = env.k_rerank, env.max_cells
    targets = env.cfg["recall_targets"]
    curves, _ = load_curves(out / "03_oracle_curves.npz")
    tune, test = subset(curves, env.tune), subset(curves, env.test)
    Dq, Iq = quantizer_topk(env.index, env.xq, C)
    X = features(Dq, env.list_sizes, Iq)
    Xtr, Xte = X[env.tune], X[env.test]
    rng = np.random.default_rng(env.cfg["seed"])

    # targets to fit
    fits = {}
    for t in [0.8, 0.9, 1.0]:
        c_star, _, reach = per_query_target(tune, t)
        fits[f"pqt{t}"] = (Xtr[reach], np.log(c_star[reach]))
    lag = lagrangian_curve(tune)
    for R in [0.8, 0.9, 0.95]:
        j = int(np.argmin(np.abs(lag["recall_mean"] - R)))
        fits[f"lagr{R}"] = (Xtr, np.log(lag["cells"][j].astype(float)))

    fixed_r, fixed_w = test.recall.mean(0), test.work.mean(0) + kr
    W_fixed = {R: interp_work_at_recall(fixed_r, fixed_w, R) for R in targets}
    rows, sav = [], []
    c_star_test_09 = per_query_target(test, 0.9)[0]
    for name, (Xf, yf) in fits.items():
        pred = ridge_fit(Xf, yf)
        p_te = np.exp(pred(Xte))
        rho = spearman(p_te, c_star_test_09)
        for ctrl in [False, True]:
            fam = name + ("_shuffled" if ctrl else "")
            base = rng.permutation(p_te) if ctrl else p_te
            for g in np.geomspace(0.05, 20, 80):
                n = np.clip(np.rint(g * base), 1, C).astype(int)
                a = at_cells(test, n)
                rows.append({"family": fam, "gamma": g, "recall": a["recall"].mean(), "work": a["work"].mean() + kr,
                             "cells_mean": n.mean(), "cells_p99": np.percentile(n, 99)})
            d = pd.DataFrame([r for r in rows if r["family"] == fam])
            s = {"family": fam, "spearman_vs_oracle_c*(0.9)": rho if not ctrl else np.nan}
            for R in targets:
                w = interp_work_at_recall(d.recall.values, d.work.values, R)
                s[f"saving@{R}"] = W_fixed[R] / w if np.isfinite(w) else np.nan
            sav.append(s)
    # oracle ceiling on test, for the same table
    lag_t = lagrangian_curve(test)
    s = {"family": "oracle_lagrangian (ceiling)", "spearman_vs_oracle_c*(0.9)": np.nan}
    for R in targets:
        s[f"saving@{R}"] = W_fixed[R] / (interp_work_at_recall(lag_t["recall_mean"], lag_t["work_mean"], R) + kr)
    sav.append(s)

    pd.DataFrame(rows).to_csv(out / "08_fitted_rule.csv", index=False)
    sav = pd.DataFrame(sav); sav.to_csv(out / "08_fitted_savings.csv", index=False)
    pd.set_option("display.width", 200)
    print("fitted on tune, evaluated on test: work saving vs fixed nprobe at matched mean recall")
    print(sav.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
