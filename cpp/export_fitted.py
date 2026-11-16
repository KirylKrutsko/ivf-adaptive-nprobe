"""Fit the static fitted rule (as in experiments/16_fitted_rule_tuned.py) and export it for the C++
benchmark. Two feature sets:
  full : log d_j/d_1 at ranks 2..256 (the 15 ranks of 08), log d_1, log mean size of the 8 nearest lists
         -> needs the 256 nearest centroids before deciding
  r32  : the same, but ranks 2..32 only -> needs the 32 nearest (the C++ code asks for 64)
  c32, c16: the compact 3-feature models of experiments/17 (d32/d1 or d16/d1, + d1 + size8)
Per variant and recall target: ridge model fitted on tune to log of the Lagrangian oracle's cells,
scale γ chosen on tune by 2-fold cross-fitting (same as 16); tuned saving on test vs the fixed
nprobe chosen on tune, with bootstrap CIs. Writes data/cpp/fitted_<variant>_R<R>.txt
(ranks, mu, sd, w, gamma) and data/cpp/cells_fitted_<variant>_R<R>.i32 (Python cells per test
query), and results/cpp_fitted_python.csv.
Run after 03:  uv run python cpp/export_fitted.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import bootstrap_weights
from ivfadapt.oracle import at_cells, lagrangian_curve, load_curves, subset
from ivfadapt.rules import quantizer_topk

VARIANTS = {"full": [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
            "r32": [2, 3, 4, 6, 8, 12, 16, 24, 32],
            "c32": [32],        # compact (17_fitted_rule_analysis): d32/d1 + d1 + size8
            "c16": [16]}        # compact: d16/d1 + d1 + size8
# C++ runs: (variant, nearest centroids requested for every query)
RUNS = [("full", 256), ("r32", 64), ("c32", 64), ("c16", 32), ("c16", 64)]
GAMMAS = np.geomspace(0.05, 20, 400)


def features(Dq, Iq, list_sizes, ranks):
    """Same as experiments/08_fitted_rule.features, for a given list of ranks."""
    d = np.sqrt(np.maximum(Dq, 1e-12))
    X = [np.log(d[:, j - 1] / d[:, 0]) for j in ranks]
    X.append(np.log(d[:, 0]))
    X.append(np.log(list_sizes[Iq[:, :8]].mean(1)))
    return np.stack(X, 1)


def ridge(X, y, lam=1e-3):
    """Same closed form as experiments/08_fitted_rule.ridge_fit, returning the parameters."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    A = Z.T @ Z + lam * np.eye(Z.shape[1]); A[-1, -1] -= lam
    w = np.linalg.solve(A, Z.T @ y)
    return mu, sd, w


def predict(X, mu, sd, w):
    return np.exp(np.c_[(X - mu) / sd, np.ones(len(X))] @ w)


def main() -> None:
    env = load_env()
    res = Path(env.cfg["results_dir"]); kr, C = env.k_rerank, env.max_cells
    out = Path(env.cfg["data_dir"]).parent / "cpp"; out.mkdir(parents=True, exist_ok=True)
    curves, _ = load_curves(res / "03_oracle_curves.npz")
    tune, test = subset(curves, env.tune), subset(curves, env.test)
    Dq, Iq = quantizer_topk(env.index, env.xq, C)
    lag = lagrangian_curve(tune)
    halves = [np.arange(len(env.tune)) % 2 == 0, np.arange(len(env.tune)) % 2 == 1]
    Wb = bootstrap_weights(len(env.test), 1000, seed=env.cfg["seed"])
    alloc = lambda pred, g: np.clip(np.rint(g * pred), 1, C).astype(int)
    rows = []
    for name, ranks in VARIANTS.items():
        X = features(Dq, Iq, env.list_sizes, ranks)
        Xtr, Xte = X[env.tune], X[env.test]
        for R in env.cfg["recall_targets"]:
            j = int(np.argmin(np.abs(lag["recall_mean"] - R)))
            y = np.log(lag["cells"][j].astype(float))
            pred_cv = np.empty(len(env.tune))
            for a, b in [(0, 1), (1, 0)]:
                pred_cv[halves[b]] = predict(Xtr[halves[b]], *ridge(Xtr[halves[a]], y[halves[a]]))
            rec = np.array([at_cells(tune, alloc(pred_cv, g))["recall"].mean() for g in GAMMAS])
            ok = np.where(rec >= R - 1e-9)[0]
            g = float(GAMMAS[ok[0]]) if len(ok) else float(GAMMAS[-1])
            mu, sd, w = ridge(Xtr, y)
            n = alloc(predict(Xte, mu, sd, w), g)
            a = at_cells(test, n)
            wk = a["work"].astype(float) + kr
            p = int(np.argmax(tune.recall.mean(0) >= R - 1e-9)) + 1
            wf = test.work[:, p - 1].astype(float) + kr
            bs = (Wb @ wf) / (Wb @ wk)
            rows.append({"variant": name, "target": R, "gamma": g, "recall": a["recall"].mean(), "work": wk.mean(),
                         "cells_mean": n.mean(), "share_over_64": (n > 64).mean(), "saving_tuned": wf.mean() / wk.mean(),
                         "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
            txt = [" ".join(map(str, ranks)), " ".join(repr(float(v)) for v in mu), " ".join(repr(float(v)) for v in sd),
                   " ".join(repr(float(v)) for v in w), repr(g)]
            (out / f"fitted_{name}_R{R}.txt").write_text("\n".join(txt) + "\n")
            n.astype(np.int32).tofile(out / f"cells_fitted_{name}_R{R}.i32")
    (out / "fitted_variants.txt").write_text("".join(f"{v} {k}\n" for v, k in RUNS))
    df = pd.DataFrame(rows)
    df.to_csv(res / "cpp_fitted_python.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
