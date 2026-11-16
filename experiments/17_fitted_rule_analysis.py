"""Which features does the static fitted rule (08 / 16) actually need?

Same questions as 15 for the stopping rule:
1. correlations between the 17 features, and the standardised weights of the R = 0.9 model;
2. refits on feature subsets (by maximum centroid rank, distance ratios only, ...);
3. greedy forward selection on the TUNE split only (cross-fitted, objective: tune work at
   recall 0.9), so the test split stays untouched;
4. every candidate scored like 16: one ridge model per target fitted on tune to the Lagrangian
   oracle's cells, scale γ chosen on tune by 2-fold cross-fitting, tuned saving on test vs the
   fixed nprobe chosen on tune, bootstrap 95% CI;
5. the compact model written as a closed form  n = A · d1^a · size^b · Π (d_j/d1)^c_j.
For speed the cost is not the features but the largest rank they need (the quantizer must
select that many nearest centroids), so subsets are also reported by their maximum rank.
Writes results/17_fitted_corr.csv, results/17_fitted_subsets.csv, results/17_fitted_greedy.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import bootstrap_weights
from ivfadapt.oracle import at_cells, lagrangian_curve, load_curves, subset
from ivfadapt.rules import quantizer_topk

RANKS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
NAMES = [f"d{j}/d1" for j in RANKS] + ["d1", "size8"]
MAXRANK = RANKS + [1, 8]            # nearest centroids each feature needs (size8: the 8 nearest)
GAMMAS = np.geomspace(0.05, 20, 400)


def all_features(Dq, Iq, list_sizes):
    d = np.sqrt(np.maximum(Dq, 1e-12))
    X = [np.log(d[:, j - 1] / d[:, 0]) for j in RANKS]
    X.append(np.log(d[:, 0]))
    X.append(np.log(list_sizes[Iq[:, :8]].mean(1)))
    return np.stack(X, 1)


def ridge(X, y, lam=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    A = Z.T @ Z + lam * np.eye(Z.shape[1]); A[-1, -1] -= lam
    return mu, sd, np.linalg.solve(A, Z.T @ y)


def predict(X, mu, sd, w):
    return np.exp(np.c_[(X - mu) / sd, np.ones(len(X))] @ w)


class Evaluator:
    def __init__(self, env, curves):
        self.env, self.C, self.kr = env, env.max_cells, env.k_rerank
        self.tune, self.test = subset(curves, env.tune), subset(curves, env.test)
        Dq, Iq = quantizer_topk(env.index, env.xq, self.C)
        self.X = all_features(Dq, Iq, env.list_sizes)
        lag = lagrangian_curve(self.tune)
        self.y = {R: np.log(lag["cells"][int(np.argmin(np.abs(lag["recall_mean"] - R)))].astype(float))
                  for R in env.cfg["recall_targets"]}
        self.halves = [np.arange(len(env.tune)) % 2 == 0, np.arange(len(env.tune)) % 2 == 1]
        self.Wb = bootstrap_weights(len(env.test), 1000, seed=env.cfg["seed"])

    def alloc(self, pred, g):
        return np.clip(np.rint(g * pred), 1, self.C).astype(int)

    def _gamma(self, Xtr, y, R):
        pred = np.empty(len(Xtr))
        for a, b in [(0, 1), (1, 0)]:
            pred[self.halves[b]] = predict(Xtr[self.halves[b]], *ridge(Xtr[self.halves[a]], y[self.halves[a]]))
        rec = np.array([at_cells(self.tune, self.alloc(pred, g))["recall"].mean() for g in GAMMAS])
        ok = np.where(rec >= R - 1e-9)[0]
        g = float(GAMMAS[ok[0]]) if len(ok) else float(GAMMAS[-1])
        return g, pred

    def tune_work(self, cols, R):
        """Cross-fitted tune work at recall R (selection objective; test is not used)."""
        Xtr = self.X[self.env.tune][:, cols]
        g, pred = self._gamma(Xtr, self.y[R], R)
        return at_cells(self.tune, self.alloc(pred, g))["work"].mean() + self.kr

    def test_saving(self, cols, R):
        Xtr, Xte = self.X[self.env.tune][:, cols], self.X[self.env.test][:, cols]
        g, _ = self._gamma(Xtr, self.y[R], R)
        mu, sd, w = ridge(Xtr, self.y[R])
        n = self.alloc(predict(Xte, mu, sd, w), g)
        a = at_cells(self.test, n)
        wk = a["work"].astype(float) + self.kr
        p = int(np.argmax(self.tune.recall.mean(0) >= R - 1e-9)) + 1
        wf = self.test.work[:, p - 1].astype(float) + self.kr
        bs = (self.Wb @ wf) / (self.Wb @ wk)
        return {"saving": wf.mean() / wk.mean(), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5),
                "recall": a["recall"].mean(), "cells_mean": n.mean(), "share_over_32": (n > 32).mean(),
                "gamma": g, "model": (mu, sd, w)}


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    env = load_env(cfg_path)
    res = Path(env.cfg["results_dir"])
    curves, _ = load_curves(res / "03_oracle_curves.npz")
    ev = Evaluator(env, curves)
    targets = env.cfg["recall_targets"]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

    # 1. correlations and weights
    Xt = ev.X[env.tune]
    corr = pd.DataFrame(np.corrcoef(Xt.T), index=NAMES, columns=NAMES)
    corr.round(3).to_csv(res / "17_fitted_corr.csv")
    print("correlation of neighbouring distance-ratio features (tune):")
    print("  " + "  ".join(f"{NAMES[i]}~{NAMES[i+1]}: {corr.iloc[i, i+1]:.3f}" for i in range(len(RANKS) - 1)))
    print(f"  corr(d1, d2/d1) {corr.loc['d1', 'd2/d1']:.3f}, corr(d1, d256/d1) {corr.loc['d1', 'd256/d1']:.3f}, "
          f"corr(size8, d1) {corr.loc['size8', 'd1']:.3f}, corr(size8, d8/d1) {corr.loc['size8', 'd8/d1']:.3f}")
    mu, sd, w = ridge(Xt, ev.y[0.9])
    print("\nstandardised weights, R = 0.9 model:")
    print("  " + ", ".join(f"{n} {v:+.2f}" for n, v in sorted(zip(NAMES, w[:-1]), key=lambda t: -abs(t[1]))))
    print("  correlation of each feature with log(oracle cells at R=0.9) on tune:")
    print("  " + ", ".join(f"{n} {np.corrcoef(Xt[:, i], ev.y[0.9])[0, 1]:+.2f}" for i, n in enumerate(NAMES)))

    # 2. hand-picked subsets
    idx = {n: i for i, n in enumerate(NAMES)}
    by_rank = lambda r: [i for i, n in enumerate(NAMES) if MAXRANK[i] <= r]
    subsets = {
        "all 17": list(range(17)),
        "ranks <= 32 (11)": by_rank(32),
        "ranks <= 16 (9)": by_rank(16),
        "ranks <= 8 (7)": by_rank(8),
        "distance ratios only (15)": list(range(15)),
        "d1 + size8 only": [idx["d1"], idx["size8"]],
    }
    # 3. greedy forward selection on tune, objective = cross-fitted tune work at R = 0.9
    chosen, rows_g = [], []
    remaining = list(range(17))
    for step in range(6):
        scores = {i: ev.tune_work(chosen + [i], 0.9) for i in remaining}
        best = min(scores, key=scores.get)
        chosen.append(best); remaining.remove(best)
        rows_g.append({"step": step + 1, "added": NAMES[best], "features": " + ".join(NAMES[i] for i in chosen),
                       "tune_work_R0.9": scores[best], "max_rank": max(MAXRANK[i] for i in chosen)})
        subsets[f"greedy {step + 1}"] = list(chosen)
    g = pd.DataFrame(rows_g); g.to_csv(res / "17_fitted_greedy.csv", index=False)
    print("\ngreedy forward selection (tune only):"); print(g.round(1).to_string(index=False))

    # 4. test scores
    rows = []
    for name, cols in subsets.items():
        for R in targets:
            r = ev.test_saving(cols, R)
            rows.append({"subset": name, "n_features": len(cols), "max_rank": max(MAXRANK[i] for i in cols),
                         "target": R, **{k: v for k, v in r.items() if k != "model"}})
    df = pd.DataFrame(rows); df.to_csv(res / "17_fitted_subsets.csv", index=False)
    wide = df.pivot_table(index=["subset", "n_features", "max_rank"], columns="target", values="saving", sort=False)
    print("\ntuned saving on test (vs fixed nprobe chosen on tune):"); print(wide.round(3).to_string())
    ci = df.pivot_table(index="subset", columns="target", values=["ci_lo", "ci_hi"], sort=False)
    print("\n95% CIs:"); print(ci.round(3).to_string())

    # 5. closed form of a compact model: log n = log γ + b + Σ w_k (x_k - mu_k)/sd_k
    for name in ["greedy 2", "greedy 3"]:
        cols = subsets[name]
        print(f"\nclosed form, {name} ({' + '.join(NAMES[i] for i in cols)}):")
        for R in targets:
            r = ev.test_saving(cols, R); mu, sd, w = r["model"]
            coef = w[:-1] / sd; logA = np.log(r["gamma"]) + w[-1] - (coef * mu).sum()
            terms = " · ".join(f"({NAMES[i]})^{c:+.2f}" for i, c in zip(cols, coef))
            print(f"  R={R}: n = {np.exp(logA):.4g} · {terms}")


if __name__ == "__main__":
    main(*sys.argv[1:])
