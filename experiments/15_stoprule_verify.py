"""Live verification and feature ablation of the stopping rule (complements 14_compact_rule.py).

1. Feature-group ablation: refit the logistic P(continue) on subsets of the 26 features with the
   exact protocol of 11 (labels = Lagrangian oracle on tune at mean recall R, theta = largest value
   reaching R on tune with 2-fold cross-fitted predictions, model refitted on all of tune).
2. The 3-feature model (log r10/d_next, log d_next/d1, log cells) rewritten as a closed-form rule:
       continue iff d_next/d1 < K * (r10/d1)^a * cells^b
   i.e. the ratio rule of the proposal with an epsilon that adapts during the scan.
3. Live verification on test: the 3-feature model and the formula are run through
   stoprule.run_rounds (real search_preassigned rounds), re-ranked exactly, and recall is measured
   against the ground truth (not read off the oracle curves). Saving is the tuned one of 11
   (vs fixed nprobe tuned on tune) with bootstrap 95% CIs over test queries.

Needs 03_oracle.py. Writes results/15_ablation.csv and results/15_compact_live.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import bootstrap_weights, interp_work_at_recall, recall_per_query
from ivfadapt.oracle import at_cells, lagrangian_curve, load_curves, subset
from ivfadapt.rules import quantizer_topk
from ivfadapt.search import rerank_exact
from ivfadapt.stoprule import cells_from_P, fit_logistic, labels_continue, run_rounds, static_features

THETAS = np.concatenate([np.geomspace(1e-6, 0.02, 40), np.linspace(0.025, 0.99, 80)])
N_BOOT = 1000
COMPACT = [1, 3, 4]          # log r10/d_next, log d_next/d1, log cells scanned
GROUPS = {
    "all 26": list(range(26)),
    "dynamic 9": list(range(9)),
    "dynamic without patience/changes": [0, 1, 2, 3, 4, 8],
    "compact 3": COMPACT,
    "static 17": list(range(9, 26)),
}


class Subset:
    """A model fitted on some feature columns, applied to the full rows run_rounds produces."""
    def __init__(self, model, cols):
        self.model, self.cols = model, cols

    def predict(self, F):
        return self.model.predict(F[:, self.cols])


class Formula:
    """Closed form of the compact model: continue iff d_next/d1 < K * (r10/d1)^a * cells^b."""
    def __init__(self, K, a, b):
        self.K, self.a, self.b = K, a, b

    def predict(self, F):
        log_r10_d1 = F[:, 1] + F[:, 3]                  # log r10/d_next + log d_next/d1
        go = F[:, 3] < np.log(self.K) + self.a * log_r10_d1 + self.b * F[:, 4]
        return go.astype(np.float32)                    # 1 = continue; any theta in (0, 1) works


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    env = load_env(cfg_path)
    out = Path(env.cfg["results_dir"]); k, kr, C = env.k, env.k_rerank, env.max_cells
    curves, _ = load_curves(out / "03_oracle_curves.npz")
    tune, test = subset(curves, env.tune), subset(curves, env.test)
    Dq, Iq = quantizer_topk(env.index, env.xq, C)
    static = static_features(Dq, Iq, env.list_sizes)
    a_tu = (env.index, env.xq[env.tune], Dq[env.tune], Iq[env.tune], static[env.tune], k, kr)
    a_te = (env.index, env.xq[env.test], Dq[env.test], Iq[env.test], static[env.test], k, kr)
    F_tune, F_test = run_rounds(*a_tu)["features"], run_rounds(*a_te)["features"]
    halves = [np.arange(len(env.tune)) % 2 == 0, np.arange(len(env.tune)) % 2 == 1]
    lag = lagrangian_curve(tune)
    fixed_r, fixed_w = test.recall.mean(0), test.work.mean(0) + kr
    Wb = bootstrap_weights(len(env.test), N_BOOT, seed=env.cfg["seed"])
    gt = env.gt[env.test]

    def fit(cols, Y):
        X = F_tune[:, :, cols]; nf = len(cols)
        P_cv = np.full(Y.shape, np.nan, np.float32)
        for a, b in [(0, 1), (1, 0)]:
            m = fit_logistic(X[halves[a]].reshape(-1, nf), Y[halves[a]].reshape(-1))
            P_cv[halves[b]] = m.predict(X[halves[b]].reshape(-1, nf)).reshape(-1, Y.shape[1])
        rec = np.array([at_cells(tune, cells_from_P(P_cv, t))["recall"].mean() for t in THETAS])
        ok = np.where(rec >= R - 1e-9)[0]
        theta = float(THETAS[ok[-1]]) if len(ok) else float(THETAS[0])
        return fit_logistic(X.reshape(-1, nf), Y.reshape(-1)), theta

    def tuned_saving(w, p):
        wf = test.work[:, p - 1].astype(float) + kr
        bs = (Wb @ wf) / (Wb @ w)
        return wf.mean() / w.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    abl, comp = [], []
    for R in env.cfg["recall_targets"]:
        j = int(np.argmin(np.abs(lag["recall_mean"] - R)))
        Y = labels_continue(lag["cells"][j].astype(int))
        p = int(np.argmax(tune.recall.mean(0) >= R - 1e-9)) + 1          # fixed nprobe tuned on tune
        for name, cols in GROUPS.items():
            m, theta = fit(cols, Y)
            P = m.predict(F_test[:, :, cols].reshape(-1, len(cols))).reshape(len(env.test), -1)
            a = at_cells(test, cells_from_P(P, theta))
            w = a["work"].astype(float) + kr
            s, lo, hi = tuned_saving(w, p)
            r = a["recall"].mean()
            abl.append({"target": R, "features": name, "n_features": len(cols), "theta": theta, "recall": r,
                        "work": w.mean(), "saving_tuned": s, "saving_tuned_ci_lo": lo, "saving_tuned_ci_hi": hi,
                        "saving_same_recall": interp_work_at_recall(fixed_r, fixed_w, r) / w.mean()})
            if name != "compact 3":
                continue
            a1, a2, a3 = m.w[:3] / m.sd
            b0 = m.w[3] - (m.w[:3] * m.mu / m.sd).sum()
            L = np.log(theta / (1 - theta)); sgap = a1 - a2
            formula = Formula(float(np.exp((b0 - L) / sgap)), float(a1 / sgap), float(a3 / sgap))
            for variant, model, th in [("compact 3 (logistic)", Subset(m, cols), theta), ("formula", formula, 0.5)]:
                run = run_rounds(*a_te, model=model, theta=th)
                _, I = rerank_exact(env.xb, env.xq[env.test], run["cand_ids"], k)
                r_live = recall_per_query(I, gt, k)
                a = at_cells(test, run["n_cells"])
                w = a["work"].astype(float) + kr
                s, lo, hi = tuned_saving(w, p)
                comp.append({"target": R, "variant": variant, "K": formula.K, "a_r10": formula.a,
                             "b_cells": formula.b, "theta": theta, "fixed_nprobe": p,
                             "recall_live": r_live.mean(), "recall_curves": a["recall"].mean(),
                             "queries_live_ne_curves": int((np.abs(r_live - a["recall"]) > 1e-6).sum()),
                             "work": w.mean(), "saving_tuned": s, "saving_tuned_ci_lo": lo, "saving_tuned_ci_hi": hi,
                             "cells": run["n_cells"].copy()})
    cells = {f"{c['variant']}@{c['target']}": c.pop("cells") for c in comp}
    same = [np.array_equal(cells[f"compact 3 (logistic)@{R}"], cells[f"formula@{R}"]) for R in env.cfg["recall_targets"]]
    abl, comp = pd.DataFrame(abl), pd.DataFrame(comp)
    abl.to_csv(out / "15_ablation.csv", index=False); comp.to_csv(out / "15_compact_live.csv", index=False)

    pd.set_option("display.width", 250)
    print("feature-group ablation (test; tuned = vs fixed nprobe tuned on tune):")
    print(abl.round(4).to_string(index=False))
    print("\ncompact rule, run live on test:")
    print(comp.round(4).to_string(index=False))
    print("\nformula picks exactly the same cells as the compact logistic model:", same)
    assert all(same)


if __name__ == "__main__":
    main(*sys.argv[1:])
