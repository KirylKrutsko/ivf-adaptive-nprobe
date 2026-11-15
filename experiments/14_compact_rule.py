"""Compact stopping rule: only three features (log d_next/d_1, log r10/d_1, log c).

Kiryl's observation: a 3-feature model does about as well as the 26-feature one. Here it is fitted
and evaluated exactly like 11 (labels from the Lagrangian oracle at R on tune, theta by 2-fold
cross-fit on tune, reported on test), on both datasets, and its decision boundary is printed as a
closed-form rule:   continue past c cells  iff  d_next / d_1 < K * (r10 / d_1)^p * c^q.
Also: the compact model moved to the other dataset without refit.
Writes results/14_compact_rule.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.config import load_config
from ivfadapt.env import load_env
from ivfadapt.metrics import interp_work_at_recall
from ivfadapt.oracle import at_cells, lagrangian_curve, load_curves, subset
from ivfadapt.rules import quantizer_topk
from ivfadapt.stoprule import SCHEDULE, cells_from_P, fit_logistic, labels_continue, run_rounds, static_features

THETAS = np.concatenate([np.geomspace(1e-6, 0.02, 40), np.linspace(0.025, 0.99, 80)])
TARGETS = [0.8, 0.9, 0.95, 0.99]
COMPACT = [0, 3, 4]                  # log r10/d1, log d_next/d1, log c   (indices in stoprule features)
ALL = list(range(26))


def prepare(cfg_path):
    env = load_env(cfg_path); res = Path(env.cfg["results_dir"]); kr = env.k_rerank
    curves, _ = load_curves(res / "03_oracle_curves.npz")
    Dq, Iq = quantizer_topk(env.index, env.xq, 256); st = static_features(Dq, Iq, env.list_sizes)
    d = {"name": env.cfg["dataset"], "kr": kr, "tune": subset(curves, env.tune), "test": subset(curves, env.test)}
    for split in ["tune", "test"]:
        ids = getattr(env, split)
        d[f"F_{split}"] = run_rounds(env.index, env.xq[ids], Dq[ids], Iq[ids], st[ids], env.k, kr)["features"]
    lag = lagrangian_curve(d["tune"]); d["lag"] = lag
    return d


def fit_and_theta(d, feats, R):
    F, Y = d["F_tune"][:, :, feats], labels_continue(d["lag"]["cells"][int(np.argmin(np.abs(d["lag"]["recall_mean"] - R)))].astype(int))
    nf = len(feats); halves = [np.arange(len(F)) % 2 == h for h in (0, 1)]
    P = np.full(Y.shape, np.nan, np.float32)
    for a, b in [(0, 1), (1, 0)]:
        m = fit_logistic(F[halves[a]].reshape(-1, nf), Y[halves[a]].reshape(-1))
        P[halves[b]] = m.predict(F[halves[b]].reshape(-1, nf)).reshape(-1, Y.shape[1])
    theta = THETAS[0]
    for t in THETAS:
        if at_cells(d["tune"], cells_from_P(P, t))["recall"].mean() >= R - 1e-9:
            theta = t
    return fit_logistic(F.reshape(-1, nf), Y.reshape(-1)), float(theta)


def evaluate(d, model, feats, theta):
    test, kr = d["test"], d["kr"]; F = d["F_test"][:, :, feats]; nf = len(feats)
    P = model.predict(F.reshape(-1, nf)).reshape(len(F), -1)
    fixed_r, fixed_w = test.recall.mean(0), test.work.mean(0) + kr
    rr, ww = zip(*[(at_cells(test, cells_from_P(P, t))["recall"].mean(), at_cells(test, cells_from_P(P, t))["work"].mean() + kr) for t in THETAS])
    row = {f"saving@{R}": interp_work_at_recall(fixed_r, fixed_w, R) / interp_work_at_recall(np.array(rr), np.array(ww), R) for R in TARGETS}
    a = at_cells(test, cells_from_P(P, theta)); wf = interp_work_at_recall(fixed_r, fixed_w, 0.9)
    row.update({"recall@theta": a["recall"].mean(), "work@theta": a["work"].mean() + kr, "saving_tuned@0.9": (test.work[:, :].mean() * 0 + wf) / (a["work"].mean() + kr)})
    return row


def formula(model, theta):
    """continue iff d_next/d1 < K * (r10/d1)^p * c^q, from the standardised weights of a COMPACT model."""
    w = model.w[:-1] / model.sd; b0 = model.w[-1] - (model.w[:-1] * model.mu / model.sd).sum()
    # w . [log r10/d1, log dn/d1, log c] + b0 > logit(theta)   and  w[1] < 0
    logit = np.log(theta / (1 - theta))
    K = np.exp((logit - b0) / w[1]); p = -w[0] / w[1]; q = -w[2] / w[1]
    return K, p, q


def main(sift_cfg="configs/sift1m.yaml", deep_cfg="configs/deep1m.yaml"):
    data = {"sift1m": prepare(sift_cfg), "deep1m": prepare(deep_cfg)}
    rows, models = [], {}
    for name, d in data.items():
        for label, feats in [("26 features", ALL), ("3 features", COMPACT)]:
            for R in TARGETS:
                m, theta = fit_and_theta(d, feats, R)
                r = evaluate(d, m, feats, theta)
                rows.append({"dataset": name, "model": label, "fit R": R, "theta": theta, "recall@theta": r["recall@theta"],
                             "saving_tuned": r["saving_tuned@0.9"] if R == 0.9 else np.nan, **{k: v for k, v in r.items() if k.startswith("saving@")}})
                if label == "3 features":
                    K, p, q = formula(m, theta)
                    rows[-1]["formula"] = f"d_next/d1 < {K:.2f} (r10/d1)^{p:.2f} c^{q:.3f}"
                    if R == 0.9: models[name] = (m, theta)
    # compact model transferred without refit (R = 0.9)
    for src, (m, theta) in models.items():
        for tgt, d in data.items():
            if tgt == src: continue
            r = evaluate(d, m, COMPACT, theta)
            rows.append({"dataset": tgt, "model": f"3 features fitted on {src}", "fit R": 0.9, "theta": theta,
                         "recall@theta": r["recall@theta"], **{k: v for k, v in r.items() if k.startswith("saving@")}})
    df = pd.DataFrame(rows); out = Path(load_config(sift_cfg)["results_dir"]); df.to_csv(out / "14_compact_rule.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(df.drop(columns=["saving_tuned"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
