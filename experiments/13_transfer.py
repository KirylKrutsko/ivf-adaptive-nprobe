"""Is the stopping rule universal across datasets? (question from Kiryl)

Three checks with the R = 0.9 models of 11_stop_rule_rounds.py:
1. Weights: cosine similarity of the standardised logistic weights fitted on SIFT1M and Deep1M.
2. Transfer without refit: each model, with its own theta, applied to the OTHER dataset's test
   split: curve-based saving vs fixed nprobe (does the ranking transfer?) and recall at the source
   theta (does the threshold transfer?).
3. One joint model fitted on the union of both tune splits, one theta chosen on the pooled tune
   (2-fold cross-fit as in 11), evaluated on each test split.
Writes results/13_transfer.csv and results/13_transfer_weights.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.config import load_config
from ivfadapt.env import load_env
from ivfadapt.metrics import interp_work_at_recall
from ivfadapt.oracle import at_cells, lagrangian_curve, load_curves, subset
from ivfadapt.rules import quantizer_topk
from ivfadapt.stoprule import LogisticModel, cells_from_P, fit_logistic, labels_continue, run_rounds, static_features

THETAS = np.concatenate([np.geomspace(1e-6, 0.02, 40), np.linspace(0.025, 0.99, 80)])
TARGETS = [0.8, 0.9, 0.95, 0.99]
FEATURE_NAMES = (["log r10/d1", "log r10/d_next", "log r1/d1", "log d_next/d1", "log c", "rounds since change",
                  "log1p changes", "log1p entered/width", "log r10_prev/r10"]
                 + [f"log d{j}/d1" for j in [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]]
                 + ["log d1", "log mean list size"])


def prepare(cfg_path):
    env = load_env(cfg_path); res = Path(env.cfg["results_dir"]); kr = env.k_rerank
    curves, _ = load_curves(res / "03_oracle_curves.npz")
    Dq, Iq = quantizer_topk(env.index, env.xq, 256); st = static_features(Dq, Iq, env.list_sizes)
    out = {"env": env, "res": res, "kr": kr, "tune": subset(curves, env.tune), "test": subset(curves, env.test)}
    for split in ["tune", "test"]:
        ids = getattr(env, split)
        out[f"F_{split}"] = run_rounds(env.index, env.xq[ids], Dq[ids], Iq[ids], st[ids], env.k, kr)["features"]
    lag = lagrangian_curve(out["tune"]); j = int(np.argmin(np.abs(lag["recall_mean"] - 0.9)))
    out["Y_tune"] = labels_continue(lag["cells"][j].astype(int))
    out["model"] = LogisticModel.load(res / "11_stop_model_R0.9.npz")
    out["theta"] = json.load(open(res / "11_params.json"))["0.9"]["theta"]
    return out


def evaluate(d, model, theta):
    """Curve-based savings on d's test split + recall/work at the given theta."""
    test, kr = d["test"], d["kr"]; F = d["F_test"]; nf = F.shape[2]
    P = model.predict(F.reshape(-1, nf)).reshape(len(F), -1)
    fixed_r, fixed_w = test.recall.mean(0), test.work.mean(0) + kr
    rr, ww = [], []
    for t in THETAS:
        a = at_cells(test, cells_from_P(P, t)); rr.append(a["recall"].mean()); ww.append(a["work"].mean() + kr)
    row = {f"saving@{R}": interp_work_at_recall(fixed_r, fixed_w, R) / interp_work_at_recall(np.array(rr), np.array(ww), R) for R in TARGETS}
    a = at_cells(test, cells_from_P(P, theta))
    row["recall@theta"] = a["recall"].mean(); row["work@theta"] = a["work"].mean() + kr
    row["fixed work@0.9"] = interp_work_at_recall(fixed_r, fixed_w, 0.9)
    return row


def choose_theta(F_list, Y_list, curves_list):
    """2-fold cross-fitted P on the pooled tune, largest theta whose pooled mean recall >= 0.9."""
    halves = [[np.arange(len(F)) % 2 == h for F in F_list] for h in (0, 1)]
    nf = F_list[0].shape[2]
    P = [np.full(Y.shape, np.nan, np.float32) for Y in Y_list]
    for a, b in [(0, 1), (1, 0)]:
        m = fit_logistic(np.concatenate([F[h].reshape(-1, nf) for F, h in zip(F_list, halves[a])]),
                         np.concatenate([Y[h].reshape(-1) for Y, h in zip(Y_list, halves[a])]))
        for i, (F, h) in enumerate(zip(F_list, halves[b])):
            P[i][h] = m.predict(F[h].reshape(-1, nf)).reshape(-1, Y_list[i].shape[1])
    best = THETAS[0]
    for t in THETAS:
        r = np.concatenate([at_cells(c, cells_from_P(p, t))["recall"] for c, p in zip(curves_list, P)]).mean()
        if r >= 0.9 - 1e-9:
            best = t
    return float(best)


def main(sift_cfg="configs/sift1m.yaml", deep_cfg="configs/deep1m.yaml"):
    data = {"sift1m": prepare(sift_cfg), "deep1m": prepare(deep_cfg)}
    out = Path(load_config(sift_cfg)["results_dir"])

    w = pd.DataFrame({k: d["model"].w[:-1] for k, d in data.items()}, index=FEATURE_NAMES)
    cos = float(w.sift1m @ w.deep1m / np.linalg.norm(w.sift1m) / np.linalg.norm(w.deep1m))
    w.to_csv(out / "13_transfer_weights.csv")

    rows = []
    for src, ds in data.items():
        for tgt, dt in data.items():
            rows.append({"model": f"fitted on {src}", "theta": ds["theta"], "evaluated on": tgt, **evaluate(dt, ds["model"], ds["theta"])})
    nf = data["sift1m"]["F_tune"].shape[2]
    joint = fit_logistic(np.concatenate([d["F_tune"].reshape(-1, nf) for d in data.values()]),
                         np.concatenate([d["Y_tune"].reshape(-1) for d in data.values()]))
    theta_j = choose_theta([d["F_tune"] for d in data.values()], [d["Y_tune"] for d in data.values()], [d["tune"] for d in data.values()])
    joint.save(out / "13_joint_model_R0.9.npz")
    for tgt, dt in data.items():
        rows.append({"model": "fitted on both tune splits", "theta": theta_j, "evaluated on": tgt, **evaluate(dt, joint, theta_j)})
    df = pd.DataFrame(rows); df.to_csv(out / "13_transfer.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"cosine similarity of standardised weights (SIFT1M vs Deep1M models, R = 0.9): {cos:.3f}")
    print("\ntransfer (test splits; savings are curve-based at matched recall; recall@theta uses the model's own theta):")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
