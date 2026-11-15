"""A stopping rule that looks at the scan itself (Li et al.-style early termination, fitted).

After each cell c the search knows: the exact distance of its current best and 10th candidate,
whether the top-10 just changed, how many of the cell's candidates entered the top-200, and the
static centroid distances. A logistic model P(continue | features at step c) is fitted on tune
(label: the Lagrangian oracle at mean recall R continues past c), and on test the scan stops at
the first c with P < θ. Sweeping θ traces the curve; the rule is a prefix rule so recall is read
off the E2 curves. Compared with the static fitted rule (08) and fixed nprobe on test.

Writes results/10_sequential_features.npz (all queries), results/10_stop_rule.csv,
results/10_stop_rule_savings.csv.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import interp_work_at_recall
from ivfadapt.oracle import at_cells, lagrangian_curve, load_curves, sequential_features, subset
from ivfadapt.rules import quantizer_topk

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
fitted = import_module("08_fitted_rule")


def step_features(seq, d, Xstatic, C):
    """(n, C-1, F): features available after scanning c = 1..C-1 cells (deciding on cell c+1)."""
    r1, r10, ch, ent = seq["r1"], seq["r10"], seq["n_changed"].astype(np.float32), seq["n_entered"].astype(np.float32)
    n = len(r10); cs = np.arange(1, C)                                 # cells scanned so far
    d1 = d[:, :1]
    F = []
    F.append(np.log(r10[:, :-1] / d1))                                 # 10th candidate vs nearest centroid
    F.append(np.log(r10[:, :-1] / d[:, 1:]))                           # 10th candidate vs next centroid
    F.append(np.log(r1[:, :-1] / d1))
    F.append(np.log(d[:, 1:] / d1))                                    # ratio rule's quantity for the next cell
    F.append(np.broadcast_to(np.log(cs)[None, :], (n, C - 1)))
    # steps since last top-10 change (capped at 32), changes in the last 4 steps, entered in last 4
    changed = ch > 0
    since = np.zeros((n, C), np.float32); run = np.zeros(n, np.float32)
    for c in range(C):
        run = np.where(changed[:, c], 0, run + 1); since[:, c] = np.minimum(run, 32)
    F.append(since[:, :-1] / 32)
    cum_ch = np.cumsum(ch, 1); cum_en = np.cumsum(ent, 1)
    last4 = lambda cum: cum[:, :-1] - np.concatenate([np.zeros((n, 4), np.float32), cum[:, :-5]], 1)[:, :C - 1]
    F.append(np.log1p(last4(cum_ch)))
    F.append(np.log1p(last4(cum_en)))
    # improvement of r10 over the last 4 steps
    prev = np.concatenate([np.repeat(r10[:, :1], 4, 1), r10[:, :-4]], 1)
    F.append(np.log(prev[:, :-1] / r10[:, :-1]))
    for j in range(Xstatic.shape[1]):
        F.append(np.broadcast_to(Xstatic[:, j:j + 1], (n, C - 1)))
    return np.stack(F, 2).astype(np.float32)


def logistic_fit(X, y, lam=1e-2, iters=25):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X), np.float32)]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):                                             # Newton / IRLS
        p = 1 / (1 + np.exp(-(Z @ w))); W = p * (1 - p)
        g = Z.T @ (p - y) + lam * np.r_[w[:-1], 0]
        H = (Z * W[:, None]).T @ Z + lam * np.diag(np.r_[np.ones(len(w) - 1), 0])
        w -= np.linalg.solve(H, g)
    return lambda Xn: 1 / (1 + np.exp(-(np.c_[(Xn - mu) / sd, np.ones(len(Xn), np.float32)] @ w)))


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    env = load_env(cfg_path)
    out = Path(env.cfg["results_dir"]); k, kr, C = env.k, env.k_rerank, env.max_cells
    targets = env.cfg["recall_targets"]
    curves, _ = load_curves(out / "03_oracle_curves.npz")
    tune, test = subset(curves, env.tune), subset(curves, env.test)
    Dq, Iq = quantizer_topk(env.index, env.xq, C)
    assert np.array_equal(Iq, curves.cells)
    seq_path = out / "10_sequential_features.npz"
    if seq_path.exists():
        seq = dict(np.load(seq_path))
    else:
        t0 = time.perf_counter()
        seq = sequential_features(env.index, env.xb, env.xq, Iq, Dq, k, kr)
        np.savez_compressed(seq_path, **seq); print(f"sequential features in {time.perf_counter() - t0:.0f}s")
    # r10 is "almost" non-increasing: the running top-200 is cut by PQ distance, so a candidate
    # with a good exact distance can be evicted when later cells bring better PQ distances.
    inc = np.diff(seq["r10"], axis=1) > 1e-4
    print(f"r10 increases in {inc.mean():.4%} of steps ({(inc.any(1)).mean():.2%} of queries), "
          f"max relative increase {np.nanmax(np.diff(seq['r10'], axis=1) / seq['r10'][:, :-1]):.3f}")

    d = np.sqrt(np.maximum(Dq, 1e-12))
    Xs = fitted.features(Dq, env.list_sizes, Iq)
    F = step_features(seq, d, Xs, C)                                   # (n, C-1, nf)
    nf = F.shape[2]
    Ftr, Fte = F[env.tune].reshape(-1, nf), F[env.test]
    cs = np.arange(1, C)

    fixed_r, fixed_w = test.recall.mean(0), test.work.mean(0) + kr
    W_fixed = {R: interp_work_at_recall(fixed_r, fixed_w, R) for R in targets}
    lag_tune = lagrangian_curve(tune)
    rows, sav = [], []
    for R in [0.8, 0.9, 0.95]:
        j = int(np.argmin(np.abs(lag_tune["recall_mean"] - R)))
        c_star = lag_tune["cells"][j].astype(int)                        # tune allocation
        y = (cs[None, :] < c_star[:, None]).astype(np.float32).reshape(-1)   # continue past c?
        t0 = time.perf_counter(); model = logistic_fit(Ftr, y); 
        P = model(Fte.reshape(-1, nf)).reshape(len(env.test), C - 1)
        print(f"fit R={R}: {time.perf_counter() - t0:.0f}s, positives {y.mean():.3f}")
        for theta in np.concatenate([np.geomspace(1e-5, 0.02, 30), np.linspace(0.04, 0.98, 48)]):
            stop = P < theta
            n = np.where(stop.any(1), stop.argmax(1) + 1, C)
            a = at_cells(test, n)
            rows.append({"family": f"stop_lagr{R}", "theta": theta, "recall": a["recall"].mean(),
                         "work": a["work"].mean() + kr, "cells_mean": n.mean(), "cells_p99": np.percentile(n, 99)})
        g = pd.DataFrame([r for r in rows if r["family"] == f"stop_lagr{R}"])
        s = {"family": f"stop_lagr{R}"}
        for Rt in targets:
            w = interp_work_at_recall(g.recall.values, g.work.values, Rt)
            s[f"saving@{Rt}"] = W_fixed[Rt] / w if np.isfinite(w) else np.nan
        sav.append(s)
    # references on the same split
    ref = pd.read_csv(out / "08_fitted_savings.csv")
    for name in ["lagr0.9", "oracle_lagrangian (ceiling)"]:
        r = ref[ref.family == name].iloc[0]
        sav.append({"family": f"ref: {name}", **{f"saving@{R}": r[f"saving@{R}"] for R in targets}})
    pd.DataFrame(rows).to_csv(out / "10_stop_rule.csv", index=False)
    sav = pd.DataFrame(sav); sav.to_csv(out / "10_stop_rule_savings.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nstopping rule fitted on tune, evaluated on test: work saving vs fixed nprobe at matched mean recall")
    print(sav.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
