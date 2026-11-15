"""The static fitted rule (08) evaluated like every row of the report's summary table.

08 reads the fitted rule's saving off its test curve (γ swept on test, work interpolated at exactly
R). Here it gets the tune -> test protocol of 05 / 11 instead:
- one ridge model per target R, fitted on tune to log of the Lagrangian oracle's cells at mean
  recall R (same 17 centroid-distance features as 08);
- γ (the scale of the allocation n = clip(round(γ·exp(pred)), 1, 256)) chosen on tune with 2-fold
  cross-fitted predictions, as the smallest γ whose tune recall reaches R;
- model refitted on all of tune, run on test with that γ; saving vs the fixed nprobe tuned on tune,
  bootstrap 95% CI over test queries (1000 resamples).
Also runs the tuned rule live on test (quantizer top-256, features, ridge prediction, one
search_preassigned call, exact re-ranking): recall is measured against the ground truth and must
equal the curve-read value; wall clock single thread, median of 5, against plain Faiss at the
tuned nprobe, components as in 12.
Needs 03_oracle.py. Writes results/16_fitted_rule_tuned.csv.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import bootstrap_weights, recall_per_query
from ivfadapt.oracle import at_cells, lagrangian_curve, load_curves, subset
from ivfadapt.rules import quantizer_topk
from ivfadapt.search import preassigned_search, rerank_exact


def timed(f, reps=5):
    f()
    ts, out = [], None
    for _ in range(reps):
        t = time.perf_counter(); out = f(); ts.append(time.perf_counter() - t)
    return out, float(np.median(ts))

GAMMAS = np.geomspace(0.05, 20, 400)
N_BOOT = 1000

_spec = importlib.util.spec_from_file_location("fitted08", Path(__file__).with_name("08_fitted_rule.py"))
fitted08 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fitted08)
features, ridge_fit = fitted08.features, fitted08.ridge_fit


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    faiss.omp_set_num_threads(1)
    env = load_env(cfg_path)
    out = Path(env.cfg["results_dir"]); kr, C = env.k_rerank, env.max_cells
    curves, _ = load_curves(out / "03_oracle_curves.npz")
    tune, test = subset(curves, env.tune), subset(curves, env.test)
    Dq, Iq = quantizer_topk(env.index, env.xq, C)
    X = features(Dq, env.list_sizes, Iq)
    Xtr, Xte = X[env.tune], X[env.test]
    lag = lagrangian_curve(tune)
    halves = [np.arange(len(env.tune)) % 2 == 0, np.arange(len(env.tune)) % 2 == 1]
    Wb = bootstrap_weights(len(env.test), N_BOOT, seed=env.cfg["seed"])

    def alloc(pred, g):
        return np.clip(np.rint(g * pred), 1, C).astype(int)

    rows = []
    for R in env.cfg["recall_targets"]:
        j = int(np.argmin(np.abs(lag["recall_mean"] - R)))
        y = np.log(lag["cells"][j].astype(float))
        pred_cv = np.empty(len(env.tune))
        for a, b in [(0, 1), (1, 0)]:
            pred_cv[halves[b]] = np.exp(ridge_fit(Xtr[halves[a]], y[halves[a]])(Xtr[halves[b]]))
        rec = np.array([at_cells(tune, alloc(pred_cv, g))["recall"].mean() for g in GAMMAS])
        ok = np.where(rec >= R - 1e-9)[0]
        g = float(GAMMAS[ok[0]]) if len(ok) else float(GAMMAS[-1])
        pred_te = np.exp(ridge_fit(Xtr, y)(Xte))
        n = alloc(pred_te, g)
        a = at_cells(test, n)
        w = a["work"].astype(float) + kr
        p = int(np.argmax(tune.recall.mean(0) >= R - 1e-9)) + 1
        wf = test.work[:, p - 1].astype(float) + kr
        bs = (Wb @ wf) / (Wb @ w)
        # --- live run and timing (single thread, set at the start of main)
        ix, xq, xb, k = env.index, env.xq[env.test], env.xb, env.k
        model = ridge_fit(Xtr, y)
        (Dl, Il), t_q = timed(lambda: quantizer_topk(ix, xq, C))
        nl, t_rule = timed(lambda: alloc(np.exp(model(features(Dl, env.list_sizes, Il))), g))
        cells = np.where(np.arange(C)[None, :] < nl[:, None], Il, -1)
        (_, Ipq), t_scan = timed(lambda: preassigned_search(ix, xq, kr, cells, Dl))
        (_, Irr), t_rr = timed(lambda: rerank_exact(xb, xq, Ipq, k))
        r_live = recall_per_query(Irr, env.gt[env.test], k)
        def plain():
            ix.nprobe = p
            return rerank_exact(xb, xq, ix.search(xq, kr)[1], k)
        _, t_plain = timed(plain)
        nq = len(xq)
        rows.append({"target": R, "gamma": g, "fixed_nprobe": p, "recall": a["recall"].mean(),
                     "fixed_recall": test.recall[:, p - 1].mean(), "work": w.mean(), "fixed_work": wf.mean(),
                     "cells_mean": n.mean(), "cells_p99": np.percentile(n, 99),
                     "saving_tuned": wf.mean() / w.mean(),
                     "saving_tuned_ci_lo": np.percentile(bs, 2.5), "saving_tuned_ci_hi": np.percentile(bs, 97.5),
                     "recall_live": r_live.mean(), "cells_live_equal": bool(np.array_equal(nl, n)),
                     "live_ne_curves": int((np.abs(r_live - a["recall"]) > 1e-6).sum()),
                     "quantizer256_ms": 1e3 * t_q / nq, "rule_ms": 1e3 * t_rule / nq, "scan_ms": 1e3 * t_scan / nq,
                     "rerank_ms": 1e3 * t_rr / nq, "total_ms": 1e3 * (t_q + t_rule + t_scan + t_rr) / nq,
                     "plain_ms": 1e3 * t_plain / nq, "speedup_vs_plain": t_plain / (t_q + t_rule + t_scan + t_rr)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "16_fitted_rule_tuned.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
