"""E4 for the round-based stopping rule (ivfadapt.stoprule): tune -> test with bootstrap CIs.

Per recall target R:
- model: logistic P(continue) fitted on the tune split; labels = the Lagrangian oracle on tune
  (at mean recall R) continues past the boundary. Features recorded by stoprule.run_rounds.
- theta: chosen on tune with 2-fold cross-fitted predictions (fit on one half of tune, predict
  the other), as the largest theta whose tune recall reaches R. Then the model is refitted on all
  of tune and run on test with that theta.
- baselines tuned on tune the same way: fixed nprobe; ratio rule with cap (cap in {16,32,48,128},
  smallest eps reaching R per cap, cap with the least tune work).
- reported on test: recall, work, saving vs fixed (bootstrap 95% CI over test queries), the
  curve-based saving at exactly R and the captured share of the Lagrangian oracle (with CIs).
Writes results/11_stop_rule_rounds.csv, results/11_params.json, results/11_stop_model_R*.npz,
results/11_curves_test.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import bootstrap_weights, interp_work_at_recall
from ivfadapt.oracle import at_cells, lagrangian_curve, load_curves, subset
from ivfadapt.rules import quantizer_topk, ratio_n_cells
from ivfadapt.stoprule import SCHEDULE, cells_from_P, fit_logistic, labels_continue, run_rounds, static_features

N_BOOT = 1000
THETAS = np.concatenate([np.geomspace(1e-6, 0.02, 40), np.linspace(0.025, 0.99, 80)])
CAPS = [16, 32, 48, 128, 256]        # 256 = max_cells, i.e. the uncapped rule
EPS_GRID = np.unique(np.concatenate([[0.0], np.geomspace(1e-3, 1.0, 150)]))


def smallest_reaching(mean_recall, R):
    hit = np.where(mean_recall >= R - 1e-9)[0]
    return int(hit[0]) if len(hit) else None


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    env = load_env(cfg_path)
    out = Path(env.cfg["results_dir"]); k, kr, C = env.k, env.k_rerank, env.max_cells
    targets = env.cfg["recall_targets"]
    curves, _ = load_curves(out / "03_oracle_curves.npz")
    tune, test = subset(curves, env.tune), subset(curves, env.test)
    Dq, Iq = quantizer_topk(env.index, env.xq, C)
    static = static_features(Dq, Iq, env.list_sizes)

    # features at every boundary for all queries (forced to the end)
    rec_tune = run_rounds(env.index, env.xq[env.tune], Dq[env.tune], Iq[env.tune], static[env.tune], k, kr)
    rec_test = run_rounds(env.index, env.xq[env.test], Dq[env.test], Iq[env.test], static[env.test], k, kr)
    F_tune, F_test = rec_tune["features"], rec_test["features"]
    nf = F_tune.shape[2]
    halves = [np.arange(len(env.tune)) % 2 == 0, np.arange(len(env.tune)) % 2 == 1]

    lag_tune = lagrangian_curve(tune); lag_test = lagrangian_curve(test)
    nprobes = np.arange(1, C + 1)
    Wb = bootstrap_weights(len(env.test), N_BOOT, seed=env.cfg["seed"])

    def per_query(cv, allocs):
        res = [at_cells(cv, a) for a in allocs]
        return np.stack([r["recall"] for r in res]).astype(np.float64), np.stack([r["work"] for r in res]).astype(np.float64)

    fx_tune = per_query(tune, [np.full(len(env.tune), c) for c in nprobes])
    fx_test = per_query(test, [np.full(len(env.test), c) for c in nprobes])
    rule_tune = {cap: per_query(tune, [np.minimum(ratio_n_cells(Dq[env.tune], e), cap) for e in EPS_GRID]) for cap in CAPS}
    lg_test = per_query(test, lag_test["cells"])

    def ci(x):
        x = np.asarray(x)[np.isfinite(x)]
        return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)

    def work_at_boot(r, w, R, b):      # r, w: (points, n_test)
        return interp_work_at_recall(r @ Wb[b], w @ Wb[b], R)

    rows, params, curve_rows = [], {}, []
    for R in targets:
        j = int(np.argmin(np.abs(lag_tune["recall_mean"] - R)))
        oracle_cells = lag_tune["cells"][j].astype(int)
        Y = labels_continue(oracle_cells)
        # cross-fitted P on tune -> theta
        P_cv = np.full(Y.shape, np.nan, np.float32)
        for a, b in [(0, 1), (1, 0)]:
            m = fit_logistic(F_tune[halves[a]].reshape(-1, nf), Y[halves[a]].reshape(-1))
            P_cv[halves[b]] = m.predict(F_tune[halves[b]].reshape(-1, nf)).reshape(-1, Y.shape[1])
        st_tune = per_query(tune, [cells_from_P(P_cv, t) for t in THETAS])
        ok = np.where(st_tune[0].mean(1) >= R - 1e-9)[0]
        theta = float(THETAS[ok[-1]]) if len(ok) else float(THETAS[0])   # largest theta still reaching R
        model = fit_logistic(F_tune.reshape(-1, nf), Y.reshape(-1))
        model.save(out / f"11_stop_model_R{R}.npz")
        P_test = model.predict(F_test.reshape(-1, nf)).reshape(len(env.test), -1)
        st_test = per_query(test, [cells_from_P(P_test, t) for t in THETAS])
        n_test = cells_from_P(P_test, theta)
        a = at_cells(test, n_test)
        # baselines tuned on tune
        p = smallest_reaching(fx_tune[0].mean(1), R); wf = fx_test[1][p]
        best = None
        for cap in CAPS:
            e = smallest_reaching(rule_tune[cap][0].mean(1), R)
            if e is not None and (best is None or rule_tune[cap][1][e].mean() < best[2]):
                best = (cap, e, rule_tune[cap][1][e].mean())
        if best is None:                     # no (cap, eps) reaches R on tune: report the uncapped rule at its largest eps
            best = (C, len(EPS_GRID) - 1, np.nan)
        cap, e, _ = best
        n_cap = np.minimum(ratio_n_cells(Dq[env.test], EPS_GRID[e]), cap); ac = at_cells(test, n_cap)
        row = {"target": R, "theta": theta, "nprobe_tuned": int(nprobes[p]), "cap_tuned": cap, "eps_tuned": float(EPS_GRID[e]),
               "fixed_recall": fx_test[0][p].mean(), "fixed_work": wf.mean() + kr,
               "cap_recall": ac["recall"].mean(), "cap_work": ac["work"].mean() + kr,
               "stop_recall": a["recall"].mean(), "stop_work": a["work"].mean() + kr,
               "stop_cells_mean": n_test.mean(), "stop_cells_p99": float(np.percentile(n_test, 99)),
               "stop_share_at_256": float((n_test == C).mean()),
               "saving_stop_tuned": (wf.mean() + kr) / (a["work"].mean() + kr),
               "saving_cap_tuned": (wf.mean() + kr) / (ac["work"].mean() + kr)}
        row["saving_stop_tuned_ci"] = ci((Wb @ wf + kr) / (Wb @ a["work"] + kr))
        row["saving_cap_tuned_ci"] = ci((Wb @ wf + kr) / (Wb @ ac["work"] + kr))
        # curve-based at exactly R on test, with CIs
        W = {"fixed": interp_work_at_recall(fx_test[0].mean(1), fx_test[1].mean(1), R),
             "stop": interp_work_at_recall(st_test[0].mean(1), st_test[1].mean(1), R),
             "oracle": interp_work_at_recall(lg_test[0].mean(1), lg_test[1].mean(1), R)}
        row.update({"W_fixed": W["fixed"], "W_stop": W["stop"], "W_oracle": W["oracle"],
                    "saving_stop_curve": (W["fixed"] + kr) / (W["stop"] + kr),
                    "captured_lagrangian": (W["fixed"] - W["stop"]) / (W["fixed"] - W["oracle"])})
        bw = {name: np.array([work_at_boot(r, w, R, b) for b in range(N_BOOT)])
              for name, (r, w) in {"fixed": fx_test, "stop": st_test, "oracle": lg_test}.items()}
        row["saving_stop_curve_ci"] = ci((bw["fixed"] + kr) / (bw["stop"] + kr))
        row["captured_lagrangian_ci"] = ci((bw["fixed"] - bw["stop"]) / (bw["fixed"] - bw["oracle"]))
        rows.append(row)
        params[str(R)] = {"nprobe": int(nprobes[p]), "cap": cap, "eps": float(EPS_GRID[e]), "theta": theta,
                          "model": f"11_stop_model_R{R}.npz"}
        curve_rows += [{"target_fit": R, "theta": t, "recall": st_test[0][i].mean(), "work": st_test[1][i].mean() + kr}
                       for i, t in enumerate(THETAS)]
        print(f"R={R}: theta={theta:.3g}  stop {row['stop_recall']:.4f} / {row['stop_work']:.0f}  fixed {row['fixed_recall']:.4f} / "
              f"{row['fixed_work']:.0f}  saving {row['saving_stop_tuned']:.3f} {row['saving_stop_tuned_ci']}  curve {row['saving_stop_curve']:.3f}"
              f"  captured {row['captured_lagrangian']:.3f} {row['captured_lagrangian_ci']}")

    df = pd.DataFrame(rows)
    for c in [c for c in df.columns if c.endswith("_ci")]:
        df[c + "_lo"] = df[c].map(lambda t: t[0]); df[c + "_hi"] = df[c].map(lambda t: t[1]); df = df.drop(columns=c)
    df.to_csv(out / "11_stop_rule_rounds.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out / "11_curves_test.csv", index=False)
    with open(out / "11_params.json", "w") as f:
        json.dump(params, f, indent=2)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(df[["target", "nprobe_tuned", "cap_tuned", "eps_tuned", "theta", "fixed_recall", "cap_recall", "stop_recall",
              "fixed_work", "cap_work", "stop_work", "saving_cap_tuned", "saving_stop_tuned", "saving_stop_tuned_ci_lo",
              "saving_stop_tuned_ci_hi", "saving_stop_curve", "captured_lagrangian", "captured_lagrangian_ci_lo",
              "captured_lagrangian_ci_hi", "stop_cells_mean", "stop_cells_p99"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
