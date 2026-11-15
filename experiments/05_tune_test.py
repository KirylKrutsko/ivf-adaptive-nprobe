"""E4: tune/test protocol and the headline RQ1 / RQ2 numbers (PLAN §4).

For each recall target R:
- tuned:  on the tune split pick the smallest nprobe (1..max_cells) and the smallest ε (E3 grid)
          whose mean recall reaches R; report both on the test split. saving = work ratio.
- curve:  on the test split, log-interpolate the work at exactly R along the fixed-nprobe,
          rule, Lagrangian-oracle and per-query-target-oracle curves.
          saving = W_fixed / W_rule; captured = (W_fixed - W_rule) / (W_fixed - W_oracle).
Work includes the k_rerank exact distances (D5); ndis-only savings are reported too.
95% CIs: bootstrap over test queries (1000 resamples). Every method is a per-query mean,
so a resample is a weighted mean of fixed per-query arrays; the tune choices stay fixed.

Needs results of 03_oracle.py and 04_rule.py. Writes results/05_tune_test.csv and
results/05_curves_test.csv (all test curves, for plotting).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import bootstrap_weights, interp_work_at_recall
from ivfadapt.oracle import at_cells, lagrangian_curve, load_curves, per_query_target, subset

N_BOOT = 1000


def family(curves, alloc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-query (recall, work) for each row of an allocation matrix (n_points, n_q) of cell counts."""
    res = [at_cells(curves, a) for a in alloc]
    return np.stack([r["recall"] for r in res]).astype(np.float64), np.stack([r["work"] for r in res]).astype(np.float64)


def smallest_reaching(mean_recall: np.ndarray, R: float) -> int | None:
    hit = np.where(mean_recall >= R - 1e-9)[0]
    return int(hit[0]) if len(hit) else None


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    env = load_env(cfg_path)
    out = Path(env.cfg["results_dir"])
    kr = env.k_rerank
    curves, _ = load_curves(out / "03_oracle_curves.npz")
    rc = np.load(out / "04_rule_cells.npz")
    eps_grid, rule_cells = rc["eps"], rc["n_cells"]
    tune, test = subset(curves, env.tune), subset(curves, env.test)
    C = curves.c_max
    nprobes = np.arange(1, C + 1)

    # per-query (recall, work) arrays on each split for every method / parameter value
    fixed_alloc = np.repeat(nprobes[:, None], len(env.xq), axis=1)
    fx_tune = family(tune, fixed_alloc[:, env.tune]); fx_test = family(test, fixed_alloc[:, env.test])
    ru_tune = family(tune, rule_cells[:, env.tune]); ru_test = family(test, rule_cells[:, env.test])
    lag = lagrangian_curve(test)
    lg_test = family(test, lag["cells"])                                          # λ points x test queries
    pq_targets = np.round(np.arange(0.1, 1.0001, 0.1), 1)
    pq_test = family(test, np.stack([per_query_target(test, t)[0] for t in pq_targets]))

    methods = {"fixed": fx_test, "rule": ru_test, "oracle_lagrangian": lg_test, "oracle_perquery": pq_test}
    params = {"fixed": nprobes, "rule": eps_grid, "oracle_lagrangian": lag["lambda"],
              "oracle_perquery": pq_targets}
    curve_rows = [{"method": m, "param": params[m][j], "recall": r[j].mean(), "ndis_mean": w[j].mean(),
                   "work_mean": w[j].mean() + kr}
                  for m, (r, w) in methods.items() for j in range(len(r))]
    pd.DataFrame(curve_rows).to_csv(out / "05_curves_test.csv", index=False)

    Wb = bootstrap_weights(len(env.test), N_BOOT, seed=env.cfg["seed"])          # (N_BOOT, n_test)
    boot = {name: (r @ Wb.T, w @ Wb.T) for name, (r, w) in methods.items()}     # each (n_points, N_BOOT)

    def work_at(name, R, b=None):
        r, w = (methods[name][0].mean(1), methods[name][1].mean(1)) if b is None else (boot[name][0][:, b], boot[name][1][:, b])
        return interp_work_at_recall(r, w, R)

    def ci(x):
        x = np.asarray(x)[np.isfinite(x)]
        return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan, np.nan)

    rows = []
    for R in env.cfg["recall_targets"]:
        row = {"target": R}
        # --- tuned on tune, evaluated on test
        p = smallest_reaching(fx_tune[0].mean(1), R)
        e = smallest_reaching(ru_tune[0].mean(1), R)
        if p is not None and e is not None:
            wf, wr = fx_test[1][p], ru_test[1][e]
            row.update({"nprobe_tuned": int(nprobes[p]), "eps_tuned": float(eps_grid[e]),
                        "fixed_recall_test": fx_test[0][p].mean(), "rule_recall_test": ru_test[0][e].mean(),
                        "fixed_ndis_test": wf.mean(), "rule_ndis_test": wr.mean(),
                        "saving_tuned": (wf.mean() + kr) / (wr.mean() + kr),
                        "saving_tuned_ndis_only": wf.mean() / wr.mean(),
                        "rule_cells_mean": rule_cells[e, env.test].mean(),
                        "rule_cells_p99": np.percentile(rule_cells[e, env.test], 99),
                        "rule_share_at_cap": (rule_cells[e, env.test] >= C).mean()})
            bs = (Wb @ wf + kr) / (Wb @ wr + kr)
            row["saving_tuned_ci"] = ci(bs)
        # --- curve-based, test split
        W = {m: work_at(m, R) for m in methods}
        row.update({f"W_{m}": W[m] for m in methods})
        row["saving_curve"] = (W["fixed"] + kr) / (W["rule"] + kr)
        row["max_saving_lagrangian"] = (W["fixed"] + kr) / (W["oracle_lagrangian"] + kr)
        row["captured_lagrangian"] = (W["fixed"] - W["rule"]) / (W["fixed"] - W["oracle_lagrangian"])
        row["captured_perquery"] = (W["fixed"] - W["rule"]) / (W["fixed"] - W["oracle_perquery"])
        bw = {m: np.array([work_at(m, R, b) for b in range(N_BOOT)]) for m in methods}
        row["saving_curve_ci"] = ci((bw["fixed"] + kr) / (bw["rule"] + kr))
        row["captured_lagrangian_ci"] = ci((bw["fixed"] - bw["rule"]) / (bw["fixed"] - bw["oracle_lagrangian"]))
        row["captured_perquery_ci"] = ci((bw["fixed"] - bw["rule"]) / (bw["fixed"] - bw["oracle_perquery"]))
        rows.append(row)

    df = pd.DataFrame(rows)
    for c in [c for c in df.columns if c.endswith("_ci")]:
        df[c + "_lo"] = df[c].map(lambda t: t[0]); df[c + "_hi"] = df[c].map(lambda t: t[1])
        df = df.drop(columns=c)
    df.to_csv(out / "05_tune_test.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print("tuned on tune, evaluated on test:")
    print(df[["target", "nprobe_tuned", "eps_tuned", "fixed_recall_test", "rule_recall_test", "fixed_ndis_test",
              "rule_ndis_test", "saving_tuned", "saving_tuned_ci_lo", "saving_tuned_ci_hi", "rule_cells_mean",
              "rule_cells_p99", "rule_share_at_cap"]].round(4).to_string(index=False))
    print("\ncurve-based at exactly R (test split):")
    print(df[["target", "W_fixed", "W_rule", "W_oracle_lagrangian", "W_oracle_perquery", "saving_curve",
              "saving_curve_ci_lo", "saving_curve_ci_hi", "max_saving_lagrangian", "captured_lagrangian",
              "captured_lagrangian_ci_lo", "captured_lagrangian_ci_hi", "captured_perquery"]].round(3).to_string(index=False))
    print("wrote", out / "05_tune_test.csv")


if __name__ == "__main__":
    main(*sys.argv[1:])
