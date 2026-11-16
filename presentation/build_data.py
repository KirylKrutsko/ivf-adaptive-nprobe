"""Collect every number and curve the slides show into presentation/data.json.

Run from the repo root after experiments 01-05, 08, 09, 11, 12, 15, 16 (SIFT1M config):
    uv run python presentation/build_data.py
then `python presentation/inject.py` writes the data into presentation/index.html.
Deep1M and nlist = 1024 numbers are not on every machine; they are copied from
report/results.md (§9, §10) and marked as such below.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.oracle import at_cells, load_curves, per_query_target, subset
from ivfadapt.rules import quantizer_topk, ratio_n_cells
from ivfadapt.stoprule import STATIC_RANKS, LogisticModel, run_rounds, static_features

OUT = Path(__file__).with_name("data.json")
R4 = [0.8, 0.9, 0.95, 0.99]


def thin(df, x, y, n=120):
    """Downsample a curve to about n points, keeping it sorted by x."""
    d = df.sort_values(x)
    if len(d) > n:
        d = d.iloc[np.unique(np.linspace(0, len(d) - 1, n).astype(int))]
    return [[round(float(a), 1), round(float(b), 4)] for a, b in zip(d[x], d[y])]


def main() -> None:
    env = load_env()
    res = Path(env.cfg["results_dir"])
    kr = env.k_rerank
    curves, _ = load_curves(res / "03_oracle_curves.npz")
    test = subset(curves, env.test)
    D: dict = {}

    # ---- recall vs work curves (test) -------------------------------------------------
    c5 = pd.read_csv(res / "05_curves_test.csv")
    fixed = pd.DataFrame({"work": test.work.mean(0) + kr, "recall": test.recall.mean(0)})
    D["curve_fixed"] = thin(fixed, "work", "recall", 140)
    D["curve_ratio"] = thin(c5[c5.method == "rule"], "work_mean", "recall")
    D["curve_oracle"] = thin(c5[c5.method == "oracle_lagrangian"], "work_mean", "recall")
    D["curve_pqt"] = thin(c5[c5.method == "oracle_perquery"], "work_mean", "recall", 20)
    rc = np.load(res / "04_rule_cells.npz")
    cap = []
    for j in range(len(rc["eps"])):
        a = at_cells(test, np.minimum(rc["n_cells"][j, env.test], 32))
        cap.append({"work": a["work"].mean() + kr, "recall": a["recall"].mean()})
    D["curve_cap32"] = thin(pd.DataFrame(cap).drop_duplicates(), "work", "recall")
    f8 = pd.read_csv(res / "08_fitted_rule.csv")
    D["curve_fitted"] = thin(f8[f8.family == "lagr0.9"], "work", "recall")
    s11 = pd.read_csv(res / "11_curves_test.csv")
    D["curve_stop"] = thin(s11[s11.target_fit == 0.9], "work", "recall")
    D["fixed_nprobe_points"] = [[int(p), round(float(test.work[:, p - 1].mean() + kr), 1),
                                 round(float(test.recall[:, p - 1].mean()), 4)] for p in [13, 26, 45, 122]]

    # ---- oracle histogram of cells needed (target 0.9), log bins ------------------------
    h = np.load(res / "03_oracle_hist.npz")
    c9 = h["c_star_0.9"]
    edges = [1, 2, 4, 8, 16, 32, 64, 128, 257]          # doubling bins: equal width on a log axis
    cnt, _ = np.histogram(c9[c9 > 0], bins=edges)
    D["hist09"] = {"edges": edges, "share": [round(float(x) / len(c9), 4) for x in cnt],
                   "never": round(float((c9 < 0).mean()), 4), "median": int(np.median(c9[c9 > 0])),
                   "le3": round(float((c9[c9 > 0] <= 3).mean()), 3), "gt20": round(float((c9[c9 > 0] > 20).mean()), 3)}

    # ---- ratio rule: where the work goes (ε tuned for R = 0.9) --------------------------
    Dq, Iq = quantizer_topk(env.index, env.xq[env.test], 256)
    d = np.sqrt(np.maximum(Dq, 0))
    eps = 0.2376
    n = ratio_n_cells(Dq, eps)
    w = at_cells(test, n)["work"].astype(float)
    c9t, _, reach = per_query_target(test, 0.9)
    gt_cells = env.cell_of[env.gt[env.test][:, :10]]
    useful = (test.cells[:, :, None] == gt_cells[:, None, :]).any(2)
    buckets = [(1, 5), (6, 20), (21, 50), (51, 128), (129, 256)]
    rows = []
    for lo, hi in buckets:
        m = (n >= lo) & (n <= hi)
        rows.append({"label": f"{lo}–{hi}", "queries": round(float(m.mean()), 3),
                     "work": round(float(w[m].sum() / w.sum()), 3),
                     "oracle_median": int(np.median(c9t[m & reach])),
                     "useful": round(float(np.mean([useful[i, :c].mean() for i, c in zip(np.where(m)[0], n[m])])), 3)})
    D["ratio_buckets"] = rows
    D["ratio_cells"] = {"mean": round(float(n.mean()), 1), "p99": int(np.percentile(n, 99)), "fixed": 26}

    # ---- two real distance profiles: an easy and a flat query --------------------------
    ratio = d / d[:, :1]
    easy = int(np.where((n == 1) & (c9t <= 2) & reach)[0][0])
    hard_c = np.where((n >= 60) & (n <= 120) & reach & (c9t <= 40))[0]
    hard = int(hard_c[np.argsort(n[hard_c])[len(hard_c) // 2]])
    D["profiles"] = {"easy": [round(float(x), 4) for x in ratio[easy, :40]],
                     "hard": [round(float(x), 4) for x in ratio[hard, :40]],
                     "easy_cells": int(n[easy]), "hard_cells": int(n[hard]),
                     "easy_need": int(c9t[easy]), "hard_need": int(c9t[hard]), "eps": eps}

    # ---- scoreboard (tuned on tune, test) ----------------------------------------------
    t5 = pd.read_csv(res / "05_tune_test.csv")
    t11 = pd.read_csv(res / "11_stop_rule_rounds.csv")
    t16 = pd.read_csv(res / "16_fitted_rule_tuned.csv")
    t15 = pd.read_csv(res / "15_compact_live.csv")
    t15 = t15[t15.variant == "compact 3 (logistic)"]
    head = pd.read_csv(res / "03_oracle_headline.csv")
    ci = lambda df, c: [[round(float(r[c]), 3), round(float(r[c + "_ci_lo"]), 3), round(float(r[c + "_ci_hi"]), 3)]
                        for _, r in df.iterrows()]
    D["score_sift"] = {
        "ratio": ci(t5, "saving_tuned"),
        "cap": [[round(float(x), 3), None, None] for x in t11.saving_cap_tuned],
        "fitted": ci(t16, "saving_tuned"),
        "stop": ci(t11, "saving_stop_tuned"),
        "compact": ci(t15, "saving_tuned"),
        "oracle": [[round(float(x), 3), None, None] for x in head.max_saving_lagrangian],
        "captured_stop": [round(float(x), 3) for x in t11.captured_lagrangian],
    }
    # copied from report/results.md (Matvei's runs; data not on this machine)
    D["score_deep"] = {"ratio": [0.87, 0.77, 0.78, 1.15], "cap": [1.12, 1.10, 1.19, 1.15],
                       "fitted": [1.20, 1.20, 1.32, 1.52],   # 16 on Deep1M (WORKLOG 2026-11-16, Matvei)
                       "stop": [1.24, 1.26, 1.36, 1.65], "compact": [1.24, 1.26, 1.36, 1.55],
                       "oracle": [1.64, 2.02, 2.64, 5.25], "captured_stop": [0.33, 0.42, 0.50, 0.53]}
    D["score_nlist1024"] = {"ratio": [0.92, 0.82, 0.82, 0.84], "cap": [1.05, 1.10, 1.12, 0.96],
                            "stop": [1.25, 1.22, 1.31, 1.40], "oracle": [1.61, 1.86, 2.23, 3.79]}
    # tune-split exploration of 07 (WORKLOG 2026-11-15 night), work saving at matched recall
    D["score_geometry"] = {"gap": [0.88, 0.90, 0.94, 1.08], "bound_thr": [0.86, 0.85, 0.88, 1.01],
                           "bound_order": [0.96, 0.94, 0.92, 0.83], "twostage": [1.00, 0.94, 0.96, 1.11],
                           "bound_stop": [0.95, 1.00, 1.05, 1.19], "patience": [1.05, 1.10, 1.14, None]}

    # ---- min(rule, nprobe) (06), best ε per nprobe -------------------------------------
    s6 = pd.read_csv(res / "06_rule_within_nprobe.csv")
    best = s6.loc[s6.groupby("nprobe")["frontier_gain"].idxmax()]
    D["dropin"] = [{"nprobe": int(r.nprobe), "eps": round(float(r.eps), 3), "drop": round(float(r.recall_drop), 4),
                    "saving": round(float(r.work_saving), 3), "gain": round(float(r.frontier_gain), 3),
                    "trimmed": round(float(r.share_reduced), 3)} for _, r in best.iterrows()]
    D["dropin_curve26"] = [[round(float(r.work), 1), round(float(r.recall), 4)]
                           for _, r in s6[s6.nprobe == 26].sort_values("work").iterrows()][::3]

    # ---- stopping rule: two real traces with the compact formula (R = 0.9) --------------
    comp = pd.read_csv(res / "15_compact_live.csv")
    cr = comp[(comp.variant == "formula") & (comp.target == 0.9)].iloc[0]
    K, pa, qb = float(cr.K), float(cr.a_r10), float(cr.b_cells)
    D["formula"] = {str(R): {"K": round(float(r.K), 3), "p": round(float(r.a_r10), 3), "q": round(float(r.b_cells), 3)}
                    for R, r in comp[comp.variant == "formula"].set_index("target").iterrows()}
    st = static_features(Dq, Iq, env.list_sizes)
    rr = run_rounds(env.index, env.xq[env.test], Dq, Iq, st, env.k, kr)
    F = rr["features"]                                              # (n, 26 decisions, 26)
    from ivfadapt.stoprule import SCHEDULE
    sched = np.array(SCHEDULE[:-1])
    log_r10_d1, log_dn_d1 = F[:, :, 0], F[:, :, 3]
    thr = np.log(K) + pa * log_r10_d1 + qb * np.log(sched)[None, :]
    go = log_dn_d1 < thr
    stop_at = np.where((~go).any(1), sched[np.argmax(~go, 1)], 256)

    def trace(i):
        rows = []
        for j, c in enumerate(sched):
            rows.append([int(c), round(float(np.exp(log_dn_d1[i, j])), 4), round(float(np.exp(thr[i, j])), 4),
                         round(float(np.exp(log_r10_d1[i, j])), 4), round(float(test.recall[i, c - 1]), 2)])
            if c >= stop_at[i] and c >= 24:
                break
        return {"rows": rows, "stop": int(stop_at[i]), "need": int(c9t[i]) if reach[i] else None}
    ea = np.where((stop_at <= 4) & (c9t <= 3) & reach & (test.recall[np.arange(len(stop_at)), np.minimum(stop_at, 256) - 1] >= 0.9))[0]
    ha = np.where((stop_at >= 24) & (stop_at <= 48) & reach & (c9t >= 16) & (c9t <= 40))[0]
    D["traces"] = {"easy": trace(int(ea[0])), "hard": trace(int(ha[len(ha) // 2]))}

    # ---- model weights (R = 0.9) and ablation ------------------------------------------
    names = ["r10 / d₁", "r10 / d_next", "r1 / d₁", "d_next / d₁", "cells scanned", "rounds since change",
             "top-10 changes", "entered top-200", "r10 improvement"] + [f"d{j} / d₁" for j in STATIC_RANKS] + ["d₁", "list size"]
    m = LogisticModel.load(res / "11_stop_model_R0.9.npz")
    D["weights"] = [{"name": nm, "w": round(float(v), 3), "dyn": i < 9} for i, (nm, v) in enumerate(zip(names, m.w[:-1]))]
    ab = pd.read_csv(res / "15_ablation.csv")
    D["ablation"] = {g: [[round(float(r.saving_tuned), 3), round(float(r.saving_tuned_ci_lo), 3), round(float(r.saving_tuned_ci_hi), 3)]
                         for _, r in ab[ab.features == g].iterrows()] for g in ab.features.unique()}

    # ---- tail (09): share of queries with recall <= 0.5 at equal mean work --------------
    t9 = pd.read_csv(res / "09_tail_recall.csv")
    D["tail"] = {str(int(b)): {r.method: round(float(r["share_recall_le_0.5"]), 4) for _, r in t9[t9.nprobe_level == b].iterrows()}
                 for b in [13, 26]}

    # ---- timing (12 rerun, 16) at R = 0.9 ----------------------------------------------
    t12 = pd.read_csv(res / "12_timing.csv")
    tt = {r.method: round(float(r.total_ms), 4) for _, r in t12[t12.target == 0.9].iterrows()}
    tt["fitted"] = round(float(t16[t16.target == 0.9].total_ms.iloc[0]), 4)
    D["timing09"] = tt
    D["speedup"] = {"cap": [round(float(x), 3) for x in t12[t12.method.str.startswith("ratio rule")].speedup_vs_plain],
                    "stop": [round(float(x), 3) for x in t12[t12.method == "stopping rule (rounds)"].speedup_vs_plain],
                    "fitted": [round(float(x), 3) for x in t16.speedup_vs_plain]}

    # ---- C++ wall clock (cpp/run.sh, cpp/run.sh fitted): speed vs plain Faiss, single thread -----
    cs = pd.read_csv(res / "cpp_stoprule_bench.csv")
    cf = pd.read_csv(res / "cpp_fitted_bench.csv")
    pick = lambda df, m: [round(float(x), 3) for x in df[df.method == m].sort_values("target").speedup_vs_plain]
    D["cpp"] = {"stop": pick(cs, "stop-lazy64"), "fitted": pick(cf, "fitted-c32-k64")}

    # ---- fitted static rule: weights, 3-feature version, formula (experiments/17) ---------------
    import importlib.util
    spec = importlib.util.spec_from_file_location("a17", Path(__file__).parents[1] / "experiments" / "17_fitted_rule_analysis.py")
    a17 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a17)
    ev = a17.Evaluator(env, curves)
    Xt = ev.X[env.tune]
    mu17, sd17, w17 = a17.ridge(Xt, ev.y[0.9])
    pretty = lambda n: "d₁" if n == "d1" else ("cell size" if n == "size8" else n.split("/")[0] + " / d₁")
    D["fitted_weights"] = [{"name": pretty(n), "w": round(float(v), 3),
                            "key": n in ("d32/d1", "d1", "size8")} for n, v in zip(a17.NAMES, w17[:-1])]
    D["fitted_corr_neighbours"] = round(float(np.min([np.corrcoef(Xt[:, i], Xt[:, i + 1])[0, 1] for i in range(3, 14)])), 3)
    idx = {n: i for i, n in enumerate(a17.NAMES)}
    cols3 = [idx["d32/d1"], idx["d1"], idx["size8"]]
    full, three, form = [], [], {}
    for R in R4:
        f = ev.test_saving(list(range(17)), R); t = ev.test_saving(cols3, R)
        full.append([round(float(f["saving"]), 3), round(float(f["ci_lo"]), 3), round(float(f["ci_hi"]), 3)])
        three.append([round(float(t["saving"]), 3), round(float(t["ci_lo"]), 3), round(float(t["ci_hi"]), 3)])
        mu, sd, w = t["model"]; coef = w[:-1] / sd
        logA = np.log(t["gamma"]) + w[-1] - (coef * mu).sum()
        form[str(R)] = {"A": float(np.exp(logA)), "a": round(float(coef[0]), 2), "b": round(float(coef[1]), 2), "c": round(float(coef[2]), 2)}
    D["fitted17"], D["fitted3"], D["fitted_formula"] = full, three, form

    # ---- Python timing breakdown at R = 0.9 (12_timing, 16), ms per query ----------------------
    r9 = t12[t12.target == 0.9].set_index("method")
    f9 = t16[t16.target == 0.9].iloc[0]
    g = lambda m, c: round(float(r9.loc[m, c]), 4) if pd.notna(r9.loc[m, c]) else 0.0
    D["py_breakdown"] = [
        {"label": "plain Faiss", "quant": g("fixed, preassigned path", "quantizer_ms"), "rule": 0.0,
         "scan": g("fixed, preassigned path", "scan_ms"), "rerank": g("fixed, preassigned path", "rerank_ms")},
        {"label": "fitted rule", "quant": round(float(f9.quantizer256_ms), 4), "rule": round(float(f9.rule_ms), 4),
         "scan": round(float(f9.scan_ms), 4), "rerank": round(float(f9.rerank_ms), 4)},
        {"label": "stopping rule", "quant": g("stopping rule (rounds)", "quantizer_ms"), "rule": g("stopping rule (rounds)", "rule_ms"),
         "scan": g("stopping rule (rounds)", "scan_ms"), "rerank": g("stopping rule (rounds)", "rerank_ms")},
        {"label": "26 cells in 26 rounds", "quant": g("fixed, preassigned in rounds (overhead check)", "quantizer_ms"), "rule": 0.0,
         "scan": g("fixed, preassigned in rounds (overhead check)", "scan+merge_ms"), "rerank": g("fixed, preassigned in rounds (overhead check)", "rerank_ms")}]

    OUT.write_text(json.dumps(D, ensure_ascii=False))
    print("wrote", OUT, f"({OUT.stat().st_size / 1024:.0f} KB)")
    print(json.dumps({k: D[k] for k in ["score_sift", "tail", "timing09", "speedup", "traces", "hist09"]}, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
