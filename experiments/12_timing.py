"""E5: wall-clock timing at matched recall (parameters tuned on tune by 11_stop_rule_rounds.py).

Single thread, all 5000 test queries in one batch, warm-up run then median of 5. For each recall
target: plain Faiss (index.search + re-rank), fixed nprobe through the preassigned path, the ratio
rule with cap, and the round-based stopping rule. Components: coarse quantizer, rule (masking, or
merge + features + model), scan (list scanning in Faiss), exact re-ranking (NumPy, identical for
all methods). Also batch-size-1 latency (p50 / p99) for plain fixed nprobe and the stopping rule.
Writes results/12_timing.csv and results/12_latency.csv.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import recall_at_k
from ivfadapt.rules import fixed_cells, quantizer_topk, ratio_cells
from ivfadapt.search import preassigned_search, rerank_exact, scanned_vectors
from ivfadapt.stoprule import SCHEDULE, LogisticModel, run_rounds, static_features

REPS = 5


def timed(f):
    f()                                                   # warm-up
    outs, ts = [], []
    for _ in range(REPS):
        t = time.perf_counter(); outs.append(f()); ts.append(time.perf_counter() - t)
    return outs[int(np.argsort(ts)[len(ts) // 2])], float(np.median(ts))


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    faiss.omp_set_num_threads(1)
    env = load_env(cfg_path)
    out = Path(env.cfg["results_dir"]); k, kr, C = env.k, env.k_rerank, env.max_cells
    params = json.load(open(out / "11_params.json"))
    ix, xb = env.index, env.xb
    xq, gt = env.xq[env.test], env.gt[env.test]; n = len(xq)
    rows = []

    def add(R, method, parts, I_rr, cells_mean, work):
        total = sum(parts.values())
        rows.append({"target": R, "method": method, **{f"{p}_ms": 1e3 * v / n for p, v in parts.items()},
                     "total_ms": 1e3 * total / n, "qps": n / total, "recall": recall_at_k(I_rr, gt, k),
                     "cells_mean": cells_mean, "work": work})

    for R, prm in params.items():
        R = float(R); p, cap, eps, theta = prm["nprobe"], prm["cap"], prm["eps"], prm["theta"]
        model = LogisticModel.load(out / prm["model"])

        # 1. plain Faiss: quantizer is inside index.search
        def plain():
            ix.nprobe = p; return ix.search(xq, kr)
        (D, I), t_search = timed(plain)
        (_, I_rr), t_rr = timed(lambda: rerank_exact(xb, xq, I, k))
        _, Iq_p = quantizer_topk(ix, xq, p)
        add(R, "fixed, plain index.search", {"quantizer+scan": t_search, "rerank": t_rr}, I_rr, p, scanned_vectors(Iq_p, env.list_sizes).mean() + kr)

        # 2. fixed nprobe through the preassigned path
        (Dq, Iq), t_q = timed(lambda: quantizer_topk(ix, xq, p))
        (_, I), t_s = timed(lambda: preassigned_search(ix, xq, kr, fixed_cells(Iq, p), Dq))
        (_, I_rr), t_rr = timed(lambda: rerank_exact(xb, xq, I, k))
        add(R, "fixed, preassigned path", {"quantizer": t_q, "scan": t_s, "rerank": t_rr}, I_rr, p, scanned_vectors(Iq, env.list_sizes).mean() + kr)

        # 2b. overhead check: the same p cells scanned in rounds of the stopping rule's schedule
        def in_rounds():
            I_all = np.full((n, kr), -1, np.int64); D_all = np.full((n, kr), np.inf, np.float32)
            c_prev = 0
            for c in [c for c in SCHEDULE if c <= p] + ([p] if p not in SCHEDULE else []):
                D_c, I_c = preassigned_search(ix, xq, kr, Iq[:, c_prev:c], Dq[:, c_prev:c])
                pq = np.concatenate([D_all, D_c], 1); ids = np.concatenate([I_all, I_c], 1)
                part = np.argpartition(pq, kr - 1, axis=1)[:, :kr]
                D_all, I_all = np.take_along_axis(pq, part, 1), np.take_along_axis(ids, part, 1)
                c_prev = c
            return D_all, I_all
        (_, I), t_sr = timed(in_rounds)
        (_, I_rr), t_rr = timed(lambda: rerank_exact(xb, xq, I, k))
        add(R, "fixed, preassigned in rounds (overhead check)", {"quantizer": t_q, "scan+merge": t_sr, "rerank": t_rr}, I_rr, p, scanned_vectors(Iq, env.list_sizes).mean() + kr)

        # 3. ratio rule with cap: needs the cap nearest centroids
        (Dq, Iq), t_q = timed(lambda: quantizer_topk(ix, xq, cap))
        cells, t_rule = timed(lambda: ratio_cells(Dq, Iq, eps))
        (_, I), t_s = timed(lambda: preassigned_search(ix, xq, kr, cells, Dq))
        (_, I_rr), t_rr = timed(lambda: rerank_exact(xb, xq, I, k))
        add(R, f"ratio rule, cap {cap}", {"quantizer": t_q, "rule": t_rule, "scan": t_s, "rerank": t_rr}, I_rr,
            (cells >= 0).sum(1).mean(), scanned_vectors(cells, env.list_sizes).mean() + kr)

        # 4. round-based stopping rule: 256 nearest centroids, static features, rounds, re-rank
        (Dq, Iq), t_q = timed(lambda: quantizer_topk(ix, xq, C))
        static, t_static = timed(lambda: static_features(Dq, Iq, env.list_sizes))
        res, _ = timed(lambda: run_rounds(ix, xq, Dq, Iq, static, k, kr, model=model, theta=theta))
        (_, I_rr), t_rr = timed(lambda: rerank_exact(xb, xq, res["cand_ids"], k))
        ncell = res["n_cells"]
        work = np.array([env.list_sizes[Iq[i, :ncell[i]]].sum() for i in range(n)]).mean() + kr
        add(R, "stopping rule (rounds)", {"quantizer": t_q, "rule": t_static + res["t_rule"], "scan": res["t_scan"], "rerank": t_rr},
            I_rr, ncell.mean(), work)

    df = pd.DataFrame(rows)
    plain_total = df[df.method.str.startswith("fixed, plain")].set_index("target")["total_ms"]
    df["speedup_vs_plain"] = df.apply(lambda r: plain_total[r.target] / r.total_ms, axis=1)
    df.to_csv(out / "12_timing.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["target", "method", "recall", "cells_mean", "work", "quantizer+scan_ms", "quantizer_ms", "rule_ms", "scan_ms", "scan+merge_ms", "rerank_ms", "total_ms", "qps", "speedup_vs_plain"]
    print(df[cols].round(4).to_string(index=False))

    # batch-size-1 latency for R = 0.9: plain fixed vs stopping rule
    prm = params["0.9"]; p, theta = prm["nprobe"], prm["theta"]; model = LogisticModel.load(out / prm["model"])
    sub = np.random.default_rng(0).choice(n, 1000, replace=False)
    lat = {"fixed, plain": [], "stopping rule (rounds)": []}
    ix.nprobe = p
    for i in sub:
        q = xq[i:i + 1]
        t = time.perf_counter(); _, I = ix.search(q, kr); rerank_exact(xb, q, I, k); lat["fixed, plain"].append(time.perf_counter() - t)
        t = time.perf_counter(); Dq, Iq = quantizer_topk(ix, q, C); st = static_features(Dq, Iq, env.list_sizes)
        r = run_rounds(ix, q, Dq, Iq, st, k, kr, model=model, theta=theta); rerank_exact(xb, q, r["cand_ids"], k)
        lat["stopping rule (rounds)"].append(time.perf_counter() - t)
    lrows = [{"method": m, "p50_ms": 1e3 * np.percentile(v, 50), "p90_ms": 1e3 * np.percentile(v, 90),
              "p99_ms": 1e3 * np.percentile(v, 99), "mean_ms": 1e3 * np.mean(v)} for m, v in lat.items()]
    ldf = pd.DataFrame(lrows); ldf.to_csv(out / "12_latency.csv", index=False)
    print("\nbatch-size-1 latency at R = 0.9 (1000 test queries, Python overhead included):")
    print(ldf.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
