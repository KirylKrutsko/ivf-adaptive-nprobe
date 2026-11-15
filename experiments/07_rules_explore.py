"""Exploration of alternative cell-selection rules (tune split only; test is kept for the end).

Every rule produces, per query, a set of cells among the max_cells nearest. Prefix rules (the
kept cells are the c nearest) are read off the E2 per-query curves; non-prefix rules are run
through the real search (search_preassigned + exact re-ranking). For each rule family the
parameter is swept, the (mean recall, mean work) curve is traced and the work at each recall
target is log-interpolated and compared with fixed nprobe on the same split.

Families (see WORKLOG for the reasoning):
  fixed            c nearest cells (reference)
  ratio            d_j <= (1+eps) d_1                                   [prefix]
  ratio_cap        min(ratio(eps), C), envelope over C in {16,32,48,128} [prefix]
  gap              d_j - d_1 <= delta (plain distances)                 [prefix]
  patience         stop after m cells in a row without a change of the exact top-10 [prefix]
  bound_stop       scan in centroid order, stop when the next cell's bound > alpha * r_10 (r_10
                   updated after every cell), from c0 = 1                [prefix]
  b1_thr, bmax_thr cells whose bisector bound <= tau (absolute)          [real search]
  bmax_rel         cells whose bisector bound <= tau * d_1               [real search]
  bmax_order       c cells with the smallest bisector bound              [real search]
  twostage         scan c0 nearest cells, r = exact 10th distance, add every cell with
                   bmax <= alpha * r; envelope over c0 in {1,2,4,8}      [real search]

Writes results/07_rules_explore.csv (one row per family and parameter), results/07_rules_savings.csv
and results/07_tune_sequential.npz (per-step exact 10th distance and top-10 change flags).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import interp_work_at_recall, recall_per_query
from ivfadapt.oracle import at_cells, load_curves, subset
from ivfadapt.rules import bisector_bounds, centroids, compact_cells, quantizer_topk, ratio_n_cells
from ivfadapt.search import preassigned_search, rerank_exact, scanned_vectors


def sequential_pass(env, xq, gt, Iq, Dq, k, kr, chunk=500):
    """Scan each query's cells one at a time (centroid order). Record after every cell the exact
    distance of the current 10th candidate and whether the exact top-10 changed."""
    n, C = Iq.shape
    r10 = np.full((n, C), np.inf, dtype=np.float32)
    changed = np.zeros((n, C), dtype=bool)
    for s in range(0, n, chunk):
        e = min(n, s + chunk); m = e - s
        xs = xq[s:e]
        cum_pq = np.full((m, kr), np.inf, np.float32); cum_ex = np.full((m, kr), np.inf, np.float32)
        cum_id = np.full((m, kr), -1, np.int64); prev = np.full((m, k), -2, np.int64)
        for c in range(C):
            D_c, I_c = preassigned_search(env.index, xs, kr, Iq[s:e, c:c + 1], Dq[s:e, c:c + 1])
            vec = env.xb[np.maximum(I_c, 0)]
            E_c = np.where(I_c >= 0, ((vec - xs[:, None, :]) ** 2).sum(2), np.inf).astype(np.float32)
            pq = np.concatenate([cum_pq, D_c], 1)
            order = np.argsort(pq, 1, kind="stable")[:, :kr]
            cum_pq = np.take_along_axis(pq, order, 1)
            cum_ex = np.take_along_axis(np.concatenate([cum_ex, E_c], 1), order, 1)
            cum_id = np.take_along_axis(np.concatenate([cum_id, I_c], 1), order, 1)
            top = np.argsort(cum_ex, 1, kind="stable")[:, :k]
            ids = np.sort(np.take_along_axis(cum_id, top, 1), 1)
            r10[s:e, c] = np.take_along_axis(cum_ex, top, 1)[:, -1]
            changed[s:e, c] = (ids != prev).any(1)
            prev = ids
        print(f"sequential pass: {e}/{n}")
    return np.sqrt(r10), changed


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    env = load_env(cfg_path)
    out = Path(env.cfg["results_dir"])
    k, kr, C = env.k, env.k_rerank, env.max_cells
    targets = env.cfg["recall_targets"]
    rows_ids = env.tune
    xq, gt = env.xq[rows_ids], env.gt[rows_ids]
    curves, _ = load_curves(out / "03_oracle_curves.npz")
    tune = subset(curves, rows_ids)
    Dq, Iq = quantizer_topk(env.index, xq, C)
    assert np.array_equal(Iq, tune.cells)
    d = np.sqrt(np.maximum(Dq, 0))
    print("bisector bounds ..."); t0 = time.perf_counter()
    b1, bmax = bisector_bounds(centroids(env.index), Dq, Iq)
    print(f"  {time.perf_counter() - t0:.0f}s; median bmax of 2nd cell {np.median(bmax[:, 1]):.1f}, "
          f"of 32nd {np.median(bmax[:, 31]):.1f}; median d_1 {np.median(d[:, 0]):.1f}")

    seq_path = out / "07_tune_sequential.npz"
    if seq_path.exists():
        z = np.load(seq_path); r10, changed = z["r10"], z["changed"]
    else:
        r10, changed = sequential_pass(env, xq, gt, Iq, Dq, k, kr)
        np.savez_compressed(seq_path, r10=r10, changed=changed, query_ids=rows_ids)
    print(f"median exact 10th-neighbour distance after 1 cell {np.median(r10[:, 0]):.1f}, after 256 {np.median(r10[:, -1]):.1f}")

    results = []

    def add_prefix(family, param, n):
        a = at_cells(tune, n)
        results.append({"family": family, "param": float(param), "recall": a["recall"].mean(),
                        "ndis": a["work"].mean(), "work": a["work"].mean() + kr,
                        "cells_mean": n.mean(), "cells_p99": np.percentile(n, 99)})

    def add_real(family, param, keep):
        keep = keep.copy(); keep[:, 0] = True
        cells, Dk = compact_cells(Iq, keep, Dq)
        _, I_pq = preassigned_search(env.index, xq, kr, cells, Dk)
        _, I_rr = rerank_exact(env.xb, xq, I_pq, k)
        r = recall_per_query(I_rr, gt, k); w = scanned_vectors(cells, env.list_sizes); n = keep.sum(1)
        results.append({"family": family, "param": float(param), "recall": r.mean(), "ndis": w.mean(),
                        "work": w.mean() + kr, "cells_mean": n.mean(), "cells_p99": np.percentile(n, 99)})

    # --- prefix families
    for c in range(1, C + 1):
        add_prefix("fixed", c, np.full(len(xq), c))
    eps_grid = np.unique(np.concatenate([[0.0], np.geomspace(1e-3, 1.0, 100)]))
    for e in eps_grid:
        add_prefix("ratio", e, ratio_n_cells(Dq, e))
    for cap in [16, 32, 48, 128]:
        for e in eps_grid:
            add_prefix(f"ratio_cap{cap}", e, np.minimum(ratio_n_cells(Dq, e), cap))
    gap = d - d[:, :1]
    for delta in np.geomspace(0.1, 150, 60):
        add_prefix("gap", delta, np.maximum((gap <= delta).sum(1), 1))
    # patience: stop after m consecutive cells with no change (always scan at least 1)
    for m in range(1, 41):
        n = np.full(len(xq), C)
        run = np.zeros(len(xq), int); done = np.zeros(len(xq), bool)
        for c in range(C):
            run = np.where(changed[:, c], 0, run + 1)
            hit = (run >= m) & ~done
            n[hit] = c + 1; done |= hit
        add_prefix("patience", m, n)
    # bound_stop: in centroid order, stop before the first cell c+1 whose bound exceeds alpha * r10(c)
    for alpha in np.geomspace(0.05, 3.0, 40):
        nxt = bmax[:, 1:]                                   # bound of cell c+1, c = 1..C-1
        stop = nxt > alpha * r10[:, :-1]
        n = np.where(stop.any(1), stop.argmax(1) + 1, C)
        add_prefix("bound_stop", alpha, n)

    # --- real-search families
    tau_abs = np.quantile(bmax[:, 1:], np.linspace(0.02, 0.9, 24))
    for tau in tau_abs:
        add_real("b1_thr", tau, b1 <= tau)
        add_real("bmax_thr", tau, bmax <= tau)
    rel = bmax / d[:, :1]
    for tau in np.quantile(rel[:, 1:], np.linspace(0.02, 0.9, 24)):
        add_real("bmax_rel", tau, rel <= tau)
    order = np.argsort(bmax, 1, kind="stable"); rank = np.empty_like(order); np.put_along_axis(rank, order, np.arange(C)[None, :], 1)
    for c in [1, 2, 3, 4, 6, 8, 11, 14, 18, 23, 28, 33, 40, 48, 58, 69, 83, 100, 120, 144, 173, 208, 256]:
        add_real("bmax_order", c, rank < c)
    for c0 in [1, 2, 4, 8]:
        r = r10[:, c0 - 1]
        for alpha in np.geomspace(0.02, 2.0, 28):
            keep = (bmax <= alpha * r[:, None]); keep[:, :c0] = True
            add_real(f"twostage_c{c0}", alpha, keep)
    print(f"{len(results)} rule evaluations done")

    df = pd.DataFrame(results)
    df.to_csv(out / "07_rules_explore.csv", index=False)

    # --- savings at the targets, vs fixed on the same split; envelopes for 2-parameter families
    fixed = df[df.family == "fixed"]
    W_fixed = {R: interp_work_at_recall(fixed.recall.values, fixed.work.values, R) for R in targets}
    fams = [f for f in df.family.unique() if f != "fixed"]
    sav_rows = []
    for f in fams:
        g = df[df.family == f]
        row = {"family": f}
        for R in targets:
            w = interp_work_at_recall(g.recall.values, g.work.values, R)
            row[f"saving@{R}"] = W_fixed[R] / w if np.isfinite(w) else np.nan
        sav_rows.append(row)
    for base in ["ratio_cap", "twostage_c"]:
        members = [f for f in fams if f.startswith(base)]
        row = {"family": base + "*(envelope)"}
        for R in targets:
            vals = [r[f"saving@{R}"] for r in sav_rows if r["family"] in members]
            row[f"saving@{R}"] = np.nanmax(vals) if len(vals) else np.nan
        sav_rows.append(row)
    sav = pd.DataFrame(sav_rows).sort_values("saving@0.9", ascending=False)
    sav.to_csv(out / "07_rules_savings.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nwork saving vs fixed nprobe at matched mean recall (tune split; >1 = better than fixed)")
    print(sav.round(3).to_string(index=False))
    print("\nfor reference, work of fixed nprobe:", {R: round(w) for R, w in W_fixed.items()})


if __name__ == "__main__":
    main(*sys.argv[1:])
