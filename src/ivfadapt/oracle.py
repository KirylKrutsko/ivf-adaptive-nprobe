"""Per-query recall/work curves vs number of cells; per-query-target and Lagrangian oracles.

For IVF-PQ with residual encoding the PQ distance of a vector depends only on the query and
the vector's own cell, so scanning cells one at a time and merging the candidate lists
cumulatively reproduces the real search with nprobe = c for every c at once (PLAN §3).
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from ivfadapt.metrics import recall_per_query
from ivfadapt.rules import quantizer_topk
from ivfadapt.search import preassigned_search


@dataclass
class Curves:
    """Arrays of shape (n_queries, C_max); column c-1 is the state after c cells."""
    recall: np.ndarray      # recall@k after exact re-ranking of the top-k_rerank candidates
    recall_raw: np.ndarray  # recall@k of the raw PQ top-k
    coverage: np.ndarray    # share of the true top-k whose cell is among the first c cells
    work: np.ndarray        # cumulative scanned vectors (ndis), int64
    cells: np.ndarray       # the C_max nearest cell ids per query
    k: int
    k_rerank: int

    @property
    def c_max(self) -> int:
        return self.recall.shape[1]


def _exact_sq_dist(xb: np.ndarray, xq: np.ndarray, I: np.ndarray) -> np.ndarray:
    """Exact squared L2 from each query to its candidate ids (inf where id < 0)."""
    vecs = xb[np.maximum(I, 0)]
    d = ((vecs - xq[:, None, :]) ** 2).sum(axis=2)
    return np.where(I >= 0, d, np.inf).astype(np.float32)


def compute_curves(
    index: faiss.IndexIVF, xb: np.ndarray, xq: np.ndarray, gt: np.ndarray, cell_of: np.ndarray,
    list_sizes: np.ndarray, c_max: int, k: int, k_rerank: int, chunk: int = 1000, verbose: bool = True,
) -> Curves:
    n = len(xq)
    recall = np.zeros((n, c_max), dtype=np.float32)
    recall_raw = np.zeros((n, c_max), dtype=np.float32)
    coverage = np.zeros((n, c_max), dtype=np.float32)
    work = np.zeros((n, c_max), dtype=np.int64)
    cells_all = np.zeros((n, c_max), dtype=np.int64)
    gt_k = gt[:, :k]

    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        xs, gts = xq[s:e], gt_k[s:e]
        m = e - s
        Dq, Iq = quantizer_topk(index, xs, c_max)
        cells_all[s:e] = Iq
        work[s:e] = np.cumsum(list_sizes[Iq], axis=1)

        # coverage: rank (in Iq) of the cell of each true neighbour
        gt_cells = cell_of[gts]                                            # (m, k)
        match = Iq[:, None, :] == gt_cells[:, :, None]                     # (m, k, c_max)
        rank = np.where(match.any(axis=2), match.argmax(axis=2), c_max)    # (m, k)
        coverage[s:e] = (rank[:, :, None] < np.arange(c_max)[None, None, :] + 1).mean(axis=1)

        # cumulative candidate list: PQ distance, exact distance, id
        cum_pq = np.full((m, k_rerank), np.inf, dtype=np.float32)
        cum_ex = np.full((m, k_rerank), np.inf, dtype=np.float32)
        cum_id = np.full((m, k_rerank), -1, dtype=np.int64)
        for c in range(c_max):
            D_c, I_c = preassigned_search(index, xs, k_rerank, Iq[:, c:c + 1], Dq[:, c:c + 1])
            E_c = _exact_sq_dist(xb, xs, I_c)
            pq = np.concatenate([cum_pq, D_c], axis=1)
            order = np.argsort(pq, axis=1, kind="stable")[:, :k_rerank]
            cum_pq = np.take_along_axis(pq, order, axis=1)
            cum_ex = np.take_along_axis(np.concatenate([cum_ex, E_c], axis=1), order, axis=1)
            cum_id = np.take_along_axis(np.concatenate([cum_id, I_c], axis=1), order, axis=1)
            top_rr = np.take_along_axis(cum_id, np.argsort(cum_ex, axis=1, kind="stable")[:, :k], axis=1)
            recall[s:e, c] = recall_per_query(top_rr, gts, k)
            recall_raw[s:e, c] = recall_per_query(cum_id[:, :k], gts, k)
        if verbose:
            print(f"oracle curves: {e}/{n} queries")
    return Curves(recall, recall_raw, coverage, work, cells_all, k, k_rerank)


def save_curves(path, curves: Curves, query_ids: np.ndarray) -> None:
    np.savez_compressed(path, recall=curves.recall, recall_raw=curves.recall_raw, coverage=curves.coverage,
                        work=curves.work, cells=curves.cells, k=curves.k, k_rerank=curves.k_rerank,
                        query_ids=query_ids)


def load_curves(path) -> tuple[Curves, np.ndarray]:
    z = np.load(path)
    return Curves(z["recall"], z["recall_raw"], z["coverage"], z["work"], z["cells"],
                  int(z["k"]), int(z["k_rerank"])), z["query_ids"]


def subset(curves: Curves, rows: np.ndarray) -> Curves:
    return Curves(curves.recall[rows], curves.recall_raw[rows], curves.coverage[rows], curves.work[rows],
                  curves.cells[rows], curves.k, curves.k_rerank)


# ---- oracle variant 1: per-query target --------------------------------------------------

def per_query_target(curves: Curves, target: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smallest c with recall_q(c) >= target. Returns (c_star, work_at_c_star, reachable).

    For unreachable queries c_star = c_max and the work is work_q(c_max); use `reachable`
    to put them in a separate bin.
    """
    hit = curves.recall >= target - 1e-6
    reachable = hit.any(axis=1)
    c_star = np.where(reachable, hit.argmax(axis=1) + 1, curves.c_max)
    work = np.take_along_axis(curves.work, (c_star - 1)[:, None], axis=1)[:, 0]
    return c_star, work, reachable


def per_query_target_curve(curves: Curves, targets: np.ndarray) -> dict[str, np.ndarray]:
    """Mean (work, recall) over queries when every query gets c*_q(t); unreachable queries
    get c_max. One point per target."""
    out = {"target": [], "work_mean": [], "recall_mean": [], "cells_mean": [], "unreachable": []}
    for t in targets:
        c_star, w, reach = per_query_target(curves, t)
        r = np.take_along_axis(curves.recall, (c_star - 1)[:, None], axis=1)[:, 0]
        out["target"].append(t); out["work_mean"].append(w.mean()); out["recall_mean"].append(r.mean())
        out["cells_mean"].append(c_star.mean()); out["unreachable"].append((~reach).sum())
    return {k: np.array(v) for k, v in out.items()}


# ---- oracle variant 2: Lagrangian (best mean recall for a given mean work) ----------------

def lagrangian_curve(curves: Curves, lambdas: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """For each λ every query picks c = argmin_c work_q(c) − λ·recall_q(c).

    Sweeping λ traces the lower envelope of achievable (mean work, mean recall) for any
    per-query cell allocation. Returns the per-λ means plus the chosen cells (n_lambda, n).
    """
    if lambdas is None:
        lambdas = np.logspace(1, 8, 300)
    work = curves.work.astype(np.float64)
    rec = curves.recall.astype(np.float64)
    out_w, out_r, out_c, chosen = [], [], [], []
    for lam in lambdas:
        c = np.argmin(work - lam * rec, axis=1)                  # 0-based
        out_w.append(np.take_along_axis(work, c[:, None], axis=1).mean())
        out_r.append(np.take_along_axis(rec, c[:, None], axis=1).mean())
        out_c.append((c + 1).mean())
        chosen.append((c + 1).astype(np.int16))
    return {"lambda": lambdas, "work_mean": np.array(out_w), "recall_mean": np.array(out_r),
            "cells_mean": np.array(out_c), "cells": np.array(chosen)}


# ---- reading any per-query cell allocation off the curves --------------------------------

def at_cells(curves: Curves, n: np.ndarray) -> dict[str, np.ndarray]:
    """Per-query recall / raw recall / coverage / work when query q scans its n[q] nearest cells.

    Valid for any rule whose cells are a prefix of the quantizer order (the ratio rule, fixed
    nprobe, the oracles): the curves already hold the result of every such prefix.
    """
    j = (np.clip(n, 1, curves.c_max) - 1).astype(np.int64)[:, None]
    return {"recall": np.take_along_axis(curves.recall, j, axis=1)[:, 0],
            "recall_raw": np.take_along_axis(curves.recall_raw, j, axis=1)[:, 0],
            "coverage": np.take_along_axis(curves.coverage, j, axis=1)[:, 0],
            "work": np.take_along_axis(curves.work, j, axis=1)[:, 0]}
