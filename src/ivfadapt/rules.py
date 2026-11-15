"""Cell-selection rules. Input: the max_cells nearest centroids per query (squared L2)."""

from __future__ import annotations

import faiss
import numpy as np


def quantizer_topk(index: faiss.IndexIVF, xq: np.ndarray, max_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Squared distances and ids of the max_cells nearest centroids, sorted ascending."""
    Dq, Iq = index.quantizer.search(xq, max_cells)
    return Dq, Iq


def ratio_cells(Dq: np.ndarray, Iq: np.ndarray, eps: float) -> np.ndarray:
    """Ratio rule (D4): keep cell j iff d_j <= (1+eps) * d_1 on plain distances.

    Dq holds squared distances, so the test is Dq[:, j] <= (1+eps)^2 * Dq[:, 0].
    Dq[:, 0] is clamped at 0 (Faiss can return tiny negatives). Non-strict comparison:
    eps = 0 keeps exactly one cell unless distances tie. Returns Iq with -1 where a cell
    is dropped; the nearest cell is always kept.
    """
    d1 = np.maximum(Dq[:, :1], 0.0)
    keep = Dq <= (1.0 + eps) ** 2 * d1
    keep[:, 0] = True
    cells = np.where(keep, Iq, -1)
    return cells


def fixed_cells(Iq: np.ndarray, c: int) -> np.ndarray:
    """The baseline expressed as a per-query cell list: the c nearest cells for every query."""
    cells = np.full_like(Iq, -1)
    cells[:, :c] = Iq[:, :c]
    return cells


def n_cells(cells: np.ndarray) -> np.ndarray:
    return (cells >= 0).sum(axis=1)


def ratio_n_cells(Dq: np.ndarray, eps: float) -> np.ndarray:
    """Number of cells the ratio rule keeps per query (same as n_cells(ratio_cells(...))).

    Dq is sorted ascending, so the kept cells are a prefix and their count is enough to
    look the result up in the oracle's per-query curves.
    """
    d1 = np.maximum(Dq[:, :1], 0.0)
    return np.maximum((Dq <= (1.0 + eps) ** 2 * d1).sum(axis=1), 1)


# ---- geometry: lower bounds on the distance from the query to a cell --------------------

def centroids(index: faiss.IndexIVF) -> np.ndarray:
    return index.quantizer.reconstruct_n(0, index.nlist)


def bisector_bounds(C: np.ndarray, Dq: np.ndarray, Iq: np.ndarray, chunk: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Lower bounds on the plain L2 distance from each query to any vector of its K nearest cells.

    A vector x stored in cell j satisfies ||x - c_j|| <= ||x - c_i|| for every i, so x lies on
    cell j's side of the bisector of (c_i, c_j). The query is closer to c_i whenever D_i < D_j,
    hence ||q - x|| >= (D_j - D_i) / (2 ||c_j - c_i||) (distance from q to that hyperplane).
    Returns (b1, bmax): the bound from the nearest centroid only (i = 1), and the max over all
    nearer centroids among the K (tighter, still valid). Both are 0 for the nearest cell.
    Dq are squared distances sorted ascending, Iq the matching cell ids.
    """
    n, K = Iq.shape
    b1 = np.zeros((n, K), dtype=np.float32)
    bmax = np.zeros((n, K), dtype=np.float32)
    Cn = (C.astype(np.float64) ** 2).sum(1)
    lower = np.tril(np.ones((K, K), dtype=bool), -1)          # [j, i] with i < j
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        I = Iq[s:e]
        D = np.maximum(Dq[s:e].astype(np.float64), 0.0)
        Cs = C[I].astype(np.float64)                              # (m, K, d)
        gram = Cs @ Cs.transpose(0, 2, 1)
        nrm = Cn[I]
        dist = np.sqrt(np.maximum(nrm[:, :, None] + nrm[:, None, :] - 2 * gram, 1e-12))
        G = (D[:, :, None] - D[:, None, :]) / (2 * dist)         # (m, j, i)
        G = np.where(lower[None], G, -np.inf)
        bm = G.max(axis=2)
        bm[:, 0] = 0.0
        bmax[s:e] = np.maximum(bm, 0.0)
        bb = G[:, :, 0].copy()
        bb[:, 0] = 0.0
        b1[s:e] = np.maximum(bb, 0.0)
    return b1, bmax


def compact_cells(Iq: np.ndarray, keep: np.ndarray, Dq: np.ndarray | None = None):
    """Turn a boolean mask over the K nearest cells into a left-packed cell matrix (-1 padded),
    keeping the quantizer order among kept cells. Also reorders Dq the same way if given."""
    order = np.argsort(~keep, axis=1, kind="stable")
    cells = np.take_along_axis(np.where(keep, Iq, -1), order, axis=1)
    if Dq is None:
        return cells
    return cells, np.take_along_axis(Dq, order, axis=1)
