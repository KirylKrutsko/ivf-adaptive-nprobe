"""Fixed-nprobe search, per-query preassigned search, exact re-ranking, scanned-vector counts."""

from __future__ import annotations

import faiss
import numpy as np


def reset_stats() -> None:
    faiss.cvar.indexIVF_stats.reset()


def stats_ndis() -> int:
    return int(faiss.cvar.indexIVF_stats.ndis)


def scanned_vectors(cells: np.ndarray, list_sizes: np.ndarray) -> np.ndarray:
    """Per-query sum of list sizes over the selected cells (-1 entries ignored)."""
    sizes = np.where(cells >= 0, list_sizes[np.maximum(cells, 0)], 0)
    return sizes.sum(axis=1).astype(np.int64)


def fixed_search(index: faiss.IndexIVF, xq: np.ndarray, k: int, nprobe: int) -> tuple[np.ndarray, np.ndarray]:
    """Plain Faiss search with a fixed nprobe. Returns (D, I) of the k best PQ candidates."""
    index.nprobe = nprobe
    return index.search(xq, k)


def preassigned_search(
    index: faiss.IndexIVF, xq: np.ndarray, k: int, cells: np.ndarray, Dq: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Search where query i scans exactly the cells in cells[i] (>= 0 entries).

    Queries are grouped by their number of cells so that each search_preassigned call has
    no padding. Rows and columns of `cells` must be ordered as the quantizer returned them
    (nearest first), with -1 after the kept cells. Returns (D, I) in the original query order.
    """
    n = len(xq)
    counts = (cells >= 0).sum(axis=1)
    D = np.full((n, k), np.inf, dtype=np.float32)
    I = np.full((n, k), -1, dtype=np.int64)
    if Dq is None:
        Dq = np.zeros_like(cells, dtype=np.float32)
    old_nprobe = index.nprobe
    try:
        for c in np.unique(counts):
            rows = np.where(counts == c)[0]
            if c == 0:
                continue
            index.nprobe = int(c)
            sub_cells = np.ascontiguousarray(cells[rows, :c]).astype(np.int64)
            sub_dq = np.ascontiguousarray(Dq[rows, :c]).astype(np.float32)
            Dr, Ir = index.search_preassigned(xq[rows], k, sub_cells, sub_dq)
            D[rows], I[rows] = Dr, Ir
    finally:
        index.nprobe = old_nprobe
    return D, I


def rerank_exact(xb: np.ndarray, xq: np.ndarray, I: np.ndarray, k: int, chunk: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Exact L2 re-ranking of the candidate ids in I (-1 = missing). Returns top-k (D, I)."""
    n, kc = I.shape
    out_d = np.empty((n, k), dtype=np.float32)
    out_i = np.full((n, k), -1, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        ids = I[s:e]
        valid = ids >= 0
        vecs = xb[np.maximum(ids, 0)]                                  # (m, kc, d)
        d = ((vecs - xq[s:e, None, :]) ** 2).sum(axis=2)              # (m, kc)
        d = np.where(valid, d, np.inf).astype(np.float32)
        order = np.argsort(d, axis=1, kind="stable")[:, :k]
        out_d[s:e] = np.take_along_axis(d, order, axis=1)
        out_i[s:e] = np.where(np.isfinite(out_d[s:e]), np.take_along_axis(ids, order, axis=1), -1)
    return out_d, out_i
