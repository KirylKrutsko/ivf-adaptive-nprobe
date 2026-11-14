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
