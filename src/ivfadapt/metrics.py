"""recall@k, interpolation at matched recall, saving factor, share of oracle saving captured."""

from __future__ import annotations

import numpy as np


def recall_per_query(I: np.ndarray, gt: np.ndarray, k: int) -> np.ndarray:
    """Share of the true top-k (gt[:, :k]) present in I[:, :k], per query."""
    hits = (I[:, :k, None] == gt[:, None, :k]).any(axis=2).sum(axis=1)
    return hits / k


def recall_at_k(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    return float(recall_per_query(I, gt, k).mean())


def interp_work_at_recall(recall: np.ndarray, work: np.ndarray, target: float) -> float:
    """Work needed to reach `target` recall, log-interpolated along a (recall, work) curve.

    The curve is sorted by work; returns nan if the target is above the best recall.
    """
    order = np.argsort(work)
    r, w = np.asarray(recall)[order], np.asarray(work)[order]
    if target > r.max():
        return float("nan")
    j = int(np.argmax(r >= target))
    if j == 0:
        return float(w[0])
    r0, r1, w0, w1 = r[j - 1], r[j], w[j - 1], w[j]
    if r1 == r0:
        return float(w1)
    t = (target - r0) / (r1 - r0)
    return float(np.exp(np.log(w0) + t * (np.log(w1) - np.log(w0))))
