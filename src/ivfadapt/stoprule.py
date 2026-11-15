"""Scan-feedback stopping rule in a form a real search can run.

Cells are scanned in rounds (SCHEDULE gives the number of cells scanned after each round). After
every round, for each still-active query, features are computed from what the search already has
(PQ distances of the running top-k_rerank candidates, the quantizer distances) and a logistic
model gives P(continue). Queries with P < theta stop; the rest scan the next round. Exact
re-ranking happens once at the end, as in every other method here.

`run_rounds` is used for three things with the same code path: generating training features
(model=None: every query runs to the end and the features at every boundary are recorded),
inference (model + theta), and timing (scan and rule time are accumulated separately).
"""

from __future__ import annotations

import time

import numpy as np

from ivfadapt.search import preassigned_search

SCHEDULE = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256]
STATIC_RANKS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
N_FEATURES = 9 + len(STATIC_RANKS) + 2


def static_features(Dq: np.ndarray, Iq: np.ndarray, list_sizes: np.ndarray) -> np.ndarray:
    """Per-query features from the quantizer alone (same as experiments/08_fitted_rule.py)."""
    d = np.sqrt(np.maximum(Dq, 1e-12))
    X = [np.log(d[:, j - 1] / d[:, 0]) for j in STATIC_RANKS]
    X.append(np.log(d[:, 0]))
    X.append(np.log(list_sizes[Iq[:, :8]].mean(1)))
    return np.stack(X, 1).astype(np.float32)


class LogisticModel:
    def __init__(self, mu, sd, w):
        self.mu, self.sd, self.w = mu, sd, w

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = ((X - self.mu) / self.sd) @ self.w[:-1] + self.w[-1]
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path):
        np.savez(path, mu=self.mu, sd=self.sd, w=self.w)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["mu"], z["sd"], z["w"])


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float = 1e-2, iters: int = 25) -> LogisticModel:
    """L2-regularised logistic regression by Newton's method (features standardised)."""
    X = X.astype(np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z.shape[1]); reg = np.r_[np.ones(Z.shape[1] - 1), 0.0]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w))); W = p * (1 - p)
        g = Z.T @ (p - y) + lam * reg * w
        H = (Z * W[:, None]).T @ Z + lam * np.diag(reg)
        w -= np.linalg.solve(H, g)
    return LogisticModel(mu.astype(np.float32), sd.astype(np.float32), w.astype(np.float32))


def run_rounds(index, xq: np.ndarray, Dq: np.ndarray, Iq: np.ndarray, static: np.ndarray, k: int, k_rerank: int,
               model: LogisticModel | None = None, theta: float = 0.0, schedule=SCHEDULE):
    """Scan cells in rounds with per-query stopping.

    Returns dict with n_cells (n,), cand_ids (n, k_rerank) PQ candidates for re-ranking,
    features (n, n_decisions, F) and P (n, n_decisions) recorded at every decision boundary
    (features only where the query was still active; NaN otherwise), and the time spent in the
    scan (search_preassigned) and in the rule (merge + features + model).
    """
    n = len(xq)
    C = schedule[-1]
    assert Iq.shape[1] >= C
    d = np.sqrt(np.maximum(Dq[:, :C + 1] if Dq.shape[1] > C else np.c_[Dq[:, :C], Dq[:, C - 1:C]], 1e-12))
    d1 = d[:, 0]
    decisions = schedule[:-1]
    feats = np.full((n, len(decisions), N_FEATURES), np.nan, np.float32)
    P = np.full((n, len(decisions)), np.nan, np.float32)
    n_cells = np.full(n, C, np.int64)
    cum_pq = np.full((n, k_rerank), np.inf, np.float32)
    cum_id = np.full((n, k_rerank), -1, np.int64)
    prev_top = np.full((n, k), -2, np.int64)
    r10_prev = np.zeros(n, np.float32)
    since = np.zeros(n, np.float32)
    active = np.ones(n, bool)
    t_scan = t_rule = 0.0
    c_prev = 0
    for i, c in enumerate(schedule):
        rows = np.where(active)[0]
        if len(rows) == 0:
            break
        width = c - c_prev
        t0 = time.perf_counter()
        D_c, I_c = preassigned_search(index, xq[rows], k_rerank, Iq[rows, c_prev:c], Dq[rows, c_prev:c])
        t_scan += time.perf_counter() - t0
        t0 = time.perf_counter()
        pq = np.concatenate([cum_pq[rows], D_c], 1)
        ids = np.concatenate([cum_id[rows], I_c], 1)
        part = np.argpartition(pq, k_rerank - 1, axis=1)[:, :k_rerank]
        new_pq = np.take_along_axis(pq, part, 1); new_id = np.take_along_axis(ids, part, 1)
        n_entered = (part >= k_rerank).sum(1)
        cum_pq[rows], cum_id[rows] = new_pq, new_id
        if i < len(decisions):                                   # a decision is taken after this round
            top = np.argpartition(new_pq, k - 1, axis=1)[:, :k]
            top_ids = np.take_along_axis(new_id, top, 1)
            r10 = np.sqrt(np.maximum(np.take_along_axis(new_pq, top, 1).max(1), 1e-12))
            r1 = np.sqrt(np.maximum(new_pq.min(1), 1e-12))
            n_changed = k - (top_ids[:, :, None] == prev_top[rows][:, None, :]).any(2).sum(1)
            since_r = np.where(n_changed == 0, since[rows] + 1, 0.0)
            r10p = np.where(i == 0, r10, r10_prev[rows])
            dn = d[rows, c]                                       # distance to the next cell's centroid
            F = np.empty((len(rows), N_FEATURES), np.float32)
            F[:, 0] = np.log(r10 / d1[rows]); F[:, 1] = np.log(r10 / dn); F[:, 2] = np.log(r1 / d1[rows])
            F[:, 3] = np.log(dn / d1[rows]); F[:, 4] = np.log(c); F[:, 5] = np.minimum(since_r, 8) / 8
            F[:, 6] = np.log1p(n_changed); F[:, 7] = np.log1p(n_entered / width); F[:, 8] = np.log(r10p / r10)
            F[:, 9:] = static[rows]
            feats[rows, i] = F
            prev_top[rows] = top_ids; r10_prev[rows] = r10; since[rows] = since_r
            if model is not None:
                p = model.predict(F)
                P[rows, i] = p
                stop = p < theta
                n_cells[rows[stop]] = c
                active[rows[stop]] = False
        t_rule += time.perf_counter() - t0
        c_prev = c
    return {"n_cells": n_cells, "cand_ids": cum_id, "features": feats, "P": P, "t_scan": t_scan, "t_rule": t_rule}


def cells_from_P(P: np.ndarray, theta: float, schedule=SCHEDULE) -> np.ndarray:
    """Offline version of the stopping decision from recorded P (n, n_decisions)."""
    stop = P < theta
    first = np.where(stop.any(1), stop.argmax(1), -1)
    return np.where(first >= 0, np.array(schedule)[np.maximum(first, 0)], schedule[-1]).astype(np.int64)


def labels_continue(oracle_cells: np.ndarray, schedule=SCHEDULE) -> np.ndarray:
    """(n, n_decisions) 1 if the oracle allocation continues past the boundary."""
    b = np.array(schedule[:-1])[None, :]
    return (oracle_cells[:, None] > b).astype(np.float32)
