"""The cumulative-merge oracle must reproduce the real search at every nprobe."""

import numpy as np
import pytest

from ivfadapt.metrics import interp_work_at_recall, recall_per_query
from ivfadapt.oracle import compute_curves, lagrangian_curve, per_query_target
from ivfadapt.search import fixed_search, rerank_exact, scanned_vectors
from ivfadapt.rules import quantizer_topk


@pytest.fixture(scope="module")
def curves(env, xq_small, gt_small):
    return compute_curves(env.index, env.xb, xq_small, gt_small, env.cell_of, env.list_sizes,
                          c_max=40, k=env.k, k_rerank=env.k_rerank, chunk=250, verbose=False)


@pytest.mark.parametrize("c", [1, 4, 32])
def test_oracle_recall_equals_real_search(env, xq_small, gt_small, curves, c):
    _, I_pq = fixed_search(env.index, xq_small, env.k_rerank, nprobe=c)
    _, I_rr = rerank_exact(env.xb, xq_small, I_pq, env.k)
    np.testing.assert_allclose(curves.recall[:, c - 1], recall_per_query(I_rr, gt_small, env.k), atol=1e-6)
    np.testing.assert_allclose(curves.recall_raw[:, c - 1], recall_per_query(I_pq, gt_small, env.k), atol=1e-6)


def test_oracle_work_equals_list_sizes(env, xq_small, curves):
    _, Iq = quantizer_topk(env.index, xq_small, 40)
    for c in [1, 7, 40]:
        np.testing.assert_array_equal(curves.work[:, c - 1], scanned_vectors(Iq[:, :c], env.list_sizes))


def test_coverage_bounds_recall(curves):
    # a true neighbour can be found only if its cell was scanned
    assert np.all(curves.recall <= curves.coverage + 1e-6)
    assert np.all(np.diff(curves.coverage, axis=1) >= 0)
    assert np.all(np.diff(curves.work, axis=1) > 0)


def test_per_query_target(curves):
    c_star, work, reachable = per_query_target(curves, 0.9)
    assert np.all(c_star >= 1) and np.all(c_star <= curves.c_max)
    rows = np.where(reachable)[0]
    assert np.all(curves.recall[rows, c_star[rows] - 1] >= 0.9)
    before = rows[c_star[rows] > 1]
    assert np.all(curves.recall[before, c_star[before] - 2] < 0.9)


def test_lagrangian_beats_fixed_nprobe(curves):
    lag = lagrangian_curve(curves)
    fixed_r, fixed_w = curves.recall.mean(0), curves.work.mean(0)
    for c in [2, 8, 32]:
        w_oracle = interp_work_at_recall(lag["recall_mean"], lag["work_mean"], fixed_r[c - 1])
        assert w_oracle <= fixed_w[c - 1] * (1 + 1e-9)
