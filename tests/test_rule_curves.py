"""Reading the ratio rule off the oracle curves must equal running it through the real search."""

import numpy as np
import pytest

from ivfadapt.metrics import recall_per_query
from ivfadapt.oracle import at_cells, compute_curves
from ivfadapt.rules import n_cells, quantizer_topk, ratio_cells, ratio_n_cells
from ivfadapt.search import preassigned_search, rerank_exact, scanned_vectors

C_MAX = 64


@pytest.fixture(scope="module")
def curves(env, xq_small, gt_small):
    return compute_curves(env.index, env.xb, xq_small, gt_small, env.cell_of, env.list_sizes,
                          c_max=C_MAX, k=env.k, k_rerank=env.k_rerank, chunk=250, verbose=False)


def test_ratio_n_cells_matches_ratio_cells(env, xq_small):
    Dq, Iq = quantizer_topk(env.index, xq_small, env.max_cells)
    for eps in [0.0, 0.01, 0.05, 0.3, 1.0]:
        np.testing.assert_array_equal(ratio_n_cells(Dq, eps), n_cells(ratio_cells(Dq, Iq, eps)))


@pytest.mark.parametrize("eps", [0.0, 0.02, 0.05])
def test_rule_from_curves_equals_real_search(env, xq_small, gt_small, curves, eps):
    Dq, Iq = quantizer_topk(env.index, xq_small, C_MAX)
    cells = ratio_cells(Dq, Iq, eps)
    n = n_cells(cells)
    ok = n < C_MAX                         # queries whose cells all fit inside the curves
    assert ok.mean() > 0.9
    _, I_pq = preassigned_search(env.index, xq_small, env.k_rerank, cells, Dq)
    _, I_rr = rerank_exact(env.xb, xq_small, I_pq, env.k)
    res = at_cells(curves, n)
    np.testing.assert_allclose(res["recall"][ok], recall_per_query(I_rr, gt_small, env.k)[ok], atol=1e-6)
    np.testing.assert_array_equal(res["work"][ok], scanned_vectors(cells, env.list_sizes)[ok])
