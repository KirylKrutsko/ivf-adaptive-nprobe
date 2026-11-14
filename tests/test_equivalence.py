"""A constant-c rule through search_preassigned must equal index.search with nprobe=c."""

import numpy as np
import pytest

from ivfadapt.rules import fixed_cells, quantizer_topk
from ivfadapt.search import fixed_search, preassigned_search, rerank_exact


def assert_same_results(D1, I1, D2, I2):
    np.testing.assert_array_equal(D1, D2)
    swapped = I1 != I2
    if swapped.any():
        # ids may differ only where the distances tie
        rows, cols = np.where(swapped)
        for r in set(rows):
            c = cols[rows == r]
            assert len(np.unique(D1[r, c])) < len(c), "ids differ without a distance tie"


@pytest.mark.parametrize("c", [1, 4, 32])
def test_constant_rule_equals_fixed_nprobe(env, xq_small, c):
    k = env.k_rerank
    D_fixed, I_fixed = fixed_search(env.index, xq_small, k, nprobe=c)
    Dq, Iq = quantizer_topk(env.index, xq_small, env.max_cells)
    D_pre, I_pre = preassigned_search(env.index, xq_small, k, fixed_cells(Iq, c), Dq)
    assert_same_results(D_fixed, I_fixed, D_pre, I_pre)


def test_mixed_widths_equal_fixed_nprobe_per_query(env, xq_small):
    """Queries with different cell counts in one batch each match their own nprobe."""
    k = env.k_rerank
    Dq, Iq = quantizer_topk(env.index, xq_small, env.max_cells)
    counts = np.array([1, 2, 5, 16] * (len(xq_small) // 4 + 1))[: len(xq_small)]
    cells = np.where(np.arange(Iq.shape[1])[None, :] < counts[:, None], Iq, -1)
    D_pre, I_pre = preassigned_search(env.index, xq_small, k, cells, Dq)
    for c in np.unique(counts):
        rows = counts == c
        D_fixed, I_fixed = fixed_search(env.index, xq_small[rows], k, nprobe=int(c))
        assert_same_results(D_fixed, I_fixed, D_pre[rows], I_pre[rows])


def test_rerank_equals_exact_search_within_one_cell(env, xq_small):
    """With k_rerank >= cell size, re-ranked PQ candidates = exact top-10 inside that cell."""
    Dq, Iq = quantizer_topk(env.index, xq_small, 1)
    small = env.list_sizes[Iq[:, 0]] <= env.k_rerank
    assert small.sum() > 10
    xq = xq_small[small]
    cells = Iq[small]
    _, I_pq = preassigned_search(env.index, xq, env.k_rerank, cells, Dq[small])
    _, I_rr = rerank_exact(env.xb, xq, I_pq, env.k)
    for i in range(len(xq)):
        members = np.where(env.cell_of == cells[i, 0])[0]
        d = ((env.xb[members] - xq[i]) ** 2).sum(1)
        exact = members[np.argsort(d, kind="stable")[: env.k]]
        assert set(I_rr[i]) == set(exact)
