"""Scanned-vector counts: Faiss's ndis must equal the sum of list sizes of the chosen cells."""

import numpy as np
import pytest

from ivfadapt.rules import fixed_cells, quantizer_topk, ratio_cells
from ivfadapt.search import fixed_search, preassigned_search, reset_stats, scanned_vectors, stats_ndis


@pytest.mark.parametrize("nprobe", [1, 7, 64])
def test_fixed_path_ndis(env, xq_small, nprobe):
    Dq, Iq = quantizer_topk(env.index, xq_small, nprobe)
    expected = scanned_vectors(Iq, env.list_sizes).sum()
    reset_stats()
    fixed_search(env.index, xq_small, env.k_rerank, nprobe)
    assert stats_ndis() == expected


@pytest.mark.parametrize("eps", [0.0, 0.05, 0.2])
def test_rule_path_ndis(env, xq_small, eps):
    Dq, Iq = quantizer_topk(env.index, xq_small, env.max_cells)
    cells = ratio_cells(Dq, Iq, eps)
    expected = scanned_vectors(cells, env.list_sizes).sum()
    reset_stats()
    preassigned_search(env.index, xq_small, env.k_rerank, cells, Dq)
    assert stats_ndis() == expected


def test_list_sizes_consistent(env):
    assert env.list_sizes.sum() == env.index.ntotal
    np.testing.assert_array_equal(np.bincount(env.cell_of, minlength=env.index.nlist), env.list_sizes)
