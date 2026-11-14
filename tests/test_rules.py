"""Ratio rule sanity (D4)."""

import numpy as np

from ivfadapt.rules import n_cells, quantizer_topk, ratio_cells


def test_eps_zero_gives_one_cell_unless_tie(env, xq_small):
    Dq, Iq = quantizer_topk(env.index, xq_small, env.max_cells)
    cnt = n_cells(ratio_cells(Dq, Iq, 0.0))
    ties = Dq[:, 1] <= np.maximum(Dq[:, 0], 0)
    assert np.all(cnt[~ties] == 1)


def test_monotone_in_eps(env, xq_small):
    Dq, Iq = quantizer_topk(env.index, xq_small, env.max_cells)
    prev = None
    for eps in [0.0, 0.01, 0.05, 0.1, 0.3, 1.0]:
        cnt = n_cells(ratio_cells(Dq, Iq, eps))
        if prev is not None:
            assert np.all(cnt >= prev)
        prev = cnt
    assert np.all(prev <= env.max_cells)


def test_squared_equals_plain_distances(env, xq_small):
    """(1+eps)^2 on squared distances == (1+eps) on plain distances."""
    Dq, Iq = quantizer_topk(env.index, xq_small, env.max_cells)
    for eps in [0.0, 0.03, 0.2]:
        cells = ratio_cells(Dq, Iq, eps)
        d = np.sqrt(np.maximum(Dq, 0))
        keep = d <= (1 + eps) * d[:, :1]
        keep[:, 0] = True
        np.testing.assert_array_equal(cells >= 0, keep)


def test_kept_cells_are_a_prefix(env, xq_small):
    """Distances are sorted, so kept cells form a prefix of each row."""
    Dq, Iq = quantizer_topk(env.index, xq_small, env.max_cells)
    cells = ratio_cells(Dq, Iq, 0.1)
    keep = cells >= 0
    first_drop = np.argmin(keep, axis=1)
    for i in range(len(keep)):
        if not keep[i].all():
            assert not keep[i, first_drop[i]:].any()
