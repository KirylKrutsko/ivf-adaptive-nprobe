"""Round-based stopping rule: online run == offline decision from recorded P; its result is a
prefix of the quantizer order, so re-ranked recall equals the oracle curves at n_cells."""

import numpy as np

from ivfadapt.metrics import recall_per_query
from ivfadapt.oracle import at_cells, compute_curves
from ivfadapt.rules import quantizer_topk
from ivfadapt.search import rerank_exact
from ivfadapt.stoprule import SCHEDULE, cells_from_P, fit_logistic, labels_continue, run_rounds, static_features


def test_rounds_online_equals_offline_and_curves(env, xq_small, gt_small):
    C = SCHEDULE[-1]
    Dq, Iq = quantizer_topk(env.index, xq_small, C)
    static = static_features(Dq, Iq, env.list_sizes)
    full = run_rounds(env.index, xq_small, Dq, Iq, static, env.k, env.k_rerank)
    assert np.all(full["n_cells"] == C) and not np.isnan(full["features"]).any()
    # a toy model: labels from a fixed "oracle" of 12 cells for everyone
    X = full["features"].reshape(-1, full["features"].shape[2])
    y = labels_continue(np.full(len(xq_small), 12)).reshape(-1)
    model = fit_logistic(X, y)
    rec = run_rounds(env.index, xq_small, Dq, Iq, static, env.k, env.k_rerank, model=model, theta=1e-9)
    for theta in [0.05, 0.3, 0.7]:
        online = run_rounds(env.index, xq_small, Dq, Iq, static, env.k, env.k_rerank, model=model, theta=theta)
        np.testing.assert_array_equal(online["n_cells"], cells_from_P(rec["P"], theta))
        assert set(np.unique(online["n_cells"])) <= set(SCHEDULE)
    # re-ranked recall of the online candidates == curve value at n_cells (prefix rule)
    curves = compute_curves(env.index, env.xb, xq_small, gt_small, env.cell_of, env.list_sizes,
                            c_max=int(online["n_cells"].max()), k=env.k, k_rerank=env.k_rerank, chunk=250, verbose=False)
    _, I_rr = rerank_exact(env.xb, xq_small, online["cand_ids"], env.k)
    np.testing.assert_allclose(recall_per_query(I_rr, gt_small, env.k), at_cells(curves, online["n_cells"])["recall"], atol=1e-6)
