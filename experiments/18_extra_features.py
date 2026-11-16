"""Do cheap data-aware features help the two learned rules save more work?

Both rules so far only see the query's distances to centroids (and, for the stopping rule, how
the scan is going). Here they also get features describing the data around the query, all free
at search time (per-cell statistics precomputed once from the base, or by-products of the search):

static (both rules), for the query's nearest cell unless stated:
  rad1     log median distance of the cell's vectors to its centroid           (offline, per cell)
  d1/rad1  log(d1 / rad1): inside or outside the cell's data cloud
  nnr1     log median 10th-neighbour distance of the cell's vectors             (offline, per cell)
  d1/nnr1  log(d1 / nnr1)
  nnr8     log mean nnr over the 8 nearest cells
  border1  share of the cell's vectors with d2/d1 < 1.05 to their own centroids (offline, per cell)
  pqerr    log(PQ reconstruction error of the query's residual / d1)
  b2/d1    log(distance to the bisector of cells 1 and 2 / d1)
dynamic (stopping rule), after c cells:
  since_contrib  log(1 + c - highest centroid rank among the cells of the current PQ top-10)
  r10/nnr1       log(r10 / nnr1): is the current 10th-best good by local standards
  r200/r10       log(worst heap distance / r10): how tight the candidate pool is

Protocols are those of 17 (fitted rule: ridge on log oracle cells, γ by cross-fit, tuned saving
on test with bootstrap CIs) and 15 (stopping rule: logistic P(continue), labels from the
Lagrangian oracle, θ by cross-fit). Work only, no timing. SIFT1M unless a config is given.
Writes results/18_cell_stats.npz (cache), results/18_fitted.csv, results/18_stop.csv.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import bootstrap_weights
from ivfadapt.oracle import at_cells, lagrangian_curve, load_curves, subset
from ivfadapt.rules import quantizer_topk
from ivfadapt.search import preassigned_search
from ivfadapt.stoprule import SCHEDULE, cells_from_P, fit_logistic, labels_continue, run_rounds, static_features

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("a17", ROOT / "experiments" / "17_fitted_rule_analysis.py")
a17 = importlib.util.module_from_spec(_s); _s.loader.exec_module(a17)

NEW_STATIC = ["rad1", "d1/rad1", "nnr1", "d1/nnr1", "nnr8", "border1", "pqerr", "b2/d1"]
NEW_DYN = ["since_contrib", "r10/nnr1", "r200/r10"]
THETAS = np.concatenate([np.geomspace(1e-6, 0.02, 40), np.linspace(0.025, 0.99, 80)])
N_BOOT = 1000


# ---------------------------------------------------------------- per-cell statistics (offline)
def cell_stats(env, path, per_cell=10):
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    t0 = time.perf_counter()
    C = env.index.quantizer.reconstruct_n(0, env.index.nlist)
    xb, cell = env.xb, env.cell_of
    nl = env.index.nlist
    # radius: median distance of the members to their centroid
    dist = np.empty(len(xb), np.float32)
    for s in range(0, len(xb), 100_000):
        e = min(len(xb), s + 100_000)
        dist[s:e] = np.sqrt(((xb[s:e] - C[cell[s:e]]) ** 2).sum(1))
    order = np.argsort(cell, kind="stable")
    bounds = np.searchsorted(cell[order], np.arange(nl + 1))
    rad = np.array([np.median(dist[order[bounds[c]:bounds[c + 1]]]) for c in range(nl)], np.float32)
    # border share: members whose 2nd nearest centroid is within 5% of the nearest
    D2, _ = env.index.quantizer.search(xb, 2)
    d2 = np.sqrt(np.maximum(D2, 1e-12))
    near = (d2[:, 1] / d2[:, 0] < 1.05).astype(np.float32)
    border = np.bincount(cell, weights=near, minlength=nl) / np.maximum(np.bincount(cell, minlength=nl), 1)
    # local neighbour radius: 10th-neighbour distance of a sample of each cell's members
    rng = np.random.default_rng(env.cfg["seed"])
    samp = np.concatenate([rng.choice(order[bounds[c]:bounds[c + 1]], min(per_cell, bounds[c + 1] - bounds[c]), replace=False)
                           for c in range(nl)])
    flat = faiss.IndexFlatL2(xb.shape[1]); flat.add(xb)
    Dn, _ = flat.search(xb[samp], 11)                       # first hit is the point itself
    r10 = np.sqrt(np.maximum(Dn[:, 10], 0))
    nnr = np.zeros(nl, np.float32)
    sc = cell[samp]
    for c in range(nl):
        v = r10[sc == c]
        nnr[c] = np.median(v) if len(v) else np.nan
    nnr = np.where(np.isfinite(nnr), nnr, np.nanmedian(nnr)).astype(np.float32)
    out = {"rad": rad, "nnr": nnr, "border": border.astype(np.float32)}
    np.savez(path, **out)
    print(f"cell statistics computed in {time.perf_counter() - t0:.0f}s -> {path}")
    return out


def new_static(env, Dq, Iq, xq, cs):
    d = np.sqrt(np.maximum(Dq, 1e-12))
    c1 = Iq[:, 0]
    C = env.index.quantizer.reconstruct_n(0, env.index.nlist)
    # PQ reconstruction error of the query residual w.r.t. its nearest centroid
    res = np.ascontiguousarray(xq - C[c1], dtype=np.float32)
    pq = env.index.pq
    rec = pq.decode(pq.compute_codes(res))
    pqerr = np.sqrt(((res - rec) ** 2).sum(1))
    # distance to the bisector of cells 1 and 2
    cc = np.sqrt(((C[Iq[:, 1]] - C[c1]) ** 2).sum(1))
    b2 = np.maximum((Dq[:, 1] - Dq[:, 0]) / (2 * np.maximum(cc, 1e-9)), 1e-6)
    X = np.stack([np.log(cs["rad"][c1]), np.log(d[:, 0] / cs["rad"][c1]), np.log(cs["nnr"][c1]),
                  np.log(d[:, 0] / cs["nnr"][c1]), np.log(cs["nnr"][Iq[:, :8]].mean(1)), cs["border"][c1],
                  np.log(np.maximum(pqerr, 1e-6) / d[:, 0]), np.log(b2 / d[:, 0])], 1)
    return X.astype(np.float32)


# ---------------------------------------------------------------- stopping rule: extra dynamic features
def dynamic_extra(env, xq, Dq, Iq, nnr1, k, kr):
    """Per decision point: since_contrib, r10/nnr1, r200/r10, computed from the same running PQ
    top-k_rerank list as stoprule.run_rounds (every query forced to the end)."""
    n = len(xq); dec = SCHEDULE[:-1]
    out = np.zeros((n, len(dec), len(NEW_DYN)), np.float32)
    rank = np.full((n, env.index.nlist), 10_000, np.int32)          # centroid rank of each cell, per query
    rank[np.arange(n)[:, None], Iq[:, :256]] = np.arange(256)[None, :]
    cum_pq = np.full((n, kr), np.inf, np.float32); cum_id = np.full((n, kr), -1, np.int64)
    c_prev = 0
    for i, c in enumerate(SCHEDULE):
        D_c, I_c = preassigned_search(env.index, xq, kr, Iq[:, c_prev:c], Dq[:, c_prev:c])
        pq = np.concatenate([cum_pq, D_c], 1); ids = np.concatenate([cum_id, I_c], 1)
        part = np.argpartition(pq, kr - 1, axis=1)[:, :kr]
        cum_pq = np.take_along_axis(pq, part, 1); cum_id = np.take_along_axis(ids, part, 1)
        if i < len(dec):
            top = np.argpartition(cum_pq, k - 1, axis=1)[:, :k]
            tid = np.take_along_axis(cum_id, top, 1)
            tcell = env.cell_of[np.maximum(tid, 0)]
            trank = np.where(tid >= 0, rank[np.arange(n)[:, None], tcell], -1).max(1)
            r10 = np.sqrt(np.maximum(np.take_along_axis(cum_pq, top, 1).max(1), 1e-12))
            r200 = np.sqrt(np.maximum(np.where(np.isfinite(cum_pq), cum_pq, np.nan).max(1), 1e-12))
            r200 = np.where(np.isfinite(r200), r200, r10)
            out[:, i, 0] = np.log1p(np.maximum(c - 1 - trank, 0))
            out[:, i, 1] = np.log(r10 / nnr1)
            out[:, i, 2] = np.log(np.maximum(r200, r10) / r10)
        c_prev = c
    return out


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    env = load_env(cfg_path)
    res = Path(env.cfg["results_dir"]); k, kr, C = env.k, env.k_rerank, env.max_cells
    curves, _ = load_curves(res / "03_oracle_curves.npz")
    tune, test = subset(curves, env.tune), subset(curves, env.test)
    targets = env.cfg["recall_targets"]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

    cs = cell_stats(env, res / "18_cell_stats.npz")
    Dq, Iq = quantizer_topk(env.index, env.xq, C)
    XN = new_static(env, Dq, Iq, env.xq, cs)                         # (10k, 8)

    # =============================== fitted static rule (protocol of 17) ===============================
    ev = a17.Evaluator(env, curves)
    base = ev.X
    ev.X = np.concatenate([base, XN], 1)
    names = a17.NAMES + NEW_STATIC
    idx = {n: i for i, n in enumerate(names)}
    new_ids = [idx[n] for n in NEW_STATIC]
    three = [idx["d32/d1"], idx["d1"], idx["size8"]]
    Xt = ev.X[env.tune]
    print("correlation of each new feature with log(oracle cells at R=0.9), tune:")
    print("  " + ", ".join(f"{n} {np.corrcoef(Xt[:, idx[n]], ev.y[0.9])[0, 1]:+.2f}" for n in NEW_STATIC))
    print("  (for reference d32/d1 {:+.2f}, d1 {:+.2f}, size8 {:+.2f})".format(
        *[np.corrcoef(Xt[:, idx[n]], ev.y[0.9])[0, 1] for n in ["d32/d1", "d1", "size8"]]))
    sets = {"17 base": list(range(17)), "17 base + 8 new": list(range(17)) + new_ids,
            "3 base": three, "3 base + 8 new": three + new_ids}
    for n in NEW_STATIC:
        sets[f"3 base + {n}"] = three + [idx[n]]
    chosen, remaining = [], list(range(len(names)))
    greedy = []
    for step in range(6):
        sc = {i: ev.tune_work(chosen + [i], 0.9) for i in remaining}
        b = min(sc, key=sc.get); chosen.append(b); remaining.remove(b)
        greedy.append(f"{step + 1}: +{names[b]} (tune work {sc[b]:.0f})")
        sets[f"greedy {step + 1}"] = list(chosen)
    print("\ngreedy forward selection on tune over all 25:\n  " + "\n  ".join(greedy))
    rows = []
    for name, cols in sets.items():
        for R in targets:
            r = ev.test_saving(cols, R)
            rows.append({"rule": "fitted", "features": name, "n": len(cols), "target": R,
                         **{kk: v for kk, v in r.items() if kk != "model"}})
    df = pd.DataFrame(rows); df.to_csv(res / "18_fitted.csv", index=False)
    show = lambda d: d.assign(s=d.apply(lambda r: f"{r.saving:.3f} [{r.ci_lo:.2f},{r.ci_hi:.2f}]", axis=1)) \
        .pivot_table(index="features", columns="target", values="s", aggfunc="first", sort=False)
    print("\nfitted rule, tuned saving on test [95% CI]:"); print(show(df).to_string())

    # =============================== stopping rule (protocol of 15) ===============================
    st = static_features(Dq, Iq, env.list_sizes)
    nnr1 = cs["nnr"][Iq[:, 0]]
    feats = {}
    for split, ids in [("tune", env.tune), ("test", env.test)]:
        F = run_rounds(env.index, env.xq[ids], Dq[ids], Iq[ids], st[ids], k, kr)["features"]   # (n, 26, 26)
        Fd = dynamic_extra(env, env.xq[ids], Dq[ids], Iq[ids], nnr1[ids], k, kr)            # (n, 26, 3)
        Fs = np.repeat(XN[ids][:, None, :], F.shape[1], axis=1)                              # (n, 26, 8)
        feats[split] = np.concatenate([F, Fd, Fs], 2)
    # consistency: our r10 must equal the one of run_rounds (feature 0 is log r10/d1)
    d1 = np.sqrt(np.maximum(Dq[env.tune][:, 0], 1e-12))
    chk = feats["tune"][:, :, 26 + 1] + np.log(nnr1[env.tune] / d1)[:, None]             # log r10/d1 via the new feature
    assert np.allclose(chk, feats["tune"][:, :, 0], atol=1e-4), "dynamic_extra does not follow run_rounds' candidate list"
    allnames = [f"f{i}" for i in range(26)] + NEW_DYN + NEW_STATIC
    groups = {"26 base": list(range(26)),
              "26 base + 3 dyn": list(range(29)),
              "26 base + 3 dyn + 8 static": list(range(37)),
              "3 base": [1, 3, 4],
              "3 base + 3 dyn": [1, 3, 4, 26, 27, 28],
              "3 base + 3 dyn + 8 static": [1, 3, 4] + list(range(26, 37))}
    for j, n in enumerate(NEW_DYN):
        groups[f"3 base + {n}"] = [1, 3, 4, 26 + j]
    halves = [np.arange(len(env.tune)) % 2 == 0, np.arange(len(env.tune)) % 2 == 1]
    lag = lagrangian_curve(tune)
    Wb = bootstrap_weights(len(env.test), N_BOOT, seed=env.cfg["seed"])
    rows = []
    for R in targets:
        j = int(np.argmin(np.abs(lag["recall_mean"] - R))); Y = labels_continue(lag["cells"][j].astype(int))
        p = int(np.argmax(tune.recall.mean(0) >= R - 1e-9)) + 1
        wf = test.work[:, p - 1].astype(float) + kr
        for name, cols in groups.items():
            Xtr, Xte = feats["tune"][:, :, cols], feats["test"][:, :, cols]; nf = len(cols)
            P_cv = np.full(Y.shape, np.nan, np.float32)
            for a, b in [(0, 1), (1, 0)]:
                m = fit_logistic(Xtr[halves[a]].reshape(-1, nf), Y[halves[a]].reshape(-1))
                P_cv[halves[b]] = m.predict(Xtr[halves[b]].reshape(-1, nf)).reshape(-1, Y.shape[1])
            rec = np.array([at_cells(tune, cells_from_P(P_cv, t))["recall"].mean() for t in THETAS])
            ok = np.where(rec >= R - 1e-9)[0]; theta = float(THETAS[ok[-1]]) if len(ok) else float(THETAS[0])
            m = fit_logistic(Xtr.reshape(-1, nf), Y.reshape(-1))
            P = m.predict(Xte.reshape(-1, nf)).reshape(len(env.test), -1)
            a_ = at_cells(test, cells_from_P(P, theta)); wk = a_["work"].astype(float) + kr
            bs = (Wb @ wf) / (Wb @ wk)
            rows.append({"rule": "stop", "features": name, "n": nf, "target": R, "saving": wf.mean() / wk.mean(),
                         "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5), "recall": a_["recall"].mean()})
    ds = pd.DataFrame(rows); ds.to_csv(res / "18_stop.csv", index=False)
    print("\nstopping rule, tuned saving on test [95% CI]:"); print(show(ds).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
