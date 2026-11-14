# Work log

Running notes on progress and decisions, newest entry first. One entry per working
session. Each entry has three parts: **Done**, **Decisions** (what was decided and why,
so the other person can object before it is built on), **Next**. Detailed specs live in
`PLAN.md`; this file records the history and the reasoning.

Status board (updated with every entry):

| Area | State |
|---|---|
| §0 Project setup | done |
| §1 Data and index | done (index built, cached in `data/`) |
| §2 Search machinery + tests | done (16 tests pass) |
| §3 Oracle | done (`oracle.py`, 7 tests) |
| E1 fixed-nprobe curve | done (`results/02_fixed.csv`) |
| E2 oracle + histogram | done (`results/03_oracle_*`) |
| E3 ε sweep | not started |
| E4 tune/test, RQ1/RQ2 | not started |
| E5 timing | not started |
| Plots | not started |
| Extensions | not started |

---

## 2026-11-15 — §3 oracle and E2

**Done**
- `oracle.py`: `compute_curves` scans each query's 256 nearest cells one at a time
  (`search_preassigned`, width 1, k = 200), computes the exact distance of every PQ candidate
  once, merges the candidate lists cumulatively by PQ distance, re-ranks the running top-200
  exactly and records, for every c = 1..256: `recall_q(c)` (re-ranked), raw PQ recall,
  **coverage** (share of true top-10 whose cell is among the first c) and cumulative work.
  Both oracles: `per_query_target` / `per_query_target_curve` (variant 1) and
  `lagrangian_curve` (variant 2, 300 λ values, log-spaced 1e1..1e8). All 10k queries in 177 s;
  arrays cached in `results/03_oracle_curves.npz` (~5 MB each).
- Tests (7 new, 23 total): oracle recall at c equals real `index.search(nprobe=c)` +
  re-ranking per query for c ∈ {1, 4, 32} (also raw PQ recall); work equals Σ list sizes;
  recall ≤ coverage; coverage and work monotone; per-query target is the first crossing;
  Lagrangian curve never needs more work than fixed nprobe at the same mean recall.
- E2 (`experiments/03_oracle.py`, test split). The fixed-nprobe curve derived from the
  oracle arrays reproduces E1 exactly (recall and ndis identical at all 31 grid points),
  which is the end-to-end check that the cumulative merge is right.

**Results (test split, 5000 queries)**

Per-query-target oracle, cells needed per query:

| target | unreachable | median | mean | p90 | p99 | need 1 cell | need ≤ 3 | need > 20 |
|---|---|---|---|---|---|---|---|---|
| 0.8 | 2 | 8 | 13.7 | 31 | 80 | 6.0% | 22.0% | 19.6% |
| 0.9 | 9 | 13 | 21.2 | 50 | 126 | 3.6% | 14.3% | 33.1% |
| 1.0 | 139 | 23 | 36.3 | 87 | 191 | 1.9% | 7.8% | 53.8% |

(With k = 10 per-query recall moves in steps of 0.1, so targets 0.95 and 0.99 are the
same as 1.0 for a single query: all ten neighbours. Report them under 1.0.)

Work (ndis) needed at each mean-recall target and the largest possible saving:

| R | fixed nprobe | Lagrangian oracle | per-query oracle | max saving (Lagr.) | max saving (per-query) |
|---|---|---|---|---|---|
| 0.80 | 3422 | 1924 | 3049 | 1.71× | 1.12× |
| 0.90 | 6707 | 3290 | 5063 | 1.98× | 1.31× |
| 0.95 | 11508 | 4774 | 7293 | 2.35× | 1.56× |
| 0.99 | 31504 | 7839 | 10176 | 3.94× | 3.06× |

(saving = (ndis_fixed + 200) / (ndis_oracle + 200), i.e. work incl. re-ranking.)
Coverage at 256 cells 0.9991, recall 0.9964: with re-ranking the PQ ceiling is ~0.3%.

**What this says about the proposal's predictions.**
- "Most queries need 1–3 cells" is **wrong for SIFT1M at nlist = 4096**: at target 0.9 only
  14% need ≤ 3 cells, the median needs 13, a third need more than 20. Queries are "hard"
  far more often than assumed: with ~244 vectors per cell, a query's 10 true neighbours are
  spread over many cells.
- Consequently the oracle saves ~2× at R = 0.9, not "several times". The saving does grow
  with the target (1.7× → 3.9×), as predicted.
- The gap between the two oracles is large (1.98× vs 1.31× at 0.9): guaranteeing 0.9 to
  *every* query costs much more than reaching 0.9 on average. The proposal's oracle
  understates the possible saving; the Lagrangian one is the right ceiling for RQ2 (D2).
- Prediction "the rule gets about half of the saving, 1.5–2× fewer scanned vectors" now
  has a ceiling of 1.98× at R = 0.9; a 1.5× result would mean capturing ~50% of the
  Lagrangian saving. E3/E4 will tell.

**Next**
- E3 ε sweep (`experiments/04_rule.py`), E4 tune/test protocol with bootstrap, `plots.py`
  for plots 1–3. Open: whether to add nlist = 1024 early, since the cell-size argument above
  suggests the histogram shape depends strongly on nlist (extension 3).

---

## 2026-11-14 (night) — D5 closed; §2 search machinery, tests, E1

**Done**
- **D5 closed** (see `PLAN.md`): recall@10 is measured after exact re-ranking of the top-200
  PQ candidates; work = ndis + 200. `k_rerank: 200` added to the config.
- `rules.py`: `quantizer_topk`, `ratio_cells` (D4: (1+ε)² on squared distances, clamp, `<=`),
  `fixed_cells`, `n_cells`. `search.py`: `fixed_search`, `preassigned_search` (queries
  grouped by cell count, one `search_preassigned` call per distinct count, no padding),
  `rerank_exact` (vectorised NumPy, chunks of 1000 queries), `scanned_vectors`, Faiss stats
  helpers. `metrics.py`: per-query recall, log-interpolation of work at a recall target.
  `env.py`: one loader for config + vectors + ground truth + split + index.
- Tests (`uv run pytest`, 16 pass in ~1 s, on 500 tune queries so the test split stays
  untouched): constant-c preassigned search == `index.search(nprobe=c)` for c ∈ {1, 4, 32}
  with bit-identical distances; mixed widths in one batch each match their own nprobe;
  re-ranking inside one small cell == exact search in that cell; Faiss `ndis` == Σ list
  sizes on both paths; ε = 0 → one cell; monotone in ε; squared == plain-distance rule;
  kept cells form a prefix.
- **E1** (`experiments/02_fixed.py`, test split, 5000 queries, 31 nprobe values):

  | nprobe | recall (re-ranked) | raw PQ recall | mean ndis | p99 ndis |
  |---|---|---|---|---|
  | 1 | 0.285 | 0.256 | 279 | 772 |
  | 4 | 0.573 | 0.446 | 1114 | 2632 |
  | 13 | 0.809 | 0.547 | 3593 | 7153 |
  | 23 | 0.892 | 0.570 | 6295 | 11201 |
  | 28 | 0.914 | 0.576 | 7634 | 13142 |
  | 48 | 0.957 | 0.584 | 12927 | 20175 |
  | 120 | 0.990 | 0.589 | 31535 | 44460 |
  | 256 | 0.996 | 0.589 | 65641 | 84495 |

  Interpolated: R = 0.8 at nprobe ≈ 12.4, R = 0.9 ≈ 25, R = 0.95 ≈ 43, R = 0.99 ≈ 120.
  Per-query recall and ndis saved in `results/02_fixed_perquery.npz` for the bootstrap.

**Observation worth knowing.** Faiss's own `ndis` counter and Σ list sizes of the
`quantizer.search` cells disagree for a handful of queries at some nprobe (e.g. −41 vectors
at nprobe 12 out of 16.6M). Cause: query 3951 has its 12th and 13th centroids at relative
distance 1.8e-6 and the quantizer search inside `index.search` rounds differently from a
standalone `quantizer.search` (different BLAS blocking). Not a bookkeeping bug; the E1 script
asserts agreement to 1e-4 and stores the gap per nprobe (`faiss_ndis_gap`). The rule path
has no such issue because it passes the cells explicitly.

**Next**
- §3 oracle: per-cell top-200 candidates via width-1 `search_preassigned`, cumulative merge,
  re-rank, `recall_q(c)` / `work_q(c)` arrays; both oracle variants (D2). Then E2, E3, E4.

---

## 2026-11-14 (evening) — §1 done; recall ceiling of PQ16 found

**Done**
- `config.py` (YAML + paths), `data.py` (fvecs/ivecs, download, ground truth, tune/test
  split), `index.py` (build on full base, cache, cell ids + list sizes),
  `experiments/01_build.py`. Output in `results/01_build.json`.
- SIFT1M downloaded from the irisa FTP (168 MB). Ground truth recomputed with
  IndexFlatL2, k = 100, in 7 s: 9944/10000 queries have the same top-10 set as the shipped
  file, all 56 differences are exact ties on the 10th neighbour. Split: 5000 tune / 5000 test,
  seed 0, saved to `data/sift1m/query_split_seed0.npz`.
- Index `IVF4096,PQ16x8` trained on the full 1M base (D3) in 71 s, add 9 s. List sizes:
  mean 244, median 225, min 59, p99 558, max 1292, no empty lists. Cell ids checked
  against `quantizer.search` on a 20k sample: identical.

**Finding: the PQ16x8 recall@10 ceiling is ≈ 0.59.** On the test split (fixed nprobe, raw
PQ ranking, no re-ranking):

| nprobe | PQ16 R@10 | PQ32 R@10 | PQ64 R@10 | IVFFlat R@10 | 10-in-top-100 (PQ16) |
|---|---|---|---|---|---|
| 1 | 0.256 | 0.274 | 0.283 | 0.285 | 0.285 |
| 4 | 0.446 | 0.515 | 0.557 | 0.573 | 0.573 |
| 16 | 0.557 | 0.681 | 0.785 | 0.842 | 0.839 |
| 64 | 0.586 | 0.733 | 0.871 | 0.973 | 0.963 |
| 256 | 0.589 | 0.740 | 0.884 | 0.999 | 0.984 |

(PQ32/PQ64/IVFFlat share the same coarse quantizer and cells as PQ16; built in ~10 s each.)
So with the index from the proposal, the targets R ∈ {0.8, 0.9, 0.95, 0.99} cannot be
reached by any cell-selection method: above nprobe ≈ 16 the misses are PQ error, not
skipped cells. The last column (share of true top-10 among the first 100 PQ candidates)
tracks IVFFlat almost exactly, i.e. it measures cell coverage with PQ noise removed.

**Open question Q1 (needs both of us): what to measure recall on.** Options:
1. Keep IVF4096,PQ16x8 and report recall@10 **after exact re-ranking of the top-100 PQ
   candidates** (IVFADC+R, as in the seminar paper's follow-up). Ceiling ≈ 0.98, so R = 0.99
   is still out of reach; PQ32 + re-rank reaches 0.999. The 100 exact distances per query
   must be counted as work (a constant offset on top of scanned vectors).
2. Switch to IVF4096,Flat (no PQ). Cleanest for the cell-selection question, recall reaches
   1.0, no "unreachable" bin in the oracle. But the proposal and seminar topic say IVF+PQ.
3. Keep PQ16 and raw PQ ranking, and move the targets down to the reachable range
   (e.g. 0.4, 0.5, 0.55, 0.58). Honest but the numbers look odd and the top target sits on
   a plateau where every method needs many cells.
Measured (2000 test queries, single thread, same cells for every column):

| nprobe | ndis | PQ16 raw | PQ16 + rerank 100 | PQ16 + rerank 200 | PQ32 + rerank 100 | IVFFlat |
|---|---|---|---|---|---|---|
| 4 | 1109 | 0.431 | 0.554 | 0.554 | 0.554 | 0.554 |
| 16 | 4398 | 0.547 | 0.829 | 0.832 | 0.832 | 0.832 |
| 32 | 8685 | 0.569 | 0.915 | 0.921 | 0.922 | 0.923 |
| 64 | 17111 | 0.577 | 0.958 | 0.968 | 0.969 | 0.970 |
| 128 | 33626 | 0.580 | 0.977 | 0.990 | 0.992 | 0.993 |
| 256 | 65756 | 0.581 | 0.982 | 0.996 | 0.998 | 0.999 |

After exact re-ranking, recall no longer depends on the PQ: PQ16 + rerank 200, PQ32 +
rerank 100 and IVFFlat agree to the third decimal. Re-ranked recall is cell coverage, so
the choice between them barely affects RQ1/RQ2; it affects cost and framing only.
Rerank 100 on PQ16 is too few candidates at high nprobe (0.982 at 256). IVFFlat is fastest
at small nprobe but ~2x slower than PQ16 at nprobe 256 (512 vs 16 bytes per vector).

Suggested: keep the proposal's index IVF4096,PQ16x8, main recall = recall@10 after exact
re-ranking of the top-200 PQ candidates (IndexRefineFlat in Faiss), work = scanned vectors
+ a constant 200 exact distances per query (measured separately in E5). Keep all four
targets. Show raw-PQ recall as a second curve: it is the PQ ceiling of extension 5 for free.
Either way the three methods still differ only in the choice of cells.

**Next**
- Decide Q1, then §2: fixed-nprobe path with ndis counting, `search_preassigned` path,
  tests, E1.

---

## 2026-11-14 — closed the four open decisions in PLAN.md

**Done**
- Reviewed `PLAN.md` against `proposalMD (1).md`. No contradictions. The plan goes beyond
  the proposal in: a dense nprobe grid up to 256 instead of powers of two up to 128, a second
  (Lagrangian) oracle, an explicit `max_cells` cap, bootstrap CIs, timing split by
  component, a rule-vs-oracle diagnostic. These are refinements, not changes of question.
- Closed the four items that were listed under "Open decisions". They are now D1–D4 at the
  end of `PLAN.md`. Config updated (`train_size: null`).

**Decisions**

- **D1. `max_cells` = 256 everywhere** (rule cap, oracle C_max, top of the nprobe grid).
  Why: the rule needs a K for the quantizer search and a cap on cells per query, otherwise
  a hard query at large ε scans hundreds of cells. Using one number for all three methods
  means they choose from the same range 1..256, which keeps RQ2 fair. 256 is a convenience
  choice, not a data-driven one, so E3 records the share of queries hitting the cap; if it
  is more than a few percent at the ε picked for R = 0.99, we rerun with 512/1024. The
  proposal says ε is the only parameter; the report has to name the cap as a second one.

- **D2. Compute both oracles, lead with the Lagrangian one.** The proposal's oracle
  (smallest c per query with recall_q ≥ t) answers "cost of a guarantee per query"; its
  mean recall is above t, so its point on plot 1 is not on the R = t line. The Lagrangian
  oracle (each query picks c minimising work − λ·recall, sweep λ) is the true lower bound on
  mean work at a given mean recall, which is the metric on the axis. Using the proposal's
  oracle as the RQ2 denominator would overstate the captured share. So: Lagrangian gives the
  headline RQ2 number and the solid line; the per-query oracle gives a second RQ2 column, the
  dashed line, the histogram (plot 2) and the rule-vs-oracle diagnostic (plot 3). Both come
  from the same `recall_q(c)`, `work_q(c)` arrays, so no extra search cost. The gap between
  the two curves is itself a result (price of a per-query guarantee).

- **D3. Train the index on the full 1M base set.** Options were the 100k learn set, a 256k
  base subsample (plan's original proposal), or the full base. The learn set is below
  Faiss's 39 × nlist guideline (24 points per centroid). 256k vs 1M: Faiss itself caps
  k-means at 256 points per centroid, so 1M is within its own default; PQ training is
  subsampled to 65 536 residuals by Faiss regardless, so only the coarse quantizer is
  affected; 1M also covers nlist = 16384 (extension 3) without a separate setting; the cost
  is a few minutes once, cached in `data/`. Training on indexed data is not a leak: queries
  are a separate file and never enter training; it is also what every production index does.
  The build script keeps an optional subsample size for quick debug builds.

- **D4. ε is defined on plain L2 distances**, as in the proposal's example (100 → 110 at
  ε = 0.1). Faiss returns squared distances, so the code tests `D_j ≤ (1+ε)² · D_1`, which
  is exactly equivalent. Implementation notes: clamp `D_1` at 0 (Faiss's BLAS path can return
  tiny negatives), use `≤` so ε = 0 gives one cell unless ties, add a test against
  `sqrt(D)` with factor (1+ε). Side finding for extension 4: the "normalised gap"
  `(d_j − d_1)/d_1 ≤ δ` is identical to the ratio rule, so it was dropped; the gap rule is
  the absolute one on plain distances (differences of squares are not equivalent).

**Next**
- §1: `data.py` (fvecs/ivecs, download, tune/test split), `index.py` (build on full base,
  cache to `data/`, cell ids and list sizes), `experiments/01_build.py`.
- §2: fixed-nprobe path with ndis counting, then `search_preassigned` path, then the three
  tests (equivalence, ndis, rule sanity). E1 after that.
