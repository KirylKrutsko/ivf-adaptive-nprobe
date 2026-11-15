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
| §3 Oracle | done (`oracle.py`, 7 tests; 27 tests total) |
| E1 fixed-nprobe curve | done (`results/02_fixed.csv`) |
| E2 oracle + histogram | done (`results/03_oracle_*`) |
| E3 ε sweep | done (`results/04_rule*`) |
| E4 tune/test, RQ1/RQ2 | done (`results/05_*`), **rule loses to fixed nprobe** |
| E5 timing | not started |
| Plots | plots 1–5 done (`results/plots/`) |
| Side exp.: rule capped at nprobe | done (`results/06_*`, plot 5) |
| Extensions | not started |

---

## 2026-11-15 (evening) — side experiment: the rule as a drop-in, min(rule(ε), nprobe)

**Question.** Keep the usual nprobe for hard queries and let the rule only *remove* cells from easy
ones: each query scans `min(n_rule(ε), nprobe)` cells. How much work/time does it save, and how
much recall does it lose? This is a different view of the capped rule (cap C = nprobe), not
the fixed-recall comparison of E4.

**Done**
- `experiments/06_rule_within_nprobe.py`: for the E4-tuned nprobe values {13, 26, 45, 122}
  and every ε of the E3 grid, on the test split: recall, work, recall drop and work saving vs
  plain nprobe, and the **control**: work plain IVF needs for the *same* recall by lowering
  nprobe for everyone (`frontier_gain` = that work / rule work; > 1 means the rule beats just
  lowering nprobe). Plot 5 (`results/plots/5_rule_within_nprobe.png`): ε traces vs fixed curve.
- `experiments/06_rule_within_nprobe_timing.py`: wall clock, single thread, median of 5, re-ranking
  included, at the SAME nprobe (so the rule's recall is slightly lower).

**Results (test split; work = ndis + 200)**

At ε = 0.238 / 0.286 (tuned for the uncapped / cap-32 rule in E4, far too aggressive here):

| nprobe (recall) | ε | recall | drop | work saving | queries trimmed | frontier_gain |
|---|---|---|---|---|---|---|
| 13 (0.809) | 0.238 / 0.286 | 0.778 / 0.794 | 0.031 / 0.015 | 1.25× / 1.16× | 34% / 23% | 1.06 / 1.07 |
| 26 (0.907) | 0.238 / 0.286 | 0.855 / 0.882 | 0.052 / 0.025 | 1.50× / 1.32× | 54% / 39% | 1.03 / 1.09 |
| 45 (0.953) | 0.238 / 0.286 | 0.886 / 0.921 | 0.068 / 0.033 | 1.88× / 1.53× | 71% / 54% | 0.94 / 1.03 |
| 122 (0.990) | 0.238 / 0.286 | 0.903 / 0.947 | 0.087 / 0.043 | 3.51× / 2.41× | 94% / 84% | 0.77 / 0.85 |

The large raw savings are mostly "fewer cells", not "better cells": at nprobe 122 lowering
nprobe for everyone is cheaper than the rule at the same recall.

Best frontier_gain over the ε grid (**ε chosen on test, so slightly optimistic**):

| nprobe | ε | recall drop | work saving | frontier_gain | time, vs plain at same nprobe |
|---|---|---|---|---|---|
| 13 | 0.329 | 0.007 | 1.12× | 1.07 | – |
| 26 | 0.329 | 0.012 | 1.22× | 1.11 | 1.08× |
| 45 | 0.378 | 0.007 | 1.25× | 1.13 | 1.10× |
| 122 | 0.547 | 0.001 | 1.20× | 1.17 | 1.12× |

Largest work saving with recall drop ≤ 0.01: 1.13× (nprobe 13), 1.20× (26), 1.28× (45), 1.60× (122);
frontier_gain of those points 1.07 / 1.11 / 1.13 / 1.06.

**Reading**
- Same method family as the capped rule of the previous entry (`min(rule(ε), C)`), seen from the
  other side: there the recall target was fixed and (ε, C) tuned on tune; here C = nprobe and ε is
  swept. Both reach the same envelope: frontier gain 1.07 / 1.11 / 1.13 / 1.17 here vs capped-rule
  savings 1.05 / 1.07 / 1.12 / 1.18 (tuned on tune: **quote those** in the report). The tuned
  caps (16 / 32 / 48 / 128) are close to the nprobe for each target (13 / 26 / 45 / 122), which is
  why the two agree.
- The rule only reliably identifies *clearly easy* queries: the useful ε (0.33–0.55) trims just
  17–32% of queries, and the useful ε grows with nprobe. In plot 5 each trace rises slightly
  above the fixed curve just before its nprobe point and falls below it as ε shrinks.
- Time saving < work saving (1.22× work → 1.08× time at nprobe 26), as the proposal predicted.
  Likely cause: per-query costs that do not shrink with cells (4096-centroid quantizer, 200-candidate
  re-rank, Python grouping); not yet measured separately (E5). The preassigned path itself costs
  the same as `index.search` (0.140 vs 0.141 ms/query). The matched-recall comparison exists only
  in work; it was not timed.

**Next**
- If this framing goes in the report: choose ε on tune (per nprobe), not on test.
- E5: split per-query time into quantizer / rule / scan / re-rank to see what limits the time saving.

---

## 2026-11-15 (later) — E3 ε sweep, E4 tune/test, plots 1–4

**Done**
- `rules.ratio_n_cells`, `oracle.at_cells`, `metrics.bootstrap_weights`.
- **E3** (`experiments/04_rule.py`): the rule keeps a prefix of the nearest cells, so its result
  at any ε is read off the E2 curves, with no search per ε. 151 ε values (0 plus 150 geometric
  steps in 1e-3..1). Checked against the real path (`search_preassigned` + re-rank) on all 5000
  test queries at ε ∈ {0.02, 0.06, 0.15}: 0 recall mismatches, 0 ndis mismatches, Faiss ndis equal.
  New test `tests/test_rule_curves.py` (4 tests) checks the same on 500 tune queries.
- **E4** (`experiments/05_tune_test.py`): nprobe (1..256) and ε tuned on tune, reported on test;
  curve-based numbers at exactly R; captured share against both oracles; bootstrap 95% CIs
  (1000 resamples of test queries, tune choices fixed).
- `experiments/plots.py`: plots 1–4 (recall vs work, cells-needed histogram, rule vs oracle
  2-D histogram, saving vs target).

**Results (test split): the ratio rule needs MORE work than fixed nprobe**

| R | nprobe / ε (tuned) | saving, tuned [95% CI] | saving, curve [95% CI] | oracle max | captured (Lagr.) |
|---|---|---|---|---|---|
| 0.80 | 13 / 0.180 | 0.77× [0.74, 0.79] | 0.79× [0.77, 0.81] | 1.71× | −65% |
| 0.90 | 26 / 0.238 | 0.73× [0.71, 0.75] | 0.72× [0.70, 0.74] | 1.98× | −79% |
| 0.95 | 45 / 0.286 | 0.76× [0.74, 0.79] | 0.73× [0.70, 0.75] | 2.35× | −66% |
| 0.99 | 122 / 0.414 | 0.93× [0.91, 0.94] | 0.93× [0.88, 0.97] | 3.94× | −10% |

(saving = work_fixed / work_rule including the 200 re-ranked; < 1 means the rule is worse.)
Test recall of the tuned choices is within ±0.004 of R for both methods. At R = 0.99 the rule
hits the 256 cap for 28% of queries (D1 threshold "a few percent" exceeded, see Next).

**Why (diagnostics, scripts not committed; numbers at the ε tuned for R = 0.9):**
- The rule *ranks* queries well: Spearman(rule cells, oracle c*(0.9)) = 0.70 (plot 3). Giving the
  same cell counts to random queries is far worse (23 343 vs 9 399 ndis at R = 0.9), so the rule
  carries real information. Cell sizes are not the issue (mean size 254 vs 273 for fixed).
- It is **mis-calibrated**: mean 39 cells, p99 239, where fixed needs 26 for all. The count
  grows explosively for queries with a flat centroid-distance profile (Spearman with d10/d1 =
  −0.96), and the hardest queries soak up hundreds of cells at little recall per vector. The
  Lagrangian oracle does the opposite: it gives up on expensive queries.
- Exploration: **capping** the rule (cap tuned on tune from {8..256}, ε re-tuned) turns it
  positive: 1.05× (R 0.8, cap 16), 1.07× (0.9, cap 32), 1.12× (0.95, cap 48), 1.18× (0.99,
  cap 128) on test. Still far below the oracle.

**Against the proposal's predictions:** the rule does not save 1.5–2×; uncapped it costs
7–39% more work than fixed nprobe; capped it saves 5–18%. The saving does grow with the target.

**Next**
- Decide how to present this (both of us): the headline is a negative result for the plain rule
  as proposed. Candidates: (a) make the cap part of the rule (two parameters, tuned on tune)
  and report it as "rule + cap"; (b) extension 6 (map the rule's score to cells via a lookup
  fitted on tune), since the ranking is good and only the mapping is wrong; (c) extension 3
  (nlist 1024), where cells are bigger and more queries are easy.
- Move the diagnostics (shuffled control, cap sweep, feature correlations) into a proper script.
- E5 timing; it matters less now that the rule does not save scanned vectors.

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
  oracle arrays reproduces E1 (ndis identical, recall equal to 1e-5 = float32 rounding, at all 31 grid points),
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
