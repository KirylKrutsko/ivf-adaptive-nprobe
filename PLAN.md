# Implementation & experiment plan — query-adaptive nprobe for IVF+PQ

Based on `proposal.pdf`. Rule under test: per query, scan every cell whose centroid
distance is ≤ (1 + ε) × the nearest centroid distance.

---

## 0. Project setup

- Bump Python to 3.11 or 3.12 (`.python-version` is 3.9; current `faiss-cpu` wheels for
  macOS arm64 target newer Pythons).
- Rename the package from `pythonproject` to `ivfadapt` and delete the PyCharm sample `src/main.py`.
- Dependencies: `faiss-cpu`, `numpy`, `matplotlib`, `pandas` (result tables), `pyyaml`, `tqdm`; dev: `pytest`.
- Add `data/` and `results/` to `.gitignore`. Commit the code and configs, not the data.

### Layout

```
src/ivfadapt/
  data.py      # fvecs/ivecs readers, SIFT1M download, fixed tune/test query split
  index.py     # build / train / save / load the IVF-PQ index (cached to data/)
  rules.py     # cells_ratio(D, eps, max_cells), later cells_gap(...)
  search.py    # fixed_search, preassigned_search (per-query cell lists), ndis counting
  oracle.py    # per-query recall-vs-cells curves, oracle curves
  metrics.py   # recall@k, interpolation at matched recall, "share of saving"
  timing.py    # single-thread timing, per-query latency
experiments/
  01_build.py  02_fixed.py  03_oracle.py  04_rule.py  05_tune_test.py  06_timing.py
  plots.py
tests/
  test_equivalence.py  test_ndis.py  test_oracle.py
configs/sift1m.yaml    # nlist, M, nbits, seeds, grids
```

Each experiment script writes a tidy CSV/NPZ to `results/`. `plots.py` only reads those
files, so a plot can be redrawn without rerunning a search.

---

## 1. Data and index (week 1)

1. **SIFT1M**: download from `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`
   (base 1M, query 10k, learn 100k, groundtruth). Read the `.fvecs`/`.ivecs` files with NumPy.
2. **Ground truth**: brute-force with `faiss.IndexFlatL2` (k = 100). Check that it matches
   the shipped `sift_groundtruth.ivecs` on the top 10 (ties may differ slightly).
3. **Index**: `IndexIVFPQ(IndexFlatL2(128), 128, nlist=4096, M=16, nbits=8)`.
   - Training: train on the **full 1M base set** (decision D3). The 100k learn set is below
     Faiss's suggested 39 × 4096 ≈ 160k points; the full base gives ≈ 244 points per
     centroid, within Faiss's own default cap of 256, and is enough for nlist = 16384 too
     (≈ 61 per centroid). Faiss subsamples PQ training to 65 536 residuals by itself, so the
     choice only affects the coarse quantizer. `train_size: null` in the config means the
     full base; a number is allowed for quick debug builds only.
   - Save the trained index to `data/` (`faiss.write_index`) so it is built only once.
   - Save the cell id of every base vector (from the invlists) and the list sizes.
     Steps 3–5 need both.
4. **Query split**: use a fixed seed to split the 10k queries into a 5k **tune** set and a
   5k **test** set. Every reported number comes from the test set.

## 2. Search machinery (week 1–2)

### Fixed nprobe (baseline)
- `index.nprobe = p; index.search(xq, 10)`. Use a dense grid, not only powers of 2:
  every integer 1..16, then about 1.2× steps up to 256. Matched-recall comparisons
  interpolate along this curve, so a coarse grid skews them.
- Work: reset `faiss.cvar.indexIVF_stats` before each run and read `.ndis`. Also compute
  Σ list_size over the probed cells. The two must agree.

### Per-query cells (our rule): this handles the "main risk"
Faiss already supports per-query cell lists:
1. `Dq, Iq = index.quantizer.search(xq, K)` with K = `max_cells` = 256.
   The distances are **squared** L2.
2. Rule mask: `Dq[:, j] <= (1+ε)² · max(Dq[:, 0], 0)` (decision D4). ε is defined on plain
   distances, exactly as in the proposal; squaring the factor is the equivalent test on
   squared distances. Clamp `Dq[:, 0]` at 0 because Faiss's BLAS L2 path can return tiny
   negative values. Use `<=` so ε = 0 gives one cell unless distances tie.
3. Set masked-out entries of `Iq` to `-1`, cut the matrix to the widest row in the batch,
   set `index.nprobe` to that width, and call `index.search_preassigned(xq, 10, Iq, Dq)`.
   Faiss skips list ids < 0, so each query scans only its own cells.
4. To keep padding small, sort queries by cell count and run them in groups.
   This is the same idea as the proposal's fallback, used here to save time rather than to work around a limitation.

`max_cells` caps the rule and is a second parameter. Decision D1: fix it at 256 for the
rule, the oracle (C_max) and the top of the nprobe grid, so all three methods choose from
the same range 1..256 cells. Report it, and in E3 record the share of queries that hit
the cap at the ε values chosen in E4; if that share exceeds a few percent at R = 0.99,
rerun E2–E3 with 512 or 1024 and show whether the conclusions change.

### Correctness tests (`tests/`)
- **Equivalence**: a "rule" that always returns c cells, run through `search_preassigned`,
  must give exactly the same labels and distances as `index.search` with `nprobe=c`.
  Test c ∈ {1, 4, 32}.
- **ndis**: the scanned-vector count from `indexIVF_stats.ndis` must equal Σ list sizes of
  the selected cells, for both the fixed and the adaptive path.
- **Rule sanity**: ε = 0 → exactly 1 cell per query (unless ties). Cell count is monotone in ε.
- **Oracle**: the cumulative-merge recall at c cells must equal real search recall at
  `nprobe=c`. Check a few values of c (see below).

## 3. Oracle (week 2)

### Computing per-query curves cheaply
For IVF-PQ with residual encoding, the PQ distance of a vector depends only on the query
and its own cell, not on how many cells are scanned. So:
- for each query q and rank c = 1..C_max, scan **only** its c-th nearest cell
  (`search_preassigned` with width 1) and keep the top-10;
- merge these lists cumulatively. The top-10 after c cells equals the search result with nprobe = c.

This yields, for every query, `recall_q(c)` and `work_q(c)` (cumulative scanned vectors) for
c = 1..C_max, in about the cost of one search at nprobe = C_max. Store them as
`(n_queries, C_max)` arrays. The oracle, the histogram and the analysis of rule vs oracle all use these arrays.

### Two oracle variants (compute and report both; decision D2)
1. **Per-query target (as in the proposal)**:
   `c*_q(t)` = smallest c with `recall_q(c) ≥ t`, for t ∈ {0.8, 0.9, 1.0}.
   Some queries never reach t even at C_max, because PQ error caps their recall.
   Report how many these are, as a separate "unreachable" bin.
   Note that its mean recall is above t (many queries jump straight to 1.0), so its point
   on plot 1 is not on the R = t line.
2. **Average-recall oracle (Lagrangian)**: requiring every query to reach
   recall 0.9 is *not* the cheapest way to get mean recall 0.9. The best allocation follows
   from a Lagrangian sweep. For each λ, every query independently picks
   `c = argmin_c work_q(c) − λ · recall_q(c)`. Sweeping λ traces the best possible
   (mean work, mean recall) curve for any per-query cell count. Variant 1 lies to its right.
   It may give a hard query 1 cell and low recall when that is cheaper for the mean: say so
   in the report.

How they are used:
- **Headline RQ2 number and the solid oracle line on plot 1: Lagrangian.** It matches the
  metric on the axis (mean recall), so "captured share" is measured at the same recall.
  Using variant 1 as the denominator would overstate the captured share.
- **Also report RQ2 against variant 1** as a second column in the E4 table.
- **Histogram (plot 2) and the rule-vs-oracle diagnostic (plot 3): variant 1**, because
  these need a cell count per query at a fixed target.
- The gap between the two oracle curves is a result in itself: the cost of a per-query
  recall guarantee compared with a guarantee on the mean.
- State in the report that the proposal described variant 1 and why the headline uses 2.

## 4. Experiment procedure

### E1: Fixed-nprobe curve (week 1)
Test set: recall@10 and mean ndis for each nprobe in the grid. Also store per-query cells and ndis.

### E2: Oracle and histogram (week 2)
Compute `recall_q(c)`, `work_q(c)` with C_max = `max_cells` = 256 (see D1 for when to
raise it). Outputs: the histogram of `c*_q(0.9)` and `c*_q(1.0)` (log-x, with the
unreachable bin), the Lagrangian oracle curve and the per-query-target oracle curve.

### E3: Rule sweep (week 2)
ε grid: start dense near 0. In 128-d space centroid distances concentrate, so the useful
ε may be small, around 0.01–0.3. A pilot with geometric steps from 0.002 to 1 (about 30
values) finds the range, then refine it. For each ε record mean recall, mean ndis, and the
cell distribution (mean, median, p99, share hitting `max_cells`).

### E4: Tune/test protocol (RQ1 + RQ2 headline numbers)
For each recall target R ∈ {0.8, 0.9, 0.95, 0.99}:
- on **tune**: the smallest nprobe with recall ≥ R, and the smallest ε with recall ≥ R;
- on **test**: report recall and mean ndis for both choices. The baseline is tuned the
  same way, so the comparison is fair.
- **RQ1**: `saving = ndis_fixed / ndis_rule` at matched recall. Also give the curve-based
  version, with ndis log-interpolated at exactly R on both test curves.
- **RQ2**: `captured = (ndis_fixed − ndis_rule) / (ndis_fixed − ndis_oracle)`, with the
  Lagrangian oracle as the headline and the per-query-target oracle as a second column.
- Uncertainty: a bootstrap over test queries (1000 resamples) for saving and captured.
  This is cheap because all values are means of per-query numbers.

### E5: Timing (week 3)
- `faiss.omp_set_num_threads(1)`; warm-up run; median of 5 repeats.
- Measure the components separately: coarse quantizer, rule (NumPy masking), list scanning.
  Run the baseline through the same `quantizer.search + search_preassigned` path as well
  as plain `index.search`. This separates the cost of the rule from the cost of the path.
- Report QPS at matched recall alongside the ndis saving. This checks the prediction that
  the time saving is smaller than the ndis saving.

### Diagnostics (cheap, high value)
- Scatter / 2-D histogram: the rule's cell count against the oracle `c*_q` per query, with
  Spearman correlation. This shows *why* the rule captures more or less of the possible saving.
- **Cell coverage / PQ ceiling** (extension 5, nearly free): the share of true top-10
  neighbours whose cell is among the probed cells. Plot it next to recall. The gap between
  the two curves is PQ error, which no cell-selection rule can fix.

## 5. Plots
1. Recall@10 vs mean scanned vectors (log-x): fixed, rule, Lagrangian oracle (solid), and
   the per-query-target oracle (dashed), with the legend stating what each oracle
   guarantees. Mark the R = 0.9 line.
2. Histogram of oracle cells needed (log-x bins, unreachable bin).
3. Rule cells vs oracle cells (2-D histogram).
4. Saving factor vs recall target (shows whether "the saving grows with the target" holds).
5. QPS vs recall (timing).

## 6. Extensions (week 3, in proposal order)
Each one reuses the same pipeline with a different config:
1. Second dataset: Deep1M (96-d) or GloVe-100 (use inner product or normalise vectors and
   use L2, and decide which). Only `data.py` and the config change.
2. Tail latency: per-query time (batch size 1, subtract Python overhead measured on an
   empty call), p50 / p99 for fixed and rule at matched mean recall.
3. nlist ∈ {1024, 4096, 16384}: rebuild the index, trained on the full base as in D3.
4. Gap rule: scan cell j if `d_j − d_1 ≤ δ` on **plain** distances (take `sqrt` of `Dq`;
   differences of squares are not equivalent: `d_j² − d_1² = (d_j − d_1)(d_j + d_1)`).
   Do not use `(d_j − d_1)/d_1 ≤ δ` as a variant: it is identical to the ratio rule with
   ε = δ. If a normalised gap is wanted, normalise by `d_2 − d_1` or by the mean distance
   to the K nearest centroids. Compare with the ratio rule at matched recall and by
   correlation with the oracle.
5. PQ ceiling: already covered by the coverage diagnostic.
6. Rule fitted to the oracle: features from `Dq` (d2/d1, d10/d1, …) → predicted `c*`.
   Fit a lookup table on tune and evaluate on test.

## 7. Schedule mapping
| Week | Deliverable |
|---|---|
| 1 | §0, §1, fixed-nprobe path + equivalence/ndis tests, E1 |
| 2 | Oracle (§3), rule, E2–E4, plots 1–3, **minimal result before pre-defense** |
| 3 | E5 timing, diagnostics, extensions 1→… |
| 4 | Bootstrap CIs, write-up, slides |

## Decisions (closed 2026-11-14)
- **D1. `max_cells` = 256** for the rule, for the oracle's C_max and as the top of the nprobe
  grid, so every method picks from the same range of 1..256 cells. Report the share of
  queries hitting the cap; raise to 512/1024 and recheck if it exceeds a few percent at the
  ε chosen for R = 0.99. Named in the report as a second parameter of the rule.
- **D2. Both oracles are computed and shown.** Lagrangian oracle: headline RQ2 number and
  solid line on plot 1, because it matches the mean-recall metric. Per-query-target oracle:
  dashed line on plot 1, second RQ2 column, histogram (plot 2) and rule-vs-oracle
  diagnostic (plot 3). The gap between them is reported as a result.
- **D3. Train the coarse quantizer and PQ on the full 1M base set**, not the 100k learn set
  (below Faiss's 39 × nlist guideline) and not a 256k subsample (too small for nlist =
  16384, one extra config knob). Queries are never used in training. The build script keeps
  an optional subsample size for debug builds only.
- **D4. ε is defined on plain L2 distances**, as in the proposal's example (100 → 110 at
  ε = 0.1). Faiss returns squared distances, so the code tests `D_j ≤ (1+ε)² · D_1` with
  `D_1` clamped at 0 and a non-strict comparison. One test checks this equals the rule on
  `sqrt(D)` with factor (1+ε). The gap rule (extension 4) is on plain distances and its
  "normalised" variant is dropped as identical to the ratio rule.
