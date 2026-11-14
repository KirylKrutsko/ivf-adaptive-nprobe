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
| §2 Search machinery + tests | not started |
| §3 Oracle | not started |
| E1 fixed-nprobe curve | not started |
| E2 oracle + histogram | not started |
| E3 ε sweep | not started |
| E4 tune/test, RQ1/RQ2 | not started |
| E5 timing | not started |
| Plots | not started |
| Extensions | not started |

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
Suggested: option 1 with PQ16 (proposal's index) and targets {0.8, 0.9, 0.95}, dropping 0.99,
plus raw-PQ recall as a secondary curve; or option 1 with PQ32 if 0.99 is wanted.
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
