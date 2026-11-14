# Work log

Running notes on progress and decisions, newest entry first. One entry per working
session. Each entry has three parts: **Done**, **Decisions** (what was decided and why,
so the other person can object before it is built on), **Next**. Detailed specs live in
`PLAN.md`; this file records the history and the reasoning.

Status board (updated with every entry):

| Area | State |
|---|---|
| §0 Project setup | done |
| §1 Data and index | not started |
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
