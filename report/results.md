# Results

Draft of the results section (structure B, see `WORKLOG.md` "FORK POINT"). All numbers are on the
**test** split (5 000 queries); every parameter was chosen on the **tune** split (the other 5 000).
Figures are in `results/plots/`.

## Setup in one paragraph

SIFT1M (1M base vectors, 128-d, 10 000 queries), index `IVF4096,PQ16x8` built with Faiss and
trained on the full base. The search returns the 200 best candidates by PQ distance; their exact
distances are computed and the 10 closest are the answer (IVFADC+R). Without this re-ranking the
16-byte codes cap recall@10 at 0.59 whatever cells are scanned, so none of the targets of the
proposal would be reachable. **Work** per query = scanned vectors + 200 exact distances; we report
it instead of time because it is deterministic, and report time separately (§7). Three methods
differ only in the number of cells each query scans: fixed `nprobe`, the adaptive rules, and an
oracle that knows the per-query recall curve. Every method is run through the same code path
(`search_preassigned` with an explicit cell list), which we checked reproduces Faiss's own search
bit for bit. With re-ranking, recall after 256 cells is 0.996 while the share of true neighbours
whose cell was scanned is 0.999: the remaining PQ loss is about 0.3%, so recall here measures
almost only which cells a method scans (the PQ ceiling of the proposal's extension 5).
Recall targets R ∈ {0.8, 0.9, 0.95, 0.99}; for each, the smallest parameter reaching R
on tune is chosen and reported on test. 95% confidence intervals are bootstraps over test queries.

## 1. Summary

Work saving relative to fixed nprobe at matched mean recall@10 (ratio of work; > 1 = less work).

| method | parameters tuned on tune | R = 0.80 | R = 0.90 | R = 0.95 | R = 0.99 | time vs plain Faiss, R = 0.9: Python / C++ |
|---|---|---|---|---|---|---|
| ratio rule as proposed, d_j ≤ (1+ε) d_1 | ε | 0.77× [0.74, 0.79] | 0.73× [0.71, 0.75] | 0.77× [0.74, 0.79] | 0.93× [0.91, 0.94] | – |
| ratio rule + cap on the number of cells | ε, cap | 1.11× [1.10, 1.12] | 1.13× [1.12, 1.15] | 1.16× [1.15, 1.17] | 1.16× [1.14, 1.17] | 1.05× / – |
| static rule fitted to the oracle (centroid distances only) | linear model per R, scale γ | 1.16× [1.15, 1.17] | 1.23× [1.22, 1.25] | 1.30× [1.28, 1.32] | 1.38× [1.36, 1.40] | 0.83× / – |
| static fitted rule, 3 features: d₃₂/d₁, d₁, cell size (§4) | linear model per R, scale γ | 1.16× [1.15, 1.17] | 1.24× [1.22, 1.25] | 1.29× [1.28, 1.31] | 1.39× [1.37, 1.41] | – / **1.02×** |
| stopping rule on scan feedback | logistic model per R, θ | 1.17× [1.16, 1.18] | 1.25× [1.23, 1.27] | 1.31× [1.29, 1.34] | 1.34× [1.32, 1.37] | 0.28× / 1.01× |
| stopping rule, 3-feature closed form (§5) | K, p, q per R | 1.17× [1.16, 1.18] | 1.25× [1.23, 1.26] | 1.32× [1.30, 1.34] | 1.29× [1.27, 1.32] | as above |
| oracle, best mean recall (ceiling) | – | 1.71× | 1.98× | 2.35× | 3.94× | – |
| share of the oracle saving captured by the stopping rule | | 32% [29, 34] | 35% [32, 37] | 37% [35, 40] | 42% [37, 44] | |

Fixed nprobe needs 13 / 26 / 45 / 122 cells (work 3 793 / 7 299 / 12 339 / 32 247) for the four
targets. Tuned choices land within 0.012 of the target recall on test (the capped, fitted and
stopping rules within 0.006; fixed nprobe overshoots by up to 0.009 because nprobe is an integer,
the plain ratio rule by up to 0.012). Every row is tuned on
tune and evaluated on test the same way (`05`, `11`, `15`, `16`, `17`); the time column is one
single-threaded run on one machine: our Python prototype on top of Faiss, and the C++
implementation inside Faiss's scan loop (§7; C++ only for the two learned rules, the fitted rule in
its 3-feature form).

In words: the rule of the proposal costs **more** work than a fixed nprobe at every target; with a
cap on the number of cells it saves 11–16%, and this is the only rule whose saving also shows in
time at every target in our implementation. A rule fitted to the oracle, on the same centroid distances, saves
16–38%, and the stopping rule, which also watches how the scan is going, 17–34%: on SIFT1M the
two are equal within the confidence intervals up to R = 0.95 and the static rule is ahead at
R = 0.99 (on Deep1M and with nlist = 1 024 the stopping rule is ahead, §5). Both capture about a
third of what an oracle could save; the remaining two thirds are not visible to the search. In
time the picture changes: implemented in C++ inside Faiss's scan loop, the fitted rule, which
decides once and scans in one go, is 0.96 / 1.02 / 1.06 / 1.12× as fast as plain Faiss; the
stopping rule, which interrupts the scan, is about break-even (0.96 / 1.01 / 0.91 / 1.04×).

## 2. How many cells does a query need? (Figure 2)

The oracle scans each query's cells in centroid order and records the recall after every cell.
Smallest number of cells per query that reaches the target:

| per-query target | median | mean | p90 | p99 | need 1 cell | need ≤ 3 | need > 20 | never reach it (of 5 000) |
|---|---|---|---|---|---|---|---|---|
| 0.8 | 8 | 13.7 | 31 | 80 | 6.0% | 22.0% | 19.6% | 2 |
| 0.9 | 13 | 21.2 | 50 | 126 | 3.6% | 14.3% | 33.1% | 9 |
| 1.0 | 23 | 36.3 | 87 | 191 | 1.9% | 7.8% | 53.8% | 139 |

The proposal expected "most queries need 1–3 cells and a few percent need tens". On SIFT1M with
4 096 cells (about 244 vectors each) it is the other way round: for recall 0.9, one query in
seven is satisfied by 3 cells and one in three needs more than 20. A query's ten nearest neighbours
are spread over many cells. The distribution is wide, so a per-query choice *can* pay: the oracle
needs 1.7–3.9× less work than a fixed nprobe (ceiling row in §1). But it is not the "several
times" the proposal expected at R = 0.9, and most of it comes from the tail of hard queries, not
from a mass of trivially easy ones.

Two oracles are reported. The proposal's oracle gives every query the cells it needs for *its own*
recall ≥ t; this guarantees recall per query and costs 1.1–1.6× less than fixed nprobe at R ≤
0.95. The oracle we use as the ceiling minimises mean work for a given *mean* recall (every query
picks `argmin_c work_q(c) − λ·recall_q(c)`, λ swept); it is the correct bound for the metric on the
axis and lies further left (Figure 1, solid vs dashed green). The gap between them is the price
of a per-query guarantee.

## 3. The proposed rule does not save work (Figures 1, 3)

Rule: scan every cell whose centroid is at most (1+ε) times as far as the nearest one (on plain
distances; Faiss returns squares, so the code tests (1+ε)²). With ε tuned on tune (0.18 / 0.24 /
0.29 / 0.41) the rule reaches each target on test but scans 7–37% **more** vectors than the fixed
nprobe that reaches the same recall (first row of §1). The confidence intervals exclude 1 at every
target. The rule also serves the worst queries *worse*: at the work budget of nprobe 26, 5.6% of
queries end with recall ≤ 0.5 under the rule against 2.5% under fixed nprobe (Figure 7).

Why. The rule ranks queries by difficulty reasonably well: Spearman correlation between its cell
count and the oracle's is 0.70 (Figure 3), and giving the same cell counts to randomly permuted
queries is far worse (shuffled control: 0.82× at R = 0.9 even for the best calibrated rule). What
fails is the *mapping* from the score to a number of cells. At the ε needed for recall 0.9 the
rule gives 39 cells on average where fixed nprobe needs 26, with a 99th percentile of 239: queries
with a flat profile of centroid distances (d_10/d_1 close to 1) receive hundreds of cells that add
almost no recall per scanned vector, and at R = 0.99 28% of queries hit the 256-cell cap. The
oracle does the opposite: it gives up on expensive queries. The rule has one parameter and it
cannot be both sensitive enough for easy queries and bounded for hard ones.

## 4. What centroid distances can tell: 1.2–1.4× (Figure 4)

Capping the rule's cell count fixes the explosion: with cap and ε tuned together on tune (cap 16 /
32 / 48 / 128), the rule saves 1.11–1.16× (second row of §1), and this translates into time (§7).
The tuned caps are close to the fixed nprobe for each target (13 / 26 / 45 / 122), and the same
rule seen from the other side gives the same answer: keep the usual nprobe and let the rule only
*remove* cells, i.e. scan min(n_rule(ε), nprobe) cells (Figure 5). Against lowering nprobe for
everyone to the same recall, this saves 7 / 11 / 13 / 17% work at nprobe 13 / 26 / 45 / 122 (best ε
per nprobe, 0.33–0.55, chosen on test, so slightly optimistic) and 6–12% wall-clock time against
plain Faiss at the same nprobe. More aggressive ε values save much more work (up to 3.5× at
nprobe 122 with ε = 0.24) but lose 3–9 points of recall, and lowering nprobe would buy the same
recall more cheaply: the rule can only safely trim the queries that are clearly easy, 17–32% of
them at the best ε.

To see whether a better function of the same information exists, we fitted a linear model from 17
features of the quantizer output (log d_j/d_1 at 15 ranks, log d_1, log mean size of the 8 nearest
lists) to the log of the oracle's cell count on tune, one model per target (the Lagrangian
oracle's allocation at mean recall R), with the scale γ of the allocation chosen on tune by
cross-fitting. On test it saves 1.16 / 1.23 / 1.30 / 1.38× (third row of §1), and run live it gives
exactly the recall read off the curves. (A single model fitted at R = 0.9 and read off the test
curve gives 1.13 / 1.19 / 1.23 / 1.33×: fitting per target matters most at R = 0.99.) Its
correlation with the oracle is 0.72, barely above the ratio rule's 0.70, yet a calibrated mapping
turns that ranking into a real saving.

Ten further rule families built on the quantizer geometry (absolute gap d_j − d_1; the distance from the query to
the cell's boundary, i.e. to the bisector hyperplane between centroids, used as a threshold, as an
ordering, or in a two-stage search that scans more cells while a boundary is closer than the
current 10th neighbour; patience-style stopping) were swept on the tune split; none beats fixed
nprobe by more than ~1.1–1.2× at R ≤ 0.95. The boundary idea fails for a geometric reason: the
bound is valid but in 128 dimensions it is only 20% of the true distance to the cell's nearest
vector in the median, while the 10th neighbour is as far away as the nearest centroid (226 vs 213),
so "boundary closer than r" holds for almost every cell and prunes nothing. Of everything built
on the quantizer output, the fitted rule is the best, and on SIFT1M it is as good as the stopping
rule of §5; on Deep1M and with nlist = 1 024 it is behind it (§5). The catch is time: its features need
the 256 nearest centroids, a quantizer call five times as expensive as for 26, which eats the
saving below R = 0.99 (§7).

**The fitted rule in three features** (`17_fitted_rule_analysis.py`). As in the stopping rule's
model, most of the 17 features are near copies: neighbouring distance ratios correlate at ≥ 0.986
from rank 8 on, and the model gives them large weights of opposite sign that cancel (d₁₂₈ +0.72,
d₉₆ +0.70, d₁₉₂ −0.54, d₂₅₆ −0.40 at R = 0.9). The correlation of a single ratio with the oracle's
cell count stops improving at rank 16 (−0.66). A greedy forward selection on the tune split picks
d₂₅₆/d₁, cell size and d₁; replacing d₂₅₆ by its 0.99-correlated copy d₃₂/d₁ gives, on test,
1.16 / 1.24 / 1.29 / 1.39× against 1.16 / 1.23 / 1.30 / 1.38× for all 17 (intervals overlap at every
target; d₁₆/d₁ instead is equal up to R = 0.9 and 1.33× at 0.99; one ratio alone 1.19–1.25×). A
ridge model on logs is a power law:

    n = A · (d₃₂/d₁)^a · d₁^b · size^c,     R = 0.9:  a = −4.34, b = −1.25, c = −0.87

(a = −2.61 / −4.34 / −5.36 / −6.72 from R = 0.8 to 0.99). A flatter distance profile gives more
cells, smoothly rather than explosively as in §3; among equally flat queries, those far from every
centroid get *fewer* cells, which is the oracle's habit of giving up on hopeless queries; large
nearby cells also mean fewer cells. Fewer features do not make the rule faster by themselves
(computing 3 numbers instead of 17 saves microseconds); what matters for time is that rank 32 is
cheap to obtain (§7).

## 5. Scan feedback: the stopping rule (Figures 1, 4, 7)

What the search learns during the scan, in addition to the centroid distances: how the result
list is evolving. After
each round of cells the search has, for free, the PQ distances of its running top-200, hence the
distance of its current best and 10th candidate, whether the top-10 changed in the last round and
how many of the round's candidates entered the top-200. We scan in rounds (1, 2, 3, 4, 5, 6, 8,
10, …, 224, 256 cells; 26 decision points), compute 9 such features plus the 17 static ones, and a
logistic model gives P(continue). The model is fitted on tune with labels from the oracle ("the
oracle would continue past this point"); the threshold θ is chosen on tune with 2-fold
cross-fitted predictions. No exact distances are used before the final re-ranking. The result is
a prefix of the quantizer order, so its recall can be read off the oracle curves; run live through
the real rounds it is lower by at most 0.0007 (7–35 of 5 000 SIFT1M test queries differ, 1 on
Deep1M). The difference comes from merging candidate lists across rounds (the single-call ratio
rule has none); the likely cause, not verified, is PQ-distance ties at the 200th candidate.

On test it saves **1.17 / 1.25 / 1.31 / 1.34×** work at the four targets (CIs ±0.02), i.e. 32–42%
of the oracle's saving, and reduces the share of badly served queries at equal work from 12.2% to
5.3% (budget of nprobe 13) and from 2.5% to 0.3% (budget of nprobe 26) (Figure 7). An earlier
variant that re-ranked after every cell and used exact distances did no better (1.22× at R = 0.9),
and a feature ablation shows what matters: the distance of the current 10th PQ candidate relative
to the next centroid, plus the number of cells scanned. The "patience" features (has the top-10
changed recently, how many candidates entered the list) add nothing once those are present.
Against the static rule of §4 fitted and tuned the same way (one model per target), the stopping
rule is **not** better on SIFT1M: 1.17 / 1.25 / 1.31 / 1.34× against 1.16 / 1.23 / 1.30 / 1.38×,
equal within the confidence intervals up to R = 0.95 and below it at R = 0.99. An earlier
comparison against a single static model fitted at R = 0.9 had suggested a gain at R = 0.99
(1.45× vs 1.33×, curve-based); it disappears once the static rule is fitted per target. On the other two configurations the picture tilts: on Deep1M the stopping rule is ahead at every
target with non-overlapping intervals (1.24 / 1.26 / 1.36 / 1.65× against 1.20 / 1.20 / 1.32 /
1.52×), and with nlist = 1 024 ahead at R ≤ 0.9 and equal above (1.25 / 1.22 / 1.31 / 1.40× against
1.19 / 1.18 / 1.28 / 1.39×). Scan feedback therefore adds something where the tail of hard queries
is longer, costs nothing elsewhere except SIFT1M at R = 0.99, and comes with an interpretable rule
(below); in 11 of the 12 configuration × target cells it is at least as good as the best static
rule.

**The rule in closed form.** Three of the 26 features carry almost all of the effect: refitting the
model with only d_next/d_1, r10/d_1 and c gives the same savings within ±0.02 at R ≤ 0.95 on both
datasets (1.20× vs 1.21× at R = 0.9 on SIFT1M, 1.25× vs 1.27× on Deep1M); at R = 0.99 the 17 static
features add 4–6% in the tuned measure (1.29× vs 1.34× on SIFT1M, 1.55× vs 1.65× on Deep1M), the one
place where they matter. Run live through the real search, the formula below picks exactly the
same cells as the 3-feature model for every test query on both datasets. Its decision boundary
is a formula. After scanning c cells, continue to the next cell iff

    d_next / d_1  <  K · (r10 / d_1)^p · c^q,        SIFT1M, R = 0.9:  K = 1.53, p = 0.64, q = −0.085,

where d_1 and d_next are the distances to the nearest centroid and to the next cell's centroid, and
r10 is the PQ distance of the current 10th candidate. This is the proposal's rule with an adaptive
ε: scan while d_next ≤ (1 + ε) d_1, but with 1 + ε = K (r10/d_1)^p c^q instead of a constant. The
data say ε has to grow with how far the current 10th candidate still is from the query relative to
the nearest centroid, and shrink slowly with the number of cells already scanned; a constant ε
cannot do either, which is why the rule of §3 fails. On Deep1M the fitted constants are K = 1.38,
p = 0.58, q = −0.067; the SIFT1M formula applied to Deep1M unchanged reaches recall 0.93 at the same
work saving.

## 6. Why the other two thirds are out of reach

The oracle's remaining advantage comes from two things the search cannot observe. First, it knows
*which* queries will never improve: it stops hard queries early and spends the budget on queries
where the next cell pays. A rule only sees how far its current candidates are and how far the next
cells lie, not whether the next cell actually holds one of the true neighbours. Second, difficulty is only moderately predictable from anything
available: the best correlation we obtain with the oracle's cell count is 0.72–0.77. On SIFT1M
with 4 096 cells, the ten neighbours of a typical query sit in many cells, and which cells is a
property of the data around the query, not of its distances to a few centroids.

## 7. Time (Figure 6)

Single thread, 5 000 test queries in one batch, warm-up then median of 5 runs, exact re-ranking
included, R = 0.9 (`12_timing.py`, and `16_fitted_rule_tuned.py` for the fitted rule, same
machine):

| method | work | ms / query | vs plain Faiss |
|---|---|---|---|
| fixed nprobe 26, plain `index.search` | 7 299 | 0.141 | 1.00 |
| fixed nprobe 26, our preassigned path | 7 299 | 0.142 | 0.99 |
| the same 26 cells scanned in the stopping rule's 26 rounds (control) | 7 299 | 0.542 | 0.26 |
| ratio rule + cap 32 | 6 438 | 0.134 | 1.05 |
| static fitted rule | 5 924 | 0.171 | 0.83 |
| stopping rule | 5 853 | 0.511 | 0.28 |

Components at R = 0.9: the coarse quantizer costs 0.009 ms for the 26 nearest centroids and 0.047 ms
for the 256 that the fitted and stopping rules ask for; exact re-ranking of 200 candidates is
0.039 ms for every method; list scanning is the rest.

- **Ratio rule + cap.** Its 1.11–1.16× work saving becomes a 1.03–1.09× speed-up across the four
  targets (1.05× at R = 0.9): one masking step and one scan call, but the quantizer and the
  re-ranking do not shrink with the number of cells.
- **Static fitted rule.** One scan call, like the capped rule, but its features use the distances
  to the 256 nearest centroids. That quantizer call (0.047 vs 0.009 ms) eats the 1.16–1.30× work
  saving at R ≤ 0.95 (0.76 / 0.83 / 0.93× of plain Faiss's speed); at R = 0.99 it is 1.15× faster.
  Features from fewer ranks, or a quantizer that only extends the list for queries predicted to
  need many cells, would remove most of that cost; we did not try it.
- **Stopping rule.** 3–4× **slower** than plain Faiss in our prototype (0.27–0.33×), and the
  control row says why: scanning the very same 26 cells in 26 calls instead of one costs 0.49 ms
  instead of 0.094 ms, because each `search_preassigned` call re-initialises per-query heaps and
  output arrays and crosses the Python boundary. The rule's own logic (merging candidates, features,
  model) is 0.056 ms per query. Within the round structure the rule does scan 25% fewer vectors
  (0.37 vs 0.49 ms), as its work count says. Realising the work saving as time requires an
  incremental scan inside the library (keeping the per-query scanner and heap alive between rounds),
  which Faiss does not expose.

Batch-size-1 latency at R = 0.9 (1 000 test queries, Python overhead included): plain Faiss p50
0.20 ms, p99 0.24 ms; stopping rule p50 1.42 ms, p99 2.10 ms.

An earlier run of `12_timing.py` on the other author's machine was about 3× slower in absolute
terms (0.46 ms per query for plain Faiss) and showed our preassigned path 1.10× faster than
`index.search`, re-ranking at 0.21 vs 0.13 ms for identical candidates, and an outlier for plain
Faiss at R = 0.99. None of this reappears in the run above, where the two paths and the re-ranking
cost the same for every method, so we quote the run above (capped rule 1.05× rather than 1.17×).

**The two learned rules in C++** (`cpp/`, `cpp/run.sh`; Faiss 1.16 C++ library on the same index
file). To separate the rules from the cost of the Python prototype we reimplemented them inside
Faiss's own scan loop, using `IndexIVF::get_InvertedListScanner` as `search_preassigned` does: one
scanner and one result heap per query, kept across all its cells. Checks: our loop at fixed
nprobe returns the same top-10 as `index.search` for every test query, with the same vector count
and the same time (1.00×); both rules pick the same number of cells as their Python versions (0–2
mismatches in 5 000), with the expected recall. Speed vs plain `index.search` at the same recall,
single thread, 5 000 test queries, median of 5, re-ranking included:

| method | R = 0.8 | R = 0.9 | R = 0.95 | R = 0.99 |
|---|---|---|---|---|
| fitted rule, 3 features (d₃₂/d₁, d₁, size); asks for the 64 nearest centroids, 256 only for queries predicted to need more | 0.96× | **1.02×** | **1.06×** | **1.12×** |
| fitted rule, 17 features (needs the 256 nearest for every query) | 0.75× | 0.83× | 0.92× | 1.14× |
| stopping rule (3-feature formula), asks for 64 centroids first, 256 for queries that get that far | 0.96× | 1.01× | 0.91× | 1.04× |
| stopping rule, 256 centroids up front | 0.75× | 0.82× | 0.91× | 1.09× |
| control: the stopping rule's rounds with the rule evaluated but ignored, fixed nprobe | 0.99× | 0.98× | 0.95× | 0.99× |

Run-to-run noise is about ±3%. What it took and what remains:

- **Rounds are cheap once the scan is not restarted** (control row ~0.98×). The rule's check needs
  the current 10th-best PQ distance; computed naively from the 200-entry heap it cost ~15 µs per
  query (the heap's array order defeats a front-to-back insertion pass); since the 10th-best can
  only shrink, collecting the heap entries ≤ the previous value branch-free and selecting among them
  brings it to ~3 µs.
- **The centroid search is the remaining cost.** Both rules need more nearest centroids than plain
  Faiss (26 at R = 0.9): selecting 26 / 64 / 256 of the 4 096 costs 0.018 / 0.025 / 0.060 ms per
  query. Queries that need the 256 get one extra batched search, which has a large fixed cost per
  call (~0.14 s, i.e. ~0.28 ms per query for a batch of 500, 0.06 ms for a batch of 3 000; not
  understood), so asking for fewer centroids first only pays where few queries exceed them.
- **Scan time falls less than vector count.** At R = 0.9 the fitted rule scans 19% fewer vectors
  and saves ~10% of scan time: each cell has a fixed setup cost (its PQ lookup table) and the
  skipped cells are the small ones of easy queries. Re-ranking (0.036 ms) and the centroid search do
  not shrink at all.

So implemented carefully, the fitted rule's 1.2–1.4× work saving becomes a small but real
speed-up from R = 0.9 up (2–12%), none at 0.8; the stopping rule's equal or larger work saving does
not survive as time, because stopping mid-scan forces it to ask for far centroids up front or
fetch them later.

## 8. The proposal's predictions

| prediction | outcome |
|---|---|
| Most queries need 1–3 cells; a few percent need tens | **No.** For recall 0.9 the median query needs 13 cells; 14% need ≤ 3, 33% need > 20. |
| The oracle needs several times fewer scanned vectors at recall 0.9 | **Partly.** 2.0× at 0.9 (1.7× at 0.8, 3.9× at 0.99). |
| The rule gets about half of that: 1.5–2× fewer scanned vectors | **No.** The rule as proposed needs 1.4× *more* at 0.9. With a cap: 1.13×. The best rules we found, a static rule fitted to the oracle and a stopping rule on scan feedback: 1.23× and 1.25×, a third of the oracle's saving. |
| The saving grows with the recall target | **Yes**, for every method: oracle 1.7× → 3.9×, fitted rule 1.16× → 1.38×, stopping rule 1.17× → 1.34×, capped rule 1.11× → 1.16×. |
| The saving in time is smaller than in scanned vectors | **Yes, much smaller.** Python prototype: capped rule 1.13× work → 1.05× time; fitted rule 1.23× → 0.83×; stopping rule 1.25× → 0.28×. In C++ inside Faiss's scan loop: fitted rule (3 features) 1.24× → 1.02×, stopping rule 1.25× → 1.01× (§7). |

## 9. Extension: nlist = 1 024 (bigger cells)

The same pipeline on an index with 1 024 cells (977 vectors each on average, four times bigger).
Fixed nprobe needs 7 / 12 / 21 / 54 cells for the four targets, and about twice the work of the
4 096-cell index at every target (13 196 vs 7 299 at R = 0.9), so this is a worse index for SIFT1M,
as the usual √N-to-4√N guideline predicts. What changes for the adaptive question:

| | R = 0.8 | R = 0.9 | R = 0.95 | R = 0.99 |
|---|---|---|---|---|
| queries done in ≤ 3 cells (oracle, target 0.9) | 26% (was 14%) | | | |
| queries needing > 20 cells | 12.5% (was 33%) | | | |
| oracle ceiling | 1.61× (1.71×) | 1.86× (1.98×) | 2.23× (2.35×) | 3.79× (3.94×) |
| ratio rule (proposal) | 0.92× (0.77×) | 0.82× (0.73×) | 0.82× (0.77×) | 0.84× (0.93×) |
| ratio rule + cap | 1.05× (1.11×) | 1.10× (1.13×) | 1.12× (1.16×) | 0.96× (1.16×) |
| stopping rule | 1.25× (1.17×) | 1.22× (1.25×) | 1.31× (1.31×) | 1.40× (1.34×) |
| captured share, stopping rule | 32% | 38% | 41% | 40% |

The proposal expected bigger cells to make more queries easy and therefore the saving larger. The
first part happens: with 1 000-vector cells a quarter of the queries are finished after 3 cells. The
second does not: the oracle's ceiling and every rule's saving stay within a few points of their
4 096-cell values. An easy query now costs ~1 000 scanned vectors instead of ~250, hard queries
still need tens of cells, and the saving is made of the ratio between the two, which the cell
size does not change. The ratio rule remains worse than fixed nprobe (0.82–0.92×); the capped
version is weaker here and loses at R = 0.99; the stopping rule is the one method whose saving is
stable across the two indexes (1.22–1.40×, 32–41% of the oracle).

## 10. Extension: a second dataset, Deep1M

The first million vectors of Yandex DEEP (96-d image descriptors, unit norm, L2) with the identical
index `IVF4096,PQ16x8` and pipeline; ground truth computed by us. Fixed nprobe needs 10 / 20 / 37 /
134 cells for the four targets (work 2 906 / 5 510 / 9 824 / 33 068).

| | R = 0.8 | R = 0.9 | R = 0.95 | R = 0.99 |
|---|---|---|---|---|
| queries done in ≤ 3 cells / needing > 20 (oracle, target 0.9) | 22% / 25% (SIFT: 14% / 33%) | | | |
| oracle ceiling | 1.64× (1.71×) | 2.02× (1.98×) | 2.64× (2.35×) | 5.25× (3.94×) |
| ratio rule (proposal) | 0.87× (0.77×) | 0.77× (0.73×) | 0.78× (0.77×) | 1.15× (0.93×) |
| ratio rule + cap | 1.12× (1.11×) | 1.10× (1.13×) | 1.19× (1.16×) | 1.15× (1.16×) |
| stopping rule | 1.24× (1.17×) | 1.26× (1.25×) | 1.36× (1.31×) | **1.65×** (1.34×) |
| captured share, stopping rule | 33% | 42% | 50% | 53% |

The picture is the same as on SIFT1M, method by method and within a few points, so the result is
not specific to SIFT. Deep1M has a longer tail of hard queries (34 queries never reach 0.9 within
256 cells, against 9), and the tail is where adaptive stopping pays: at R = 0.99 the oracle could
save 5.2× and the stopping rule saves 1.65×, half of that, its best result in the project. It also
leaves 0.8% of queries at recall ≤ 0.5 at the budget of nprobe 20, against 4.4% for fixed nprobe.
The ratio rule of the proposal is below fixed nprobe at R ≤ 0.95 here as well, and beats it only at
R = 0.99 (1.15×), the one configuration in the project where it does; there fixed nprobe needs 134
cells for everyone, and the rule reaches the 256-cell cap for 27% of queries.

Is the stopping rule tied to the dataset it was fitted on? A model fitted on one dataset, run on
the other without refitting, loses at most 0.02 of saving at every target. (We do not compare the
two weight vectors feature by feature: several of the 26 features are almost collinear, e.g. log
d_96/d_1, d_128/d_1 and d_192/d_1 correlate at 0.997–0.999, and log r10/d_1, log r10/d_next and log
d_next/d_1 are exactly linearly dependent, so individual weights are not identifiable. The compact
3-feature form of §5 is the one to compare: K = 1.53 vs 1.38, p = 0.64 vs 0.58, q = −0.085 vs
−0.067 at R = 0.9.) What does not transfer is the threshold: the SIFT model's θ stops Deep
queries far too early (recall 0.68) and the Deep model's θ stops SIFT queries too late (0.96, twice
the work). A single model fitted on both tune splits with one threshold reaches 0.904 on SIFT and
0.901 on Deep at the same θ with the same savings as the per-dataset models (1.20× and 1.28× at
R = 0.9), so one rule serves both datasets; a new dataset needs a few hundred tune queries to set θ,
not a new model.

## 11. Limitations

- Two datasets (SIFT1M, Deep1M) and two values of nlist on SIFT1M. Both datasets are dense
  descriptors of similar dimension; a very different one (e.g. GloVe, angular, strong hubness)
  was not tried.
- The cap of 256 cells is a second parameter of every rule; the proposed rule hits it for 28% of
  queries at R = 0.99, the stopping rule for fewer.
- With k = 10, per-query recall moves in steps of 0.1, so per-query targets 0.95 and 0.99 mean
  "all ten" (reported under 1.0 in §2); mean-recall targets are unaffected.
- Time was measured both in Python on top of Faiss's public API and in C++ inside Faiss's scan
  loop (§7), single-threaded, on one machine, SIFT1M only. The C++ numbers are what the rules can
  realistically give; a cheaper way to get the far centroids is the remaining lever.
- Exploration of alternative rules (§4) used the tune split for both choosing and scoring, and
  the min(rule, nprobe) view of §4 chose ε on test; the numbers quoted in §1 and §5 are all
  tune → test.
- Recall of the round-based stopping rule is read off the oracle curves; run live it is up to
  0.0007 lower (§5). All other methods match the live search exactly.
- Times (§7) are single-threaded runs on one laptop; absolute milliseconds differ about 3× between
  our two machines (0.46 vs 0.14 ms per query for plain Faiss at nprobe 26), so only ratios
  measured within one run are compared. Timing was not rerun for nlist = 1 024 or Deep1M.
- The fitted static rule was evaluated tune → test on all three configurations (§5), but the
  tables of §9 and §10 do not include it; see `WORKLOG.md` for its numbers there.
