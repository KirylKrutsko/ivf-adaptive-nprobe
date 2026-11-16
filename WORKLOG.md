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
| E5 timing | done (`results/12_*`), rerun 2026-11-16 on Kiryl's machine (quoted in the report): capped rule 1.03–1.09×, stopping rule 0.27–0.33× |
| Plots | 1, 1b, 2, 3, 4, 4b, 5, 6, 7 (`results/plots/`), structure B |
| Side exp.: rule capped at nprobe | done (`results/06_*`, plot 5) |
| Alternative rules (bounds, two-stage, patience, fitted) | explored on tune (`results/07_*`), fitted rule tune→test (`results/08_*`) |
| Tail recall at equal work | done (`results/09_*`) |
| Scan-feedback stopping rule (fitted) | per-cell exact version (`results/10_*`); **implementable round version with CIs** (`stoprule.py`, `results/11_*`): 1.17 / 1.25 / 1.31 / 1.34× |
| Fitted static rule, tune → test + live + timing | done on all three configs (`16_fitted_rule_tuned.py`): equal to the stopping rule on SIFT1M/4096 (ahead at 0.99), behind it on Deep1M and nlist 1024 |
| Compact stopping rule, live check | done (`15_stoprule_verify.py`, SIFT1M): formula == model; 1.17 / 1.25 / 1.32 / 1.29× tuned |
| Extensions | 3 (nlist = 1024): `results/nlist1024/`; 1 (Deep1M): `configs/deep1m.yaml`, `results/deep1m/` |

---

## 2026-11-16 — fitted static rule tune → test on Deep1M and nlist 1024 (Kiryl's request)

`16_fitted_rule_tuned.py configs/deep1m.yaml` and `configs/sift1m_nlist1024.yaml` (1–2 min each).
Tuned saving vs fixed nprobe [95% CI], static fitted rule vs stopping rule (11), same protocol:

| config | rule | R = 0.8 | R = 0.9 | R = 0.95 | R = 0.99 |
|---|---|---|---|---|---|
| SIFT1M, nlist 4096 | static fitted | 1.16 [1.15, 1.17] | 1.23 [1.22, 1.25] | 1.30 [1.28, 1.32] | **1.38** [1.36, 1.40] |
| | stopping rule | 1.17 [1.16, 1.18] | 1.25 [1.23, 1.27] | 1.31 [1.29, 1.34] | 1.34 [1.32, 1.37] |
| Deep1M | static fitted | 1.20 [1.19, 1.21] | 1.20 [1.19, 1.21] | 1.32 [1.30, 1.33] | 1.52 [1.49, 1.54] |
| | stopping rule | **1.24** [1.22, 1.25] | **1.26** [1.24, 1.28] | **1.36** [1.33, 1.38] | **1.65** [1.61, 1.69] |
| SIFT1M, nlist 1024 | static fitted | 1.19 [1.18, 1.20] | 1.18 [1.17, 1.19] | 1.28 [1.27, 1.30] | 1.39 [1.37, 1.41] |
| | stopping rule | **1.25** [1.24, 1.26] | **1.22** [1.20, 1.23] | 1.31 [1.30, 1.34] | 1.40 [1.38, 1.43] |

Live run of the static rule: recall == curves for every query, as on SIFT1M. Time of the static rule
vs plain Faiss: Deep1M 0.70 / 0.78 / 0.88 / 1.24×, nlist 1024 0.91 / 0.92 / 1.01 / 1.20× (the
256-centroid quantizer call again).

**Reading.** On SIFT1M/4096 the two calibrated rules are equal (static ahead at 0.99). On Deep1M the
stopping rule is ahead at every target with non-overlapping CIs (1.65 vs 1.52 at 0.99, 1.26 vs
1.20 at 0.9); on nlist 1024 ahead at 0.8/0.9, equal at 0.95/0.99. So the scan feedback does add
something where the tail of hard queries is longer (Deep1M), and never costs anything except the
one cell SIFT1M/0.99. Decision for the report (Matvei, structure B kept): present the two
calibrated rules together as "what is reachable" (both ≈ a third of the oracle); the stopping rule
stays the headline because it is at least as good in 11 of 12 cells, more robust across
configurations, and has the closed form; the static rule is the cheaper one in time at R = 0.99.
`report/results.md` §4/§5/§11 updated accordingly.

---

## 2026-11-16 — report brought up to date; fitted static rule tuned → test; timing rerun (Kiryl)

**Done**
- `experiments/16_fitted_rule_tuned.py`: the static fitted rule of 08 with the protocol of every
  other row of §1: one ridge model per target (labels = Lagrangian oracle on tune at mean recall
  R), scale γ chosen on tune by 2-fold cross-fitting, tuned saving vs the fixed nprobe chosen on
  tune, bootstrap CIs; plus a live run (quantizer, features, prediction, one scan call, re-rank)
  with timing.
- `12_timing.py` rerun on my machine with nothing else running.
- `report/results.md` updated (details below).

**Results**

Fitted static rule, SIFT1M test, tuned saving [95% CI]: **1.16 [1.15, 1.17] / 1.23 [1.22, 1.25] /
1.30 [1.28, 1.32] / 1.38 [1.36, 1.40]** at R = 0.8 / 0.9 / 0.95 / 0.99. Stopping rule (11):
1.17 / 1.25 / 1.31 / 1.34. **Equal within the CIs up to 0.95, and the static rule is better at
0.99** (CIs do not overlap). Live run: recall equals the curve-read value for every query (0
mismatches), cells identical. 08's single model fitted at R = 0.9 gave 1.13 / 1.19 / 1.23 / 1.33
curve-based; fitting per target is what closes the gap, mostly at 0.99.

Timing rerun (single thread, ms per query at R = 0.9): plain 0.141, preassigned path 0.142
(0.99×, not 1.10×), re-rank 0.039 for every method (no 0.21 vs 0.13 discrepancy), no R = 0.99
outlier. Speed-up vs plain: capped rule 1.03 / 1.05 / 1.04 / 1.09×; stopping rule 0.30 / 0.28 /
0.27 / 0.33×; fitted static rule 0.76 / 0.83 / 0.93 / 1.15× (its 256-centroid quantizer call costs
0.047 ms vs 0.009 for 26). Latency b = 1: plain p50 0.20 / p99 0.24 ms, stopping rule 1.42 / 2.10 ms.

**What this changes in the report (all edited in `report/results.md`)**
- §1: fitted-rule row now tuned → test with CIs; compact-formula row added; time column from the
  rerun; "in words" no longer says scan feedback beats centroid distances; recall-overshoot
  sentence corrected (fixed nprobe overshoots by up to 0.009, the ratio rule by 0.012; the old
  "within 0.004" was not true for those two).
- §4: title 1.2–1.4×; per-target fit described; "~1.2× is the limit of centroid distances"
  removed; the min(rule, nprobe) side experiment (06, Figure 5) added.
- §5: the comparison with the static rule rewritten (not better on SIFT1M); the cause of the
  live-vs-curves recall gap marked as likely, not verified.
- §6: the "top-10 is still moving" sentence aligned with the ablation.
- §7: table and text from the rerun, fitted rule added, the earlier run's discrepancies noted.
- §8: rows updated (time prediction: Yes, with the rerun numbers).
- §10: the comparison of the two 26-feature weight vectors dropped (features are collinear, weights
  not identifiable); the compact constants compared instead.
- Setup: the PQ ceiling after re-ranking (recall 0.996 vs coverage 0.999). §11: three limitations.

**Decision needed (both).** Structure B made the stopping rule the main method because it was
ahead of the static rule. Measured the same way it is not: on SIFT1M the two are equal, the static
rule is ahead at 0.99, and neither saves time below 0.99 in our implementation. What is still
unique to the stopping rule: the interpretable closed form (§5), robustness across nlist and
datasets (§9–10, not yet compared with the static rule there), and the tail result.
Before choosing: run `16_fitted_rule_tuned.py configs/deep1m.yaml` and `configs/sift1m_nlist1024.yaml`
(Deep1M is where the stopping rule is strongest, 1.65× at 0.99).

---

## 2026-11-16 — Kiryl's live check rerun on Deep1M; report edits applied

`15_stoprule_verify.py configs/deep1m.yaml` (42 s). Same picture as on SIFT1M:
- formula == 3-feature model for every test query at all four targets;
- tuned saving, compact vs all 26: 1.24 / 1.26 / 1.36 / **1.55** vs 1.24 / 1.26 / 1.36 / **1.65**
  (CIs ±0.02 at R ≤ 0.95, [1.51, 1.59] vs [1.61, 1.69] at 0.99): static features add ~6% at 0.99
  on Deep1M (4% on SIFT1M), nothing below;
- patience / top-10-change features add nothing (dynamic-6 == compact-3 within CI);
- live vs curve-read recall: 1 of 5 000 test queries differs (SIFT1M: 7–35).
Compact formula on Deep1M per target: K ≈ 1.37–1.38, p 0.52 → 0.66, q −0.104 → −0.019.

Report (`report/results.md` §5) updated as Kiryl asked: the "top-10 settled" sentence replaced by
the ablation result (r10 vs the next centroid + cells scanned is the signal), the live-vs-curves
caveat added, and the compact rule's 0.99 numbers quoted in the tuned measure with the 4–6% the
static features add.

---

## 2026-11-16 — compact stopping rule: live check, tuned savings with CIs (Kiryl)

Complements Matvei's `14_compact_rule.py` entry below (same finding, same coefficients). That
entry evaluates curve-based savings with one model fitted at R = 0.9; this one checks the compact
rule the way 11 reports the full model, and runs it for real.

**Done.** `experiments/15_stoprule_verify.py` (SIFT1M only so far, 46 s):
- feature-group ablation with the exact protocol of 11 (one model per target, labels from the
  Lagrangian oracle on tune, θ by 2-fold cross-fit on tune), tuned saving vs the fixed nprobe
  chosen on tune, bootstrap 95% CIs (1000 resamples of test queries);
- the 3-feature model and its closed form `d_next/d_1 < K (r10/d_1)^p c^q` run **live** through
  `stoprule.run_rounds` on all 5000 test queries, re-ranked exactly, recall measured against the
  ground truth instead of read off the E2 curves.

**Results (SIFT1M test; tuned saving [95% CI])**

| features | R = 0.8 | R = 0.9 | R = 0.95 | R = 0.99 |
|---|---|---|---|---|
| all 26 (model of 11) | 1.169 [1.155, 1.182] | 1.247 [1.229, 1.265] | 1.314 [1.294, 1.336] | **1.340** [1.315, 1.366] |
| dynamic 9 | 1.165 | 1.253 | 1.276 | 1.285 |
| dynamic without patience / top-10 changes (6) | 1.169 | 1.246 | 1.322 | 1.294 |
| **compact 3** | 1.169 [1.157, 1.181] | 1.247 [1.230, 1.265] | 1.322 [1.301, 1.344] | **1.292** [1.267, 1.317] |
| static 17 (cannot stop: no feature changes during the scan) | 0.09 | 0.14 | 0.21 | 0.50 |

Formula coefficients (K, p, q) per target: 0.8: 1.558, 0.53, −0.130; 0.9: 1.527, 0.64, −0.085;
0.95: 1.546, 0.69, −0.069; 0.99: 1.551, 0.73, −0.039 (same as 14 at R = 0.9).

**Findings**
1. **The formula is exactly the model.** Run as a plain inequality it picks the same cells as the
   3-feature logistic model for every test query at all four targets (asserted in the script).
2. **Up to R = 0.95 the compact rule equals the full model** in the tuned measure too (identical
   numbers or within the CI).
3. **At R = 0.99 it does not: 1.29× vs 1.34×**, CIs just touching (1.317 vs 1.315). The gap is the
   static features: the 9 dynamic features alone also give 1.29×. The curve-based measure of 14
   (1.51 vs 1.49) does not show it because the compact rule overshoots the target slightly
   (recall 0.9921 vs 0.9916), which the tuned measure charges for and the curve measure does not.
   So "within ±0.02 everywhere" holds for the curve measure, not for the tuned one at 0.99.
4. **Patience / "has the top-10 settled" adds nothing.** Removing those features changes nothing;
   the signal is r10 relative to the next centroid plus the number of cells scanned. The report's
   sentence in §5 ("the information is in 'has the top-10 settled'") should be reworded.
5. **Live recall is slightly below the curve-read recall** for the round-based rule: 7 / 19 / 22 / 35
   of 5000 test queries differ, mean recall lower by 0.0002–0.0007 (e.g. 0.9019 live vs 0.9022 at
   R = 0.9). This applies to the numbers of 11 and 14 as well, which read recall off the curves.
   The single-call ratio rule had 0 mismatches (04), so it comes from merging candidates across
   rounds; likely cause: PQ-distance ties at the 200th candidate (SIFT1M has duplicate vectors),
   not verified. Too small to change any conclusion, but the report should not say the round
   rule's recall equals the curves exactly.

**Next (Matvei, since the report is yours)**
- §5 / the compact-rule paragraph: quote 1.29× at R = 0.99 for the compact rule in the tuned
  measure, or say the static features add ~4% there; reword the "top-10 settled" sentence; soften
  "equals the real search" for the round rule.
- Optional: run `15_stoprule_verify.py configs/deep1m.yaml` to check the same on Deep1M.

---

## 2026-11-16 — compact stopping rule: three features are enough (Kiryl's observation, confirmed)

`experiments/14_compact_rule.py`: the stopping rule refitted with only log d_next/d_1, log r10/d_1
and log c, same protocol as 11 (labels from the Lagrangian oracle on tune, θ by cross-fit on tune,
reported on test), both datasets, all four targets. Results in `results/14_compact_rule.csv`.

| dataset | model (fit R = 0.9) | saving @0.8 / 0.9 / 0.95 / 0.99 |
|---|---|---|
| SIFT1M | 26 features | 1.15 / 1.21 / 1.26 / 1.49 |
| SIFT1M | **3 features** | 1.14 / 1.20 / 1.28 / 1.51 |
| Deep1M | 26 features | 1.15 / 1.27 / 1.44 / 1.81 |
| Deep1M | **3 features** | 1.14 / 1.25 / 1.43 / 1.78 |

Within ±0.02 everywhere; same for the models fitted at the other targets. The 23 other features
add nothing measurable. The decision boundary of the 3-feature model is a closed-form rule.
Continue past c cells iff

    d_next / d_1  <  K · (r10 / d_1)^p · c^q        SIFT1M, R = 0.9:  K = 1.53, p = 0.64, q = −0.085
                                                      Deep1M, R = 0.9:  K = 1.38, p = 0.58, q = −0.067

where d_next is the distance to the next cell's centroid, d_1 to the nearest, r10 the PQ distance
of the current 10th candidate, c the cells scanned so far. (Kiryl's Claude derived the SIFT
formula independently; the coefficients agree exactly.) Fitting at other targets changes p and q
a little (p 0.53 → 0.73 from R = 0.8 to 0.99 on SIFT) and K not at all.

**Reading.** This is the proposal's ratio rule with an adaptive ε: scan the next cell while
d_next ≤ (1 + ε_eff) d_1, with 1 + ε_eff = K (r10/d_1)^p c^q instead of a constant. The data say
ε must grow with how far the current 10th candidate still is relative to the nearest centroid
(far → keep scanning) and shrink slowly with the number of cells already scanned. A constant ε
cannot do this, which is the failure of §3 in one line. The whole model has 4 numbers (K, p, q, θ
folded into K), no learning library, and the same cost as the ratio rule plus one PQ-distance
lookup per round.

Transfer without refit (compact, R = 0.9): SIFT rule on Deep reaches recall 0.934 (full model:
0.68), Deep rule on SIFT 0.853 (full model: 0.96); savings within 0.02 of native. The compact
rule's threshold transfers much better than the 26-feature one, still not exactly; K differs
(1.53 vs 1.38) between datasets.

**For the report**: present this formula as the method. It is the proposal's rule, repaired by the
data; the 26-feature model becomes a footnote showing nothing more is to be gained from the rest.

---

## 2026-11-16 — is the stopping rule universal across datasets? (Kiryl's question)

`experiments/13_transfer.py`, R = 0.9 models from 11 (results in `results/13_transfer*.csv`).

1. **Weights.** Cosine similarity of the standardised logistic weights fitted on SIFT1M and on
   Deep1M: 0.84. The large weights agree in sign and size: log d_192/d_1 (−3.5 / −6.0), log c
   (−2.6 / −2.2), log d_256/d_1 (+2.2 / +4.6), log d_next/d_1 (−1.3 / −1.8), log r10/d_next
   (+1.3 / +1.9), rounds since the last top-10 change (−0.6 / −0.8). Differences sit in the
   mid-rank static ratios (d_64, d_128), which are collinear with their neighbours, so single
   weights there are not meaningful.
2. **Transfer without refit.** Each model applied to the other dataset's test split:

   | model | evaluated on | saving @0.8 / 0.9 / 0.95 / 0.99 | recall at the model's own θ |
   |---|---|---|---|
   | SIFT → SIFT | SIFT | 1.15 / 1.21 / 1.26 / 1.49 | 0.903 |
   | SIFT → Deep | Deep | 1.13 / 1.26 / 1.43 / 1.77 | **0.68** |
   | Deep → SIFT | SIFT | 1.14 / 1.19 / 1.24 / 1.47 | **0.96** (2× the work) |
   | Deep → Deep | Deep | 1.15 / 1.27 / 1.44 / 1.81 | 0.901 |

   The *curves* transfer: a model trained on the other dataset loses ≤ 0.02 of saving at every
   target. The *threshold* does not: at θ = 0.123 the SIFT model on Deep stops far too early
   (recall 0.68) and the Deep model on SIFT far too late (0.96, twice the work). Same ranking of
   "should I continue", different calibration of P.
3. **One joint model** fitted on both tune splits with one θ chosen on the pooled tune (0.123):
   recall 0.904 on SIFT test, 0.901 on Deep test, savings 1.14 / 1.20 / 1.26 / 1.49 (SIFT) and
   1.15 / 1.28 / 1.45 / 1.81 (Deep), i.e. equal to the per-dataset models. Saved as
   `results/13_joint_model_R0.9.npz`.

**Answer.** Yes, the rule is close to dataset-independent: the same 26 standardised features with
nearly the same weights work on both datasets, and a single model with a single threshold trained
on both reaches the target on both with no loss. What is not transferable is the threshold of a
model trained on one dataset alone; a new dataset needs either a few hundred tune queries to pick
θ (no refit) or a model trained on more than one dataset. For the report: the joint model is the
one to present if "universal" matters; its numbers are the same as the per-dataset ones.

---

## 2026-11-16 — extension 1: second dataset, Deep1M

**Done.** Deep1M = the first 1M vectors of Yandex DEEP (96-d, unit-norm, L2), fetched from the
big-ann-benchmarks mirror with an HTTP range request (384 MB) plus the public 10k queries; ground
truth computed by us (no shipped GT for this prefix). `data.load_dataset` dispatches on
`cfg["dataset"]`; `configs/deep1m.yaml` keeps everything else identical (`IVF4096,PQ16x8`, 96/16 =
6 dims per sub-quantizer, full-base training, k_rerank 200, same grids). Same pipeline, 15 min.
Two robustness fixes on the way: `04_rule.py` now tolerates rare near-tie swaps in the quantizer
order between the chunked oracle pass and a full-batch call (1.4e-4 of entries on Deep; 0 on
SIFT), and `11_stop_rule_rounds.py` includes cap = 256 (the uncapped rule) so that a target the
capped rule cannot reach on tune does not crash it.

Index: 4 096 cells of 244 vectors (43–782), k-means 86 s. Raw PQ16 recall@10 ceiling 0.56,
re-ranked 0.994 at 256 cells. Fixed nprobe needs 10 / 20 / 37 / 134 cells (SIFT: 13 / 26 / 45 / 122).

**Results, test split, tune → test (SIFT1M nlist 4096 in brackets)**

| | R = 0.8 | R = 0.9 | R = 0.95 | R = 0.99 |
|---|---|---|---|---|
| work of fixed nprobe | 2 906 (3 793) | 5 510 (7 299) | 9 824 (12 339) | 33 068 (32 247) |
| oracle ceiling | 1.64× (1.71×) | 2.02× (1.98×) | 2.64× (2.35×) | 5.25× (3.94×) |
| ratio rule (proposal) | 0.87× (0.77×) | 0.77× (0.73×) | 0.78× (0.77×) | 1.15× (0.93×) |
| ratio + cap | 1.12× (1.11×) | 1.10× (1.13×) | 1.19× (1.16×) | 1.15× (1.16×) |
| fitted static rule | 1.12× (1.13×) | 1.22× (1.19×) | 1.30× (1.23×) | 1.35× (1.33×) |
| **stopping rule** | **1.24× (1.17×)** | **1.26× (1.25×)** | **1.36× (1.31×)** | **1.65× (1.34×)** |
| captured share (stopping rule) | 33% (32%) | 42% (35%) | 50% (37%) | 53% (42%) |

CIs for the stopping rule: ±0.015 at R ≤ 0.95, [1.61, 1.69] at 0.99. Cells needed (oracle, 0.9):
median 9 (13), ≤ 3 cells 22% (14%), > 20 cells 25% (33%), never 34 (9). Tail at the nprobe-20
budget: recall ≤ 0.5 for 4.4% of queries under fixed nprobe, 4.6% ratio, 0.8% stopping rule.

**Reading**
- Every conclusion from SIFT1M holds on Deep1M, with the same ordering of methods and numbers
  within a few points, except that the hard tail matters more here: the oracle ceiling at R = 0.99
  is 5.2× and the stopping rule gets 1.65× of it (half of the possible saving), its best result in
  the project. Deep has more "never reach it" queries (34 vs 9 at 0.9) and a longer tail, which is
  exactly where "stop when nothing moves" pays.
- The ratio rule is the one method whose verdict changes with the target: still 0.77–0.87× at
  R ≤ 0.95, but 1.15× at R = 0.99 (fixed nprobe needs 134 cells there, and at that point even an
  over-allocating rule beats a flat 134 for everyone). This is the only configuration in the project
  where the proposal's rule beats fixed nprobe; it does so at the cap (27% of queries at 256).
- Static fitted and stopping rules are very close up to 0.95 here too; the scan feedback separates
  them at 0.99 (1.65× vs 1.35×), as on SIFT.

Figures: `results/deep1m/plots/` (no 6_timing). `report/results.md` §10 added.

**Next**: nothing mandatory left from the plan except timing on the extensions (optional). Write-up.

---

## 2026-11-16 — extension 3: nlist = 1024

**Done.** `configs/sift1m_nlist1024.yaml` (only nlist and `results_dir` differ; `load_config` now
takes `results_dir` from the config, `plots.py` takes the env from the config path). The whole
pipeline rerun with the same scripts: 01 → 02 → 03 → 04 → 05 → 08 → 11 → 09 → plots, 5 min total.
Index: 1 024 cells of 977 vectors on average (318–3 891), k-means 5 s. Timing (12) not rerun.

**Results, test split, tune → test (nlist 4096 in brackets)**

| | R = 0.8 | R = 0.9 | R = 0.95 | R = 0.99 |
|---|---|---|---|---|
| fixed nprobe needed | 7 (13) | 12 (26) | 21 (45) | 54 (122) |
| work of fixed nprobe | 7 856 (3 793) | 13 196 (7 299) | 22 660 (12 339) | 56 530 (32 247) |
| oracle ceiling | 1.61× (1.71×) | 1.86× (1.98×) | 2.23× (2.35×) | 3.79× (3.94×) |
| ratio rule (proposal) | 0.92× (0.77×) | 0.82× (0.73×) | 0.82× (0.77×) | 0.84× (0.93×) |
| ratio + cap | 1.05× (1.11×) | 1.10× (1.13×) | 1.12× (1.16×) | 0.96× (1.16×) |
| fitted static rule | 1.12× (1.13×) | 1.18× (1.19×) | 1.21× (1.23×) | 1.23× (1.33×) |
| **stopping rule** | **1.25× (1.17×)** | **1.22× (1.25×)** | **1.31× (1.31×)** | **1.40× (1.34×)** |
| captured share (stopping rule) | 32% (32%) | 38% (35%) | 41% (37%) | 40% (42%) |

Cells needed per query (oracle, target 0.9): median 7 (13), need ≤ 3 cells 26% (14%), need > 20
cells 12.5% (33%). Tail at the nprobe-12 budget: queries with recall ≤ 0.5 are 3.1% fixed, 4.1%
ratio, 0.7% stopping rule (4096 at nprobe 26: 2.5 / 5.6 / 0.3%).

**Reading**
- The proposal's expectation for bigger cells was "more queries are easy, so the saving grows". The
  first half is true: with ~1 000-vector cells, a quarter of the queries are done in 3 cells and
  only one in eight needs more than 20. The second half is not: the oracle ceiling is the same or
  slightly lower (1.86× vs 1.98× at 0.9), and every rule lands within a few points of its nlist-4096
  value. Easy queries now cost ~1 000 vectors each, hard ones still need tens of cells, and the
  *ratio* of the two is what the saving is made of.
- Absolute work doubles at every target (13 196 vs 7 299 at 0.9): nlist = 1024 is simply a worse
  index for this base, as expected from the Faiss guideline of ~4√N cells. Adaptive nprobe does not
  change that ranking.
- The ratio rule is still worse than fixed nprobe (0.82–0.92×), less badly so, because the cap at
  256 now equals a quarter of the index and the explosion is milder (p99 76 cells at R = 0.9 vs
  239). Ratio + cap is weaker here and even loses at 0.99 (cap 128 chosen on tune, 0.96× on test).
- The stopping rule is the only method whose saving is stable across the two indexes
  (1.22–1.40×, 32–41% of the oracle), which is the kind of robustness the report can claim for it.

Figures for this run are in `results/nlist1024/plots/` (same names; no 6_timing).

**Next**: add §10 to `report/results.md` (done in this commit). Remaining optional items: second
dataset, timing for nlist 1024, coarser round schedule for the stopping rule.

---

## 2026-11-16 — plots for structure B

`experiments/plots.py` extended (Kiryl's plots kept; the proposal-only versions saved as 1b / 4b).
`09_tail_recall.py` now includes the stopping rule. Figures, in the order the report would use them:

| file | shows | source |
|---|---|---|
| `1_recall_vs_work.png` | recall vs work on test for fixed, ratio, ratio + cap 32, fitted static, stopping rule, both oracles | 05, 04 + 03, 08, 11 |
| `2_cells_needed_hist.png` | cells each query needs (oracle), targets 0.9 and 1.0 | 03 |
| `3_rule_vs_oracle.png` | ratio rule cells vs oracle cells per query, Spearman 0.70 | 04, 03 |
| `4_saving_vs_target.png` | saving vs fixed at each target, tuned on tune, 95% CI: ratio, ratio + cap, stopping rule, oracle | 05, 11 |
| `5_rule_within_nprobe.png` | Kiryl's side experiment, min(rule, nprobe) | 06 |
| `6_timing.png` | E5: time per query by component at R = 0.9 incl. the 26-call control; work saving vs speed-up by target | 12 |
| `7_tail_recall.png` | share of queries with recall ≤ 0.5 at equal mean work, budgets of nprobe 13 and 26 | 09 |

Plot 1 reads: the stopping rule and the fitted static rule lie left of fixed nprobe everywhere and
almost on top of each other up to 0.95, the stopping rule pulling ahead above; the ratio rule is
right of fixed at every recall; the cap-32 curve ends at 0.927 (its ceiling).
Plot 7 reads: at the nprobe-26 budget the stopping rule leaves 0.3% of queries at recall ≤ 0.5 vs
2.5% for fixed nprobe and 5.6% for the ratio rule.

**Next**: results table and text for the report (structure B); nlist = 1024 if time allows.

---

## 2026-11-16 — FORK POINT: report structure (decision pending with Kiryl)

Everything up to here is tagged **`fork-report-structure`** (git tag). Two ways to frame the
report from this state; the code and results are the same for both, only the emphasis differs.

- **A. "The proposed rule, diagnosed and repaired"**: ratio rule (negative result) → why →
  ratio + cap as the fix (1.1×, works in time too) → the oracle as the ceiling; the stopping
  rule as an extension (6).
- **B. "Adaptive nprobe: what is achievable from what the search knows"** (Matvei's choice,
  proceeding with it from here): ratio rule (negative) → diagnosis → ratio + cap and the fitted
  rule as the limit of centroid-distance information (~1.1–1.2×) → the stopping rule on scan
  feedback as the main method (1.25–1.34×, a third of the oracle saving; time gain needs an
  in-library implementation) → oracle ceiling and why the rest is unobservable.

Both answer RQ1/RQ2 of the proposal; the topic (per-query number of cells for IVF+PQ, same
index, oracle, metrics and tune/test protocol) is unchanged. To go back to A: `git checkout
fork-report-structure`, or simply reorder the sections; nothing after this point removes results.

Work from here (structure B): plots for the new structure (plot 1 with stopping rule and
ratio + cap, time vs recall from E5, tail recall), a results table for the report, then
nlist = 1024 if time allows.

---

## 2026-11-16 (later) — stopping rule made implementable, bootstrap CIs, E5 timing

**Done**
- `src/ivfadapt/stoprule.py`: the stopping rule in a form a real search can run. Cells are
  scanned in rounds (`SCHEDULE`: 1, 2, 3, 4, 5, 6, 8, 10, …, 224, 256; 26 decision points).
  After each round the features come from **PQ distances only** (best and 10th PQ candidate vs
  d_1 and vs the next centroid, PQ top-10 changes, candidates entering the top-200, rounds since
  the last change, r10 improvement, the 17 static features): no re-ranking until the end, unlike
  the per-cell exact version of 10. One function `run_rounds` generates training features
  (everyone runs to the end), does inference, and is what gets timed, so train/serve cannot
  diverge. `tests/test_stoprule.py` (28 tests total): online stopping == offline decision from
  recorded P; the re-ranked result equals the oracle curves at n_cells.
- `experiments/11_stop_rule_rounds.py`: E4 protocol for it. Model fitted on tune (labels: the
  Lagrangian oracle at R continues past the boundary); θ chosen on tune from 2-fold
  cross-fitted predictions (largest θ still reaching R); baselines (fixed nprobe; ratio rule with
  cap ∈ {16, 32, 48, 128}, ε and cap tuned on tune) the same way; bootstrap 95% CIs over test
  queries (1000 resamples). Parameters saved to `results/11_params.json` for the timing.
- `experiments/12_timing.py` (E5): single thread, 5000 test queries in one batch, warm-up +
  median of 5, components separated; batch-size-1 latency at R = 0.9. Rerun on a quiet machine
  (the first run overlapped a background job and had outliers).
- `oracle.sequential_features` now also records the PQ-based quantities (cache recomputed; the
  per-cell exact results of 10 are unchanged).

**Results on test, parameters tuned on tune (work = ndis + 200)**

| R | fixed nprobe | ratio + cap (cap, ε) | stopping rule θ | saving ratio+cap | saving stopping rule [95% CI] | curve-based | captured (Lagr.) [95% CI] |
|---|---|---|---|---|---|---|---|
| 0.80 | 13 | 16, 0.227 | 0.147 | 1.11× | **1.17×** [1.16, 1.18] | 1.15× | 32% [29, 34] |
| 0.90 | 26 | 32, 0.286 | 0.123 | 1.13× | **1.25×** [1.23, 1.27] | 1.21× | 35% [32, 37] |
| 0.95 | 45 | 48, 0.396 | 0.098 | 1.16× | **1.31×** [1.29, 1.34] | 1.27× | 37% [35, 40] |
| 0.99 | 122 | 128, 0.547 | 0.037 | 1.16× | **1.34×** [1.32, 1.37] | 1.45× | 42% [37, 44] |

Test recall of every tuned choice is within 0.004 of R. The round/PQ version loses nothing
against the per-cell exact version of 10 (1.22× at 0.9 there): the information is in "has the
top-10 settled", and the PQ top-10 carries it as well as the exact one.
(Ratio + cap here is slightly better than in Kiryl's E4 entry, 1.13× vs 1.07× at 0.9: different
ε grid and the cap chosen by least tune work rather than a fixed list.)

**E5 timing (single thread, ms per query, batch of 5000; recall and work as above)**

| R | method | quantizer | rule | scan | re-rank | total | speed-up vs plain |
|---|---|---|---|---|---|---|---|
| 0.9 | fixed, plain `index.search` | 0.26 (incl. scan) | – | – | 0.21 | 0.46 | 1.00 |
| 0.9 | fixed, preassigned path | 0.017 | – | 0.27 | 0.13 | 0.42 | 1.10 |
| 0.9 | **same 26 cells scanned in the rule's 26 rounds** | 0.017 | – | 1.21 (incl. merge) | 0.13 | 1.36 | 0.34 |
| 0.9 | ratio rule, cap 32 | 0.019 | 0.0001 | 0.25 | 0.13 | 0.40 | 1.17 |
| 0.9 | stopping rule (rounds) | 0.086 | 0.19 | 0.92 | 0.13 | 1.32 | 0.35 |

Same pattern at every target: ratio + cap 1.04–1.17× faster than plain Faiss (work saving
1.11–1.16×), the stopping rule 0.33–0.41× (i.e. ~3× slower). Batch-size-1 latency at R = 0.9:
plain p50 0.67 ms / p99 2.3 ms, stopping rule p50 4.5 ms / p99 11 ms.

**Reading of E5**
- The stopping rule's time is not its logic (features + model: 0.19 ms, could be ~0 in C++) but
  **the per-call overhead of `search_preassigned`**: the control row shows that scanning the
  very same 26 cells in 26 calls instead of 1 costs 1.21 ms instead of 0.27 ms (4.4×). Each call
  re-initialises per-query heaps, output arrays and Python/SWIG round trips; Faiss has no API to
  resume a scan. The rule does scan 20% fewer vectors (0.92 vs 1.21 ms inside the same round
  structure), as the work numbers say.
- So, in this prototype: the ratio + cap rule turns its 1.13× work saving into a 1.17× speed-up
  (one extra masking step, one scan call); the stopping rule's 1.25× work saving cannot be
  realised without an incremental scan inside Faiss (C++: keep the per-query scanner and heap
  alive across rounds). The proposal's prediction "time saving smaller than work saving" holds
  for the first and is an understatement for the second.
- Quantizer cost: the stopping rule asks for the 256 nearest centroids up front (0.086 ms vs
  0.017 for 26). A lazy extension (ask for 32, extend only for queries that continue) would
  remove most of it.
- The plain `index.search` row at R = 0.99 (0.98 ms vs 0.40 ms through the preassigned path with
  identical ndis) is anomalous; not investigated, not used in any conclusion.

**Next**
- Both: this settles the report's structure. Suggested: ratio + cap as "the simple rule that works
  a little, in work and in time"; the stopping rule as "what the scan feedback is worth" with the
  honest statement that its time gain needs an in-library implementation; ratio as the negative
  result; oracle as the ceiling.
- Plots: add stopping-rule and ratio+cap curves to plot 1 (data in `11_curves_test.csv`), a
  time-vs-recall plot from `12_timing.csv`, the tail-recall figure from `09_tail_recall.csv`.
- Optional: a coarser schedule (e.g. 8 decision points) to show the work/time trade-off of the
  stopping rule; nlist = 1024.

---

## 2026-11-16 — scan-feedback stopping rule (best so far) and tail recall at equal work

**Done**
- `oracle.sequential_features`: one pass over all 10k queries (175 s, cached in
  `results/10_sequential_features.npz`) recording after every cell what a real search knows at
  that moment: exact distance of the current best and 10th candidate (r1, r10), how many top-10
  ids changed, how many of the cell's PQ candidates entered the running top-200.
  r10 is not strictly monotone (0.008% of steps, 1.9% of queries, ≤ 1.8%): the running top-200 is
  cut by PQ distance, so a good exact candidate can be evicted later. Real behaviour, kept.
- `experiments/10_stop_rule.py`: logistic model P(continue after c cells) on 26 features
  (log r10/d_1, log r10/d_{c+1}, log r1/d_1, log d_{c+1}/d_1, log c, steps since the last top-10
  change, changes and entries in the last 4 cells, r10 improvement over 4 cells, + the 17 static
  features of 08). Labels: the Lagrangian oracle on **tune** continues past c (fit for R = 0.8,
  0.9, 0.95). On **test** the scan stops at the first c with P < θ; θ swept. Prefix rule, so read
  off the E2 curves. Newton fit in 3 s on 1.28M rows.
- `experiments/09_tail_recall.py`: per-query recall distribution at equal mean work on test.

**Results, tune → test, work saving vs fixed nprobe at matched mean recall**

| rule | @0.8 | @0.9 | @0.95 | @0.99 |
|---|---|---|---|---|
| stopping rule, labels from oracle at R = 0.9 | 1.15 | **1.22** | 1.25 | 1.49 |
| stopping rule, labels at R = 0.95 | 1.13 | 1.22 | **1.28** | **1.55** |
| static fitted rule (08), for reference | 1.13 | 1.19 | 1.23 | 1.33 |
| ratio + cap (E4), for reference | 1.05 | 1.07 | 1.12 | 1.18 |
| oracle ceiling | 1.71 | 1.98 | 2.35 | 3.94 |

Captured share of the oracle saving at R = 0.9: 37% (was 32% static, 13% ratio+cap).
Scan feedback adds a little over the static features at 0.8–0.95 and clearly more at 0.99
(1.55× vs 1.33×): at high recall the question "has the top-10 stopped moving?" carries
information that centroid distances do not.

**Tail recall at equal mean work (test; budget = work of fixed nprobe 13 / 26):**

| method | recall mean | share recall ≤ 0.5 | p10 recall |
|---|---|---|---|
| fixed 13 | 0.809 | 12.2% | 0.5 |
| ratio rule | 0.757 | 16.1% | 0.5 |
| ratio + cap 32 | 0.783 | 12.4% | 0.5 |
| fitted (08) | 0.832 | **6.4%** | 0.6 |
| oracle | 0.914 | 1.3% | 0.8 |
| fixed 26 | 0.907 | 2.5% | 0.7 |
| ratio rule | 0.861 | 5.6% | 0.6 |
| fitted (08) | 0.927 | **0.8%** | 0.8 |

The proposal's ratio rule is worse than fixed nprobe on the tail as well (it gives the extra cells
to the wrong queries), so "fairer" is not a rescue for it. The fitted rule halves the share of
badly served queries at the same budget: that is a second, positive way to present it.

**Where this leaves the project.** Three honest statements for the report:
1. The proposed ratio rule does not save work (0.73–0.93×) and does not help the tail.
2. Any calibrated rule on centroid distances saves 1.1–1.2×; adding scan feedback 1.15–1.55×
   (the only method here that is a real improvement at every target, with 37% of the oracle
   saving at R = 0.9). Both are tune→test, no model beyond a linear/logistic fit.
3. The oracle shows 2–4× is possible per query, but the difficulty of a query is mostly not
   observable from what IVF knows at search time on SIFT1M at nlist = 4096.

**Next**
- Both: pick the report's "method" (suggest: stopping rule as main, ratio + cap as the simple
  variant, ratio as the negative result). Then E5 timing for that method: the stopping rule needs
  per-cell sequential scanning, whose overhead in Python is not representative; time it in a
  batched form (scan cells in rounds, re-evaluate P every round).
- Bootstrap CIs for the stopping rule (same machinery as E4).
- nlist = 1024 / second dataset if time remains.

---

## 2026-11-15 (night) — alternative cell-selection rules: none beats fixed nprobe by much

**Question (Kiryl).** The ratio rule loses to fixed nprobe. Are there other ways to tell useful
cells from useless ones at query time? In particular a two-stage search: scan a few cells,
take the current 10th-neighbour distance r, then scan every cell whose *boundary* is closer
than r.

**Done**
- `rules.bisector_bounds`: for each of the 256 nearest cells, a valid lower bound on the distance
  from the query to any vector stored in that cell: the distance from q to the bisector
  hyperplane of (c_i, c_j), (D_j − D_i)/(2‖c_j − c_i‖), maximised over all nearer centroids i.
  Checked on 1200 (query, cell) pairs: 0 violations. `rules.compact_cells` packs an arbitrary
  cell mask for `search_preassigned` (non-prefix rules need the real search path).
- `experiments/07_rules_explore.py` (**tune split only**, 1108 rule evaluations, 7 min): prefix
  rules are read off the E2 curves, the others run through the real search. One sequential pass
  records after every cell the exact 10th-candidate distance r₁₀(c) and whether the exact
  top-10 changed (`results/07_tune_sequential.npz`), so patience / stopping rules are prefix
  rules too.
- `experiments/08_fitted_rule.py` (extension 6, **fit on tune, evaluate on test**): ridge
  regression from 17 features (log d_j/d_1 at 15 ranks, log d_1, log mean size of the 8 nearest
  lists) to log oracle cells; allocation γ·exp(prediction), γ swept; shuffled control.

**Results: work saving vs fixed nprobe at matched mean recall** (tune split, so slightly
optimistic for one-parameter rules and more so for envelopes over a second parameter):

| rule | @0.8 | @0.9 | @0.95 | @0.99 | note |
|---|---|---|---|---|---|
| ratio (proposal) | 0.80 | 0.76 | 0.79 | 0.93 | reference |
| ratio + cap, envelope over C ∈ {16,32,48,128} | 1.07 | 1.11 | 1.15 | 1.18 | Kiryl's capped rule |
| patience m (stop after m cells without a top-10 change) | 1.05 | 1.10 | 1.14 | – | Li et al.-style, no learning |
| bound_stop α (stop when next cell's bound > α·r₁₀, r updated) | 0.95 | 1.00 | 1.05 | 1.19 | Kiryl's idea, iterative form |
| two-stage (c0 cells, then all cells with bound ≤ α·r), envelope c0 ∈ {1,2,4,8} | 1.00 | 0.94 | 0.96 | 1.11 | Kiryl's idea, literal form |
| gap d_j − d_1 ≤ δ | 0.88 | 0.90 | 0.94 | 1.08 | extension 4 |
| bound threshold (bmax ≤ τ, absolute / relative to d_1) | 0.86 / 0.83 | 0.85 / 0.81 | 0.88 / 0.85 | 1.01 / 1.01 | |
| c cells in order of bound instead of centroid distance | 0.96 | 0.94 | 0.92 | 0.83 | worse ordering |

Fitted rule, tune → **test** (`08_fitted_savings.csv`); fixed-nprobe work on test 3422 / 6706 / 11507 / 31498:

| fit target | Spearman vs oracle c*(0.9) | @0.8 | @0.9 | @0.95 | @0.99 |
|---|---|---|---|---|---|
| Lagrangian cells at R = 0.9 | 0.72 | 1.13 | **1.19** | 1.23 | 1.33 |
| Lagrangian cells at R = 0.95 | 0.75 | 1.11 | 1.19 | 1.25 | 1.40 |
| per-query c*(0.8) | 0.75 | 1.06 | 1.14 | 1.22 | 1.41 |
| same counts, shuffled over queries | – | 0.86 | 0.82 | 0.78 | 0.68 |
| oracle (Lagrangian) ceiling | – | 1.71 | 1.98 | 2.35 | 3.94 |

Captured share of the oracle saving by the best fitted rule at R = 0.9: (1 − 1/1.19)/(1 − 1/1.98) ≈ 32%.

**Why the boundary idea does not work here (the important finding).**
- The bound is valid but loose: in the median it is **20% of the true distance** to the nearest
  vector of the cell (max 70%). In 128-d the 10th true neighbour is about as far as the nearest
  centroid (median r₁₀ = 226 vs d_1 = 213), while the distance to a neighbouring cell's boundary
  is 10–70. So "bound ≤ r" is true for almost every cell and prunes nothing; the usable α are
  0.1–0.2, at which the bound is just another heuristic score, and a slightly worse one than the
  centroid-distance ratio. Branch-and-bound pruning dies of the curse of dimensionality.
- All centroid-geometry scores rank queries by difficulty about equally well (Spearman 0.70–0.77
  with the oracle) and far from perfectly. The gain from calibrating the mapping score → cells
  (cap, patience, fitted) is real but small: ~1.1–1.2× at R = 0.9 against a 2× ceiling. Most of
  a query's difficulty is not visible in its centroid distances.
- The ratio rule's specific failure (explosive cell counts for flat profiles) is a calibration
  problem; any cap or fit removes it, and that is where the ~1.1× comes from.

**Recommendation for the report**
- Headline: the proposed rule does not save work; capped/calibrated variants save 10–20%;
  the oracle shows 2–4× is possible but not from centroid distances alone.
- Quote the tune→test numbers (E4 capped rule, 08 fitted rule), not the tune-split exploration.
- If one more experiment: nlist = 1024 (extension 3). With ~1000 vectors per cell more queries
  may be truly easy and the geometry signal may be stronger; the whole pipeline is reusable.

**Next**
- Both: decide the report's framing. Candidate for the "method" the report evaluates beyond the
  plain rule: the fitted rule (two lines of NumPy, no model), with ratio+cap as the simple variant.
- Proper tune→test E4-style run for patience and bound_stop if they go in the report.

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
