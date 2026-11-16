# Adaptive nprobe: speaker notes

Deck: `presentation/adaptive-nprobe.html` (→ / space / click = next step, ← = back, `f` = fullscreen).
Target length: about 30 minutes; the bracketed timings per slide add up to that. Leave the audience questions on slides 3 and 17 short (10–20 s each); the twist is slide 28.
Numbers are SIFT1M test queries unless a slide says otherwise; every parameter was tuned on
the other half of the queries.

---

## 1. Title [0:30]
- Who we are; a few weeks ago we explained IVF + PQ, today: one knob of it, `nprobe`.
- The picture: two queries, and the cells a smarter search would scan for each one, 2 cells for the easy one, several for the one sitting where cells meet.
- The question on the slide is literally our project: how many cells should *each* query scan?

## 2. IVF + PQ in one slide [0:45]
- Step 1: k-means splits the vectors into `nlist` cells (here 16; we use 4096).
- Step 2: a query compares itself with all centroids and scans only the `nprobe` nearest cells. Larger `nprobe` → better recall, more work.
- Step 3: inside the cells the vectors are PQ codes, 16 bytes instead of 512. Distances from codes are approximate, so we take the 200 best candidates and recompute their exact distances (re-ranking).
- Keep in mind: `nprobe` is one number for all queries.

## 3. Same nprobe for every query? [1:05]  ← ask the audience
- Show two queries A and B, both scanning the same 4 cells.
- Step 1 question: do they need the same number of cells? Give them a few seconds.
  - A sits in the middle of its cell: its neighbours are almost surely in that cell.
  - B sits where three cells meet: its neighbours are spread around.
- Step 2 question: what do we already know before scanning anything? → the distances to all centroids; IVF computes them anyway in step 1 of every search. That is free information.
- Step 3 (cells change): what each query would actually need.

## 4. Our project [0:45]
- Idea: pick `nprobe` per query; easy queries get few cells, hard ones many.
- RQ1: at the *same recall*, how much less work than one fixed `nprobe`?
- RQ2: how much of the *best possible* saving do we get? Best possible = an oracle (next slides).
- Work = vectors scanned (not cells: cells differ in size). Best possible = an oracle that knows each query's answer (slide 8).
- Our proposal started from one simple rule: scan every cell whose centroid is at most (1 + ε) times as far as the nearest one. One parameter, no training (slide 10).

## 5. Setup [0:45]
- SIFT1M: 1M SIFT image descriptors, 128 dimensions, 10 000 queries with exact top-10 computed by brute force. Later Deep1M (96-d image embeddings) as a second dataset.
- Index: `IVF4096,PQ16x8` in Faiss, the standard setting for SIFT1M, trained on the full base.
- Re-ranking (why): with 16-byte codes alone, recall@10 never exceeds 0.59, whatever cells you scan, so the targets 0.8–0.99 would be impossible. With re-ranking of the 200 best, recall after 256 cells is 0.996, i.e. recall now measures almost only *which cells* were scanned, exactly the thing we change.
- Protocol: 5 000 queries to tune every parameter, 5 000 to report. Confidence intervals by bootstrap over test queries.
- The key point (bottom line of the slide): one trained index (same cells, same PQ codes, same re-ranking) for every method; a method only decides which cells each query scans, so every difference in recall or work comes from that choice alone. We check this: with a constant cell count our code returns bit-identical results to Faiss's own search.

## 6. Measure: recall against work [0:45]
- Two numbers per method: recall@10 (share of the true 10 neighbours found) and work = vectors whose distance we compute + the 200 re-ranked.
- Why work and not time: deterministic and independent of the machine; time comes at the end.
- The blue curve: fixed `nprobe` from 1 to 256. Step: our four targets, recall 0.8 / 0.9 / 0.95 / 0.99, need 13 / 26 / 45 / 122 cells.
- Every method later is compared with this curve *at the same recall*: "how much less work to reach recall 0.9".

## 7. How many cells does a query really need? [1:05]
- For every query we scan its cells one at a time and record recall after each, the per-query truth.
- Histogram: smallest number of cells that gets *this* query to recall 0.9.
- Bins double (1, 2–3, 4–7, …), so each bar has the same width on the log axis. Green = done in at most 3 cells.
- Step: median 13; only 14% are done in 3 cells; a third need more than 20; a handful (grey) never get there within 256.
- Our proposal predicted "most queries need 1–3 cells". That was wrong: on SIFT with 4096 cells (~244 vectors each), the 10 neighbours of a typical query are spread over many cells.
- But the spread is wide (1 to 200+), so choosing per query *can* pay.

## 8. The oracle [1:05]
- Oracle = knows each query's recall curve and spends the budget where it buys the most recall. Formally: each query picks c minimising work − λ·recall, sweep λ (a Lagrangian). This is the best possible mean work for a given mean recall: no per-query rule can beat it.
- Dashed line: the oracle our proposal described, every query must reach the target on its own; it costs much more, because guaranteeing recall per query is more expensive than on average. We use the solid one as the ceiling.
- Step: at recall 0.9 the oracle needs 2.0× less work than fixed `nprobe`; 1.7× at 0.8, 3.9× at 0.99.
- So the headroom is real but moderate, not the "several times" we expected; and it grows with the target, because high recall forces fixed `nprobe` to serve the rare hardest queries.

## 9. The scoreboard [0:35]
- From now on every result looks like this: one bar per recall target, height = how many times less work than fixed `nprobe` at the same recall.
- Blue line = fixed `nprobe` (1×). Above = better, below = worse. Dashed line = the oracle ceiling.
- All numbers: parameters tuned on the tune half, measured on the test half.

## 10. Idea 1: the ratio rule [1:05]
- The proposal's rule: scan every centroid within (1 + ε)·d₁ (d₁ = distance to the nearest centroid). One parameter ε.
- Two real SIFT queries: bars = distance to each of the 40 nearest centroids, divided by d₁.
- Step 1: the threshold line at ε tuned for recall 0.9 (ε ≈ 0.24).
- Step 2: the easy query scans 1 cell; the flat query, where many centroids are almost equidistant, scans dozens, far more than it needs.
- Intuition it is built on: "several centroids almost equally close = hard query". Question for the audience: will it beat fixed `nprobe`?

## 11. Idea 1: result [0:45]
- No. Below the line at every target: 0.73× at recall 0.9, i.e. 37% *more* work than fixed `nprobe`. CIs exclude 1.
- This was the main hypothesis of the proposal, and it failed clearly, so the next question is why.

## 12. Idea 1: why [1:05]
- Queries grouped by how many cells the rule gives them.
- Grey = share of queries, orange = share of total work (step 1): the 25% of queries that get more than 50 cells eat 63% of all work.
- Step 2: of the cells scanned for the hardest group, only 4% contain even one true neighbour.
- The rule is not stupid: its ranking of queries by difficulty correlates 0.70 with the oracle, and handing its cell counts to random queries is far worse. The problem is the *amount*: in 128 dimensions distances concentrate, so for some queries dozens of centroids fall inside any relative band; the count explodes (roughly ∝ d₁⁴).
- One ε cannot be both generous enough for the easy queries and bounded for the hard ones.

## 13. Idea 2: geometry [1:05]
- Different signal: lower bounds from geometry. Everything in cell j is on the far side of the boundary between my cell and cell j, so nothing in cell j can be closer than b, the distance to that boundary (step 1). Exact, cheap: b = (d_j² − d₁²) / (2‖c_j − c₁‖).
- Step 2, two-stage search (Kiryl's idea): scan a few cells, take the current 10th-best distance r; any cell whose boundary is farther than r cannot improve the top 10, skip it.
- Variants we tried: absolute gap d_j − d₁ ≤ δ (bands that do not grow with d₁), ordering cells by b instead of centroid distance, stopping when the next b exceeds α·r.
- In 2D this works beautifully (the picture).

## 14. Idea 2: result [0:45]
- None does much better than fixed `nprobe`; most are below it. Two-stage and boundary-stop reach ~1.0–1.19× only at the highest target.
- Tag: exploration on the tune split only; since nothing came close to useful, we did not spend a test run on them.

## 15. Idea 2: why [1:05]
- All distances on one axis (medians): the boundary to a neighbouring cell is ~60 away, the nearest centroid 213, the 10th true neighbour 226.
- So the ball of radius r around the query swallows almost every boundary: 99% of cells pass "b < r", but only 8% actually contain a neighbour.
- The bound is valid but loose, about 0.2 × the real distance to the nearest point in the cell: in 128-d a cell is cut off by many boundaries, not one.
- Classic curse of dimensionality: branch-and-bound pruning (k-d trees, ball trees) works in low dimensions and dies here.

## 16. Idea 3: cap it [0:55]
- The ratio rule ranks queries right, only the *amount* is wrong. Simplest fix: never more than C cells, min(rule, C), with ε and C tuned together on tune.
- Step 1: finally above the line, 1.11 / 1.13 / 1.16 / 1.16×. But modest: the tuned caps (16 / 32 / 48 / 128) are about the `nprobe` you would use anyway, so for half of the queries the rule is just fixed `nprobe`.
- (If asked) the same thing seen as a drop-in, cap = your usual `nprobe`: 7–17% less work than simply lowering `nprobe` to the same recall.

## 17. Idea 4: the distance profile says more than one count [1:05]  ← ask the audience
- The ratio rule reduces the whole list of centroid distances to one number: how many fall within (1 + ε)·d₁. But we have much more for free (step 1):
  - how flat the profile is, d_j / d₁ at many ranks (2 … 256),
  - how far the query is from everything, d₁,
  - how big the nearby cells are,
  - and the old count itself.
- Step 1, with the signals: the question to the audience: many signals, each a little informative, interacting. Which matter, and how to combine them? Let them answer.
- Step 2, "Let's do ML!": we have the oracle's answer for every training query, so let a model learn the mapping.

## 18. Learn how many cells each query needs [1:05]
- Train lane, on the tune queries: 17 features from the centroid distances (15 profile ratios, d₁, size of the 8 nearest cells) → label: the number of cells the oracle gives this query (the Lagrangian oracle at the target recall, so it already includes "give up on hopeless queries") → ridge regression on log(cells) → one scale factor γ, the smallest that reaches the target recall on tune.
- Why γ: the regression predicts a typical value and underestimates hard queries; under-allocating a hard query costs much more recall than over-allocating an easy one costs work. γ comes out at ~2 (3.7 at 0.99). Chosen with cross-fitting (fit on one half of tune, choose γ on the other), so it is not tuned on its own training data.
- Search lane: nearest centroids → predict n → scan the n nearest cells in one go → re-rank. Nothing is decided during the scan.
- One model per recall target.

## 19. Idea 4: result [1:00]
- 1.16 / 1.23 / 1.30 / 1.38× less work than fixed `nprobe`, CIs about ±0.015, from the centroid distances alone. Reaches every target on test within 0.003 recall.
- Run live through the real search: recall equals what the oracle data predicts, query by query.
- Control: give the same predicted counts to random queries → 0.68–0.86×. The model really knows which queries are hard (rank correlation 0.72 with the oracle).
- Only work for now; time comes later.

## 20. Open the model [1:05]
- The 12 largest of the 17 weights (standardised features, so magnitudes compare).
- Step 1: the biggest weights are profile ratios at ranks 96 / 128 / 192 / 256 with opposite signs. Neighbouring ratios are near copies (correlation ≥ 0.99 from rank 8 on), so these weights cancel each other: individually they mean nothing.
- Step 2: what carries the model: how flat the profile is (any one rank ≥ 16 says the same), d₁, and the cell size.
- Natural question: are three enough?

## 21. Keep only three? [0:55]
- Refit with only d₃₂/d₁, d₁ and cell size, same protocol: 1.16 / 1.24 / 1.29 / 1.39× against 1.16 / 1.23 / 1.30 / 1.38× with all 17. Same within the CIs at every target.
- Why rank 32 and not 256: the greedy search on tune picked d₂₅₆/d₁, but it is a 0.99-correlated copy of d₃₂/d₁, and rank 32 is much cheaper to get (matters later for time).
- Background if asked: d₁₆/d₁ instead of d₃₂ is equal up to 0.9 and loses at 0.99 (1.33×); a single ratio alone gives ~1.19–1.25×.
- Bottom line of the slide: we also tried giving the model information about the data itself, all free at search time (per-cell statistics precomputed once): radius of the nearest cell and d₁ relative to it, the typical 10th-neighbour distance inside the cell (local density), the share of the cell's points that sit near its border, how badly PQ codes represent the query, the distance to the boundary of the 2nd cell. Each one alone, all together, and a greedy search over all 25: no change beyond noise (e.g. 1.24× → 1.24× at 0.9). They correlate with difficulty (~0.47), but say nothing the profile and d₁ do not already say. So what is visible before scanning seems exhausted by the three features.

## 22. Three features, one formula [1:05]
- Ridge on logs = a power law: n = A · (d₃₂/d₁)^a · d₁^b · size^c.
- Step 1, what it learned:
  - flatter profile (d₃₂ close to d₁) → more cells: the signal the ratio rule tried to use, but smooth instead of exploding (at 0.9 the exponent is −4.3: 10% flatter → ~1.5× the cells);
  - farther from every centroid → *fewer* cells: the oracle's habit of giving up on hopeless queries, learned;
  - bigger nearby cells → fewer cells: they already hold more candidates and cost more to scan.
- Step 2: the exponents per recall target; all grow with the target (the higher the target, the more the hard queries get).

## 23. Idea 5: decide while scanning [1:05]
- Everything so far decides *before* scanning. But the search learns as it goes (step 1): how far the current 10th-best candidate is, how far the next cell is, whether the top-10 still changes, how many cells so far.
- Step 2: same recipe as idea 4, so no new machinery: scan in rounds (1, 2, 3, 4, 6, 8 … 256 cells), after each round a logistic model gives P(continue) from 26 features (these 9 + the 17 static ones), labels again from the oracle ("would it keep scanning?"), stop if P < θ, θ chosen on tune.
- Hand-made single signals (stop after m cells without a change, "patience") give only ~1.1×.

## 24. Idea 5: result [0:55]
- 1.17 / 1.25 / 1.31 / 1.34×: the same as the fitted rule (1.16 / 1.23 / 1.30 / 1.38) within the CIs up to 0.95, slightly below it at 0.99.
- So on SIFT1M watching the scan does not see more than a well-fitted look at the centroid distances. On Deep1M it does (slide 26).

## 25. Again: three features, one formula [1:05]
- Same analysis as for the fitted rule: three features carry it: the next centroid's distance, the current 10th-best, the cells scanned.
- Its decision boundary: keep scanning while d_next/d₁ < K·(r10/d₁)^p·c^q. Step 1: the left side is exactly the proposal's rule, so this is the ratio rule with an ε that adapts: grows while the best candidates found are still far, shrinks slowly with cells scanned (a soft cap). The constant ε was the problem.
- Step 2: 3 features = 26 up to 0.95 on both datasets (at 0.99 the full model is 4–6% better).
- Also step 2: extra mid-search signals (cells scanned since the last cell that contributed a top-10 candidate, the 10th-best relative to the local neighbour density, how spread the candidate pool is) plus the data features of slide 21: ±2% and not consistent (better at 0.8 / 0.9 / 0.99, worse at 0.95; added to the 3-feature formula they even hurt slightly). Constants at 0.9: K = 1.53, p = 0.64, q = −0.085 on SIFT; K = 1.38, p = 0.58, q = −0.067 on Deep1M.

## 26. Second dataset: Deep1M [1:00]
- Deep1M: 96-d deep image embeddings, same index type and pipeline, refitted on its own tune half.
- Same ordering for the simple rules: ratio rule below the line, cap slightly above.
- Here the two learned rules separate: stopping rule 1.24 / 1.26 / 1.36 / 1.65×, fitted rule 1.20 / 1.20 / 1.32 / 1.52× (CIs do not overlap). Deep has a heavier tail of hard queries (the oracle could save 5.2× at 0.99), and that is where watching the scan pays: for the hardest queries the centroid distances alone cannot tell how far to go.
- (3-feature stopping formula on Deep: 1.24 / 1.26 / 1.36 / 1.55×. SIFT with nlist = 1024: stopping rule ahead at 0.8–0.9, equal above.)
- Step: transfer. Fit on SIFT, use on Deep1M without refitting, threshold for recall 0.9: the 3-feature formula reaches 0.93, the 26-feature model stops far too early (0.68). Simple rule = robust rule.

## 27. Same average work: who gets a bad answer? [0:45]
- Give every method the work of fixed `nprobe` 26 (recall ≈ 0.9; say this, it is no longer on the slide) and count the queries that find at most 5 of their 10 neighbours.
- Fixed 2.5%, ratio rule 5.6% (gives cells to the wrong queries), cap 1.6%; step 1: fitted rule 0.8%, stopping rule 0.3%, oracle 0%.
- Same cost, far fewer bad answers: a second way to sell both learned rules.

## 28. 1.25× less work, so 1.25× faster? [0:55]  ← the twist
- So far, so good: everybody happy? Not quite. We measured wall-clock time (step 1): single thread, all test queries, Python on top of Faiss, against plain Faiss at the same recall.
- Fitted rule 0.76 / 0.83 / 0.93 / 1.15×: slower except at 0.99. Stopping rule 0.30 / 0.28 / 0.27 / 0.33×: three to four times *slower*.

## 29. Where the time goes [1:05]
- Time per query at recall 0.9, split by stage. Plain Faiss: 0.14 ms, most of it scanning; centroid search 0.009, re-ranking 0.039.
- Step 1, fitted rule: scanning is cheaper (0.084 vs 0.094), but its features need the 256 nearest centroids, and selecting 256 of 4096 costs 0.047 ms instead of 0.009: more than the scan saves.
- Step 2, stopping rule: each round is a separate Faiss call that sets up its result heaps and output arrays from scratch and crosses Python ↔ C++. The grey control: the very same 26 cells scanned in 26 rounds instead of one call cost 0.49 ms instead of 0.09. The rule's own logic is only 0.06 ms.
- So both problems are implementation, not the rules. Next: fix the implementation.

## 30. Rewritten by hand in C++ [1:15]
- We reimplemented both rules in C++ on Faiss's own scan code (`InvertedListScanner`, what `index.search` uses inside): one heap per query kept across all cells; the fitted rule (3 features) asks for only 64 nearest centroids and fetches 256 only for queries predicted to need more; the stopping rule checks its formula between rounds without restarting anything. Checked: same cells and recall as Python; our loop at fixed `nprobe` reproduces Faiss exactly.
- Step 1: fitted rule 0.96 / 1.02 / 1.06 / 1.12×: a small but real gain from recall 0.9 up, none at 0.8. Stopping rule 0.96 / 1.01 / 0.91 / 1.04×: about break-even.
- Why not 1.2–1.3×: re-ranking and the centroid search do not shrink with fewer cells (a third of the time); the cells a rule skips are the small, cheap ones of easy queries (19% fewer vectors → ~10% less scan time); and both rules need more nearest centroids than plain Faiss asks for. The stopping rule pays on top for interrupting the scan: checking between rounds still costs about what its extra saving would bring.
- Conclusion: the stopping rule is the most promising in theory but hard to cash in, because of its structure (stop mid-scan); the fitted rule decides once and scans in one go, so its saving survives as time.

## 31. At a glance [0:45]
- Recall 0.9, everything on one axis: the proposal's rule below 1, geometry at 1, cap 1.13×, both learned rules ~1.24×, the oracle at 2×.
- Takeaways (step):
  1. A per-query `nprobe` can save up to 2× at recall 0.9 (more at higher recall).
  2. Both learned rules reach about a third of that, each with three features and a one-line formula; on SIFT1M they tie, on Deep1M the stopping rule saves more.
  3. Only the fitted rule turns it into time: up to 1.12× faster in C++.
- Proposal check: "most queries need 1–3 cells": no; "oracle saves several ×": partly (2×); "rule gets 1.5–2×": no (1.24× for the best); "saving grows with the target": yes; "time saving smaller than work saving": yes, much smaller.

## 32. Limitations and next steps [0:45]
- Limitations: two dense image-descriptor datasets, one index family; the time gain is small (≤ 1.12×, none at recall 0.8, measured on SIFT1M only); the two rules tie only on SIFT1M, on Deep1M the stopping rule saves more work, but we could not turn that into time; every method capped at 256 cells.
- Next: a cheaper way to get far centroids (the remaining time cost, including the unexplained fixed cost of the batched 256-centroid search); other, more diverse datasets (text embeddings, GloVe with angular distance, strong hubness); parallel and GPU evaluation (all our timing is single-threaded CPU; with per-query budgets threads finish at different times, and GPUs prefer equal work per query); billion-scale indexes (Deep1B / SIFT1B), where scanning dominates even more and the work saving should turn into time more fully.
- Already tried, so not listed as next steps: data-aware features (cell radius, local density, border share, PQ error) and extra mid-search signals: no gain (slides 21 and 25).

## 33. Recap [0:30]
- Measure: an oracle tells you how much a per-query budget can give (2× at 0.9).
- Diagnose: the simple ratio rule fails because ε is constant; it ranks right but floods the hard queries.
- Learn: two rules learned from the oracle save 1.16–1.38× work, each reducing to three features; the fitted rule, which decides before scanning, is the one that also saves time (up to 1.12× in C++).
- Questions; our Instagram links, as last time.
