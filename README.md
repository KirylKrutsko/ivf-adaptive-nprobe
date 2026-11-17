# ivfadapt — query-adaptive nprobe for IVF+PQ

Kiryl Krutsko, Matvei Sendetskii.

Instead of scanning a fixed `nprobe` cells per query, choose the number of cells per query. We
compare the proposal's rule (scan every cell whose centroid is at most `(1 + ε)` times as far as
the nearest one), calibrated variants of it, a static rule fitted to an oracle and a stopping rule
on scan feedback with fixed `nprobe` and with an oracle on SIFT1M and Deep1M, measuring recall@10
against scanned vectors, and wall-clock time in Python and in C++.

Where to read what:

- `proposal.pdf` — the project proposal
- `PLAN.md` — the implementation and experiment plan, with the design decisions D1–D5
- `WORKLOG.md` — the project diary, newest entry first: what was done in each session, the
  numbers, the reasoning behind every decision, and what comes next; a status board at the top
- `report/results.md` — draft of the results section; `report/guide_ru.md` — presenter's guide (Russian)
- `presentation/` — the slides and speaker notes (see below)

## Setup

```bash
uv sync            # creates .venv with Python 3.12 and all dependencies
uv run python experiments/01_build.py   # download SIFT1M, ground truth, build + cache the index (~2 min)
uv run pytest                            # correctness tests (need the built index)
uv run python experiments/02_fixed.py   # E1: fixed-nprobe baseline -> results/
uv run python experiments/03_oracle.py  # E2: per-query curves, both oracles, histogram data (~3 min)
uv run python experiments/04_rule.py    # E3: ε sweep of the ratio rule (read off the E2 curves, checked vs real search)
uv run python experiments/05_tune_test.py  # E4: tune/test, RQ1/RQ2 with bootstrap CIs
uv run python experiments/plots.py      # plots 1-4 -> results/plots/
uv run python experiments/06_rule_within_nprobe.py         # side exp.: min(rule(ε), nprobe), plot 5
uv run python experiments/06_rule_within_nprobe_timing.py  # its wall-clock check (single thread)
uv run python experiments/07_rules_explore.py   # alternative rules (bounds, two-stage, patience) on tune (~7 min)
uv run python experiments/08_fitted_rule.py     # extension 6: rule fitted to the oracle, tune -> test
uv run python experiments/09_tail_recall.py     # per-query recall distribution at equal work
uv run python experiments/10_stop_rule.py       # scan-feedback stopping rule, per-cell exact features, tune -> test (~3 min first run)
uv run python experiments/11_stop_rule_rounds.py  # implementable round-based stopping rule (stoprule.py): tune -> test, bootstrap CIs (~2 min)
uv run python experiments/12_timing.py          # E5: single-thread timing at matched recall, components, batch-1 latency (~10 min)

# every script takes a config path; the extensions are the same pipeline with another config:
#   uv run python experiments/<script>.py configs/sift1m_nlist1024.yaml   -> results/nlist1024/   (nlist = 1024)
#   uv run python experiments/<script>.py configs/deep1m.yaml            -> results/deep1m/      (Deep1M, 96-d; downloads 388 MB)
uv run python experiments/13_transfer.py        # stopping rule across datasets: weights, transfer without refit, one joint model
uv run python experiments/14_compact_rule.py    # 3-feature stopping rule and its closed form: d_next/d1 < K (r10/d1)^p c^q
uv run python experiments/15_stoprule_verify.py   # live check of the compact rule + feature ablation, tuned savings with CIs
uv run python experiments/16_fitted_rule_tuned.py   # static fitted rule tune->test with CIs, live check and timing
uv run python experiments/17_fitted_rule_analysis.py  # fitted rule: feature correlations, subsets, greedy selection, 3-feature formula
uv run python experiments/18_extra_features.py        # data-aware extra features (cell radius, local density, ...) for both rules: no gain
```

C++ timing of the two learned rules inside Faiss's scan loop (macOS: `brew install faiss`, 1.16 = the pip version):

```bash
uv run python cpp/export_inputs.py   # test queries, ground truth, stopping-rule constants -> data/cpp/
uv run python cpp/export_fitted.py   # fitted-rule models (17 / 11 / 3 features) -> data/cpp/
./cpp/run.sh                         # stopping rule vs plain Faiss  -> results/cpp_stoprule_bench.csv
./cpp/run.sh fitted                  # fitted rule vs plain Faiss    -> results/cpp_fitted_bench.csv
```

Presentation (open `presentation/adaptive-nprobe.html` in a browser; → / space / click = next
step, ← = back, `f` = fullscreen):

```bash
uv run python presentation/build_data.py && python3 presentation/inject.py   # refresh the deck's numbers from results/
```

## Layout

- `proposal.pdf`, `PLAN.md`, `WORKLOG.md`, `README.md` — proposal, plan, project diary, this file
- `src/ivfadapt/` — library code (data, index, rules, search, oracle, metrics, the stopping rule)
- `experiments/` — numbered experiment scripts (`01`–`18`) and `plots.py`; write to `results/`
- `configs/` — dataset/index/grid settings (`sift1m.yaml`, `sift1m_nlist1024.yaml`, `deep1m.yaml`)
- `tests/` — equivalence, scanned-vector-count, rule, oracle and stopping-rule tests
- `cpp/` — C++ benchmark of the stopping and fitted rules on Faiss's scan code
  - `stoprule_bench.cpp` — plain Faiss vs our scan loop vs both rules, single thread
  - `export_inputs.py`, `export_fitted.py` — write the benchmark's inputs to `data/cpp/`
  - `run.sh` — build and run (`./cpp/run.sh` stopping rule, `./cpp/run.sh fitted` fitted rule)
- `report/` — `results.md` (results section draft), `guide_ru.md` (presenter's guide, Russian)
- `presentation/` — the talk
  - `adaptive-nprobe.html` — the slides: one self-contained file (no build step to view it), same
    engine and design as our IVF+PQ deck; all charts are drawn from data embedded in the file
  - `speaker-notes.md` — what to say on each slide, with timings (~31 min)
  - `build_data.py` — collects every number and curve the slides show from `results/` into `data.json`
    (Deep1M and nlist 1024 numbers are copied from the report, marked in the code)
  - `data.json` — that data, as embedded in the slides
  - `inject.py` — writes `data.json` into the HTML
- `data/`, `results/`, `build/` — git-ignored (datasets and indexes, experiment outputs, C++ binary)
