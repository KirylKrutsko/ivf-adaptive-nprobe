# ivfadapt — query-adaptive nprobe for IVF+PQ

Kiryl Krutsko, Matvei Sendetskii.

Instead of scanning a fixed `nprobe` cells per query, scan every cell whose centroid is at most
`(1 + ε)` times as far as the nearest one. We compare this rule with fixed `nprobe` and with an
oracle on SIFT1M, measuring recall@10 against the number of scanned vectors.

See `proposal.pdf` for the proposal, `PLAN.md` for the implementation and experiment plan,
and `WORKLOG.md` for progress notes and the reasoning behind decisions.

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

# every script takes a config path; extension 3 (nlist = 1024) is the same pipeline with
#   uv run python experiments/<script>.py configs/sift1m_nlist1024.yaml   -> results/nlist1024/
```

## Layout

- `src/ivfadapt/` — library code (data, index, rules, search, oracle, metrics, timing)
- `experiments/` — numbered experiment scripts, write to `results/`
- `configs/` — dataset/index/grid settings
- `tests/` — equivalence, scanned-vector-count and oracle tests
- `data/`, `results/` — git-ignored
