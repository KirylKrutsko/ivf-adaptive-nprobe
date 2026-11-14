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
```

## Layout

- `src/ivfadapt/` — library code (data, index, rules, search, oracle, metrics, timing)
- `experiments/` — numbered experiment scripts, write to `results/`
- `configs/` — dataset/index/grid settings
- `tests/` — equivalence, scanned-vector-count and oracle tests
- `data/`, `results/` — git-ignored
