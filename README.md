# ivfadapt — query-adaptive nprobe for IVF+PQ

Kiryl Krutsko, Matvei Sendetskii.

Instead of scanning a fixed `nprobe` cells per query, scan every cell whose centroid is at most
`(1 + ε)` times as far as the nearest one. We compare this rule with fixed `nprobe` and with an
oracle on SIFT1M, measuring recall@10 against the number of scanned vectors.

See `proposal.pdf` for the proposal and `PLAN.md` for the implementation and experiment plan.

## Setup

```bash
uv sync            # creates .venv with Python 3.12 and all dependencies
uv run pytest      # correctness tests
```

## Layout

- `src/ivfadapt/` — library code (data, index, rules, search, oracle, metrics, timing)
- `experiments/` — numbered experiment scripts, write to `results/`
- `configs/` — dataset/index/grid settings
- `tests/` — equivalence, scanned-vector-count and oracle tests
- `data/`, `results/` — git-ignored
