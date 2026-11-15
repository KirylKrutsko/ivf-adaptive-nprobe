"""fvecs/ivecs readers, SIFT1M download, ground truth, fixed tune/test query split."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import faiss
import numpy as np

SIFT1M_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT1M_FILES = {
    "base": "sift/sift_base.fvecs",
    "query": "sift/sift_query.fvecs",
    "learn": "sift/sift_learn.fvecs",
    "groundtruth": "sift/sift_groundtruth.ivecs",
}


def read_fvecs(path: str | Path) -> np.ndarray:
    """Read a .fvecs file: each row is int32 dim followed by dim float32 values."""
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()


def read_ivecs(path: str | Path) -> np.ndarray:
    """Read a .ivecs file: each row is int32 dim followed by dim int32 values."""
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:].copy()


def download_sift1m(data_dir: str | Path) -> Path:
    """Download and unpack SIFT1M into data_dir unless the four files are already there."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if all((data_dir / f).exists() for f in SIFT1M_FILES.values()):
        return data_dir
    archive = data_dir / "sift.tar.gz"
    if not archive.exists():
        print(f"downloading {SIFT1M_URL} -> {archive}")
        urllib.request.urlretrieve(SIFT1M_URL, archive)
    print(f"extracting {archive}")
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir)
    missing = [f for f in SIFT1M_FILES.values() if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"after extraction, missing: {missing}")
    return data_dir


def load_sift1m(data_dir: str | Path) -> dict[str, np.ndarray]:
    """Return base (1M x 128), query (10k x 128), learn (100k x 128), groundtruth (10k x 100)."""
    data_dir = download_sift1m(data_dir)
    return {
        "base": read_fvecs(data_dir / SIFT1M_FILES["base"]),
        "query": read_fvecs(data_dir / SIFT1M_FILES["query"]),
        "learn": read_fvecs(data_dir / SIFT1M_FILES["learn"]),
        "groundtruth": read_ivecs(data_dir / SIFT1M_FILES["groundtruth"]),
    }


def compute_ground_truth(
    xb: np.ndarray, xq: np.ndarray, k: int = 100, cache: str | Path | None = None
) -> np.ndarray:
    """Exact k nearest neighbours (L2) of each query, as int64 ids. Cached to `cache` if given."""
    if cache is not None and Path(cache).exists():
        gt = np.load(cache)
        if gt.shape == (len(xq), k):
            return gt
    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)
    _, gt = index.search(xq, k)
    if cache is not None:
        np.save(cache, gt)
    return gt


def compare_ground_truth(gt_ours: np.ndarray, gt_shipped: np.ndarray, k: int = 10) -> dict:
    """How well our top-k sets agree with the shipped ones (ties may reorder or swap ids)."""
    ours = np.sort(gt_ours[:, :k], axis=1)
    shipped = np.sort(gt_shipped[:, :k], axis=1)
    exact = np.all(ours == shipped, axis=1)
    overlap = np.array(
        [len(np.intersect1d(a, b, assume_unique=True)) for a, b in zip(ours, shipped)]
    )
    return {
        "k": k,
        "queries_with_identical_topk": int(exact.sum()),
        "n_queries": int(len(ours)),
        "mean_overlap": float(overlap.mean() / k),
        "min_overlap": int(overlap.min()),
    }


def query_split(n_queries: int, tune_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed random split of query ids into (tune, test). Both arrays are sorted."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_queries)
    n_tune = int(round(tune_fraction * n_queries))
    return np.sort(perm[:n_tune]), np.sort(perm[n_tune:])


# ---- Deep1M: first 1M base vectors of Yandex DEEP (96-d, L2), from big-ann-benchmarks -----

DEEP_BASE_URL = "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin"
DEEP_QUERY_URL = "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin"
DEEP_N_BASE = 1_000_000


def read_fbin(path: str | Path, n_max: int | None = None) -> np.ndarray:
    """Read a .fbin file (int32 n, int32 d, float32 data), optionally only the first n_max rows."""
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
        n = int(n) if n_max is None else min(int(n), n_max)
        return np.fromfile(f, dtype=np.float32, count=n * int(d)).reshape(n, int(d))


def download_deep1m(data_dir: str | Path) -> Path:
    """Fetch the first 1M rows of DEEP base.10M.fbin with an HTTP range request, and the queries."""
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    base, query = data_dir / "base.1M.fbin", data_dir / "query.public.10K.fbin"
    if not base.exists():
        n_bytes = 8 + DEEP_N_BASE * 96 * 4
        req = urllib.request.Request(DEEP_BASE_URL, headers={"Range": f"bytes=0-{n_bytes - 1}"})
        print(f"downloading first {DEEP_N_BASE} rows of {DEEP_BASE_URL}")
        with urllib.request.urlopen(req) as r, open(base, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
    if not query.exists():
        urllib.request.urlretrieve(DEEP_QUERY_URL, query)
    return data_dir


def load_deep1m(data_dir: str | Path) -> dict[str, np.ndarray]:
    """base (1M x 96), query (10k x 96). No learn set and no shipped ground truth for this prefix."""
    data_dir = download_deep1m(data_dir)
    xb = read_fbin(data_dir / "base.1M.fbin", DEEP_N_BASE)
    assert xb.shape == (DEEP_N_BASE, 96), xb.shape
    return {"base": xb, "query": read_fbin(data_dir / "query.public.10K.fbin")}


def load_dataset(cfg: dict) -> dict[str, np.ndarray]:
    """Dispatch on cfg['dataset']. Keys: base, query, and for sift1m also learn, groundtruth."""
    name = cfg["dataset"]
    if name == "sift1m":
        return load_sift1m(cfg["data_dir"])
    if name == "deep1m":
        return load_deep1m(cfg["data_dir"])
    raise ValueError(f"unknown dataset {name!r}")
