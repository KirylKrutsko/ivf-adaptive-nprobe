"""Load everything an experiment needs: config, vectors, ground truth, split, index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from ivfadapt.config import load_config
from ivfadapt.data import load_sift1m
from ivfadapt.index import load_index


@dataclass
class Env:
    cfg: dict
    xb: np.ndarray          # base vectors (float32)
    xq: np.ndarray          # all queries
    gt: np.ndarray          # exact top-100 ids for all queries
    tune: np.ndarray        # query ids of the tune split
    test: np.ndarray        # query ids of the test split
    index: faiss.IndexIVFPQ
    cell_of: np.ndarray     # cell id of every base vector
    list_sizes: np.ndarray  # vectors per cell

    @property
    def k(self) -> int:
        return int(self.cfg["k"])

    @property
    def k_rerank(self) -> int:
        return int(self.cfg["search"]["k_rerank"])

    @property
    def max_cells(self) -> int:
        return int(self.cfg["search"]["max_cells"])


def load_env(cfg_path: str | Path = "configs/sift1m.yaml") -> Env:
    """Requires experiments/01_build.py to have run (index and ground truth cached)."""
    cfg = load_config(cfg_path)
    data_dir = Path(cfg["data_dir"])
    data = load_sift1m(data_dir)
    gt = np.load(data_dir / "groundtruth_k100.npy")
    split = np.load(data_dir / f"query_split_seed{cfg['seed']}.npz")
    index, cell_of, list_sizes = load_index(cfg)
    return Env(cfg, data["base"], data["query"], gt, split["tune"], split["test"], index, cell_of, list_sizes)
