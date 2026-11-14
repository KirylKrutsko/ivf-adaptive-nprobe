"""Build / train / save / load the IVF-PQ index; cell id of every base vector; list sizes."""

from __future__ import annotations

import time
from pathlib import Path

import faiss
import numpy as np


def index_tag(cfg: dict) -> str:
    ix = cfg["index"]
    ts = ix.get("train_size")
    train = "full" if ts is None else f"sub{int(ts)}"
    return f"ivf{ix['nlist']}_pq{ix['pq_m']}x{ix['pq_nbits']}_train-{train}_seed{cfg['seed']}"


def index_paths(cfg: dict) -> dict[str, Path]:
    d = Path(cfg["data_dir"])
    tag = index_tag(cfg)
    return {"index": d / f"{tag}.faiss", "cells": d / f"{tag}_cells.npz"}


def training_set(xb: np.ndarray, cfg: dict) -> np.ndarray:
    """Full base (D3) unless cfg.index.train_size is an int, then a fixed random subsample."""
    ts = cfg["index"].get("train_size")
    if ts is None:
        return xb
    rng = np.random.default_rng(cfg["seed"])
    return xb[np.sort(rng.choice(len(xb), size=int(ts), replace=False))]


def build_index(xb: np.ndarray, cfg: dict, verbose: bool = True) -> tuple[faiss.IndexIVFPQ, dict]:
    """Train IndexIVFPQ(IndexFlatL2) on the training set and add the whole base."""
    ix = cfg["index"]
    d = xb.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, ix["nlist"], ix["pq_m"], ix["pq_nbits"])
    index.cp.seed = cfg["seed"]
    index.pq.cp.seed = cfg["seed"]
    index.verbose = verbose

    xt = training_set(xb, cfg)
    t0 = time.perf_counter()
    index.train(xt)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    index.add(xb)
    t_add = time.perf_counter() - t0
    index.verbose = False
    info = {"n_train": int(len(xt)), "train_seconds": t_train, "add_seconds": t_add}
    return index, info


def extract_cells(index: faiss.IndexIVFPQ) -> tuple[np.ndarray, np.ndarray]:
    """From the inverted lists: cell id of every base vector (int32) and list sizes (int64)."""
    invlists = index.invlists
    nlist = index.nlist
    list_sizes = np.array([invlists.list_size(l) for l in range(nlist)], dtype=np.int64)
    cell_of = np.full(index.ntotal, -1, dtype=np.int32)
    for l in range(nlist):
        n = int(list_sizes[l])
        if n == 0:
            continue
        ids = faiss.rev_swig_ptr(invlists.get_ids(l), n).copy()
        cell_of[ids] = l
    assert (cell_of >= 0).all(), "some base vector is in no list"
    assert list_sizes.sum() == index.ntotal
    return cell_of, list_sizes


def save_index(index: faiss.IndexIVFPQ, cfg: dict) -> dict[str, Path]:
    paths = index_paths(cfg)
    paths["index"].parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(paths["index"]))
    cell_of, list_sizes = extract_cells(index)
    np.savez(paths["cells"], cell_of=cell_of, list_sizes=list_sizes)
    return paths


def load_index(cfg: dict) -> tuple[faiss.IndexIVFPQ, np.ndarray, np.ndarray]:
    """Load the cached index plus (cell_of, list_sizes)."""
    paths = index_paths(cfg)
    index = faiss.read_index(str(paths["index"]))
    z = np.load(paths["cells"])
    return index, z["cell_of"], z["list_sizes"]


def load_or_build_index(xb: np.ndarray, cfg: dict) -> tuple[faiss.IndexIVFPQ, np.ndarray, np.ndarray, dict]:
    paths = index_paths(cfg)
    if paths["index"].exists() and paths["cells"].exists():
        index, cell_of, list_sizes = load_index(cfg)
        return index, cell_of, list_sizes, {"cached": True}
    index, info = build_index(xb, cfg)
    save_index(index, cfg)
    cell_of, list_sizes = extract_cells(index)
    info["cached"] = False
    return index, cell_of, list_sizes, info
