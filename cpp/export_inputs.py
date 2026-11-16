"""Write the inputs of cpp/stoprule_bench.cpp to data/cpp/ (SIFT1M test split).

- queries.f32: test queries (nq x d float32), gt10.i64: their exact top-10 ids
- meta.txt: nq d / index path / base .fvecs path / one line per recall target:
  R nprobe_tuned K p q   (nprobe from 11, compact-formula constants from 15)
- cells_R<R>.i32: number of cells the Python round-based formula scans per test query, so the
  C++ rule can be checked query by query.
Run after experiments 11 and 15:  uv run python cpp/export_inputs.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from ivfadapt.data import SIFT1M_FILES
from ivfadapt.env import load_env
from ivfadapt.index import index_paths
from ivfadapt.rules import quantizer_topk
from ivfadapt.stoprule import run_rounds, static_features

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify15", ROOT / "experiments" / "15_stoprule_verify.py")
verify15 = importlib.util.module_from_spec(spec); spec.loader.exec_module(verify15)


def main() -> None:
    env = load_env()
    res = Path(env.cfg["results_dir"])
    out = Path(env.cfg["data_dir"]).parent / "cpp"; out.mkdir(parents=True, exist_ok=True)
    xq = np.ascontiguousarray(env.xq[env.test], dtype=np.float32)
    xq.tofile(out / "queries.f32")
    np.ascontiguousarray(env.gt[env.test][:, :env.k], dtype=np.int64).tofile(out / "gt10.i64")
    t11 = pd.read_csv(res / "11_stop_rule_rounds.csv").set_index("target")
    t15 = pd.read_csv(res / "15_compact_live.csv")
    t15 = t15[t15.variant == "formula"].set_index("target")
    Dq, Iq = quantizer_topk(env.index, xq, env.max_cells)
    st = static_features(Dq, Iq, env.list_sizes)
    lines = [f"{len(xq)} {xq.shape[1]}", str(index_paths(env.cfg)["index"]),
             str(Path(env.cfg["data_dir"]) / SIFT1M_FILES["base"]), str(len(env.cfg["recall_targets"]))]
    for R in env.cfg["recall_targets"]:
        f = t15.loc[R]
        lines.append(f"{R} {int(t11.loc[R, 'nprobe_tuned'])} {float(f.K)!r} {float(f.a_r10)!r} {float(f.b_cells)!r}")
        model = verify15.Formula(float(f.K), float(f.a_r10), float(f.b_cells))
        run = run_rounds(env.index, xq, Dq, Iq, st, env.k, env.k_rerank, model=model, theta=0.5)
        run["n_cells"].astype(np.int32).tofile(out / f"cells_R{R}.i32")
    (out / "meta.txt").write_text("\n".join(lines) + "\n")
    print("wrote", out, "\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
