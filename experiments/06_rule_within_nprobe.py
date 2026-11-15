"""Side experiment: the rule as a drop-in for a fixed nprobe.

Every query scans min(n_rule(ε), nprobe) cells: hard queries keep the usual nprobe, easy ones
get fewer. For the nprobe values tuned in E4 (recall targets 0.8 .. 0.99) and every ε of the E3
grid, report recall and work on the test split, the recall lost and the work saved relative to
plain nprobe, and the control: the work plain IVF needs to reach the *same* recall by lowering
nprobe for everyone (log-interpolated on the fixed curve). frontier_gain > 1 means the rule's
trade beats simply lowering nprobe.

Needs 03_oracle.py, 04_rule.py, 05_tune_test.py. Writes results/06_rule_within_nprobe.csv
and results/plots/5_rule_within_nprobe.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ivfadapt.env import load_env
from ivfadapt.metrics import interp_work_at_recall
from ivfadapt.oracle import at_cells, load_curves, subset

HIGHLIGHT_EPS = [0.238, 0.286]


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    env = load_env(cfg_path)
    res = Path(env.cfg["results_dir"])
    kr = env.k_rerank
    curves, _ = load_curves(res / "03_oracle_curves.npz")
    test = subset(curves, env.test)
    rc = np.load(res / "04_rule_cells.npz")
    eps_grid, rule = rc["eps"], rc["n_cells"][:, env.test]
    head = pd.read_csv(res / "05_tune_test.csv")
    nprobes = head["nprobe_tuned"].astype(int).tolist()

    fixed_r, fixed_w = test.recall.mean(0), test.work.mean(0) + kr              # c = 1..C_max
    eps_ids = sorted({int(np.argmin(np.abs(eps_grid - e))) for e in HIGHLIGHT_EPS} | set(range(len(eps_grid))))

    rows = []
    for p in nprobes:
        base_r, base_w = fixed_r[p - 1], fixed_w[p - 1]
        for j in eps_ids:
            n = np.minimum(rule[j], p)
            a = at_cells(test, n)
            r, w = a["recall"].mean(), a["work"].mean() + kr
            w_same_recall = interp_work_at_recall(fixed_r, fixed_w, r)
            rows.append({"nprobe": p, "eps": eps_grid[j], "recall": r, "work": w,
                         "recall_fixed": base_r, "work_fixed": base_w,
                         "recall_drop": base_r - r, "work_saving": base_w / w,
                         "share_reduced": (n < p).mean(), "cells_mean": n.mean(),
                         "fixed_work_same_recall": w_same_recall, "frontier_gain": w_same_recall / w,
                         "highlight": any(abs(eps_grid[j] - e) < 1e-3 for e in HIGHLIGHT_EPS)})
    df = pd.DataFrame(rows)
    df.to_csv(res / "06_rule_within_nprobe.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["nprobe", "eps", "recall_fixed", "recall", "recall_drop", "work_fixed", "work", "work_saving",
            "share_reduced", "cells_mean", "fixed_work_same_recall", "frontier_gain"]
    print("highlighted ε (test split):")
    print(df[df.highlight][cols].round(4).to_string(index=False))
    print("\nbest frontier_gain over the whole ε grid, per nprobe:")
    best = df.loc[df.groupby("nprobe")["frontier_gain"].idxmax()]
    print(best[cols].round(4).to_string(index=False))
    for p in nprobes:
        d = df[(df.nprobe == p) & (df.recall_drop <= 0.01)].sort_values("work_saving").tail(1)
        if len(d):
            print(f"nprobe {p}: largest saving with recall drop <= 0.01: {d.work_saving.iloc[0]:.3f}x "
                  f"(ε {d.eps.iloc[0]:.3f}, drop {d.recall_drop.iloc[0]:.4f}, frontier_gain {d.frontier_gain.iloc[0]:.3f})")

    # plot: fixed curve + one ε trace per nprobe
    import plots  # noqa: E402  (same folder; reuses the style)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(fixed_w, fixed_r, color=plots.FIXED, label="fixed nprobe")
    for p in nprobes:
        d = df[df.nprobe == p].sort_values("work")
        ax.plot(d.work, d.recall, color=plots.RULE, linewidth=1.5,
                label="min(rule(ε), nprobe), ε swept" if p == nprobes[0] else None)
        ax.plot([fixed_w[p - 1]], [fixed_r[p - 1]], "o", color=plots.FIXED, markersize=7)
        ax.annotate(f"nprobe {p}", (fixed_w[p - 1], fixed_r[p - 1]), xytext=(6, -12),
                    textcoords="offset points", color=plots.INK2, fontsize=8)
        h = d[d.highlight]
        ax.plot(h.work, h.recall, "o", color=plots.RULE, markersize=6,
                label="ε = 0.238 / 0.286" if p == nprobes[0] else None)
    ax.set_xscale("log"); ax.set_xlim(fixed_w[0] * 0.9, fixed_w[min(200, len(fixed_w)) - 1])
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("work per query: scanned vectors + 200 re-ranked (log scale)")
    ax.set_ylabel("recall@10 (after re-ranking)")
    ax.set_title("Rule as a drop-in: never more than nprobe cells", loc="left", color=plots.INK)
    ax.legend(loc="lower right")
    fig.tight_layout(); (res / "plots").mkdir(exist_ok=True)
    fig.savefig(res / "plots" / "5_rule_within_nprobe.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main(*sys.argv[1:])
