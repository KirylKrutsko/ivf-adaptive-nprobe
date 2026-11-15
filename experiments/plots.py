"""Plots 1-4 of PLAN §5 from the files in results/ (no searching). Writes results/plots/*.png."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ivfadapt.config import load_config

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FIXED, RULE, ORACLE = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
    "lines.linewidth": 2,
})


def plot_recall_vs_work(curves: pd.DataFrame, head: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    styles = {"fixed": ("fixed nprobe", FIXED, "-"), "rule": ("ratio rule (ε sweep)", RULE, "-"),
              "oracle_lagrangian": ("oracle, best mean recall (Lagrangian)", ORACLE, "-"),
              "oracle_perquery": ("oracle, every query reaches the target", ORACLE, "--")}
    for m, (label, color, ls) in styles.items():
        d = curves[curves.method == m].sort_values("work_mean")
        d = d[d.recall >= 0.5]
        ax.plot(d.work_mean, d.recall, ls, color=color, label=label,
                marker="o" if m == "oracle_perquery" else None, markersize=4)
    ax.axhline(0.9, color=INK2, linewidth=0.8, linestyle=":")
    r9 = head[head.target == 0.9].iloc[0]
    rule_txt = (f"rule {r9.saving_curve:.2f}× less work" if r9.saving_curve >= 1
                else f"rule {1 / r9.saving_curve:.2f}× more work")
    ax.annotate(f"at recall 0.9 vs fixed nprobe: {rule_txt}, oracle {r9.max_saving_lagrangian:.2f}× less",
                xy=(r9.W_rule + 200, 0.9), xytext=(0.02, 0.95), textcoords="axes fraction",
                color=INK2, fontsize=9, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("work per query: scanned vectors + 200 re-ranked (log scale)")
    ax.set_ylabel("recall@10 (after re-ranking)")
    ax.set_title("Recall vs work, SIFT1M test queries", loc="left", color=INK)
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_histogram(hist: np.lib.npyio.NpzFile, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    edges = np.unique(np.round(np.geomspace(1, 257, 18)).astype(int))
    for ax, t in zip(axes, ["0.9", "1.0"]):
        c = hist[f"c_star_{t}"]
        reach = c[c > 0]
        counts, _ = np.histogram(reach, bins=edges)
        share = counts / len(c)
        ax.bar(edges[:-1], share, width=np.diff(edges) * 0.9, align="edge", color=FIXED)
        unreachable = (c < 0).mean()
        ax.bar([330], [unreachable], width=80, color=INK2)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30, 100, 360]); ax.set_xticklabels(["1", "3", "10", "30", "100", "never"])
        ax.set_title(f"target {t}: median {int(np.median(reach))} cells", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("cells this query needs (oracle)")
    axes[0].set_ylabel("share of test queries")
    fig.suptitle("How many cells each query needs to reach its recall target", x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_rule_vs_oracle(cfg: dict, head: pd.DataFrame, path: Path) -> None:
    res = Path(cfg["results_dir"])
    rc, hist = np.load(res / "04_rule_cells.npz"), np.load(res / "03_oracle_hist.npz")
    eps = float(head[head.target == 0.9].iloc[0].eps_tuned)
    j = int(np.argmin(np.abs(rc["eps"] - eps)))
    rule = rc["n_cells"][j, hist["query_ids"]]
    orac = hist["c_star_0.9"]
    ok = orac > 0
    rule, orac = rule[ok], orac[ok]
    rho = pd.Series(rule).rank().corr(pd.Series(orac).rank())       # Spearman = Pearson on ranks
    fig, ax = plt.subplots(figsize=(5.6, 5))
    bins = np.geomspace(1, 257, 30)
    h = ax.hist2d(orac, rule, bins=[bins, bins], cmap="Blues", cmin=1)
    ax.plot([1, 256], [1, 256], color=INK2, linewidth=1, linestyle=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(False)
    ax.set_xlabel("cells needed for recall 0.9 (oracle)")
    ax.set_ylabel(f"cells the rule picks (ε = {eps:.3f})")
    ax.set_title(f"Rule vs oracle per query, Spearman ρ = {rho:.2f}", loc="left", color=INK)
    fig.colorbar(h[3], ax=ax, label="queries")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_saving_vs_target(head: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = head.target.astype(str)
    ax.plot(x, head.max_saving_lagrangian, "-o", color=ORACLE, label="oracle (Lagrangian)", markersize=8)
    ax.errorbar(x, head.saving_curve, yerr=[head.saving_curve - head.saving_curve_ci_lo,
                head.saving_curve_ci_hi - head.saving_curve], fmt="-o", color=RULE, label="ratio rule",
                markersize=8, capsize=3)
    ax.axhline(1, color=INK2, linewidth=0.8)
    ax.set_xlabel("recall@10 target"); ax.set_ylabel("saving vs fixed nprobe (> 1 = less work)")
    ax.set_title("Saving vs fixed nprobe at matched recall (test, 95% CI)", loc="left", color=INK)
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    cfg = load_config(cfg_path)
    res = Path(cfg["results_dir"]); out = res / "plots"; out.mkdir(exist_ok=True)
    curves, head = pd.read_csv(res / "05_curves_test.csv"), pd.read_csv(res / "05_tune_test.csv")
    plot_recall_vs_work(curves, head, out / "1_recall_vs_work.png")
    plot_histogram(np.load(res / "03_oracle_hist.npz"), out / "2_cells_needed_hist.png")
    plot_rule_vs_oracle(cfg, head, out / "3_rule_vs_oracle.png")
    plot_saving_vs_target(head, out / "4_saving_vs_target.png")
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main(*sys.argv[1:])
