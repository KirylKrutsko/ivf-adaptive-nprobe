"""Plots of PLAN §5 from the files in results/ (no searching). Writes results/plots/*.png.

1 recall vs work (all methods), 2 cells-needed histogram, 3 rule vs oracle, 4 saving vs target
(all methods, CIs), 6 time vs work (E5), 7 tail recall at equal work. Plot 5 is made by
06_rule_within_nprobe.py. Plot 1 of the proposal (ratio rule only) is kept as 1b."""

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
CAP, STOP, FITTED = "#b8860b", "#8e44ad", "#7f8c8d"

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
    ax.set_title(f"Recall vs work, {DATASET} test queries (nlist {NLIST})", loc="left", color=INK)
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


def method_curves(cfg: dict) -> pd.DataFrame:
    """(method, recall, work) curves on the test split for every method, from stored results."""
    from ivfadapt.oracle import at_cells, load_curves, subset
    from ivfadapt.env import load_env
    res = Path(cfg["results_dir"]); kr = cfg["search"]["k_rerank"]
    c05 = pd.read_csv(res / "05_curves_test.csv").rename(columns={"work_mean": "work"})
    rows = [c05[c05.method == m][["method", "recall", "work"]] for m in ["fixed", "rule", "oracle_lagrangian", "oracle_perquery"]]
    rows[1] = rows[1].assign(method="ratio")
    # ratio + cap 32 (ε swept) read off the per-query curves
    env = load_env(cfg["config_path"]); curves, _ = load_curves(res / "03_oracle_curves.npz"); test = subset(curves, env.test)
    rc = np.load(res / "04_rule_cells.npz")
    cap = [{"method": "ratio_cap32", "recall": at_cells(test, np.minimum(n[env.test], 32))["recall"].mean(),
            "work": at_cells(test, np.minimum(n[env.test], 32))["work"].mean() + kr} for n in rc["n_cells"]]
    rows.append(pd.DataFrame(cap))
    c11 = pd.read_csv(res / "11_curves_test.csv"); c11 = c11[c11.target_fit == 0.9]
    rows.append(pd.DataFrame({"method": "stop", "recall": c11.recall, "work": c11.work}))
    c08 = pd.read_csv(res / "08_fitted_rule.csv"); c08 = c08[c08.family == "lagr0.9"]
    rows.append(pd.DataFrame({"method": "fitted", "recall": c08.recall, "work": c08.work}))
    return pd.concat(rows, ignore_index=True)


STYLES = {"fixed": ("fixed nprobe", FIXED, "-"), "ratio": ("ratio rule (proposal)", RULE, "-"),
          "ratio_cap32": ("ratio rule + cap 32", CAP, "-"), "fitted": ("fitted static rule", FITTED, "-"),
          "stop": ("stopping rule (scan feedback)", STOP, "-"),
          "oracle_lagrangian": ("oracle, best mean recall", ORACLE, "-"),
          "oracle_perquery": ("oracle, every query reaches target", ORACLE, "--")}


NLIST = 4096
DATASET = "SIFT1M"


def plot_recall_vs_work_all(mc: pd.DataFrame, stop_head: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, (label, color, ls) in STYLES.items():
        d = mc[mc.method == m].sort_values("work"); d = d[d.recall >= 0.6]
        ax.plot(d.work, d.recall, ls, color=color, label=label, linewidth=2.2 if m in ("fixed", "stop") else 1.6,
                marker="o" if m == "oracle_perquery" else None, markersize=4)
    ax.axhline(0.9, color=INK2, linewidth=0.8, linestyle=":")
    r9 = stop_head[stop_head.target == 0.9].iloc[0]
    ax.annotate(f"at recall 0.9: stopping rule {r9.saving_stop_curve:.2f}× less work than fixed nprobe, "
                f"oracle {(r9.W_fixed + 200) / (r9.W_oracle + 200):.2f}× less",
                xy=(0.02, 0.96), xycoords="axes fraction", color=INK2, fontsize=9, va="top")
    ax.set_xscale("log"); ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("work per query: scanned vectors + 200 re-ranked (log scale)")
    ax.set_ylabel("recall@10 (after re-ranking)")
    ax.set_title(f"Recall vs work, {DATASET} test queries (nlist {NLIST})", loc="left", color=INK)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_saving_vs_target_all(head05: pd.DataFrame, head11: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = head05.target.astype(str)
    ax.plot(x, head05.max_saving_lagrangian, "-o", color=ORACLE, label="oracle (ceiling)", markersize=7)
    ax.errorbar(x, head11.saving_stop_tuned, yerr=[head11.saving_stop_tuned - head11.saving_stop_tuned_ci_lo,
                head11.saving_stop_tuned_ci_hi - head11.saving_stop_tuned], fmt="-o", color=STOP,
                label="stopping rule", markersize=7, capsize=3)
    ax.errorbar(x, head11.saving_cap_tuned, yerr=[head11.saving_cap_tuned - head11.saving_cap_tuned_ci_lo,
                head11.saving_cap_tuned_ci_hi - head11.saving_cap_tuned], fmt="-o", color=CAP,
                label="ratio rule + cap", markersize=7, capsize=3)
    ax.errorbar(x, head05.saving_tuned, yerr=[head05.saving_tuned - head05.saving_tuned_ci_lo,
                head05.saving_tuned_ci_hi - head05.saving_tuned], fmt="-o", color=RULE,
                label="ratio rule (proposal)", markersize=7, capsize=3)
    ax.axhline(1, color=INK2, linewidth=0.8)
    ax.set_yscale("log"); ax.set_yticks([0.7, 1, 1.5, 2, 3, 4]); ax.set_yticklabels(["0.7×", "1×", "1.5×", "2×", "3×", "4×"])
    ax.set_xlabel("recall@10 target"); ax.set_ylabel("work of fixed nprobe / work of method")
    ax.set_title("Saving vs fixed nprobe, tuned on tune, measured on test (95% CI)", loc="left", color=INK, fontsize=10.5)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_timing(tim: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    d = tim[tim.target == 0.9].copy()
    names = {"fixed, plain index.search": "fixed, plain\nFaiss", "fixed, preassigned path": "fixed, pre-\nassigned path",
             "fixed, preassigned in rounds (overhead check)": "same cells,\n26 calls", "ratio rule, cap 32": "ratio rule\n+ cap 32",
             "stopping rule (rounds)": "stopping\nrule"}
    d = d[d.method.isin(names)]; d["label"] = d.method.map(names)
    comps = [("quantizer+scan_ms", "quantizer + scan", FIXED), ("quantizer_ms", "quantizer", "#9ec5f0"), ("rule_ms", "rule", STOP),
             ("scan_ms", "scan", FIXED), ("scan+merge_ms", "scan + merge", "#5b9be0"), ("rerank_ms", "re-rank", INK2)]
    bottom = np.zeros(len(d))
    for col, label, color in comps:
        v = d[col].fillna(0).values
        if v.sum() == 0: continue
        ax.bar(d.label, v, bottom=bottom, color=color, label=label, width=0.65); bottom += v
    for i, (t, w) in enumerate(zip(d.total_ms, d.work)):
        ax.text(i, t + 0.03, f"{t:.2f} ms\n{w:.0f} work", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_ylabel("ms per query (1 thread, batch of 5000)"); ax.set_ylim(0, d.total_ms.max() * 1.55)
    ax.set_title("Time per query at recall 0.9", loc="left", color=INK, fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    ax = axes[1]
    for m, color, label in [("ratio rule, cap", CAP, "ratio rule + cap"), ("stopping rule (rounds)", STOP, "stopping rule")]:
        g = tim[tim.method.str.startswith(m)].sort_values("target")
        plain = tim[tim.method == "fixed, plain index.search"].set_index("target").loc[g.target]
        ax.plot(g.target.astype(str), plain.work.values / g.work.values, "--o", color=color, label=f"{label}: work saving", markersize=6)
        ax.plot(g.target.astype(str), g.speedup_vs_plain, "-o", color=color, label=f"{label}: time speed-up", markersize=6)
    ax.axhline(1, color=INK2, linewidth=0.8)
    ax.set_xlabel("recall@10 target"); ax.set_ylabel("vs plain Faiss (> 1 = better)")
    ax.set_title("Work saving vs measured speed-up", loc="left", color=INK, fontsize=10.5)
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_tail(tail: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    order = [("fixed", "fixed\nnprobe", FIXED), ("ratio", "ratio\nrule", RULE), ("ratio_cap32", "ratio\n+ cap 32", CAP),
             ("fitted (08)", "fitted\nstatic", FITTED), ("stopping rule (11)", "stopping\nrule", STOP), ("oracle_lagrangian", "oracle", ORACLE)]
    levels = sorted(tail.nprobe_level.unique())[:2]
    for ax, lvl in zip(axes, levels):
        d = tail[tail.nprobe_level == lvl].set_index("method")
        vals = [100 * d.loc[m, "share_recall_le_0.5"] for m, _, _ in order]
        ax.bar([l for _, l, _ in order], vals, color=[c for _, _, c in order], width=0.65)
        for i, (v, m) in enumerate(zip(vals, [m for m, _, _ in order])):
            ax.text(i, v + 0.2, f"{v:.1f}%\nR={d.loc[m, 'recall_mean']:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_title(f"budget of fixed nprobe {lvl} (work ≈ {d.loc['fixed', 'work']:.0f})", loc="left", color=INK, fontsize=10)
    axes[0].set_ylabel("queries with recall@10 ≤ 0.5 (%)")
    axes[0].set_ylim(0, tail[tail.nprobe_level == levels[0]]["share_recall_le_0.5"].max() * 100 * 1.35)
    fig.suptitle("Badly served queries at equal mean work (test split)", x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main(cfg_path: str = "configs/sift1m.yaml") -> None:
    cfg = load_config(cfg_path)
    global NLIST, DATASET; NLIST = cfg["index"]["nlist"]; DATASET = {"sift1m": "SIFT1M", "deep1m": "Deep1M"}.get(cfg["dataset"], cfg["dataset"])
    res = Path(cfg["results_dir"]); out = res / "plots"; out.mkdir(exist_ok=True)
    curves, head = pd.read_csv(res / "05_curves_test.csv"), pd.read_csv(res / "05_tune_test.csv")
    head11 = pd.read_csv(res / "11_stop_rule_rounds.csv")
    plot_recall_vs_work_all(method_curves(cfg), head11, out / "1_recall_vs_work.png")
    plot_recall_vs_work(curves, head, out / "1b_recall_vs_work_ratio_only.png")
    plot_histogram(np.load(res / "03_oracle_hist.npz"), out / "2_cells_needed_hist.png")
    plot_rule_vs_oracle(cfg, head, out / "3_rule_vs_oracle.png")
    plot_saving_vs_target_all(head, head11, out / "4_saving_vs_target.png")
    plot_saving_vs_target(head, out / "4b_saving_vs_target_ratio_only.png")
    if (res / "12_timing.csv").exists():
        plot_timing(pd.read_csv(res / "12_timing.csv"), out / "6_timing.png")
    plot_tail(pd.read_csv(res / "09_tail_recall.csv"), out / "7_tail_recall.png")
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main(*sys.argv[1:])
