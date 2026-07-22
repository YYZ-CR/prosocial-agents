"""Generate paper figures (vector PDF) for the norm-evolution grid.

Figures -> norm_evolution/figures/
  fig_reward_by_model.pdf   Mean reward by model x optimizer (grouped bars)
  fig_trajectories.pdf      Reward vs optimizer iteration, pooled (the null result)
  fig_gain_survival.pdf     Total gain + survival months by model (two panels)
  fig_noise.pdf             Optimizer gain vs the fixed-norm noise floor

Palette: dataviz reference categorical slots 1-3 (blue/green/magenta), validated
light-mode. Contrast WARN on magenta is relieved by direct value labels.
"""
from __future__ import annotations

import json
import os
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ = os.path.join(REPO, "scripts", "norm_opt_runs", "grid_v1")
RESULTS = os.path.join(REPO, "simulation", "results", "grid_v1")
FIGDIR = os.path.join(REPO, "norm_evolution", "figures")
os.makedirs(FIGDIR, exist_ok=True)

import sys
sys.path.insert(0, os.path.join(REPO, "scripts"))
from summarize_fishing_log import summarize, parse_log  # noqa: E402
from pathlib import Path  # noqa: E402

CELLS = ["gemma4-31b", "luna", "sonnet5-off", "sonnet5-on", "opus48", "sol"]
NICE = {
    "gemma4-31b": "Gemma 4\n31B",
    "luna": "GPT-5.6\nLuna",
    "sonnet5-off": "Sonnet 5\n(no reas.)",
    "sonnet5-on": "Sonnet 5\n(reasoning)",
    "opus48": "Opus 4.8",
    "sol": "GPT-5.6\nSol",
}
SHORT = {
    "gemma4-31b": "Gemma 4 31B", "luna": "GPT-5.6 Luna",
    "sonnet5-off": "Sonnet 5 (no reas.)", "sonnet5-on": "Sonnet 5 (reas.)",
    "opus48": "Opus 4.8", "sol": "GPT-5.6 Sol",
}
METHODS = ["baseline", "textgrad", "ace"]
MLABEL = {"baseline": "Baseline (fixed norm)", "textgrad": "TextGrad", "ace": "ACE"}
# dataviz reference categorical, light mode, slots 1-3
C = {"baseline": "#2a78d6", "textgrad": "#008300", "ace": "#e87ba4"}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e6e2"
SINGLE = "#2a78d6"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#9a9a94",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "legend.frameon": False,
})


def traj(cell, method):
    p = os.path.join(TRAJ, cell, f"trajectory_{method}.json")
    return json.load(open(p))["trajectory"]


def rewards(cell, method):
    return [i["reward"] for i in traj(cell, method)]


# ---------------------------------------------------------------- fig 1
def fig_reward_by_model():
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    x = np.arange(len(CELLS))
    w = 0.26
    for k, m in enumerate(METHODS):
        vals = [mean(rewards(c, m)) for c in CELLS]
        pos = x + (k - 1) * w
        ax.bar(pos, vals, w * 0.92, label=MLABEL[m], color=C[m],
               edgecolor="#fcfcfb", linewidth=0.8)
        for xi, v in zip(pos, vals):
            ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                        xytext=(0, 2 if v >= 0 else -9), ha="center",
                        fontsize=6.0, color=INK2)
    ax.axhline(0, color="#9a9a94", linewidth=0.8)
    ax.axhline(2.5, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.annotate("theoretical maximum (2.5)", (len(CELLS) - 0.45, 2.5),
                textcoords="offset points", xytext=(0, 3), ha="right",
                fontsize=6.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([NICE[c] for c in CELLS], fontsize=7.5)
    ax.set_ylabel("Mean reward")
    ax.set_ylim(-0.75, 2.95)
    ax.legend(loc="lower left", fontsize=7, ncol=3, bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_reward_by_model.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig_trajectories():
    pooled = [c for c in CELLS if c != "sol"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)

    ax = axes[0]
    for m in METHODS:
        ys = [mean([rewards(c, m)[i] for c in pooled]) for i in range(5)]
        es = [stdev([rewards(c, m)[i] for c in pooled]) for i in range(5)]
        ax.plot(range(5), ys, color=C[m], linewidth=2, marker="o",
                markersize=4.5, label=MLABEL[m], zorder=3)
        ax.fill_between(range(5), np.array(ys) - np.array(es),
                        np.array(ys) + np.array(es), color=C[m], alpha=0.10, lw=0)
    ax.set_xlabel("Optimizer iteration"); ax.set_ylabel("Mean reward")
    ax.set_xticks(range(5))
    ax.set_title("Pooled across 5 non-collapse models", fontsize=8, color=INK, pad=6)
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    for m in METHODS:
        ys = rewards("sol", m)
        ax.plot(range(5), ys, color=C[m], linewidth=2, marker="o", markersize=4.5)
    ax.axhline(0, color="#9a9a94", linewidth=0.8)
    ax.set_xlabel("Optimizer iteration"); ax.set_xticks(range(5))
    ax.set_title("GPT-5.6 Sol (collapse regime)", fontsize=8, color=INK, pad=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_trajectories.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def cell_metrics():
    """Mean paper-metrics per cell over every completed episode."""
    out = {}
    for c in CELLS:
        gains, survs = [], []
        pat = os.path.join(RESULTS, c, "*", "*", "log_env.json")
        import glob
        for lg in glob.glob(pat):
            try:
                rep = summarize(parse_log(Path(lg)), capacity=100.0,
                                expected_regen=2.0, collapse_threshold=0.0)
            except Exception:  # noqa: BLE001
                continue
            pm = rep["paper_metrics"]
            gains.append(pm["total_gain_R"]); survs.append(pm["survival_time_m"])
        out[c] = (mean(gains) if gains else 0, mean(survs) if survs else 0, len(gains))
    return out


def fig_gain_survival(cm):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    x = np.arange(len(CELLS))
    for ax, idx, lab, cap in [
        (axes[0], 0, "Total gain $R$", None),
        (axes[1], 1, "Survival months $m$", 12),
    ]:
        vals = [cm[c][idx] for c in CELLS]
        ax.bar(x, vals, 0.62, color=SINGLE, edgecolor="#fcfcfb", linewidth=0.8)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.0f}" if idx == 0 else f"{v:.1f}", (xi, v),
                        textcoords="offset points", xytext=(0, 2), ha="center",
                        fontsize=6.5, color=INK2)
        if cap:
            ax.axhline(cap, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
            ax.annotate("horizon", (len(CELLS) - 0.4, cap), textcoords="offset points",
                        xytext=(0, 3), ha="right", fontsize=6.5, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[c] for c in CELLS], fontsize=7,
                           rotation=32, ha="right")
        ax.set_ylabel(lab)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_gain_survival.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig_noise():
    """The diagnostic: optimizer gain vs the fixed-norm noise floor."""
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    x = np.arange(len(CELLS))
    noise = [max(rewards(c, "baseline")) - min(rewards(c, "baseline")) for c in CELLS]
    ax.bar(x, noise, 0.62, color="#d8d8d2", edgecolor="#fcfcfb", linewidth=0.8,
           label="Noise floor (spread of a FIXED norm)", zorder=1)
    # The one point that exceeds its noise bar is Sol/TextGrad, whose "best" is
    # the iteration-0 (un-optimized) norm drawing a lucky episode -- not an
    # optimizer gain. Render it hollow and label it so the figure can't mislead.
    sol_i = CELLS.index("sol")
    for k, m in enumerate(["textgrad", "ace"]):
        gains = [max(rewards(c, m)) - max(rewards(c, "baseline")) for c in CELLS]
        xs = x + (k - 0.5) * 0.18
        face = [C[m]] * len(CELLS)
        edge = ["#fcfcfb"] * len(CELLS)
        artifact = (m == "textgrad")
        if artifact:
            face[sol_i] = "#fcfcfb"; edge[sol_i] = C[m]
        ax.scatter(xs, gains, s=44, c=face, zorder=3,
                   label=f"{MLABEL[m]} gain over baseline",
                   edgecolors=edge, linewidth=1.1)
        if artifact:
            ax.annotate("iteration-0 artifact\n(not an optimizer gain)",
                        (xs[sol_i], gains[sol_i]), textcoords="offset points",
                        xytext=(-6, -2), ha="right", va="center",
                        fontsize=6.2, color=INK2)
    ax.axhline(0, color="#9a9a94", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([NICE[c] for c in CELLS], fontsize=7.5)
    ax.set_ylabel("Reward difference")
    ax.set_ylim(-0.55, 1.62)
    ax.legend(fontsize=6.6, loc="upper left", ncol=1, handletextpad=0.5,
              borderaxespad=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_noise.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    cm = cell_metrics()
    fig_reward_by_model()
    fig_trajectories()
    fig_gain_survival(cm)
    fig_noise()
    print("wrote figures to", FIGDIR)
    for c in CELLS:
        g, s, n = cm[c]
        print(f"  {c:12s} gain={g:6.1f} surv={s:5.2f}  (n={n})")


if __name__ == "__main__":
    main()
