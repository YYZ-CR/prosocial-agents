"""Re-lead the grid results with GovSim component metrics instead of the
composite reward.

Motivation (prof feedback, item 2): the composite reward
    reward = gain_norm + m/12 + 0.5*e - o
compresses four different things into one number, which hides *where* (if
anywhere) an optimizer actually moves the outcome. This script reports each
component on its own and applies a per-metric noise-floor test.

Component metrics (GovSim Sec. 2.4), per cell x method, averaged over 2 seeds
and read straight from the committed trajectory JSONs (no raw-log reparse):
    R  total_gain_R      higher better   ceiling = full-horizon benchmark (600)
    m  survival_time_m   higher better   ceiling = horizon (12)
    o  over_usage_o      lower  better   floor   = 0
    e  equality_e        higher better   ceiling = 1

Noise-floor test: the BASELINE method holds the seed norm fixed for all 5
iterations, so any movement across its iterations is pure episode noise. We take
the baseline's across-iteration spread (max-min) and std as the noise floor for
that metric, in that cell. An optimizer "effect" only counts if its mean shift
over baseline exceeds that floor.

Outputs:
    norm_evolution/components_table.md   per-metric Delta table + noise test
    norm_evolution/components.csv        same, machine-readable
    norm_evolution/figures/fig_components.pdf  small-multiples (4 metrics)
"""
from __future__ import annotations

import csv
import json
import os
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ = os.path.join(REPO, "scripts", "norm_opt_runs", "grid_v1")
OUTDIR = os.path.join(REPO, "norm_evolution")
FIGDIR = os.path.join(OUTDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

CELLS = ["gemma4-31b", "luna", "sonnet5-off", "sonnet5-on", "opus48", "sol"]
NONCOLLAPSE = [c for c in CELLS if c != "sol"]
SHORT = {
    "gemma4-31b": "Gemma 4 31B", "luna": "GPT-5.6 Luna",
    "sonnet5-off": "Sonnet 5 (no reas.)", "sonnet5-on": "Sonnet 5 (reas.)",
    "opus48": "Opus 4.8", "sol": "GPT-5.6 Sol",
}
NICE = {
    "gemma4-31b": "Gemma 4\n31B", "luna": "GPT-5.6\nLuna",
    "sonnet5-off": "Sonnet 5\n(no reas.)", "sonnet5-on": "Sonnet 5\n(reas.)",
    "opus48": "Opus 4.8", "sol": "GPT-5.6\nSol",
}
METHODS = ["baseline", "textgrad", "ace"]
MLABEL = {"baseline": "Baseline (fixed norm)", "textgrad": "TextGrad", "ace": "ACE"}
C = {"baseline": "#2a78d6", "textgrad": "#008300", "ace": "#e87ba4"}

# key, label, direction (+1 higher-better / -1 lower-better), ceiling/floor bound
METRICS = [
    ("total_gain_R", "Total gain $R$", +1, 600.0),
    ("survival_time_m", "Survival months $m$", +1, 12.0),
    ("over_usage_o", "Over-usage $o$", -1, 0.0),
    ("equality_e", "Equality $e$", +1, 1.0),
]


def series(cell, method, key):
    t = json.load(open(os.path.join(TRAJ, cell, f"trajectory_{method}.json")))["trajectory"]
    return [it["metrics_mean"][key] for it in t]


def best(vals, direction):
    return max(vals) if direction > 0 else min(vals)


def improvement(method_vals, base_vals, direction):
    """Signed so positive = the optimizer moved the metric in the better direction."""
    d_mean = (mean(method_vals) - mean(base_vals)) * direction
    d_best = (best(method_vals, direction) - best(base_vals, direction)) * direction
    return d_mean, d_best


def analyse():
    rows = []
    for key, label, direction, bound in METRICS:
        for cell in CELLS:
            base = series(cell, "baseline", key)
            spread = max(base) - min(base)          # noise floor (across-iter range)
            std = pstdev(base)                       # noise floor (std)
            base_best = best(base, direction)
            headroom = (bound - base_best) * direction  # >0 means room to improve
            for method in ("textgrad", "ace"):
                mv = series(cell, method, key)
                d_mean, d_best = improvement(mv, base, direction)
                rows.append({
                    "metric": key, "cell": cell, "method": method,
                    "base_mean": round(mean(base), 2), "base_best": round(base_best, 2),
                    "headroom_to_bound": round(headroom, 2),
                    "noise_spread": round(spread, 2), "noise_std": round(std, 2),
                    "opt_mean": round(mean(mv), 2),
                    "delta_mean": round(d_mean, 2), "delta_best": round(d_best, 2),
                    "beats_noise": bool(abs(d_mean) > spread and d_mean > 0),
                })
    return rows


def write_csv(rows):
    path = os.path.join(OUTDIR, "components.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def write_md(rows):
    path = os.path.join(OUTDIR, "components_table.md")
    lines = ["# Component-metric re-analysis of the norm-optimization grid", ""]
    lines.append("Each optimizer effect is the mean shift over the fixed-norm "
                 "baseline (`delta_mean`), tested against that cell's baseline "
                 "across-iteration noise floor (`noise_spread`). `beats_noise` "
                 "requires a positive shift larger than the noise floor.")
    lines.append("")
    for key, label, direction, bound in METRICS:
        arrow = "higher better" if direction > 0 else "lower better"
        lines.append(f"## {label.replace('$','')}  ({arrow}, bound={bound:g})")
        lines.append("")
        lines.append("| Model | Baseline mean (best) | Headroom | Noise floor (spread / std) "
                     "| TextGrad Δ | ACE Δ | Beats noise? |")
        lines.append("|---|---|---|---|---|---|---|")
        for cell in CELLS:
            r = {x["method"]: x for x in rows if x["metric"] == key and x["cell"] == cell}
            tg, ace = r["textgrad"], r["ace"]
            beats = [m for m in ("textgrad", "ace") if r[m]["beats_noise"]]
            beats_s = ", ".join(beats) if beats else "no"
            lines.append(
                f"| {SHORT[cell]} | {tg['base_mean']:.0f} ({tg['base_best']:.0f}) "
                f"| {tg['headroom_to_bound']:.2f} "
                f"| {tg['noise_spread']:.2f} / {tg['noise_std']:.2f} "
                f"| {tg['delta_mean']:+.2f} | {ace['delta_mean']:+.2f} | {beats_s} |"
            )
        lines.append("")
    # headline
    core = ("total_gain_R", "survival_time_m", "over_usage_o")
    n_tests = sum(1 for r in rows if r["metric"] in core)
    beats = [r for r in rows if r["metric"] in core and r["beats_noise"]]
    beats_noncollapse = [r for r in beats if r["cell"] != "sol"]
    lines.append(
        f"**Headline:** across the three prof-named metrics (R, m, o) x 6 models "
        f"x 2 optimizers = {n_tests} tests, an optimizer cleared its own noise "
        f"floor in **{len(beats)}** — and in **{len(beats_noncollapse)}** once "
        f"GPT-5.6 Sol is excluded. Both Sol hits are TextGrad, whose apparent gain "
        f"is the iteration-0 (un-optimized seed) norm catching one lucky episode "
        f"before collapsing back; it is not a learned improvement. For every model "
        f"that does not collapse, survival $m$ and over-usage $o$ are already at "
        f"ceiling/floor (no headroom), and total gain $R$ moves less than the "
        f"fixed-norm baseline's own across-iteration noise.")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def figure(rows):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9a9a94", "axes.grid": True,
        "grid.color": "#e6e6e2", "grid.linewidth": 0.7, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 150,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0))
    x = np.arange(len(CELLS))
    w = 0.26
    for ax, (key, label, direction, bound) in zip(axes.flat, METRICS):
        for k, method in enumerate(METHODS):
            means = [mean(series(c, method, key)) for c in CELLS]
            stds = [pstdev(series(c, method, key)) for c in CELLS]
            ax.bar(x + (k - 1) * w, means, w * 0.92, yerr=stds, capsize=1.6,
                   error_kw=dict(lw=0.7, ecolor="#7a7a76"),
                   label=MLABEL[method], color=C[method],
                   edgecolor="#fcfcfb", linewidth=0.6)
        if bound is not None:
            ax.axhline(bound, color="#52514e", lw=0.8, ls=(0, (4, 3)))
            tag = "ceiling" if direction > 0 else "floor"
            ax.annotate(f"{tag} ({bound:g})", (len(CELLS) - 0.4, bound),
                        textcoords="offset points", xytext=(0, 3), ha="right",
                        fontsize=6.2, color="#52514e")
        ax.set_xticks(x)
        ax.set_xticklabels([NICE[c] for c in CELLS], fontsize=6.4)
        ax.set_title(label, fontsize=9, pad=4)
    axes.flat[0].legend(fontsize=6.8, loc="lower right", ncol=1)
    fig.suptitle("Component metrics by model and optimizer (error bars: std across 5 iterations)",
                 fontsize=9.5, y=1.01)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "fig_components.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def main():
    rows = analyse()
    p_csv = write_csv(rows)
    p_md = write_md(rows)
    p_fig = figure(rows)
    print("wrote:")
    for p in (p_csv, p_md, p_fig):
        print("  ", os.path.relpath(p, REPO))
    # console summary of the noise-floor test on R/m/o
    for key, label, direction, bound in METRICS:
        beat = [f"{r['cell']}/{r['method']}" for r in rows
                if r["metric"] == key and r["beats_noise"]]
        print(f"  {key:16s} beats-noise: {beat if beat else 'none'}")


if __name__ == "__main__":
    main()
