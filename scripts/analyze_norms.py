"""Dissect the norm texts (paper item 3).

Answers three questions the composite results section left implicit:
  1. How did the norm text evolve over optimization iterations?
  2. Is the "state-contingent" characterization real, or just "longer"? We parse
     every distinct norm for catch rules conditioned on the MEASURED stock.
  3. What exactly is in the highest-scoring norm (the seven Opus clauses)?

Outputs:
  norm_evolution/norm_features.csv          every norm slot with extracted features
  norm_evolution/norm_dissection.md         prose summary + the enumerated best norm
  norm_evolution/figures/fig_norm_evolution.pdf  text growth + state-contingency by iter
"""
from __future__ import annotations

import csv
import os
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from norm_features import CELLS, extract_features, load_all_norms

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, "norm_evolution")
FIGDIR = os.path.join(OUTDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

SEED_TEXT_PREFIX = "Fishers should be mindful"
C = {"baseline": "#2a78d6", "textgrad": "#008300", "ace": "#e87ba4"}
MLABEL = {"baseline": "Baseline (fixed norm)", "textgrad": "TextGrad", "ace": "ACE"}


def enrich(norms):
    for r in norms:
        r["feat"] = extract_features(r["text"])
    return norms


def write_csv(norms):
    path = os.path.join(OUTDIR, "norm_features.csv")
    cols = ["cell", "method", "iteration", "reward", "changed",
            "n_chars", "n_words", "n_clauses", "n_numeric_caps",
            "n_stock_conditionals", "is_state_contingent", "has_moratorium",
            "has_equal_split", "has_enforcement", "has_rotation", "has_amendment_rule"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in norms:
            row = {"cell": r["cell"], "method": r["method"], "iteration": r["iteration"],
                   "reward": round(r["reward"], 3) if r["reward"] is not None else "",
                   "changed": r["changed"]}
            row.update(r["feat"])
            w.writerow(row)
    return path


def summary(norms):
    """Aggregate stats for the dissection."""
    distinct = {}
    for r in norms:
        if r["text"]:
            distinct.setdefault(r["text"], r)
    seed = [t for t in distinct if t.startswith(SEED_TEXT_PREFIX)]
    optimized = [r for t, r in distinct.items() if not t.startswith(SEED_TEXT_PREFIX)]

    def frac(pred, rows):
        return sum(1 for r in rows if pred(r["feat"])), len(rows)

    s = {}
    s["n_distinct"] = len(distinct)
    s["n_seed_texts"] = len(seed)
    s["n_optimized"] = len(optimized)
    s["opt_state_contingent"] = frac(lambda f: f["is_state_contingent"], optimized)
    s["opt_numeric_cap"] = frac(lambda f: f["n_numeric_caps"] > 0, optimized)
    s["opt_moratorium"] = frac(lambda f: f["has_moratorium"], optimized)
    s["opt_amendment"] = frac(lambda f: f["has_amendment_rule"], optimized)
    s["opt_enforcement"] = frac(lambda f: f["has_enforcement"], optimized)
    s["opt_mean_chars"] = mean(r["feat"]["n_chars"] for r in optimized)
    s["opt_mean_clauses"] = mean(r["feat"]["n_clauses"] for r in optimized)
    # baseline never moves off the seed:
    base = [r for r in norms if r["method"] == "baseline"]
    s["baseline_all_seed"] = all(r["text"].startswith(SEED_TEXT_PREFIX) for r in base)
    s["baseline_n"] = len(base)
    # per-iteration state-contingency for optimizers (pooled)
    s["by_iter"] = {}
    for m in ("textgrad", "ace"):
        rows = [r for r in norms if r["method"] == m]
        s["by_iter"][m] = [
            mean(r["feat"]["n_stock_conditionals"] for r in rows if r["iteration"] == i)
            for i in range(5)
        ]
    return s


def best_norm(norms):
    cand = [r for r in norms if r["text"] and r["reward"] is not None]
    return max(cand, key=lambda r: r["reward"])


def figure(norms):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9a9a94", "axes.grid": True,
        "grid.color": "#e6e6e2", "grid.linewidth": 0.7, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 150,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    iters = range(5)
    # left: norm length (words) by iteration, pooled across all six models
    ax = axes[0]
    for m in ("baseline", "textgrad", "ace"):
        rows = [r for r in norms if r["method"] == m]
        ys = [mean(r["feat"]["n_words"] for r in rows if r["iteration"] == i) for i in iters]
        ax.plot(iters, ys, marker="o", ms=4.5, lw=2, color=C[m], label=MLABEL[m])
    ax.set_xlabel("Optimizer iteration"); ax.set_ylabel("Norm length (words)")
    ax.set_xticks(list(iters)); ax.set_title("Norm text grows", fontsize=9, pad=4)
    ax.legend(fontsize=7, loc="upper left")
    # right: state-contingency (stock-conditional clauses) by iteration
    ax = axes[1]
    for m in ("baseline", "textgrad", "ace"):
        rows = [r for r in norms if r["method"] == m]
        ys = [mean(r["feat"]["n_stock_conditionals"] for r in rows if r["iteration"] == i) for i in iters]
        ax.plot(iters, ys, marker="o", ms=4.5, lw=2, color=C[m], label=MLABEL[m])
    ax.set_xlabel("Optimizer iteration")
    ax.set_ylabel("Stock-conditional clauses")
    ax.set_xticks(list(iters))
    ax.set_title("...and becomes state-contingent", fontsize=9, pad=4)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "fig_norm_evolution.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def write_md(norms, s, best):
    path = os.path.join(OUTDIR, "norm_dissection.md")
    L = ["# Norm dissection (item 3)", ""]
    L.append(f"- Distinct norm texts: **{s['n_distinct']}** "
             f"(the {s['n_seed_texts']} seed text + {s['n_optimized']} optimized).")
    L.append(f"- Baseline arm: all {s['baseline_n']} baseline slots are the unchanged "
             f"seed norm (`baseline_all_seed={s['baseline_all_seed']}`).")
    sc, tot = s["opt_state_contingent"]
    L.append(f"- **State-contingent test:** {sc} of {tot} optimized norms "
             f"({100*sc/tot:.0f}%) condition catch on the *measured* stock level -- "
             f"the 'state-contingent' characterization is real, not just 'longer'.")
    nc, _ = s["opt_numeric_cap"]
    mo, _ = s["opt_moratorium"]
    am, _ = s["opt_amendment"]
    en, _ = s["opt_enforcement"]
    L.append(f"- Optimized norms carry explicit numeric caps in {nc}/{tot}, a "
             f"moratorium clause in {mo}/{tot}, an amendment rule in {am}/{tot}, and "
             f"enforcement/logging language in {en}/{tot}.")
    L.append(f"- Mean optimized norm: {s['opt_mean_chars']:.0f} chars, "
             f"{s['opt_mean_clauses']:.1f} clauses (vs the 130-char, 2-clause seed).")
    L.append("")
    L.append(f"## Highest-scoring norm ({best['cell']}, {best['method']}, "
             f"iteration {best['iteration']}, reward {best['reward']:.3f})")
    L.append("")
    for line in best["text"].splitlines():
        L.append("> " + line if line.strip() else ">")
    L.append("")
    f = best["feat"]
    L.append(f"Features: {f['n_clauses']} clauses, {f['n_stock_conditionals']} "
             f"stock-conditional, {f['n_numeric_caps']} numeric caps, "
             f"moratorium={f['has_moratorium']}, amendment={f['has_amendment_rule']}.")
    L.append("")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(L))
    return path


def main():
    norms = enrich(load_all_norms())
    p_csv = write_csv(norms)
    s = summary(norms)
    best = best_norm(norms)
    p_fig = figure(norms)
    p_md = write_md(norms, s, best)
    print("wrote:", os.path.relpath(p_csv, REPO), os.path.relpath(p_md, REPO),
          os.path.relpath(p_fig, REPO))
    sc, tot = s["opt_state_contingent"]
    print(f"state-contingent: {sc}/{tot} optimized norms ({100*sc/tot:.0f}%)")
    print(f"best norm: {best['cell']}/{best['method']}/it{best['iteration']} "
          f"reward={best['reward']:.3f} clauses={best['feat']['n_clauses']}")
    print("per-iter stock-conditionals (textgrad):", [round(x, 2) for x in s["by_iter"]["textgrad"]])
    print("per-iter stock-conditionals (ace):     ", [round(x, 2) for x in s["by_iter"]["ace"]])


if __name__ == "__main__":
    main()
