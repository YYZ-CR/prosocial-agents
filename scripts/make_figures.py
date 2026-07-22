"""
Generate the paper figures (vector PDF) from scripts/qwen_aggregated.json.

Figures (saved to paper/figures/):
  fig_gain.pdf     Total gain R vs prosociality, 3 contract lines  (the "Angelo Fig-2" replica)
  fig_panel.pdf    2x2: survival m, over-usage o, equality e, efficiency u vs prosociality
  fig_contract.pdf Contract effect on total gain: paper (deterministic) vs Qwen
  fig_proso.pdf    Prosociality effect (p5-p0): Qwen vs paper, per metric (normalized bars)
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(REPO, "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)

AGG = json.load(open(os.path.join(REPO, "scripts", "qwen_aggregated.json")))
CELLS = AGG["cells"]

# ---- validated colorblind-safe categorical palette (dataviz reference, light) ----
C = {"no_contract": "#2a78d6", "nl": "#1baf7a", "code_law": "#eda100"}
LABEL = {"no_contract": "No contract", "nl": "NL contract", "code_law": "Code contract"}
COND_ORDER = ["no_contract", "nl", "code_law"]
PS = list(range(6))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e2",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "figure.dpi": 150,
})


def series(cond, metric):
    xs, ys, los, his = [], [], [], []
    for p in PS:
        cell = CELLS[f"{cond}|p{p}"][metric]
        xs.append(p)
        ys.append(cell["mean"])
        los.append(cell["ci_lo"])
        his.append(cell["ci_hi"])
    return xs, ys, los, his


def line_panel(ax, metric, ylabel, direct_labels=True, label_dy=None):
    for cond in COND_ORDER:
        xs, ys, los, his = series(cond, metric)
        ax.fill_between(xs, los, his, color=C[cond], alpha=0.13, linewidth=0)
        ax.plot(xs, ys, "-", color=C[cond], linewidth=2, zorder=3)
        ax.plot(xs, ys, "o", color=C[cond], markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        if direct_labels:
            dy = 0 if label_dy is None else label_dy.get(cond, 0)
            ax.annotate(LABEL[cond], xy=(xs[-1], ys[-1]), xytext=(6, dy),
                        textcoords="offset points", color=C[cond],
                        fontsize=8.5, fontweight="bold", va="center")
    ax.set_xlabel("Prosocial agents (of 5)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(PS)
    ax.set_xlim(-0.2, 5.9)


# ---------- Figure 1: total gain R (the headline replica) ----------
fig, ax = plt.subplots(figsize=(6.2, 4.0))
line_panel(ax, "total_gain_R", "Total gain  $R$", label_dy={"no_contract": 8, "nl": -2, "code_law": -8})
ax.set_title("Prosociality raises total gain across every contract condition",
             fontsize=11, fontweight="bold", loc="left", pad=10)
fig.tight_layout()
fig.subplots_adjust(right=0.80)
fig.savefig(os.path.join(FIGDIR, "fig_gain.pdf"))
plt.close(fig)

# ---------- Figure 2: 2x2 panel of the other metrics ----------
fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
line_panel(axes[0, 0], "survival_time_m", "Survival  $m$ (months)", direct_labels=False)
axes[0, 0].axhline(12, color="#999", lw=0.8, ls="--")
axes[0, 0].annotate("horizon = 12", xy=(0, 12), xytext=(2, -10),
                    textcoords="offset points", fontsize=7.5, color="#777")
line_panel(axes[0, 1], "over_usage_o", "Over-usage  $o$", direct_labels=False)
line_panel(axes[1, 0], "equality_e", "Equality  $e$", direct_labels=False)
line_panel(axes[1, 1], "efficiency_u", "Efficiency  $u$", direct_labels=False)
# one shared legend
handles = [plt.Line2D([0], [0], color=C[c], lw=2, marker="o", markersize=4.5,
                      markeredgecolor="white", label=LABEL[c]) for c in COND_ORDER]
fig.legend(handles=handles, ncol=3, loc="upper center", frameon=False,
           bbox_to_anchor=(0.5, 1.02), fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(FIGDIR, "fig_panel.pdf"))
plt.close(fig)

# ---------- Figure 3: contract effect on total gain, paper vs Qwen ----------
# Paper (deterministic regime) contract effect on total gain, relative to no-contract:
paper_contract = {"no_contract": 0.0, "nl": 31.8, "code_law": -27.7}
qwen_contract = {c: AGG["contract_effect"][c]["total_gain_R_delta_vs_nocontract"] for c in COND_ORDER}
qwen_err = {c: (AGG["contract_effect"][c]["total_gain_R_delta_ci_lo"],
               AGG["contract_effect"][c]["total_gain_R_delta_ci_hi"]) for c in COND_ORDER}

fig, ax = plt.subplots(figsize=(6.2, 3.8))
import numpy as np
contract_conds = ["nl", "code_law"]  # no_contract is the 0 baseline, not a bar
x = np.arange(len(contract_conds))
w = 0.36
paper_vals = [paper_contract[c] for c in contract_conds]
qwen_vals = [qwen_contract[c] for c in contract_conds]
qwen_lo = [qwen_contract[c] - qwen_err[c][0] for c in contract_conds]
qwen_hi = [qwen_err[c][1] - qwen_contract[c] for c in contract_conds]
ax.bar(x - w/2, paper_vals, w, color="#9aa0a6", label="Paper (5-model avg, det.)",
       zorder=3)
ax.bar(x + w/2, qwen_vals, w, color="#2a78d6",
       yerr=[qwen_lo, qwen_hi], capsize=3, ecolor="#123", error_kw={"lw": 1},
       label="This work (Qwen3-235B)", zorder=3)
ax.axhline(0, color="#333", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels([LABEL[c] for c in contract_conds])
ax.set_ylabel(r"$\Delta$ total gain vs no-contract")
ax.set_title("Contracts help in the paper but hurt for Qwen",
             fontsize=11, fontweight="bold", loc="left", pad=10)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig_contract.pdf"))
plt.close(fig)

# ---------- Figure 4: prosociality effect (p5-p0) Qwen vs paper, normalized ----------
pe = AGG["prosociality_effect"]
paper_proso = {  # arXiv 2605.08426 reported prosociality deltas (mean +/- 95% CI)
    "total_gain_R": (119.6, 41.1),
    "survival_time_m": (4.15, 1.52),
    "over_usage_o": (-0.31, 0.14),
    "equality_e": (0.062, 0.026),
}
metric_lab = {"total_gain_R": r"$\Delta R$", "survival_time_m": r"$\Delta m$",
              "over_usage_o": r"$\Delta o$", "equality_e": r"$\Delta e$"}
order = ["total_gain_R", "survival_time_m", "over_usage_o", "equality_e"]
# normalize each metric's Qwen and paper delta by the paper's |delta| so bars are comparable
fig, ax = plt.subplots(figsize=(6.2, 3.8))
x = np.arange(len(order))
paper_norm, qwen_norm, qwen_err_norm = [], [], []
for m in order:
    pv, pci = paper_proso[m]
    base = abs(pv)
    paper_norm.append(pv / base)
    qwen_norm.append(pe[m]["delta"] / base)
    lo = (pe[m]["delta"] - pe[m]["delta_ci_lo"]) / base
    hi = (pe[m]["delta_ci_hi"] - pe[m]["delta"]) / base
    qwen_err_norm.append((lo, hi))
ax.bar(x - w/2, paper_norm, w, color="#9aa0a6", label="Paper (=1.0 ref.)", zorder=3)
ax.bar(x + w/2, qwen_norm, w, color="#1baf7a",
       yerr=np.array(qwen_err_norm).T, capsize=3, ecolor="#123", error_kw={"lw": 1},
       label="This work (Qwen3-235B)", zorder=3)
ax.axhline(0, color="#333", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels([metric_lab[m] for m in order])
ax.set_ylabel("Effect, normalized to paper $|\\Delta|$")
ax.set_title("Prosociality effect: same sign, larger magnitude than the paper",
             fontsize=10.5, fontweight="bold", loc="left", pad=10)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig_proso.pdf"))
plt.close(fig)

print("Wrote figures to", FIGDIR)
for f in ["fig_gain.pdf", "fig_panel.pdf", "fig_contract.pdf", "fig_proso.pdf"]:
    p = os.path.join(FIGDIR, f)
    print(f"  {f}: {os.path.getsize(p)} bytes")
