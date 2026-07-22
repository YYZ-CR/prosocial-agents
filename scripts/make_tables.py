"""Emit LaTeX table fragments from scripts/qwen_aggregated.json into paper/tables/."""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TBL = os.path.join(REPO, "paper", "tables")
os.makedirs(TBL, exist_ok=True)
D = json.load(open(os.path.join(REPO, "scripts", "qwen_aggregated.json")))
CELLS = D["cells"]
COND = ["no_contract", "nl", "code_law"]
LAB = {"no_contract": "No contract", "nl": "NL contract", "code_law": "Code contract"}

# ---- Table: prosociality effect, Qwen vs paper ----
pe = D["prosociality_effect"]
paper = {  # arXiv 2605.08426 reported deltas (mean +/- 95% CI), pooled models/regimes
    "total_gain_R": ("$+119.6 \\pm 41.1$", "$R$ (total gain)"),
    "survival_time_m": ("$+4.15 \\pm 1.52$", "$m$ (survival, months)"),
    "over_usage_o": ("$-0.31 \\pm 0.14$", "$o$ (over-usage)"),
    "equality_e": ("$+0.062 \\pm 0.026$", "$e$ (equality)"),
}
order = ["total_gain_R", "survival_time_m", "over_usage_o", "equality_e"]
rows = []
for m in order:
    v = pe[m]
    half = (v["delta_ci_hi"] - v["delta_ci_lo"]) / 2.0
    if m == "total_gain_R":
        q = f"$+{v['delta']:.1f} \\pm {half:.1f}$"
    elif m == "survival_time_m":
        q = f"$+{v['delta']:.2f} \\pm {half:.2f}$"
    else:
        sign = "+" if v["delta"] >= 0 else "-"
        q = f"${sign}{abs(v['delta']):.3f} \\pm {half:.3f}$"
    agree = "\\checkmark"
    rows.append(f"{paper[m][1]} & {paper[m][0]} & {q} & {agree} \\\\")
with open(os.path.join(TBL, "prosociality.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(rows) + "\n")

# ---- Table: contract effect on total gain, Qwen vs paper ----
ce = D["contract_effect"]
paper_c = {"nl": "$+31.8$", "code_law": "$-27.7$"}
rows = []
for c in ["nl", "code_law"]:
    dl = ce[c]["total_gain_R_delta_vs_nocontract"]
    lo = ce[c]["total_gain_R_delta_ci_lo"]
    hi = ce[c]["total_gain_R_delta_ci_hi"]
    half = (hi - lo) / 2.0
    q = f"${dl:+.1f} \\pm {half:.1f}$"
    rows.append(f"{LAB[c]} & {paper_c[c]} & {q} \\\\")
with open(os.path.join(TBL, "contract.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(rows) + "\n")

# ---- Appendix: full per-cell grid (R and m) ----
rows = []
for c in COND:
    cells = []
    for p in range(6):
        R = CELLS[f"{c}|p{p}"]["total_gain_R"]["mean"]
        cells.append(f"{R:.0f}")
    rows.append(f"{LAB[c]} & " + " & ".join(cells) + " \\\\")
with open(os.path.join(TBL, "grid_R.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(rows) + "\n")

rows = []
for c in COND:
    cells = []
    for p in range(6):
        m = CELLS[f"{c}|p{p}"]["survival_time_m"]["mean"]
        cells.append(f"{m:.1f}")
    rows.append(f"{LAB[c]} & " + " & ".join(cells) + " \\\\")
with open(os.path.join(TBL, "grid_m.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(rows) + "\n")

print("Wrote table fragments to", TBL)
