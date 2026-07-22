"""
Aggregate the Qwen3-235B reproduction grid into per-(condition x prosociality)
paper metrics, with bootstrap 95% CIs, and dump JSON/CSV for the write-up.

Walks every fishing_run_summary.json under simulation/results/qwen235-*, parses
the contract condition and prosocial-count from the folder name, and aggregates
the five paper metrics (m, R, u, e, o) across seed replicates.
"""
from __future__ import annotations

import json
import os
import re
import glob
import random
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "simulation", "results")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_fishing_log import summarize, parse_log  # noqa: E402
from pathlib import Path  # noqa: E402

COND_LABEL = {"no_contract": "No contract", "nl": "NL contract", "code_law": "Code contract"}
COND_ORDER = ["no_contract", "nl", "code_law"]

PATH_RE = re.compile(r"repro_qwen235-(no_contract|code_law|nl)-p(\d)")


def _model_from_configs() -> str:
    """The model id the grid actually ran, read from the runs' own Hydra configs."""
    import glob as _glob

    import yaml as _yaml

    seen: set[str] = set()
    pattern = os.path.join(RESULTS, "qwen235-*", "**", ".hydra", "config.yaml")
    for path in _glob.glob(pattern, recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                cfg = _yaml.safe_load(handle) or {}
        except Exception:  # noqa: BLE001
            continue
        model = (cfg.get("llm") or {}).get("path")
        if model:
            seen.add(str(model))
    if not seen:
        return "unknown"
    return sorted(seen)[0] if len(seen) == 1 else "MIXED: " + ", ".join(sorted(seen))
CAPACITY = 100.0
EXPECTED_REGEN = 2.0
COLLAPSE_THRESHOLD = 0.0

METRICS = ["survival_time_m", "total_gain_R", "efficiency_u", "equality_e", "over_usage_o"]


def bootstrap_ci(values, n=5000, seed=0):
    """95% CI of the mean via nonparametric bootstrap."""
    vals = [v for v in values if v is not None]
    if not vals:
        return (None, None, None)
    if len(vals) == 1:
        return (vals[0], vals[0], vals[0])
    rng = random.Random(seed)
    means = []
    k = len(vals)
    for _ in range(n):
        sample = [vals[rng.randrange(k)] for _ in range(k)]
        means.append(sum(sample) / k)
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n)]
    return (sum(vals) / k, lo, hi)


def load_runs():
    """Compute paper metrics from every log_env.json in the Qwen grid.

    (Most run dirs have no pre-computed fishing_run_summary.json, so we recompute
    from the raw env log to get all 90 attempted runs.)"""
    runs = []
    pattern = os.path.join(RESULTS, "qwen235-*", "**", "log_env.json")
    for path in glob.glob(pattern, recursive=True):
        m = PATH_RE.search(path.replace("\\", "/"))
        if not m:
            continue
        cond, pk = m.group(1), int(m.group(2))
        try:
            records = parse_log(Path(path))
            data = summarize(records, capacity=CAPACITY, expected_regen=EXPECTED_REGEN,
                             collapse_threshold=COLLAPSE_THRESHOLD)
        except Exception as e:
            print(f"  !! could not summarize {path}: {e}")
            continue
        pm = data.get("paper_metrics", {})
        rounds = data.get("rounds_played", 0)
        # A degenerate / crashed run has essentially no gameplay.
        degenerate = rounds == 0 or data.get("num_agents", 0) != 5
        runs.append({
            "cond": cond,
            "prosocial": pk,
            "rounds": rounds,
            "degenerate": degenerate,
            "path": path,
            **{mk: pm.get(mk) for mk in METRICS},
        })
    return runs


def aggregate(runs):
    cells = defaultdict(list)
    for r in runs:
        if r["degenerate"]:
            continue
        cells[(r["cond"], r["prosocial"])].append(r)

    table = {}
    for (cond, pk), rs in cells.items():
        cell = {"n": len(rs)}
        for mk in METRICS:
            mean, lo, hi = bootstrap_ci([r[mk] for r in rs], seed=hash((cond, pk, mk)) & 0xFFFF)
            cell[mk] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
        table[f"{cond}|p{pk}"] = cell
    return table


def bootstrap_delta_ci(a, b, n=5000, seed=0):
    """95% CI for mean(b) - mean(a), resampling each group independently."""
    a = [v for v in a if v is not None]
    b = [v for v in b if v is not None]
    if not a or not b:
        return (None, None, None)
    rng = random.Random(seed)
    ka, kb = len(a), len(b)
    diffs = []
    for _ in range(n):
        ma = sum(a[rng.randrange(ka)] for _ in range(ka)) / ka
        mb = sum(b[rng.randrange(kb)] for _ in range(kb)) / kb
        diffs.append(mb - ma)
    diffs.sort()
    delta = sum(b) / kb - sum(a) / ka
    return (delta, diffs[int(0.025 * n)], diffs[int(0.975 * n)])


def prosociality_effect(runs):
    """Delta between fully-prosocial (p5) and fully-selfish (p0), pooled across
    all three contract conditions -- mirrors the paper's headline prosociality
    effect (their Delta R = +119.6, Delta m = +4.15, etc.)."""
    p0 = [r for r in runs if r["prosocial"] == 0 and not r["degenerate"]]
    p5 = [r for r in runs if r["prosocial"] == 5 and not r["degenerate"]]
    out = {}
    for mk in METRICS:
        m0, _, _ = bootstrap_ci([r[mk] for r in p0], seed=1)
        m5, _, _ = bootstrap_ci([r[mk] for r in p5], seed=2)
        delta, dlo, dhi = bootstrap_delta_ci([r[mk] for r in p0], [r[mk] for r in p5],
                                             seed=(hash(mk) & 0xFFFF))
        out[mk] = {"p0_mean": m0, "p5_mean": m5, "delta": delta,
                   "delta_ci_lo": dlo, "delta_ci_hi": dhi, "n_p0": len(p0), "n_p5": len(p5)}
    return out


def contract_effect(runs):
    """Mean metric per contract condition, pooled across prosociality --
    compare to paper's deterministic-regime NL +31.8, code -27.7 on total gain."""
    out = {}
    for cond in COND_ORDER:
        rs = [r for r in runs if r["cond"] == cond and not r["degenerate"]]
        out[cond] = {}
        for mk in METRICS:
            mean, lo, hi = bootstrap_ci([r[mk] for r in rs], seed=3)
            out[cond][mk] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(rs)}
    # deltas relative to no_contract on total gain, with bootstrap CI
    nc = [r["total_gain_R"] for r in runs if r["cond"] == "no_contract" and not r["degenerate"]]
    for cond in COND_ORDER:
        cc = [r["total_gain_R"] for r in runs if r["cond"] == cond and not r["degenerate"]]
        delta, dlo, dhi = bootstrap_delta_ci(nc, cc, seed=(hash(cond) & 0xFFFF))
        out[cond]["total_gain_R_delta_vs_nocontract"] = delta
        out[cond]["total_gain_R_delta_ci_lo"] = dlo
        out[cond]["total_gain_R_delta_ci_hi"] = dhi
    return out


def main():
    runs = load_runs()
    n_total = len(runs)
    n_degen = sum(1 for r in runs if r["degenerate"])
    print(f"Loaded {n_total} run summaries ({n_degen} degenerate/failed).")

    for cond in COND_ORDER:
        for pk in range(6):
            rs = [r for r in runs if r["cond"] == cond and r["prosocial"] == pk and not r["degenerate"]]
            print(f"  {cond:12s} p{pk}: {len(rs)} runs")

    out = {
        "meta": {
            # Read from the runs' own .hydra/config.yaml rather than hardcoded: the grid
            # actually ran qwen/qwen3-235b-a22b-2507 (a distinct, later release), not the
            # base qwen/qwen3-235b-a22b that was previously asserted here.
            "model": _model_from_configs(),
            "regime": "deterministic (r_t = 2.0 constant)",
            "n_runs_total": n_total,
            "n_degenerate": n_degen,
            "n_used": n_total - n_degen,
        },
        "cells": aggregate(runs),
        "prosociality_effect": prosociality_effect(runs),
        "contract_effect": contract_effect(runs),
    }

    out_path = os.path.join(REPO, "scripts", "qwen_aggregated.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    # flat CSV of per-run metrics for reference
    csv_path = os.path.join(REPO, "scripts", "qwen_per_run.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("cond,prosocial,rounds,degenerate," + ",".join(METRICS) + "\n")
        for r in sorted(runs, key=lambda x: (x["cond"], x["prosocial"])):
            f.write(f"{r['cond']},{r['prosocial']},{r['rounds']},{int(r['degenerate'])}," +
                    ",".join(str(r[mk]) for mk in METRICS) + "\n")
    print(f"Wrote {csv_path}")

    # quick echo of headline numbers
    pe = out["prosociality_effect"]
    print("\n=== Prosociality effect (p5 - p0, pooled across conditions) ===")
    for mk in METRICS:
        print(f"  {mk:18s}: p0={pe[mk]['p0_mean']:.3f}  p5={pe[mk]['p5_mean']:.3f}  delta={pe[mk]['delta']:+.3f}")
    print("\n=== Contract effect on total gain R (pooled across prosociality) ===")
    for cond in COND_ORDER:
        ce = out["contract_effect"][cond]
        print(f"  {COND_LABEL[cond]:14s}: R={ce['total_gain_R']['mean']:7.2f}  "
              f"delta_vs_nocontract={ce['total_gain_R_delta_vs_nocontract']:+7.2f}  (n={ce['total_gain_R']['n']})")


if __name__ == "__main__":
    main()
