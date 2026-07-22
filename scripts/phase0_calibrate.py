"""Phase 0 calibration: one 12-round episode per model config, measure real token
usage and price it against live OpenRouter rates. Replaces the $16.87-derived
estimate in the proposal with measurement.

Runs the five Phase-0 cells concurrently (each is its own subprocess episode):
  gemma-4-31b-it | sonnet-5 reasoning off | sonnet-5 reasoning on | 5.6-luna | 5.6-sol

Usage:
  .venv/Scripts/python.exe scripts/phase0_calibrate.py
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.scenarios.fishing.optimization.episode import run_episode  # noqa: E402

SEED_NORM = (
    "Each fisher should take only a fair, sustainable share so the lake keeps "
    "regenerating. Aim for roughly equal catches and avoid depleting the stock."
)

# (label, model, reasoning). reasoning None = model default.
CELLS = [
    ("gemma-4-31b",     "google/gemma-4-31b-it",      None),
    ("sonnet5-off",     "anthropic/claude-sonnet-5",  "off"),
    ("sonnet5-on",      "anthropic/claude-sonnet-5",  "on"),
    ("gpt56-luna",      "openai/gpt-5.6-luna",         None),
    ("gpt56-sol",       "openai/gpt-5.6-sol",          None),
]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "norm_opt_runs")
RESULTS_PREFIX = "phase0_calib"


def live_prices() -> dict[str, tuple[float, float]]:
    """model id -> ($/input token, $/output token)."""
    ms = httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
    out = {}
    for m in ms:
        pr = m.get("pricing", {})
        out[m["id"]] = (float(pr.get("prompt", 0)), float(pr.get("completion", 0)))
    return out


def run_one(label: str, model: str, reasoning: str | None):
    res = run_episode(
        SEED_NORM, model=model, seed=0, condition="nl", prosocial=0,
        max_rounds=12, reasoning=reasoning, results_prefix=RESULTS_PREFIX,
        timeout=9000,
    )
    usage = None
    if res.storage_dir:
        p = os.path.join(res.storage_dir, "token_usage.json")
        if os.path.exists(p):
            usage = json.load(open(p))
    return label, model, reasoning, res, usage


def main():
    prices = live_prices()
    print(f"Launching {len(CELLS)} calibration episodes concurrently...\n", flush=True)

    rows = []
    with cf.ThreadPoolExecutor(max_workers=len(CELLS)) as ex:
        futs = {ex.submit(run_one, *c): c[0] for c in CELLS}
        for fut in cf.as_completed(futs):
            label, model, reasoning, res, usage = fut.result()
            in_price, out_price = prices.get(model, (0.0, 0.0))
            if usage:
                t = usage["totals"]
                cost = t["prompt_tokens"] * in_price + t["completion_tokens"] * out_price
                rows.append({
                    "label": label, "model": model, "reasoning": reasoning,
                    "ok": res.ok, "calls": t["calls"],
                    "prompt_tokens": t["prompt_tokens"],
                    "completion_tokens": t["completion_tokens"],
                    "episode_cost": round(cost, 4),
                    "cell_cost_30ep": round(cost * 30, 2),
                    "error": res.error,
                })
                print(f"[done] {label:14s} calls={t['calls']:4d} "
                      f"in={t['prompt_tokens']:>8d} out={t['completion_tokens']:>7d} "
                      f"${cost:6.3f}/ep -> ${cost*30:8.2f}/cell", flush=True)
            else:
                rows.append({"label": label, "model": model, "reasoning": reasoning,
                             "ok": res.ok, "error": res.error or "no token_usage.json"})
                print(f"[FAIL] {label:14s} ok={res.ok} err={res.error}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "phase0_calibration.json")
    json.dump(rows, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")

    total = sum(r.get("cell_cost_30ep", 0) for r in rows)
    print(f"\n=== Measured full-grid cost (5 cells x 30 ep) = ${total:.2f} ===")
    print("(proposal estimate was ~$1,164)")


if __name__ == "__main__":
    main()
