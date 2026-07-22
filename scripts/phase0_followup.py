"""Phase 0 follow-up:
  1. Re-run gpt-5.6-sol on TWO fresh seeds -- the calibration run collapsed the
     lake in round 0, and one episode can't distinguish a real behavioural
     signature from a bad draw.
  2. Measure claude-opus-4.8 (flagship rung candidate) for real cost.

Usage: .venv/Scripts/python.exe scripts/phase0_followup.py
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

# (label, model, reasoning, seed)
CELLS = [
    ("sol-seed1",   "openai/gpt-5.6-sol",       None, 1),
    ("sol-seed2",   "openai/gpt-5.6-sol",       None, 2),
    ("opus48-seed0", "anthropic/claude-opus-4.8", None, 0),
]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "norm_opt_runs")
RESULTS_PREFIX = "phase0_followup"


def live_prices() -> dict[str, tuple[float, float]]:
    ms = httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
    return {m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"]))
            for m in ms}


def run_one(label, model, reasoning, seed):
    res = run_episode(
        SEED_NORM, model=model, seed=seed, condition="nl", prosocial=0,
        max_rounds=12, reasoning=reasoning, results_prefix=RESULTS_PREFIX,
        timeout=9000,
    )
    usage = None
    if res.storage_dir:
        p = os.path.join(res.storage_dir, "token_usage.json")
        if os.path.exists(p):
            usage = json.load(open(p))
    return label, model, seed, res, usage


def main():
    prices = live_prices()
    print(f"Launching {len(CELLS)} follow-up episodes concurrently...\n", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(max_workers=len(CELLS)) as ex:
        futs = [ex.submit(run_one, *c) for c in CELLS]
        for fut in cf.as_completed(futs):
            label, model, seed, res, usage = fut.result()
            pin, pout = prices.get(model, (0.0, 0.0))
            rounds = None
            if res.metrics:
                rounds = res.metrics.get("survival_time_m")
            if usage:
                t = usage["totals"]
                cost = t["prompt_tokens"] * pin + t["completion_tokens"] * pout
                rows.append({
                    "label": label, "model": model, "seed": seed, "ok": res.ok,
                    "survival_m": rounds, "calls": t["calls"],
                    "prompt_tokens": t["prompt_tokens"],
                    "completion_tokens": t["completion_tokens"],
                    "episode_cost": round(cost, 4),
                    "cell_cost_30ep": round(cost * 30, 2),
                    "metrics": res.metrics,
                })
                print(f"[done] {label:14s} survival={str(rounds):>5s} calls={t['calls']:5d} "
                      f"${cost:6.3f}/ep -> ${cost*30:8.2f}/cell", flush=True)
            else:
                rows.append({"label": label, "model": model, "seed": seed,
                             "ok": res.ok, "error": res.error})
                print(f"[FAIL] {label:14s} {res.error}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "phase0_followup.json")
    json.dump(rows, open(p, "w"), indent=2)
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
