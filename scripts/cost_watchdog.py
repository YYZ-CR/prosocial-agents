"""Cost watchdog for the norm-evolution grid.

Polls every completed episode's token_usage.json, prices it at live OpenRouter
rates, and KILLS the grid if spend exceeds a hard ceiling.

Why a ceiling and not just a report: the grid runs unattended for many hours on a
shared lab key. A silent cost blow-up (a model that never terminates, a pricing
change, an unexpected reasoning-token explosion) would otherwise run all night.

NOTE ON LAG: token_usage.json is written when an episode FINISHES, so in-flight
episodes are not yet counted. We add an in-flight allowance so the ceiling is
enforced against a conservative estimate of true spend, not an undercount.

Usage:
  .venv/Scripts/python.exe scripts/cost_watchdog.py --budget 569 --ceiling 700
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime

import httpx
import psutil
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "simulation", "results", "grid_v1")
LOG_PATH = os.path.join(REPO, "scripts", "norm_opt_runs", "grid_v1", "cost_log.jsonl")
# Rough per-episode cost of the priciest cell; used as the in-flight allowance.
INFLIGHT_UNIT = 6.76


def live_prices() -> dict[str, tuple[float, float]]:
    ms = httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
    return {m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"]))
            for m in ms}


def scan(prices) -> tuple[float, int, dict[str, dict]]:
    """Total $ spent, episodes counted, and a per-model breakdown."""
    total, n = 0.0, 0
    by_model: dict[str, dict] = {}
    for usage_path in glob.glob(os.path.join(RESULTS, "**", "token_usage.json"), recursive=True):
        try:
            usage = json.load(open(usage_path))
        except Exception:  # noqa: BLE001 - a half-written file is fine to skip
            continue
        for model, row in usage.get("by_model", {}).items():
            pin, pout = prices.get(model, (0.0, 0.0))
            cost = row["prompt_tokens"] * pin + row["completion_tokens"] * pout
            total += cost
            b = by_model.setdefault(model, {"cost": 0.0, "calls": 0, "episodes": 0})
            b["cost"] += cost
            b["calls"] += row["calls"]
        n += 1
        # attribute the episode to its configured model
        cfg_path = os.path.join(os.path.dirname(usage_path), ".hydra", "config.yaml")
        try:
            m = (yaml.safe_load(open(cfg_path)) or {}).get("llm", {}).get("path")
            if m in by_model:
                by_model[m]["episodes"] += 1
        except Exception:  # noqa: BLE001
            pass
    return total, n, by_model


def grid_procs() -> list[psutil.Process]:
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cl = " ".join(p.info.get("cmdline") or [])
        if "launch_grid.py" in cl or "optimize_norms.py" in cl or "simulation.main" in cl:
            out.append(p)
    return out


def running_episodes() -> int:
    """Count LOGICAL in-flight episodes.

    On Windows, .venv/Scripts/python.exe is a shim that re-execs the base
    interpreter, so every episode shows up twice (shim + real child). Counting
    raw matches doubles the number and would trip the ceiling early. Skip any
    process whose parent is also an episode process.
    """
    eps = {}
    for p in psutil.process_iter(["pid", "ppid", "cmdline"]):
        cl = " ".join(p.info.get("cmdline") or [])
        if "simulation.main" in cl:
            eps[p.info["pid"]] = p.info["ppid"]
    return sum(1 for pid, ppid in eps.items() if ppid not in eps)


def kill_grid() -> int:
    """Kill launcher first (so it stops spawning), then the workers."""
    killed = 0
    procs = grid_procs()
    procs.sort(key=lambda p: 0 if "launch_grid.py" in " ".join(p.info.get("cmdline") or []) else 1)
    for p in procs:
        try:
            for child in p.children(recursive=True):
                try:
                    child.kill(); killed += 1
                except psutil.Error:
                    pass
            p.kill(); killed += 1
        except psutil.Error:
            pass
    return killed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=569.0, help="expected total; warn above this")
    ap.add_argument("--ceiling", type=float, default=700.0, help="hard stop: kill the grid above this")
    ap.add_argument("--interval", type=int, default=300, help="poll seconds")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    prices = live_prices()
    print(f"[watchdog] budget=${args.budget:.0f} ceiling=${args.ceiling:.0f} "
          f"poll={args.interval}s", flush=True)

    while True:
        spent, n_eps, by_model = scan(prices)
        running = running_episodes()
        projected = spent + running * INFLIGHT_UNIT
        ts = datetime.now().isoformat(timespec="seconds")

        rec = {"ts": ts, "spent": round(spent, 2), "episodes_done": n_eps,
               "episodes_running": running, "projected": round(projected, 2),
               "by_model": {m: round(v["cost"], 2) for m, v in by_model.items()}}
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")

        flag = ""
        if projected > args.budget:
            flag = "  ** OVER BUDGET **"
        print(f"[{ts}] spent=${spent:7.2f} done={n_eps:3d} running={running} "
              f"projected=${projected:7.2f}{flag}", flush=True)

        if projected > args.ceiling:
            print(f"\n!!! CEILING BREACHED: projected ${projected:.2f} > ${args.ceiling:.2f}", flush=True)
            k = kill_grid()
            print(f"!!! GRID KILLED ({k} processes). Spend stopped at ~${spent:.2f}.", flush=True)
            for m, v in sorted(by_model.items(), key=lambda x: -x[1]["cost"]):
                print(f"      {m:34s} ${v['cost']:8.2f}  {v['episodes']} ep", flush=True)
            return 2

        if not grid_procs():
            print(f"\n[watchdog] grid finished. Final spend ~${spent:.2f} over {n_eps} episodes.", flush=True)
            for m, v in sorted(by_model.items(), key=lambda x: -x[1]["cost"]):
                print(f"      {m:34s} ${v['cost']:8.2f}  {v['episodes']} ep", flush=True)
            return 0

        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
