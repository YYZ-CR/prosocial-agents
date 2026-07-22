"""Launch the full six-cell norm-evolution grid.

Each cell runs baseline / TextGrad / ACE at 5 iterations x 2 seeds = 30 episodes.
Six cells = 180 episodes.

The OPTIMIZER llm is held FIXED across all cells (--opt-model). Otherwise "which
model plays the commons better" is confounded with "which model writes better
norms" -- two different questions. We only want the first to vary.

Cells run sequentially (each already parallelises internally across methods and
seeds); within a cell the 5 optimizer iterations are inherently sequential.

Usage: .venv/Scripts/python.exe scripts/launch_grid.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPT_MODEL = "anthropic/claude-sonnet-5"   # fixed norm-writer for every cell

# (cell_id, model, reasoning)
CELLS = [
    ("gemma4-31b",   "google/gemma-4-31b-it",      None),
    ("luna",         "openai/gpt-5.6-luna",         None),
    ("sonnet5-off",  "anthropic/claude-sonnet-5",  "off"),
    ("sonnet5-on",   "anthropic/claude-sonnet-5",  "on"),
    ("opus48",       "anthropic/claude-opus-4.8",   None),
    ("sol",          "openai/gpt-5.6-sol",          None),
]

ITERS = 5
SEEDS = ["0", "1"]


def main() -> int:
    started = time.time()
    for i, (cell_id, model, reasoning) in enumerate(CELLS, 1):
        out_dir = os.path.join(REPO, "scripts", "norm_opt_runs", "grid_v1", cell_id)
        cmd = [
            sys.executable, "scripts/optimize_norms.py",
            "--methods", "baseline", "textgrad", "ace",
            "--iters", str(ITERS),
            "--seeds", *SEEDS,
            "--model", model,
            "--opt-model", OPT_MODEL,
            "--rounds", "12",
            "--prosocial", "0",
            "--condition", "nl",
            "--parallel-methods",
            "--seed-workers", "2",
            "--results-prefix", f"grid_v1/{cell_id}",
            "--out-dir", out_dir,
        ]
        if reasoning:
            cmd += ["--reasoning", reasoning]

        print(f"\n{'='*72}\n[{i}/{len(CELLS)}] CELL {cell_id}  model={model} "
              f"reasoning={reasoning}\n{'='*72}", flush=True)
        env = dict(os.environ, PYTHONPATH=".", PYTHONUTF8="1",
                   OPENAI_MAX_CONCURRENCY="6")
        r = subprocess.run(cmd, cwd=REPO, env=env)
        status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
        print(f"[{i}/{len(CELLS)}] {cell_id}: {status} "
              f"| elapsed {(time.time()-started)/60:.0f} min", flush=True)

    print(f"\nGRID COMPLETE in {(time.time()-started)/60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
