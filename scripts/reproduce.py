"""
Windows-friendly reproduction driver for the fishing-commons sweeps.

Mirrors scripts/standard/examples_models.sh, but calls the project's own
Python interpreter directly (no `uv`, no bash). Runs on Windows PowerShell.

Usage (from repo root, with venv Python):
    .venv/Scripts/python.exe scripts/reproduce.py --help

Examples:
    # Tiny smoke test: 1 condition, 1 prosocial count, 1 seed
    .venv/Scripts/python.exe scripts/reproduce.py \
        --model anthropic/claude-sonnet-4.5 --name claude45 \
        --group-prefix smoke --conditions code_law \
        --prosocial 0 --seeds 0

    # Full replication grid for one model (3 conditions x prosocial 0..5 x seeds 0..4)
    .venv/Scripts/python.exe scripts/reproduce.py \
        --model anthropic/claude-sonnet-4.5 --name claude45 \
        --group-prefix repro_claude45
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

# Load API keys from the repo-root .env into os.environ (the sim itself does not
# load .env). Silently no-op if python-dotenv or the file is missing.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), os.pardir, ".env"))
except Exception:
    pass

# condition -> (experiment config, extra hydra overrides)
CONDITIONS = {
    "nl":          "fish_iid_stochastic_code_nl",
    "code_law":    "fish_iid_stochastic_code_law",
    "no_contract": "fish_iid_stochastic_no_contract",
}

TOTAL_AGENTS = 5


def build_command(python: str, *, model: str, name: str, group_prefix: str,
                  condition: str, prosocial: int, seed: int,
                  disclose_variance: bool) -> list[str]:
    experiment = CONDITIONS[condition]
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    group_name = f"{name}-p{prosocial}-{stamp}/{group_prefix}-{condition}-p{prosocial}"

    # The experiment configs hardcode the contracting "coding agent" to
    # openai/gpt-5.4 (backend OpenAI). Route it through the same model/backend as
    # the main agents so a single OpenRouter key is enough. Backend is derived the
    # same way the main routing does (openai/ -> OpenAI, else OpenRouter).
    coding_backend = "OpenAI" if model.startswith("openai/") else "OpenRouter"

    cmd = [
        python, "-m", "simulation.main",
        f"experiment={experiment}",
        f"group_name={group_name}",
        f"llm.path={model}",
        f"seed={seed}",
        f"experiment.env.regen_seed={seed}",
        f"experiment.contracting.coding_llm.path={model}",
        f"experiment.contracting.coding_llm.backend={coding_backend}",
    ]
    # "standard" family hides the regen variance from agents (tells them a flat 2.0);
    # "stochastic" family discloses the real [1.5, 2.5] range. Actual draws are always
    # stochastic either way (hardcoded in regen.py).
    if not disclose_variance:
        cmd.append("experiment.env.regen_factor_range=[2.0,2.0]")

    # first `prosocial` agents are prosocial, the rest selfish
    for i in range(TOTAL_AGENTS):
        persona = "prosocial_fisherman" if i < prosocial else "selfish_fisherman"
        cmd.append(f"experiment/persona@experiment.personas.persona_{i}={persona}")
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(description="Reproduce fishing-commons sweeps (Windows-friendly).")
    p.add_argument("--model", default="anthropic/claude-sonnet-4.5",
                   help="model id; non-openai/ ids route to OpenRouter")
    p.add_argument("--name", default="claude45", help="label used in result folder names")
    p.add_argument("--group-prefix", default="repro", help="prefix for experiment group folders")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                   choices=list(CONDITIONS), help="which contract conditions to run")
    p.add_argument("--prosocial", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5],
                   help="prosocial agent counts to sweep (0..5)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                   help="random seeds")
    p.add_argument("--disclose-variance", action="store_true",
                   help="tell agents the real [1.5,2.5] range (stochastic family). "
                        "Default hides it (standard family).")
    p.add_argument("--dry-run", action="store_true", help="print commands without running")
    p.add_argument("--parallel", type=int, default=1,
                   help="number of runs to execute concurrently (default 1 = sequential). "
                        "When >1, each run's stdout/stderr is redirected to a per-run log "
                        "under scripts/logs/reproduce/.")
    args = p.parse_args()

    # key check
    needs = "OPENAI_API_KEY" if args.model.startswith("openai/") else "OPENROUTER_API_KEY"
    if not args.dry_run and not os.environ.get(needs):
        print(f"ERROR: {needs} is not set (required for model {args.model}).", file=sys.stderr)
        return 1

    python = sys.executable  # the interpreter running this script = the venv python
    runs = [(c, pc, s) for c in args.conditions for pc in args.prosocial for s in args.seeds]
    print(f"Planned runs: {len(runs)}  (conditions={args.conditions} "
          f"prosocial={args.prosocial} seeds={args.seeds}) parallel={args.parallel}\n")

    # PYTHONUTF8=1 forces UTF-8 for all text file I/O; without it, Windows'
    # default cp1252 codec crashes when the sim writes unicode (e.g. "→") to
    # per-agent memory markdown files.
    env = dict(os.environ, PYTHONPATH=".", PYTHONUTF8="1")

    if args.dry_run:
        for idx, (condition, prosocial, seed) in enumerate(runs, 1):
            cmd = build_command(python, model=args.model, name=args.name,
                                group_prefix=args.group_prefix, condition=condition,
                                prosocial=prosocial, seed=seed,
                                disclose_variance=args.disclose_variance)
            print(f"[{idx}/{len(runs)}] {condition} p{prosocial} seed{seed}")
            print("   " + " ".join(cmd) + "\n")
        return 0

    log_dir = os.path.join(os.path.dirname(__file__), "logs", "reproduce")

    def launch(job):
        idx, (condition, prosocial, seed) = job
        cmd = build_command(python, model=args.model, name=args.name,
                            group_prefix=args.group_prefix, condition=condition,
                            prosocial=prosocial, seed=seed,
                            disclose_variance=args.disclose_variance)
        label = f"{condition}_p{prosocial}_seed{seed}"
        if args.parallel > 1:
            os.makedirs(log_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_path = os.path.join(log_dir, f"{args.name}_{label}_{stamp}.log")
            with open(log_path, "w", encoding="utf-8") as log:
                rc = subprocess.run(cmd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
            return idx, label, rc, log_path
        rc = subprocess.run(cmd, env=env).returncode
        return idx, label, rc, None

    failures = 0
    if args.parallel <= 1:
        for idx, job in enumerate(runs, 1):
            condition, prosocial, seed = job
            print(f"[{idx}/{len(runs)}] {condition} p{prosocial} seed{seed}")
            _, _, rc, _ = launch((idx, job))
            if rc != 0:
                failures += 1
                print(f"   !! run failed (exit {rc})", file=sys.stderr)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        done = 0
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(launch, (i, job)): (i, job)
                       for i, job in enumerate(runs, 1)}
            for fut in as_completed(futures):
                done += 1
                idx, label, rc, log_path = fut.result()
                status = "ok" if rc == 0 else f"FAILED (exit {rc})"
                print(f"[{done}/{len(runs)}] {label}: {status}  ->  {log_path}")
                if rc != 0:
                    failures += 1

    print(f"\nDone. {len(runs) - failures}/{len(runs)} runs succeeded.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
