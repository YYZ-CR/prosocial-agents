"""Drive norm optimization: evolve a fishing-commons norm with Baseline/TextGrad/ACE.

Dry run (offline, no spend) -- verifies the whole loop with a mock sim + mock LLM:
    .venv/Scripts/python.exe scripts/optimize_norms.py --dry-run --iters 4

Real trial (paid; runs the actual sim per proposed norm):
    .venv/Scripts/python.exe scripts/optimize_norms.py \
        --methods baseline textgrad ace --iters 3 --seeds 0 \
        --model qwen/qwen3-235b-a22b --rounds 6 --prosocial 0

Outputs one trajectory JSON per method under --out-dir (default scripts/norm_opt_runs/).
Each records, per iteration: the norm text, whether it changed, reward, and all paper metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
except Exception:
    pass

from simulation.scenarios.fishing.optimization.episode import (
    EpisodeResult, NormEvaluation, evaluate_norm,
)
from simulation.scenarios.fishing.optimization.loop import run_optimization
from simulation.scenarios.fishing.optimization.optimizers import (
    ACEOptimizer, BaselineOptimizer, MockLLM, TextGradOptimizer,
)
from simulation.scenarios.fishing.optimization.optimizer_llm import OptimizerLLM
from simulation.scenarios.fishing.optimization.reward import compute_reward

# A deliberately vague starting norm: optimization should sharpen it into concrete rules.
INITIAL_NORM = (
    "Fishers should be mindful of the shared lake and avoid taking so much that the "
    "fish cannot recover. Take a reasonable, fair share."
)


def make_mock_evaluator(max_rounds: int):
    """Offline evaluator: synthesizes plausible metrics from the norm's per-fisher cap.

    Peak reward at cap ~= 10 (sustainable). Higher caps collapse the fishery; lower caps
    waste yield; no cap is mediocre. Lets us validate the full loop with zero spend.
    """
    def _evaluate(norm: str) -> NormEvaluation:
        nums = re.findall(r"\b(\d{1,3})\b", norm)
        cap = int(nums[0]) if nums else None
        if cap is None:
            metrics = {"survival_time_m": max_rounds * 0.6, "total_gain_R": 180.0,
                       "efficiency_u": 0.5, "equality_e": 0.7, "over_usage_o": 0.35}
        elif cap > 10:
            over = min(1.0, (cap - 10) / 15)
            surv = max(1.0, max_rounds * (1 - over))
            metrics = {"survival_time_m": surv, "total_gain_R": 150 + (12 - cap),
                       "efficiency_u": max(0.2, 0.9 - over), "equality_e": 0.85,
                       "over_usage_o": 0.2 + 0.6 * over}
        elif cap < 10:
            under = (10 - cap) / 10
            metrics = {"survival_time_m": float(max_rounds), "total_gain_R": 250 * (1 - under),
                       "efficiency_u": max(0.2, 0.95 - under), "equality_e": 0.9,
                       "over_usage_o": 0.05}
        else:  # cap == 10, the sustainable optimum
            metrics = {"survival_time_m": float(max_rounds), "total_gain_R": 300.0,
                       "efficiency_u": 0.97, "equality_e": 0.95, "over_usage_o": 0.02}
        reward = compute_reward(metrics, max_rounds=max_rounds)
        dummy = EpisodeResult(norm=norm, seed=0, ok=True, reward=reward, metrics=metrics)
        # Mock retention: vague norms get rewritten, concrete sustainable ones get kept.
        kept = 1.0 if cap == 10 else (0.5 if cap is not None else 0.0)
        retention = {"seed_retained_frac": kept,
                     "n_amendments_mean": 0.0 if kept == 1.0 else 1.0,
                     "first_amendment_round_mean": None if kept == 1.0 else 3.0,
                     "max_support_when_not_adopted": 3, "n_runs": 1}
        return NormEvaluation(norm=norm, reward=reward, reward_std=0.0, episodes=[dummy],
                              metrics_mean=metrics, retention=retention,
                              final_norms=[norm])
    return _evaluate


def make_real_evaluator(args):
    def _evaluate(norm: str) -> NormEvaluation:
        return evaluate_norm(
            norm, model=args.model, seeds=args.seeds, condition=args.condition,
            prosocial=args.prosocial, max_rounds=args.rounds,
            max_workers=args.seed_workers, freeze=args.freeze,
            results_prefix=args.results_prefix, reasoning=args.reasoning,
        )
    return _evaluate


def build_optimizer(name: str, llm, max_rounds: int):
    if name == "baseline":
        return BaselineOptimizer(INITIAL_NORM)
    if name == "textgrad":
        return TextGradOptimizer(INITIAL_NORM, llm, max_rounds=max_rounds)
    if name == "ace":
        return ACEOptimizer(INITIAL_NORM, llm, max_rounds=max_rounds)
    raise ValueError(f"unknown method: {name}")


def main() -> int:
    p = argparse.ArgumentParser(description="Evolve a fishing-commons norm (Baseline/TextGrad/ACE).")
    p.add_argument("--methods", nargs="+", default=["baseline", "textgrad", "ace"],
                   choices=["baseline", "textgrad", "ace"])
    p.add_argument("--iters", type=int, default=4, help="outer optimization iterations")
    p.add_argument("--seeds", nargs="+", type=int, default=[0], help="sim seeds per norm eval")
    p.add_argument("--model", default="qwen/qwen3-235b-a22b")
    p.add_argument("--opt-model", default=None,
                   help="model for the optimizer's own reasoning (default: --model). "
                        "Hold this FIXED across model cells so 'which model plays better' "
                        "is not confounded with 'which model writes better norms'.")
    p.add_argument("--reasoning", default=None, choices=["on", "off"],
                   help="extended reasoning for the fisher agents (OpenRouter). "
                        "Omit for the model's own default.")
    p.add_argument("--rounds", type=int, default=12,
                   help="max_num_rounds per episode. Keep at 12 (the paper/replication "
                        "horizon): short horizons let a norm strip-mine the lake and still "
                        "bank most of the achievable gain, so they teach the optimizer that "
                        "overfishing is fine.")
    p.add_argument("--parallel-methods", action="store_true",
                   help="run the methods concurrently (they are independent); ~Nx faster")
    p.add_argument("--seed-workers", type=int, default=1,
                   help="concurrent episodes per norm evaluation (seeds are independent)")
    p.add_argument("--prosocial", type=int, default=0, help="# prosocial fishers (0..5)")
    p.add_argument("--condition", default="nl", choices=["nl", "code_law", "no_contract"])
    p.add_argument("--dry-run", action="store_true", help="offline mock sim + mock LLM, no spend")
    p.add_argument("--freeze", action="store_true",
                   help="hold the seeded norm fixed (no renegotiation). Default is "
                        "SEED-AND-AMEND: agents may amend/replace the seeded norm, and we "
                        "measure whether they keep it.")
    p.add_argument("--results-prefix", default="norm_opt_v2",
                   help="subfolder under simulation/results/ for this run's episodes")
    p.add_argument("--out-dir", default=os.path.join("scripts", "norm_opt_runs", "seed_amend"))
    args = p.parse_args()

    if not args.dry_run and not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set (needed for a real run).", file=sys.stderr)
        return 1

    evaluator = make_mock_evaluator(args.rounds) if args.dry_run else make_real_evaluator(args)
    os.makedirs(os.path.join(REPO, args.out_dir), exist_ok=True)

    mode = "FROZEN (no renegotiation)" if args.freeze else "SEED-AND-AMEND (agents may amend)"
    print(f"{'DRY RUN (offline)' if args.dry_run else 'REAL RUN'} | mode={mode}\n"
          f"methods={args.methods} iters={args.iters} seeds={args.seeds} model={args.model} "
          f"rounds={args.rounds} prosocial={args.prosocial} condition={args.condition}\n"
          f"episodes -> simulation/results/{args.results_prefix}/ | trajectories -> {args.out_dir}\n")
    print(f"Initial seed norm:\n  {INITIAL_NORM}\n")

    def run_method(method: str) -> tuple[str, dict]:
        llm = MockLLM() if args.dry_run else OptimizerLLM(args.opt_model or args.model)
        optimizer = build_optimizer(method, llm, args.rounds)
        log_path = os.path.join(REPO, args.out_dir, f"trajectory_{method}.json")
        traj = run_optimization(optimizer, evaluator, n_iters=args.iters, log_path=log_path)
        return method, {
            "best_reward": optimizer.best_reward,
            "best_norm": optimizer.best_norm,
            "final_reward": traj[-1].reward if traj else None,
            "llm_calls": getattr(llm, "calls", 0),
            "log_path": log_path,
        }

    summary = {}
    if args.parallel_methods and len(args.methods) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"(running {len(args.methods)} methods concurrently)\n")
        with ThreadPoolExecutor(max_workers=len(args.methods)) as pool:
            futures = [pool.submit(run_method, m) for m in args.methods]
            for fut in as_completed(futures):
                method, s = fut.result()
                summary[method] = s
    else:
        for method in args.methods:
            print(f"===== {method.upper()} =====")
            method, s = run_method(method)
            summary[method] = s
            print(f"  -> saved {s['log_path']}\n")

    print("===== SUMMARY =====")
    for method, s in summary.items():
        print(f"{method:9s} best_reward={s['best_reward']:.3f}  (LLM calls: {s['llm_calls']})")
        print(f"          best norm: {s['best_norm'][:140].replace(chr(10), ' ')}")
    with open(os.path.join(REPO, args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
