"""Outer optimization loop: propose -> evaluate -> observe, with trajectory logging.

Answers the professor's three questions directly:
  - "do norms evolve?"     -> norm_text per iteration + a change flag
  - "the resulting norms"  -> full norm text logged every iteration
  - "how well are they?"   -> reward + all paper metrics per iteration
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable

from .episode import NormEvaluation
from .optimizers import NormOptimizer

Evaluator = Callable[[str], NormEvaluation]


@dataclass
class IterationRecord:
    iteration: int
    optimizer: str
    norm: str                    # the norm the optimizer SEEDED this iteration
    changed: bool                # did the optimizer change its proposal vs last iteration?
    reward: float
    reward_std: float
    metrics_mean: dict
    ok: bool
    retention: dict = field(default_factory=dict)   # what the fishers did with it
    final_norms: list = field(default_factory=list)  # what they ended the game under
    optimizer_stats: dict = field(default_factory=dict)


def run_optimization(
    optimizer: NormOptimizer,
    evaluator: Evaluator,
    *,
    n_iters: int,
    log_path: str | None = None,
    verbose: bool = True,
) -> list[IterationRecord]:
    """Run `n_iters` of propose/evaluate/observe; return + optionally persist the trajectory."""
    trajectory: list[IterationRecord] = []
    prev_norm: str | None = None
    for it in range(n_iters):
        norm = optimizer.propose()
        ev = evaluator(norm)
        optimizer.observe(ev)
        changed = prev_norm is not None and norm.strip() != prev_norm.strip()
        rec = IterationRecord(
            iteration=it,
            optimizer=optimizer.name,
            norm=norm,
            changed=changed,
            reward=ev.reward,
            reward_std=ev.reward_std,
            metrics_mean=ev.metrics_mean,
            ok=ev.ok,
            retention=ev.retention,
            final_norms=list(ev.final_norms),
            optimizer_stats=optimizer.stats(),
        )
        trajectory.append(rec)
        prev_norm = norm
        if verbose:
            flag = "" if it == 0 else ("  [seed changed]" if changed else "  [seed unchanged]")
            kept = ev.retention.get("seed_retained_frac")
            keep_s = f" kept={kept:.0%}" if kept is not None else ""
            print(f"[{optimizer.name}] iter {it}: reward={ev.reward:.3f} "
                  f"m={ev.metrics_mean.get('survival_time_m', 0):.1f} "
                  f"o={ev.metrics_mean.get('over_usage_o', 0):.3f}{keep_s}{flag}")
            print(f"        seeded: {norm[:110].replace(chr(10), ' ')}"
                  f"{'...' if len(norm) > 110 else ''}")
        if log_path:
            _write_log(log_path, optimizer, trajectory)
    if verbose:
        print(f"[{optimizer.name}] best reward = {optimizer.best_reward:.3f}")
    return trajectory


def _write_log(log_path: str, optimizer: NormOptimizer, trajectory: list[IterationRecord]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    payload = {
        "optimizer": optimizer.name,
        "best_norm": optimizer.best_norm,
        "best_reward": optimizer.best_reward,
        "trajectory": [asdict(r) for r in trajectory],
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
