"""Run the fishing sim as an evaluation function for a single seeded norm.

Design: SEED-AND-AMEND. The optimizer proposes a *starting* norm; it is installed as
the active law and primed so agents see it in round 0 (see contracting/runtime.py).
Agents then negotiate every round as normal -- they may amend it, replace it, or keep
it. So we measure two things:

  1. the outcome (paper metrics -> reward), and
  2. what the fishers DID with the norm (retention / amendments).

(2) matters as much as (1): a norm agents immediately rewrite lacked legitimacy,
however elegant it looked. Feeding both back is what lets TextGrad/ACE learn to
propose norms that selfish agents will actually adopt and keep.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean

from .reward import RewardWeights, DEFAULT_WEIGHTS, reward_breakdown

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))
RESULTS_ROOT = os.path.join(REPO_ROOT, "simulation", "results")


def _load_summarizer():
    """Import scripts/summarize_fishing_log.py by path (scripts/ isn't a package)."""
    path = os.path.join(REPO_ROOT, "scripts", "summarize_fishing_log.py")
    spec = importlib.util.spec_from_file_location("summarize_fishing_log", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


_SUM = _load_summarizer()

CONDITION_EXPERIMENT = {
    "nl": "fish_iid_stochastic_code_nl",
    "code_law": "fish_iid_stochastic_code_law",
    "no_contract": "fish_iid_stochastic_no_contract",
}
TOTAL_AGENTS = 5


@dataclass
class NormTrajectory:
    """What the fishers did to the seeded norm across one episode."""

    final_norm: str | None = None
    n_amendments: int = 0
    amendment_rounds: list[int] = field(default_factory=list)
    rounds_until_first_amendment: int | None = None
    seed_retained: bool = True
    # Amendment needs 4 of 5 (runtime.py clamps min_agree_agents to >=4). Track the most
    # support a failed change got, so "kept because unanimity is hard" is distinguishable
    # from "kept because everyone liked it".
    max_support_when_not_adopted: int = 0


@dataclass
class EpisodeResult:
    norm: str
    seed: int
    ok: bool
    reward: float
    metrics: dict = field(default_factory=dict)
    breakdown: dict = field(default_factory=dict)
    storage_dir: str | None = None
    error: str | None = None
    trajectory: NormTrajectory = field(default_factory=NormTrajectory)


@dataclass
class NormEvaluation:
    """Aggregate of one seeded norm evaluated over one or more seeds."""

    norm: str
    reward: float
    reward_std: float
    episodes: list[EpisodeResult]
    metrics_mean: dict = field(default_factory=dict)
    retention: dict = field(default_factory=dict)
    final_norms: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return any(ep.ok for ep in self.episodes)


def parse_norm_trajectory(storage_dir: str, seed_norm: str) -> NormTrajectory:
    """Reconstruct the in-episode norm history from contracting_results.jsonl."""
    traj = NormTrajectory(final_norm=seed_norm)
    path = os.path.join(storage_dir, "contracting_results.jsonl")
    if not os.path.exists(path):
        return traj
    seen_rounds: set[int] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "negotiation":
                continue
            data = entry.get("data") or {}
            contract = data.get("contract") or {}
            meta = contract.get("metadata") or {}
            nl = meta.get("nl_contract") or contract.get("content")
            if nl:
                traj.final_norm = nl
            created = contract.get("round_created")
            # round_created == -1 is the seeded/default contract; >=0 means agents adopted
            # a new law in that round.
            if isinstance(created, int) and created >= 0 and created not in seen_rounds:
                seen_rounds.add(created)
                traj.amendment_rounds.append(created)
            result = data.get("result") or {}
            if not result.get("nl_contract"):
                agreements = result.get("agreements") or {}
                traj.max_support_when_not_adopted = max(
                    traj.max_support_when_not_adopted, len(agreements)
                )
    traj.amendment_rounds.sort()
    traj.n_amendments = len(traj.amendment_rounds)
    traj.seed_retained = traj.n_amendments == 0
    traj.rounds_until_first_amendment = (
        traj.amendment_rounds[0] if traj.amendment_rounds else None
    )
    return traj


def _build_command(
    python: str,
    *,
    model: str,
    group_name: str,
    condition: str,
    prosocial: int,
    seed: int,
    max_rounds: int,
    reasoning: str | None = None,
) -> list[str]:
    experiment = CONDITION_EXPERIMENT[condition]
    # Route everything through OpenRouter: the only live key is the shared
    # OpenRouter key, and reasoning control requires the OpenRouter backend.
    # (openai/* on the default "OpenAI" backend would hit api.openai.com with a
    # placeholder key and fail.)
    cmd = [
        python, "-m", "simulation.main",
        f"experiment={experiment}",
        f"group_name={group_name}",
        f"llm.path={model}",
        "llm.backend=OpenRouter",
        f"seed={seed}",
        f"experiment.env.regen_seed={seed}",
        f"experiment.env.max_num_rounds={max_rounds}",
        # Deterministic regen (r_t=2.0), matching the replication's default family.
        "experiment.env.regen_factor_range=[2.0,2.0]",
        f"experiment.contracting.coding_llm.path={model}",
        "experiment.contracting.coding_llm.backend=OpenRouter",
    ]
    # Reasoning applies to the fisher agents (the experimental subject). The
    # coding agent is not the manipulated variable, so it keeps model default.
    if reasoning in ("on", "off"):
        cmd.append(f"llm.reasoning={reasoning}")
    for i in range(TOTAL_AGENTS):
        persona = "prosocial_fisherman" if i < prosocial else "selfish_fisherman"
        cmd.append(f"experiment/persona@experiment.personas.persona_{i}={persona}")
    return cmd


def run_episode(
    norm: str,
    *,
    model: str,
    seed: int,
    condition: str = "nl",
    prosocial: int = 0,
    max_rounds: int = 12,
    reasoning: str | None = None,
    freeze: bool = False,
    results_prefix: str = "norm_opt_v2",
    capacity: float = 100.0,
    expected_regen: float = 2.0,
    weights: RewardWeights = DEFAULT_WEIGHTS,
    timeout: int = 10800,
) -> EpisodeResult:
    """Run one episode seeded with `norm`; agents may amend it unless freeze=True."""
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    group_name = f"{results_prefix}/{run_id}"
    cmd = _build_command(
        sys.executable, model=model, group_name=group_name, condition=condition,
        prosocial=prosocial, seed=seed, max_rounds=max_rounds, reasoning=reasoning,
    )
    env = dict(os.environ, PYTHONPATH=".", PYTHONUTF8="1")
    env["FISHING_SEED_NORM"] = norm
    env["FISHING_FREEZE_NORM"] = "1" if freeze else "0"

    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, timeout=timeout,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        return EpisodeResult(norm, seed, False, 0.0, error="timeout")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-800:]
        return EpisodeResult(norm, seed, False, 0.0, error=f"exit {proc.returncode}: {tail}")

    logs = glob.glob(
        os.path.join(RESULTS_ROOT, results_prefix, run_id, "**", "log_env.json"),
        recursive=True,
    )
    if not logs:
        return EpisodeResult(norm, seed, False, 0.0, error="no log_env.json produced")
    storage_dir = os.path.dirname(logs[0])
    try:
        from pathlib import Path

        records = _SUM.parse_log(Path(logs[0]))
        report = _SUM.summarize(
            records, capacity=capacity, expected_regen=expected_regen,
            collapse_threshold=0.0,
        )
    except Exception as e:  # noqa: BLE001 - surface any parse/scoring failure
        return EpisodeResult(norm, seed, False, 0.0, storage_dir=storage_dir,
                             error=f"summarize: {e}")

    metrics = report["paper_metrics"]
    breakdown = reward_breakdown(
        metrics, max_rounds=max_rounds, capacity=capacity,
        expected_regen=expected_regen, weights=weights,
    )
    return EpisodeResult(
        norm=norm, seed=seed, ok=True, reward=breakdown["reward"],
        metrics=metrics, breakdown=breakdown, storage_dir=storage_dir,
        trajectory=parse_norm_trajectory(storage_dir, norm),
    )


def evaluate_norm(
    norm: str,
    *,
    model: str,
    seeds: list[int],
    max_workers: int = 1,
    **kwargs,
) -> NormEvaluation:
    """Evaluate a seeded norm across seeds; reward is the mean over successful episodes.

    Seeds are independent episodes (separate subprocesses), so they run concurrently
    when max_workers > 1.
    """
    if max_workers > 1 and len(seeds) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(max_workers, len(seeds))) as pool:
            episodes = list(pool.map(
                lambda s: run_episode(norm, model=model, seed=s, **kwargs), seeds
            ))
    else:
        episodes = [run_episode(norm, model=model, seed=s, **kwargs) for s in seeds]

    ok = [ep for ep in episodes if ep.ok]
    if not ok:
        return NormEvaluation(norm, 0.0, 0.0, episodes)
    rewards = [ep.reward for ep in ok]
    keys = ("survival_time_m", "total_gain_R", "efficiency_u", "equality_e", "over_usage_o")
    metrics_mean = {k: mean(float(ep.metrics.get(k, 0.0) or 0.0) for ep in ok) for k in keys}
    first = [ep.trajectory.rounds_until_first_amendment for ep in ok
             if ep.trajectory.rounds_until_first_amendment is not None]
    retention = {
        "seed_retained_frac": mean(1.0 if ep.trajectory.seed_retained else 0.0 for ep in ok),
        "n_amendments_mean": mean(ep.trajectory.n_amendments for ep in ok),
        "first_amendment_round_mean": (mean(first) if first else None),
        "max_support_when_not_adopted": max(
            ep.trajectory.max_support_when_not_adopted for ep in ok
        ),
        "n_runs": len(ok),
    }
    std = (sum((r - mean(rewards)) ** 2 for r in rewards) / len(rewards)) ** 0.5
    return NormEvaluation(
        norm, mean(rewards), std, episodes, metrics_mean, retention,
        [ep.trajectory.final_norm or "" for ep in ok],
    )
