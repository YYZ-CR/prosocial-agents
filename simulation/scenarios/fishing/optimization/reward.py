"""Scalar reward over a norm's episode outcome.

This function *is* the operational definition of "which norms are better" for the
optimization study, so it is deliberately small, explicit, and tunable. It maps the
GovSim paper metrics (Section 2.4, computed by scripts/summarize_fishing_log.py) to
one number the optimizers maximize.

A good commons norm should be:
  - sustainable  -> high survival m, low over-usage o
  - productive   -> high total gain R over the WHOLE horizon
  - fair         -> high equality e (1 - Gini)

reward = w_g*gain_norm + w_m*(m/max_rounds) + w_e*e - w_o*o

IMPORTANT -- why we do NOT optimize the paper's efficiency_u:
    u = 1 - max(0, sum_t g_t - R) / sum_t g_t, where the sum runs only over rounds
    ACTUALLY PLAYED. Two pathologies make it unusable as an optimizer objective:
      1. It saturates at 1.0, so over-extraction is never penalized.
      2. Collapsing early shrinks the denominator, which *inflates* u. A run that
         dies in 2 rounds with R=120 scores u=1.0 -- a perfect score for destroying
         the fishery.
    Maximizing u therefore rewards collapse. We instead normalize gain by the
    full-horizon benchmark (max_rounds * per-round full-pool yield), which is a fixed
    denominator: dying early simply forfeits the remaining rounds' yield. We still
    report u in the logs for continuity with the paper -- we just don't optimize it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    gain: float = 1.0
    survival: float = 1.0
    equality: float = 0.5
    over_usage: float = 1.0


def full_horizon_benchmark(max_rounds: int, capacity: float, expected_regen: float) -> float:
    """Total catch achievable over the whole horizon by harvesting the sustainable
    full-pool yield every round: max_rounds * (H - H/r). Fixed denominator, so an
    early collapse forfeits yield instead of being rewarded for it."""
    if capacity <= 0 or expected_regen <= 0 or max_rounds <= 0:
        return 0.0
    return max_rounds * max(0.0, capacity - capacity / expected_regen)


DEFAULT_WEIGHTS = RewardWeights()


def compute_reward(
    paper_metrics: dict,
    *,
    max_rounds: int,
    capacity: float = 100.0,
    expected_regen: float = 2.0,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> float:
    """Scalarize one episode's paper_metrics into a reward the optimizer maximizes."""
    return reward_breakdown(
        paper_metrics, max_rounds=max_rounds, capacity=capacity,
        expected_regen=expected_regen, weights=weights,
    )["reward"]


def reward_breakdown(
    paper_metrics: dict,
    *,
    max_rounds: int,
    capacity: float = 100.0,
    expected_regen: float = 2.0,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> dict:
    """Per-term contributions plus the reward, for logging / analysis."""
    R = float(paper_metrics.get("total_gain_R", 0.0) or 0.0)
    m = float(paper_metrics.get("survival_time_m", 0.0) or 0.0)
    e = float(paper_metrics.get("equality_e", 0.0) or 0.0)
    o = float(paper_metrics.get("over_usage_o", 0.0) or 0.0)
    u = float(paper_metrics.get("efficiency_u", 0.0) or 0.0)  # reported, not optimized

    benchmark = full_horizon_benchmark(max_rounds, capacity, expected_regen)
    gain_norm = (R / benchmark) if benchmark > 0 else 0.0
    m_norm = (m / max_rounds) if max_rounds > 0 else 0.0
    reward = (
        weights.gain * gain_norm
        + weights.survival * m_norm
        + weights.equality * e
        - weights.over_usage * o
    )
    return {
        "gain_norm": gain_norm,
        "survival_norm": m_norm,
        "equality_e": e,
        "over_usage_o": o,
        "efficiency_u_reported_not_optimized": u,
        "full_horizon_benchmark": benchmark,
        "reward": reward,
    }
