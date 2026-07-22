"""Norm optimizers: Baseline (control), TextGrad, and ACE.

All share one interface:
    propose()      -> the norm text to evaluate this iteration
    observe(eval)  -> learn from the NormEvaluation just measured

They differ only in how the next norm is produced:
  - Baseline: never changes the norm (control for "does optimization beat nothing?")
  - TextGrad: LLM writes a *textual gradient* (critique) then applies it to revise the norm
  - ACE:      LLM grows a *playbook* (Generator/Reflector/Curator) and drafts norms from it
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from .episode import NormEvaluation

_NORM_RE = re.compile(r"<NORM>(.*?)</NORM>", re.DOTALL | re.IGNORECASE)


def extract_norm(text: str, fallback: str) -> str:
    """Pull the norm out of an LLM reply; tolerate a missing wrapper."""
    m = _NORM_RE.search(text)
    candidate = (m.group(1) if m else text).strip()
    # Guard against empty / degenerate outputs.
    return candidate if len(candidate) >= 15 else fallback


def format_metrics(ev: NormEvaluation, max_rounds: int) -> str:
    m = ev.metrics_mean
    out = (
        f"reward (objective) = {ev.reward:.3f} (+/- {ev.reward_std:.3f} across seeds)\n"
        f"survival months m  = {m.get('survival_time_m', 0):.1f} / {max_rounds} "
        "(higher = fishery lasted longer)\n"
        f"total gain R       = {m.get('total_gain_R', 0):.1f} (higher = more fish caught overall)\n"
        f"efficiency u        = {m.get('efficiency_u', 0):.3f} in [0,1] (gain vs full-pool benchmark)\n"
        f"equality e          = {m.get('equality_e', 0):.3f} in [0,1] (1 - Gini of agent gains)\n"
        f"over-usage o        = {m.get('over_usage_o', 0):.3f} in [0,1] (LOWER = more sustainable)"
    )
    r = ev.retention
    if r:
        out += (
            "\n\n--- What the fishers DID with your proposed norm ---\n"
            f"kept it unamended for the whole game: {r.get('seed_retained_frac', 0):.0%} of runs\n"
            f"times they rewrote the law: {r.get('n_amendments_mean', 0):.1f} on average\n"
        )
        fr = r.get("first_amendment_round_mean")
        if fr is not None:
            out += f"first rewrite happened around round {fr:.1f} of {max_rounds}\n"
        out += (
            f"most fishers who ever backed a change that FAILED to pass: "
            f"{r.get('max_support_when_not_adopted', 0)} of 5 (4 of 5 are required to amend)\n"
        )
        if ev.final_norms and ev.final_norms[0].strip() != ev.norm.strip():
            out += (
                "the law they actually ended the game under:\n"
                f"  \"{ev.final_norms[0][:400]}\"\n"
            )
    return out


OBJECTIVE = (
    "Setting: 5 self-interested fishers share a lake (capacity 100). Each month the "
    "remaining fish double back up toward 100 (regeneration factor ~2.0), so a full lake "
    "can sustainably yield about 50 fish total per month (10 per fisher). Overfishing "
    "shrinks next month's stock and can collapse the fishery.\n"
    "IMPORTANT: the norm is NOT automatically enforced. It is a written rule the fishers "
    "read; each fisher still freely chooses their own catch, and they may negotiate every "
    "month to amend or replace the rule (4 of the 5 must agree to change it). So a norm "
    "only works if self-interested fishers find it worth FOLLOWING and worth KEEPING -- "
    "a rule they ignore or vote away is worthless no matter how well designed.\n"
    "A GOOD norm makes the fishery survive all months, keeps over-usage low, keeps total "
    "gain high, keeps catches equal across fishers, and is one the fishers choose to keep."
)


class NormOptimizer(ABC):
    name: str = "base"

    def __init__(self, initial_norm: str):
        self.initial_norm = initial_norm
        self._current = initial_norm
        self.best_norm = initial_norm
        self.best_reward = float("-inf")

    @abstractmethod
    def propose(self) -> str: ...

    def observe(self, ev: NormEvaluation) -> None:
        if ev.ok and ev.reward > self.best_reward:
            self.best_reward = ev.reward
            self.best_norm = ev.norm

    def stats(self) -> dict:
        return {"best_reward": self.best_reward}


class BaselineOptimizer(NormOptimizer):
    """Control: hold the initial norm fixed. Shows what learning has to beat."""

    name = "baseline"

    def propose(self) -> str:
        return self.initial_norm


class TextGradOptimizer(NormOptimizer):
    """TextGrad: critique (textual gradient) -> apply -> revised norm.

    Includes TextGrad's validation-based rollback: a revision that scores worse than the
    best-so-far is discarded and the next step continues from the best norm instead. Without
    this the optimizer compounds bad steps -- it keeps revising a norm it already knows is
    worse -- which matters most when the step budget is small (each step here costs hours,
    so we run ~6 steps vs the paper's typical 12).
    """

    name = "textgrad"

    def __init__(self, initial_norm: str, llm, *, max_rounds: int = 12, revert: bool = True):
        super().__init__(initial_norm)
        self.llm = llm
        self.max_rounds = max_rounds
        self.revert = revert
        self.last_gradient = ""
        self.n_reverts = 0

    def propose(self) -> str:
        return self._current

    def observe(self, ev: NormEvaluation) -> None:
        prev_best = self.best_reward
        super().observe(ev)
        if not ev.ok:
            return
        # Validation-based rollback: if this step regressed, revise from the best norm.
        base_norm = ev.norm
        if self.revert and prev_best > float("-inf") and ev.reward < prev_best:
            self.n_reverts += 1
            base_norm = self.best_norm
        gradient = self.llm.chat(
            system=(
                "You are TextGrad, an optimizer that improves a text 'variable' by writing a "
                "textual gradient: concrete, actionable feedback on how to change it to reduce loss."
            ),
            user=(
                f"{OBJECTIVE}\n\nThe current NORM (variable being optimized):\n"
                f"<NORM>\n{ev.norm}\n</NORM>\n\n"
                f"Measured outcome when fishers lived under this norm:\n{format_metrics(ev, self.max_rounds)}\n\n"
                "Write a short textual gradient: 2-4 sentences on the SINGLE most important, concrete "
                "change to this norm to raise reward (be specific about numbers/rules). Do not rewrite "
                "the norm yet."
            ),
        )
        self.last_gradient = gradient.strip()
        revised = self.llm.chat(
            system="You revise a governance norm by applying feedback. Output only the improved norm.",
            user=(
                f"{OBJECTIVE}\n\nCurrent NORM:\n<NORM>\n{base_norm}\n</NORM>\n\n"
                f"Feedback (textual gradient):\n{self.last_gradient}\n\n"
                "Apply the feedback. Return the improved norm as a concise set of rules fishers can "
                "follow, wrapped exactly as <NORM>...</NORM>."
            ),
        )
        self._current = extract_norm(revised, fallback=base_norm)

    def stats(self) -> dict:
        return {"best_reward": self.best_reward, "last_gradient": self.last_gradient,
                "n_reverts": self.n_reverts}


class ACEOptimizer(NormOptimizer):
    """ACE: evolve a playbook (Generator/Reflector/Curator), draft norms from it."""

    name = "ace"

    def __init__(self, initial_norm: str, llm, *, max_rounds: int = 12, max_bullets: int = 20):
        super().__init__(initial_norm)
        self.llm = llm
        self.max_rounds = max_rounds
        self.max_bullets = max_bullets
        self.playbook: list[str] = []
        self._proposed = initial_norm

    def _playbook_text(self) -> str:
        if not self.playbook:
            return "(empty)"
        return "\n".join(f"- {b}" for b in self.playbook)

    def propose(self) -> str:
        if not self.playbook:
            self._proposed = self.initial_norm
            return self._proposed
        draft = self.llm.chat(  # Generator
            system="You are the Generator in ACE. Draft a governance norm using the playbook of lessons.",
            user=(
                f"{OBJECTIVE}\n\nPlaybook of lessons learned so far:\n{self._playbook_text()}\n\n"
                "Draft the best norm you can, applying the playbook. Return it wrapped exactly as "
                "<NORM>...</NORM>."
            ),
        )
        self._proposed = extract_norm(draft, fallback=self.best_norm)
        return self._proposed

    def observe(self, ev: NormEvaluation) -> None:
        super().observe(ev)
        if not ev.ok:
            return
        lessons = self.llm.chat(  # Reflector
            system="You are the Reflector in ACE. Distill transferable lessons from an outcome.",
            user=(
                f"{OBJECTIVE}\n\nNorm that was tried:\n<NORM>\n{ev.norm}\n</NORM>\n\n"
                f"Outcome:\n{format_metrics(ev, self.max_rounds)}\n\n"
                "List 1-3 concrete, transferable lessons about what makes a commons norm score higher or "
                "lower. One lesson per line, each starting with '- '. Be specific about numbers/mechanisms."
            ),
        )
        new_bullets = [ln.strip()[2:].strip() for ln in lessons.splitlines()
                       if ln.strip().startswith("- ")]
        # Curator: append genuinely new bullets, cheap dedup by lowercased prefix.
        existing = {b.lower()[:60] for b in self.playbook}
        for b in new_bullets:
            if b and b.lower()[:60] not in existing:
                self.playbook.append(b)
                existing.add(b.lower()[:60])
        if len(self.playbook) > self.max_bullets:
            self.playbook = self.playbook[-self.max_bullets:]

    def stats(self) -> dict:
        return {"best_reward": self.best_reward, "playbook_size": len(self.playbook),
                "playbook": list(self.playbook)}


# --------------------------------------------------------------------------------------
# Offline mock LLM for dry runs (no network / no spend). Nudges any numeric per-fisher
# cap in the norm toward the sustainable value (10) so the loop visibly "improves" and the
# real extract/parse paths get exercised.
# --------------------------------------------------------------------------------------
class MockLLM:
    def __init__(self):
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    @staticmethod
    def _current_cap(text: str) -> int | None:
        nums = re.findall(r"\b(\d{1,3})\b", text)
        return int(nums[0]) if nums else None

    def chat(self, system: str, user: str, *, temperature: float | None = None) -> str:
        self.calls += 1
        cap = self._current_cap(user)
        if cap is None:
            cap = 20
        # move a third of the way toward 10 each step
        new_cap = int(round(cap + (10 - cap) / 3)) if cap != 10 else 10
        s = system.lower()
        if "textual gradient" in s or "reflector" in s:
            # critique / lessons: plain text
            return (f"- A per-fisher monthly cap near 10 is sustainable; current {cap} is "
                    f"{'too high' if cap > 10 else 'too low' if cap < 10 else 'about right'}.\n"
                    "- Add an explicit numeric cap and a penalty for exceeding it.")
        # generator / revise: return a wrapped norm with the nudged cap
        return (f"<NORM>\nEach fisher may catch at most {new_cap} fish per month. "
                "Exceeding the cap forfeits the excess to restock the lake. Shares are equal "
                "across all five fishers so the lake regenerates each month.\n</NORM>")
