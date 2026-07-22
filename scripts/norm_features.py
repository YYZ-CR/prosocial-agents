"""Feature extraction over the norm texts produced in the optimization grid.

Shared by the norm-dissection analysis (paper item 3) and the specificity-index
analysis (item 4). We deliberately use transparent regex heuristics rather than an
LLM classifier: the features are simple, auditable, and reproducible, and a
reviewer can check every rule by eye.

extract_features(text) -> dict of:
    n_chars, n_words          size
    n_clauses                 numbered items if present, else sentence count
    n_numeric_caps            explicit numeric catch limits (tokens like "10 tons",
                              "8%", "no more than 5")
    n_stock_conditionals      catch rules conditioned on the MEASURED stock level
                              -- the operational test of "state-contingent"
    is_state_contingent       n_stock_conditionals >= 1
    has_moratorium            an explicit zero-catch / stop-fishing clause
    has_equal_split           an equal-share / fair-division clause
    has_enforcement           logging / announcing / sanction / verification
    has_rotation              sequential or rotating turn order
    has_amendment_rule        an explicit vote / amendment threshold

load_all_norms() -> list of records (cell, method, iteration, kind, text, reward,
metrics) pulled straight from the committed trajectory JSONs.
"""
from __future__ import annotations

import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ = os.path.join(REPO, "scripts", "norm_opt_runs", "grid_v1")

CELLS = ["gemma4-31b", "luna", "sonnet5-off", "sonnet5-on", "opus48", "sol"]
METHODS = ["baseline", "textgrad", "ace"]

# --- regex library (case-insensitive) ------------------------------------
_NUM = r"\d+(?:\.\d+)?"
RE_NUMERIC_CAP = re.compile(
    rf"(?:{_NUM}\s*(?:tons?|fish|%|percent)\b)"
    rf"|(?:\b(?:no more than|up to|at most|maximum of|max(?:\.|imum)?|cap(?:ped)? at|limit of)\s+{_NUM})"
    rf"|(?:{_NUM}\s+(?:per|each)\s+(?:fisher|person|agent|month|head))",
    re.IGNORECASE,
)
# Catch rules that reference the *measured* stock level -> genuinely state-contingent.
RE_STOCK_COND = re.compile(
    rf"(?:if\s+(?:the\s+)?(?:current\s+)?(?:stock|lake|pool|remaining|level|s)\b[^.]*?[<>=≤≥])"
    rf"|(?:\b(?:stock|lake|pool|remaining|level|s)\s*[<>=≤≥]+\s*{_NUM})"
    rf"|(?:\b(?:if|when|while|whenever)\b[^.]*?\b(?:stock|lake|pool)\b[^.]*?\b(?:below|above|under|over|at least|reaches|drops|falls|exceeds|less than|greater than)\b)"
    rf"|(?:\b(?:below|above|under|over)\s+{_NUM}\s*(?:tons?|fish)?\b[^.]*?(?:fish|catch|take|moratorium|stop))"
    rf"|(?:%\s*of\s*(?:current\s*)?(?:stock|s\b|the\s+stock))"
    rf"|(?:proportional\s+to\s+(?:the\s+)?(?:current\s+)?stock)",
    re.IGNORECASE,
)
RE_MORATORIUM = re.compile(
    r"moratorium|no\s+fishing|do\s+not\s+fish|don't\s+fish|zero\s+catch|"
    r"stop\s+fishing|halt\s+(?:all\s+)?fishing|catch\s+(?:of\s+)?0\b|0\s+catch|"
    r"suspend\s+fishing|no\s+(?:one|fisher)\s+(?:may\s+)?(?:fish|catch)",
    re.IGNORECASE,
)
RE_EQUAL = re.compile(
    r"equal(?:ly|-split|\s+split|\s+share)?|split\s+equally|fair\s+share|"
    r"divided\s+equally|same\s+amount|per\s+fisher|each\s+fisher\s+(?:may|takes|gets)",
    re.IGNORECASE,
)
RE_ENFORCE = re.compile(
    r"enforc|sanction|penalt|\blog(?:ged|ging|s)?\b|announc|verif|"
    r"public(?:ly)?\s+report|report(?:ed|ing)|transparen|accountab",
    re.IGNORECASE,
)
RE_ROTATION = re.compile(
    r"rotat|sequential|fixed\s+order|turn\s+order|in\s+order|one\s+at\s+a\s+time|"
    r"rotating\s+order|takes\s+turns",
    re.IGNORECASE,
)
RE_AMEND = re.compile(
    r"amend|four\s+of\s+(?:the\s+)?five|4\s*(?:/|\s+of\s+(?:the\s+)?)\s*5|"
    r"majority\s+vote|unanim|\bvote\b|revise\s+this\s+(?:rule|norm)|"
    r"(?:by|with|requires?|needs?)\s+(?:the\s+)?agreement|agreement\s+(?:from|of)",
    re.IGNORECASE,
)
RE_NUMBERED = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+")
RE_SENT = re.compile(r"[.!?]+(?:\s|$)")


def _count(rx, text):
    return len(rx.findall(text))


def n_clauses(text: str) -> int:
    numbered = _count(RE_NUMBERED, text)
    if numbered >= 2:
        return numbered
    return max(1, _count(RE_SENT, text))


def extract_features(text: str) -> dict:
    text = text or ""
    n_stock = _count(RE_STOCK_COND, text)
    return {
        "n_chars": len(text),
        "n_words": len(text.split()),
        "n_clauses": n_clauses(text),
        "n_numeric_caps": _count(RE_NUMERIC_CAP, text),
        "n_stock_conditionals": n_stock,
        "is_state_contingent": n_stock >= 1,
        "has_moratorium": bool(RE_MORATORIUM.search(text)),
        "has_equal_split": bool(RE_EQUAL.search(text)),
        "has_enforcement": bool(RE_ENFORCE.search(text)),
        "has_rotation": bool(RE_ROTATION.search(text)),
        "has_amendment_rule": bool(RE_AMEND.search(text)),
    }


def load_all_norms() -> list[dict]:
    """Every proposed norm across the grid, with its episode outcome."""
    out = []
    for cell in CELLS:
        for method in METHODS:
            p = os.path.join(TRAJ, cell, f"trajectory_{method}.json")
            if not os.path.exists(p):
                continue
            traj = json.load(open(p, encoding="utf-8"))["trajectory"]
            for it in traj:
                out.append({
                    "cell": cell, "method": method, "iteration": it["iteration"],
                    "text": (it.get("norm") or "").strip(),
                    "reward": it.get("reward"),
                    "metrics": it.get("metrics_mean", {}),
                    "changed": it.get("changed"),
                })
    return out


if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    norms = load_all_norms()
    seen, distinct = set(), []
    for r in norms:
        if r["text"] and r["text"] not in seen:
            seen.add(r["text"])
            distinct.append(r)
    print(f"{len(norms)} norm slots, {len(distinct)} distinct texts\n")
    hdr = f"{'cell':11s} {'method':8s} it  chars clause ncap scond  SC mor eq enf rot amd  reward"
    print(hdr)
    for r in norms:
        f = extract_features(r["text"])
        flags = "".join("Y" if f[k] else "." for k in
                        ("is_state_contingent", "has_moratorium", "has_equal_split",
                         "has_enforcement", "has_rotation", "has_amendment_rule"))
        rw = f"{r['reward']:.2f}" if r["reward"] is not None else "  -"
        print(f"{r['cell']:11s} {r['method']:8s} {r['iteration']:2d} "
              f"{f['n_chars']:6d} {f['n_clauses']:5d} {f['n_numeric_caps']:4d} "
              f"{f['n_stock_conditionals']:5d}  {flags[0]}   {flags[1]}   {flags[2]}  "
              f"{flags[3]}   {flags[4]}   {flags[5]}   {rw}")
