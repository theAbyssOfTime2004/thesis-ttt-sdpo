"""
Domain dispatch (single source of truth) for the TTT-SDPO scripts 07/08/09.

Task domains, selectable via `--domain {code,math,aime}`:
  - code: LiveCodeBench v6 (LCBv6), dense pass-ratio reward.
  - math: MATH-500, binary (0.0/1.0) reward.
  - aime: AIME 2026 (MathArena), binary reward (reuses math evaluator).

Each domain exposes a uniform interface so the scripts route problem loading,
prompt construction, scoring, and logging through one place. The CODE domain
reproduces the previous hard-coded behavior byte-for-byte.
"""

from __future__ import annotations

import json
from typing import Any

from experiments.ttt_trl.evaluator import (
    evaluate_solution,
    evaluate_solution_math,
    load_aime_split,
    load_lcbv6_split,
    load_math_split,
)

# --------------------------------------------------------------------------- #
# Per-domain directives appended to the user turn.
# --------------------------------------------------------------------------- #
# CODE directive: UNCHANGED from 07/08 (regression-critical, must stay byte-for-byte).
_CODE_DIRECTIVE = (
    "\n\nRespond with ONLY the complete Python solution inside a single "
    "```python ... ``` block. Do not explain. Read input from stdin, print to stdout."
)
# MATH directive: MUST allow reasoning (do NOT say "do not explain"; thinking needs it).
_MATH_DIRECTIVE = (
    "\n\nSolve the problem. Reason step by step, then give the final answer as "
    "\\boxed{your_answer}. You MUST give the final answer as \\boxed{N}. "
    "Do not state it any other way."
)


def build_privileged_context(row: dict, max_cases: int = 5, max_len: int = 200) -> str:
    """
    Privileged hint for the SDPO teacher (Path B feedback) — CODE domain only.
    Uses PUBLIC test cases only (private tests stay for reward scoring -> no leak).
    The student gets no hint; the teacher sees these expected I/O pairs.

    Moved verbatim from 07_discovery_curve.py so the dispatch is the single source
    of truth; 07 re-exports it for backward compatibility.
    """
    pts = row.get("public_test_cases") or []
    # Raw LCBv6 rows store this as a JSON string; parse if needed.
    if isinstance(pts, str):
        pts = pts.strip()
        if not pts:
            return ""
        try:
            pts = json.loads(pts)
        except Exception:
            return f"Reference (public tests):\n{pts[:max_len * max_cases]}"
    if not isinstance(pts, list) or not pts:
        return ""
    lines = ["Your solution must satisfy these test cases:"]
    for t in pts[:max_cases]:
        if not isinstance(t, dict):
            continue
        inp = str(t.get("input", ""))[:max_len]
        out = str(t.get("output", ""))[:max_len]
        lines.append(f"- Input: {inp}  ->  Expected: {out}")
    return "\n".join(lines)


class Domain:
    """Base interface. Subclasses fill in the domain-specific behavior."""

    name: str = ""

    def load_split(self):
        raise NotImplementedError

    def problem_text(self, row: dict) -> str:
        raise NotImplementedError

    def evaluate(self, solution: str, row: dict, split: str = "train") -> dict:
        raise NotImplementedError

    def directive(self) -> str:
        raise NotImplementedError

    def reference_answer(self, row: dict) -> str:
        raise NotImplementedError

    def difficulty(self, row: dict) -> Any:
        raise NotImplementedError

    def problem_id(self, row: dict) -> Any:
        raise NotImplementedError

    def privileged_context(self, row: dict) -> str:
        raise NotImplementedError


class CodeDomain(Domain):
    name = "code"

    def load_split(self):
        return load_lcbv6_split()

    def problem_text(self, row: dict) -> str:
        return str(row.get("question_content", ""))

    def evaluate(self, solution: str, row: dict, split: str = "train") -> dict:
        return evaluate_solution(solution, row, split=split)

    def directive(self) -> str:
        return _CODE_DIRECTIVE

    def reference_answer(self, row: dict) -> str:
        return ""

    def difficulty(self, row: dict) -> Any:
        return row.get("difficulty", "unknown")

    def problem_id(self, row: dict) -> Any:
        return row.get("question_id") or row.get("problem_id")

    def privileged_context(self, row: dict) -> str:
        return build_privileged_context(row)


class MathDomain(Domain):
    name = "math"

    def load_split(self):
        return load_math_split()

    def problem_text(self, row: dict) -> str:
        return str(row.get("problem", ""))

    def evaluate(self, solution: str, row: dict, split: str = "train") -> dict:
        return evaluate_solution_math(solution, row, split=split)

    def directive(self) -> str:
        return _MATH_DIRECTIVE

    def reference_answer(self, row: dict) -> str:
        return str(row.get("answer", ""))

    def difficulty(self, row: dict) -> Any:
        return f"L{row.get('level', '?')}"

    def problem_id(self, row: dict) -> Any:
        return row.get("unique_id")

    def privileged_context(self, row: dict) -> str:
        # Leak regime: teacher always sees the reference answer (LLM judge catches copy).
        return f"Reference: the correct final answer is {row.get('answer', '')}."


class AimeDomain(Domain):
    name = "aime"

    def load_split(self):
        return load_aime_split()

    def problem_text(self, row: dict) -> str:
        return str(row.get("problem", ""))

    def evaluate(self, solution: str, row: dict, split: str = "train") -> dict:
        # Integer answers in row["answer"] stringify cleanly for boxed matching.
        return evaluate_solution_math(solution, row, split=split)

    def directive(self) -> str:
        return _MATH_DIRECTIVE

    def reference_answer(self, row: dict) -> str:
        return str(row.get("answer", ""))

    def difficulty(self, row: dict) -> Any:
        return "AIME2026"

    def problem_id(self, row: dict) -> Any:
        return f"aime2026_{row.get('problem_idx', '?')}"

    def privileged_context(self, row: dict) -> str:
        return f"Reference: the correct final answer is {row.get('answer', '')}."


_DOMAINS: dict[str, Domain] = {
    "code": CodeDomain(),
    "math": MathDomain(),
    "aime": AimeDomain(),
}


def get_domain(name: str) -> Domain:
    if name not in _DOMAINS:
        raise ValueError(f"Unknown domain: {name!r}. Choices: {sorted(_DOMAINS)}")
    return _DOMAINS[name]
