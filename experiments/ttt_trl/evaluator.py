"""
Evaluator wrapper for LiveCodeBench v6 code problems.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Ensure repo root is importable when this file is executed directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from data.utils.livecodebench import _translate_private_test_cases
from verl.utils.reward_score.feedback import code as code_reward


def _ensure_python_fence(solution_code: str) -> str:
    """
    code_reward.extract_code expects a fenced markdown block.
    Auto-wrap plain Python source to avoid incorrect-format false negatives.
    """
    text = solution_code.strip()
    if "```" in text:
        return text
    return f"```python\n{text}\n```"


def load_lcbv6_split():
    """
    Load LiveCodeBench v6 split.

    Uses the same fallback strategy as `02_lcbv6_loader.py` for environments
    where datasets script execution is disabled.
    """
    try:
        return load_dataset("livecodebench/code_generation_lite", split="test6")
    except Exception:
        jsonl_path = hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            repo_type="dataset",
            filename="test6.jsonl",
        )
        return load_dataset("json", data_files=str(jsonl_path), split="train")


def evaluate_solution(
    solution_code: str,
    raw_lcbv6_row: dict,
    split: str = "train",
) -> dict:
    """
    Run solution against private test cases of an LCBv6 problem.

    Args:
        solution_code: full Python source to evaluate
        raw_lcbv6_row: raw row from LCBv6 HF dataset (must have
            'private_test_cases', 'metadata' fields)
        split: "train" (dense reward) or "test" (sparse reward).
            Default "train" for richer signal during POC.

    Returns:
        {
          "score": float,
          "n_total": int,
          "n_passed": int,
          "details": Any
        }
    """
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split: {split}. Expected 'train' or 'test'.")

    if "private_test_cases" not in raw_lcbv6_row:
        raise KeyError("raw_lcbv6_row missing required key: private_test_cases")

    metadata_raw = raw_lcbv6_row.get("metadata")
    if isinstance(metadata_raw, str) and metadata_raw.strip():
        metadata = json.loads(metadata_raw)
    else:
        metadata = metadata_raw or {}
    fn_name = metadata.get("func_name", "")

    ground_truth = _translate_private_test_cases(
        raw_lcbv6_row["private_test_cases"], fn_name=fn_name
    )

    normalized_solution = _ensure_python_fence(solution_code)

    result = code_reward.compute_score(
        solution=normalized_solution,
        ground_truth=ground_truth,
        extra_info={"split": split, "truncated": False},
    )

    test_cases = json.loads(ground_truth)
    n_total = len(test_cases.get("inputs", []))
    score_value = float(result.get("score", 0.0))

    # Estimated from reward semantics:
    # - split="train" -> dense score = pass ratio
    # - split="test" -> sparse score = 1.0 if all pass else 0.0
    if split == "test":
        n_passed = n_total if score_value == 1.0 else 0
    else:
        n_passed = int(round(score_value * n_total))
    n_passed = max(0, min(n_total, n_passed))

    return {
        "score": score_value,
        "n_total": n_total,
        "n_passed": n_passed,
        "details": result,
    }


def load_math_split():
    """
    Load the MATH-500 test split.

    Fields per row: problem (str), answer (str, ground-truth), solution (str),
    subject (str), level (int 1-5), unique_id (str).
    """
    return load_dataset("HuggingFaceH4/MATH-500", split="test")


def _normalize_math_answer(s: str) -> str:
    """
    Strip LaTeX-cosmetic noise so two expressions that differ only in presentation
    compare equal. Conservative: only removes formatting that never changes the
    mathematical value (sizing, spacing, display directives, frac aliases).

    e.g. r"\\left( 3, \\frac{\\pi}{2} \\right)" and r"(3, \\frac{\\pi}{2})"
         both normalize to "(3,\\frac{\\pi}{2})".
    """
    import re

    s = str(s).strip()
    # Drop surrounding math delimiters.
    s = s.replace("$", "")
    # Sizing wrappers: \left( \right) \bigl \bigr ...
    s = re.sub(r"\\(left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr|big|Big|bigg|Bigg)\b", "", s)
    # frac aliases -> \frac (value-identical).
    s = re.sub(r"\\[dt]frac\b", r"\\frac", s)
    # Display / text directives that carry no value.
    s = re.sub(r"\\(displaystyle|textstyle|scriptstyle|mathrm|text|mbox)\b", "", s)
    # LaTeX spacing commands.
    s = re.sub(r"\\[,;:! ]", "", s)
    s = re.sub(r"\\(quad|qquad)\b", "", s)
    # Trailing units like "\\text{ degrees}" already stripped; drop remaining braces-spaces.
    s = re.sub(r"\s+", "", s)
    # Cosmetic-only trailing period.
    s = s.rstrip(".")
    return s


def evaluate_solution_math(
    solution_text: str,
    row: dict,
    split: str = "train",
) -> dict:
    """
    Score a math solution against a MATH-500 row's ground-truth answer.

    The reward is BINARY (0.0/1.0), not a dense pass-ratio (unlike the code path).
    `correctness_feedback=True` puts us in the LEAK regime: when the answer is
    wrong, the returned feedback reveals the correct answer.

    Returns the SAME shape as the code `evaluate_solution`:
        {"score": float, "n_total": 1, "n_passed": int(score), "details": result}
    where `details` is the raw math.compute_score dict (contains "feedback").

    FALLBACK (math.py is left untouched per spec): math_verify cannot match
    tuple/coordinate/interval/set answers like r"\\left( 3, \\frac{\\pi}{2} \\right)"
    vs r"(3, \\frac{\\pi}{2})", producing false negatives that wreck the frontier
    scan. So when math.py scores 0 but a boxed answer WAS extracted, we retry with
    a conservative LaTeX-cosmetic normalization and only flip 0 -> 1 on an exact
    normalized match (negligible false-positive risk).
    """
    # Lazy import: math.py pulls in math_verify at module load. Keeping it lazy
    # means the code domain path never requires math_verify to be installed.
    from verl.utils.reward_score.feedback import math as math_reward

    ground_truth = str(row["answer"])
    result = math_reward.compute_score(
        solution_str=solution_text,
        ground_truth=ground_truth,
        extra_info={"split": split, "truncated": False},
        correctness_feedback=True,  # leak regime: feedback reveals the answer
    )

    score_value = float(result.get("score", 0.0))

    if score_value < 1.0:
        pred = result.get("pred") or ""
        if pred and _normalize_math_answer(pred) == _normalize_math_answer(ground_truth):
            score_value = 1.0
            # Keep the raw dict but record the override for transparency/logging.
            result = {**result, "score": 1.0, "acc": 1.0, "normalized_match": True,
                      "feedback": ""}

    return {
        "score": score_value,
        "n_total": 1,
        "n_passed": int(score_value),
        "details": result,
    }
