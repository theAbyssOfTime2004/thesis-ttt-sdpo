"""
02_lcbv6_loader.py

Load one LiveCodeBench v6 problem and verify the data pipeline.
No model or training involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def _load_lcbv6_split():
    """
    Reuse baseline pattern (`datasets.load_dataset`) while handling script-based repo layout.
    For this repo, v6 corresponds to `test6.jsonl`.
    """
    try:
        # Preferred direct route (kept for compatibility if datasets lib allows scripts).
        return load_dataset("livecodebench/code_generation_lite", split="test6")
    except Exception:
        jsonl_path = hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            repo_type="dataset",
            filename="test6.jsonl",
        )
        return load_dataset("json", data_files=str(jsonl_path), split="train")


def _parse_test_cases(raw_value: Any) -> Any:
    if isinstance(raw_value, (list, dict)):
        return raw_value
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        value = raw_value.strip()
        if not value:
            return []
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Keep raw string if provider changes format.
            return raw_value
    return raw_value


def _count_cases(parsed: Any) -> int:
    if isinstance(parsed, list):
        return len(parsed)
    if isinstance(parsed, dict):
        # Handles either {"input": [...], "output": [...]} or nested test dicts.
        if "input" in parsed and isinstance(parsed["input"], list):
            return len(parsed["input"])
        return len(parsed)
    if isinstance(parsed, str):
        return 1 if parsed else 0
    return 0


def _select_problem(dataset, problem_id: str | None):
    if problem_id is None:
        return dataset[0]
    for row in dataset:
        if str(row.get("question_id", "")) == problem_id:
            return row
    raise ValueError(f"problem_id not found in v6 split: {problem_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load one LiveCodeBench v6 sample.")
    parser.add_argument(
        "--problem-id",
        type=str,
        default=None,
        help="Specific question_id to load for reproducibility (default: first row).",
    )
    args = parser.parse_args()

    dataset = _load_lcbv6_split()
    total = len(dataset)
    row = _select_problem(dataset, args.problem_id)

    problem = {
        "problem_id": str(row.get("question_id", "")),
        "question_content": str(row.get("question_content", "")),
        "starter_code": row.get("starter_code", ""),
        "public_test_cases": _parse_test_cases(row.get("public_test_cases")),
        "private_test_cases": _parse_test_cases(row.get("private_test_cases")),
        "difficulty": row.get("difficulty", ""),
    }

    public_count = _count_cases(problem["public_test_cases"])
    private_count = _count_cases(problem["private_test_cases"])

    print(f"Total problems in split: {total}")
    print(f"Selected problem_id: {problem['problem_id']}")
    print(f"question_content[0:200]: {problem['question_content'][:200]}")
    print(f"Public test cases: {public_count}")
    print(f"Private test cases: {private_count}")

    assert isinstance(problem["problem_id"], str), "problem_id must be string"
    assert problem["problem_id"], "problem_id must be non-empty"
    assert isinstance(problem["question_content"], str) and problem["question_content"].strip(), "question_content empty"
    assert public_count >= 1, "at least 1 public test case is required"

    out_path = Path("outputs/02_lcbv6_sample.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(problem, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Saved sample to: {out_path}")


if __name__ == "__main__":
    main()
