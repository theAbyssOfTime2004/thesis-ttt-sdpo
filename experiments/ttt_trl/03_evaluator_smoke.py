"""
03_evaluator_smoke.py

Smoke test for evaluator wrapper on one LCBv6 problem.
"""

from __future__ import annotations

import pathlib
import sys
import time

# Ensure repo root is importable when this file is executed directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.ttt_trl.evaluator import evaluate_solution, load_lcbv6_split


CORRECT_SOLUTION = """
# Hardcoded correct solution for abc387_b.
# Problem: 9x9 multiplication table, sum of i*j for all 1<=i,j<=9
# where i*j != X. Read X from stdin.
import sys
x = int(sys.stdin.read().strip())
total = 0
for i in range(1, 10):
    for j in range(1, 10):
        if i * j != x:
            total += i * j
print(total)
"""


WRONG_SOLUTION = """
# Always prints 0 — should fail.
print(0)
"""


def _run_eval(label: str, code: str, row: dict, split: str = "test"):
    start = time.time()
    result = evaluate_solution(solution_code=code, raw_lcbv6_row=row, split=split)
    elapsed = time.time() - start
    print(
        f"{label}: score={result['score']:.4f}, "
        f"n_total={result['n_total']}, n_passed={result['n_passed']}, "
        f"runtime_sec={elapsed:.2f}"
    )
    return result, elapsed


def main() -> None:
    dataset = load_lcbv6_split()
    row = dataset[0]
    problem_id = row.get("question_id", row.get("problem_id", "unknown"))
    print(f"Loaded problem_id: {problem_id}")

    # Use split="test" to enforce sparse rewards (all-pass => 1.0 else 0.0).
    correct_result, correct_runtime = _run_eval(
        label="CORRECT",
        code=CORRECT_SOLUTION,
        row=row,
        split="test",
    )
    wrong_result, wrong_runtime = _run_eval(
        label="WRONG",
        code=WRONG_SOLUTION,
        row=row,
        split="test",
    )

    assert correct_result["score"] >= 0.9, "correct solution should pass"
    assert wrong_result["score"] <= 0.1, "wrong solution should fail"

    print(f"Runtime summary: correct={correct_runtime:.2f}s, wrong={wrong_runtime:.2f}s")
    print("OK: evaluator smoke test passed.")


if __name__ == "__main__":
    # Required for compute_score's multiprocessing path.
    main()
