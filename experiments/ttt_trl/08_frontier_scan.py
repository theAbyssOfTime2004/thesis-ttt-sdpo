"""
08_frontier_scan.py

Find problems at the model's "frontier" — where one-shot pass rate is in (0,1).
Those are the good testbeds for TTT discovery: easy problems (pass=1.0) hit a
ceiling (nothing to learn), impossible ones (pass=0.0) give no signal.

No training — just PRE-eval each problem with N samples and report pass rate.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import tokenization_utils_base
from transformers.utils import hub as hub_utils

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.ttt_trl.evaluator import evaluate_solution, load_lcbv6_split


def _prepare_tokenizer(model_name: str, thinking: bool = False):
    original_list_repo_templates = hub_utils.list_repo_templates

    def _safe_list_repo_templates(*args, **kwargs):
        try:
            return original_list_repo_templates(*args, **kwargs)
        except Exception:
            return []

    hub_utils.list_repo_templates = _safe_list_repo_templates
    tokenization_utils_base.list_repo_templates = _safe_list_repo_templates
    tok = AutoTokenizer.from_pretrained(model_name)

    if not thinking:
        _orig_apply = tok.apply_chat_template

        def _no_think_apply(*a, **kw):
            kw.setdefault("enable_thinking", False)
            try:
                return _orig_apply(*a, **kw)
            except TypeError:
                kw.pop("enable_thinking", None)
                return _orig_apply(*a, **kw)

        tok.apply_chat_template = _no_think_apply
    return tok


@torch.no_grad()
def pass_rate_for(model, tokenizer, row, n_samples, max_new_tokens) -> dict:
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    question_content = str(row.get("question_content", ""))
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question_content}], add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        num_return_sequences=n_samples,
        use_cache=True,
        pad_token_id=pad_id,
    )
    scores = []
    for i in range(out.shape[0]):
        code = tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True)
        try:
            scores.append(float(evaluate_solution(code, row, split="train")["score"]))
        except Exception:
            scores.append(0.0)
    pass_rate = sum(1 for s in scores if s >= 1.0) / len(scores)
    return {"pass_rate": pass_rate, "mean_score": statistics.mean(scores), "scores": scores}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan LCBv6 for frontier problems (one-shot pass rate in (0,1)).")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--num_problems", type=int, default=40, help="How many LCBv6 problems to scan from index 0.")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="outputs/08_frontier_scan")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Expected Colab L4.")
    set_seed(args.seed)

    print(f"GPU: {torch.cuda.get_device_name(0)} | model: {args.model_name} | thinking: {args.thinking}")
    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    lcb = load_lcbv6_split()
    n = min(args.num_problems, len(lcb))
    tokenizer = _prepare_tokenizer(args.model_name, thinking=args.thinking)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    results = []
    print(f"\nScanning {n} problems with {args.n_samples} samples each ...\n")
    print("idx | problem_id | difficulty | pass_rate | mean | frontier?")
    for idx in range(n):
        row = lcb[idx]
        pid = row.get("question_id", row.get("problem_id", f"idx_{idx}"))
        diff = row.get("difficulty", "?")
        try:
            r = pass_rate_for(model, tokenizer, row, args.n_samples, args.max_new_tokens)
        except Exception as exc:
            print(f"{idx} | {pid} | {diff} | ERROR: {exc}")
            results.append({"idx": idx, "problem_id": pid, "difficulty": diff, "error": str(exc)})
            continue
        frontier = 0.0 < r["pass_rate"] < 1.0
        mark = "  <-- FRONTIER" if frontier else ("  (ceiling)" if r["pass_rate"] >= 1.0 else "  (too hard)")
        print(f"{idx} | {pid} | {diff} | {r['pass_rate']:.2f} | {r['mean_score']:.2f}{mark}")
        results.append({
            "idx": idx, "problem_id": pid, "difficulty": diff,
            "pass_rate": r["pass_rate"], "mean_score": r["mean_score"], "frontier": frontier,
        })

    frontier_idxs = [r["idx"] for r in results if r.get("frontier")]
    too_hard = [r["idx"] for r in results if r.get("pass_rate") == 0.0]
    ceiling = [r["idx"] for r in results if r.get("pass_rate") == 1.0]

    print("\n=== SCAN SUMMARY ===")
    print(f"Frontier (0<pass<1, BEST for TTT): {frontier_idxs}")
    print(f"Ceiling (pass=1, too easy):        {ceiling}")
    print(f"Too hard (pass=0):                 {too_hard}")
    print(f"\nRecommend running 07 with --problem_index from the FRONTIER list.")

    with open(output_root / "scan.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results,
                   "frontier_idxs": frontier_idxs}, f, indent=2)
    print(f"Saved: {output_root / 'scan.json'}")


if __name__ == "__main__":
    main()
