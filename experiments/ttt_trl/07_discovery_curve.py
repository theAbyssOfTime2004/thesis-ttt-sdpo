"""
07_discovery_curve.py

Phase 2a: discovery curve + proof-of-effectiveness on a single problem.
- Same TTT-SDPO setup as 06 (Qwen2.5-1.5B + LoRA, feedback path ON).
- Default: 1 problem (abc387_b, index 0), 15 steps, seed=0.
- NEW vs 06:
  * PRE-eval: sample N solutions with the BASE model -> pass rate / mean score.
  * TTT: train max_steps steps; log per-step reward (the discovery curve).
  * POST-eval: sample N solutions with the TTT'd model -> pass rate / mean score.
  * Question answered: does TTT make the model BETTER at this problem
    (post pass-rate > pre pass-rate), not just "loss != 0".
"""

from __future__ import annotations

import argparse
import gc
import inspect
import math
import os
import pathlib
import statistics
import sys
import time
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import tokenization_utils_base
from transformers.utils import hub as hub_utils
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer

# Ensure repo root is importable when this file is executed directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.ttt_trl.evaluator import evaluate_solution, load_lcbv6_split


def _filter_supported_kwargs(cls_or_fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(cls_or_fn)
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def _extract_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return " ".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _prepare_tokenizer(model_name: str, thinking: bool = False):
    # Some repos do not expose `additional_chat_templates/`; treat it as optional.
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
        # Qwen3 enables thinking by default -> long <think> blocks eat the token
        # budget and the model never emits a ```python``` block, so code extraction
        # fails and reward is always 0. Force enable_thinking=False everywhere
        # (both the trainer's internal prompt rendering and our eval). Harmless for
        # tokenizers whose template doesn't accept the kwarg (TypeError -> retry).
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


def _build_messages(question_content: str) -> list[dict[str, str]]:
    # Force a code-first answer: the model otherwise rambles for the whole token
    # budget and never emits a ```python``` block -> reward 0 during training.
    directive = (
        "\n\nRespond with ONLY the complete Python solution inside a single "
        "```python ... ``` block. Do not explain. Read input from stdin, print to stdout."
    )
    return [{"role": "user", "content": question_content + directive}]


def build_privileged_context(row: dict, max_cases: int = 5, max_len: int = 200) -> str:
    """
    Privileged hint for the SDPO teacher (Path B feedback).
    Uses PUBLIC test cases only (private tests stay for reward scoring -> no leak).
    The student gets no hint; the teacher sees these expected I/O pairs.
    """
    import json

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


def build_dynamic_feedback(solution_code: str, row: dict, max_len: int = 1500) -> str:
    """
    REAL environment feedback (Phase 2b): run the model's own attempt through the
    evaluator and return the evaluator's feedback text (which can reveal failure
    modes invisible to public-test hints, e.g. timeouts on large hidden tests).
    This is the verl rich-feedback path (same one baseline multiturn.py consumed).
    """
    if not solution_code.strip():
        return ""
    try:
        out = evaluate_solution(solution_code, row, split="train")
    except Exception as exc:
        print(f"[dyn-feedback] evaluation failed: {exc}")
        return ""
    # If the attempt already passes everything, there is NOTHING to correct.
    # (A misleading "not fully correct" message here actively damaged the policy:
    # idx 38 greedy collapsed 1.0 -> 0.15 after one step of distilling toward a
    # teacher that was told a perfect solution was wrong.)
    if float(out.get("score", 0.0)) >= 1.0:
        return ""
    fb = ""
    det = out.get("details")
    if isinstance(det, dict):
        fb = str(det.get("feedback", "") or "")
    if not fb:
        fb = (f"Your earlier attempt passed {out.get('n_passed')}/{out.get('n_total')} "
              f"hidden tests (score {out.get('score'):.2f}). It is not fully correct.")
    return "Result of running your earlier attempt:\n" + fb[:max_len]


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    row: dict,
    n_samples: int,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> dict:
    """
    Generate solutions with `model` and score them on the problem's tests.
    Used for PRE (base model) vs POST (TTT'd model) comparison = effectiveness.
    Returns pass_rate (fraction fully passing), mean/max score, greedy score.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    question_content = str(row.get("question_content", ""))
    prompt_text = tokenizer.apply_chat_template(
        _build_messages(question_content), add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    def _score(code: str) -> float:
        try:
            return float(evaluate_solution(code, row, split="train")["score"])
        except Exception as exc:
            print(f"Eval score error: {exc}")
            return 0.0

    # Sampled solutions (batched via num_return_sequences).
    scores: list[float] = []
    if n_samples > 0:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            num_return_sequences=n_samples,
            use_cache=True,
            pad_token_id=pad_id,
        )
        for i in range(out.shape[0]):
            code = tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True)
            sc = _score(code)
            scores.append(sc)
            if i == 0:
                # Print the FULL completion so we can read it directly and judge
                # whether it contains code (fenced OR raw) — counting fences is a
                # proxy that misses unfenced code.
                print(f"[eval-debug] sample[0] len={len(code)} chars score={sc:.3f}")
                print("[eval-debug] FULL COMPLETION BELOW >>>>>>>>>>")
                print(code)
                print("[eval-debug] <<<<<<<<<< END FULL COMPLETION")

    # Greedy solution (deterministic reference).
    g = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_id,
    )
    greedy_code = tokenizer.decode(g[0, prompt_len:], skip_special_tokens=True)
    greedy_score = _score(greedy_code)

    if was_training:
        model.train()

    pass_rate = (sum(1 for s in scores if s >= 1.0) / len(scores)) if scores else 0.0
    return {
        "n_samples": n_samples,
        "scores": scores,
        "mean_score": statistics.mean(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "pass_rate": pass_rate,
        "greedy_score": greedy_score,
        "greedy_code": greedy_code,
    }


def safe_evaluate_model(model, tokenizer, row, n_samples, max_new_tokens, label="") -> dict:
    """Wrap evaluate_model so a generation failure does not lose training results."""
    try:
        return evaluate_model(model, tokenizer, row, n_samples, max_new_tokens)
    except Exception as exc:
        print(f"[{label}] evaluate_model FAILED: {exc}")
        return {
            "n_samples": n_samples,
            "scores": [],
            "mean_score": 0.0,
            "max_score": 0.0,
            "pass_rate": 0.0,
            "greedy_score": 0.0,
            "greedy_code": "",
            "error": str(exc),
        }


def _build_lora_config(lora_r: int) -> LoraConfig:
    return LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )


def _build_sdpo_config(
    problem_output_dir: pathlib.Path,
    num_generations: int,
    max_steps: int,
    max_new_tokens: int,
    seed: int,
    temperature: float = 1.0,
    policy_loss_mode: str = "hybrid",
    thinking: bool = False,
    report_to: str = "none",
) -> SDPOConfig:
    requested = {
        "output_dir": str(problem_output_dir),
        "per_device_train_batch_size": 1,
        # Ensure generation_batch_size == num_generations on single GPU.
        "gradient_accumulation_steps": num_generations,
        "num_generations": num_generations,
        "max_steps": max_steps,
        "learning_rate": 1e-5,
        "temperature": temperature,
        "max_completion_length": max_new_tokens,
        # TRL default max_prompt_length=512 TOKENS silently truncates long problem
        # statements (grid problems!) -> model continues the cut-off text instead
        # of answering -> rambling rollouts, reward 0. Long problems need room.
        "max_prompt_length": 4096,
        # max_time must comfortably exceed the time to generate max_new_tokens
        # (8B + thinking 20k tokens needs 15-20 min). 600s silently truncated
        # mid-think -> all-zero training rewards while eval (no max_time) scored.
        "generation_kwargs": {"max_new_tokens": max_new_tokens, "max_time": 2400.0},
        # Thinking-off must reach the TRAINER's internal rendering, not just our
        # eval. The monkey-patch on tok.apply_chat_template did NOT propagate to
        # the student-rollout prompt (GRPOTrainer renders via the trl
        # data_utils.apply_chat_template wrapper with **chat_template_kwargs).
        # This is the official, supported channel -> covers student rollout
        # (grpo_trainer line 1951) AND teacher reprompt tokenization. Without it
        # Qwen3 emits <think> blocks, blows the token budget, never produces a
        # ```python``` block -> every training rollout scores 0 (the artifact).
        "chat_template_kwargs": {"enable_thinking": thinking},
        "gradient_checkpointing": True,
        # Logit-level top-k SDPO (paper: logit > token > sequence), reverse KL.
        # Old TRL (<=1.5) names AND new TRL 1.6 names both included; the
        # _filter_supported_kwargs call keeps whichever the installed trl has.
        "distillation_topk": 20,
        "distillation_alpha": 1.0,
        "full_logit_distillation": True,            # trl <=1.5
        "distillation_mode": "topk_logits",          # trl 1.6
        # hybrid = GRPO policy term (reinforces high-reward rollouts) + distill.
        "sdpo_policy_loss_mode": policy_loss_mode,   # trl <=1.5
        # trl 1.6: blend via weight: (1-w)*policy + w*distillation.
        "distillation_weight": 0.5 if policy_loss_mode == "hybrid" else 1.0,
        # Feedback path (Path B): teacher gets the public-test hint.
        "include_environment_feedback": True,
        "environment_feedback_only_without_solution": True,
        # L4 (Ada) supports bf16 natively: stable, no GradScaler needed.
        "bf16": True,
        "fp16": False,
        "seed": seed,
        "report_to": report_to,
    }
    filtered = _filter_supported_kwargs(SDPOConfig, requested)
    dropped = set(requested) - set(filtered)
    if dropped:
        print(f"[WARN] SDPOConfig dropped unsupported keys: {sorted(dropped)}")
    return SDPOConfig(**filtered)


def _last_finite_loss(trainer: SDPOTrainer) -> float:
    losses = []
    for log in getattr(trainer.state, "log_history", []):
        for key in ("loss", "train_loss"):
            if key in log:
                value = float(log[key])
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite {key}: {value}")
                losses.append(value)
    if not losses:
        raise RuntimeError("No loss values found in trainer.state.log_history")
    return losses[-1]


def _flatten_rewards(step_rewards: list[list[float]]) -> list[float]:
    flat = []
    for batch in step_rewards:
        flat.extend(batch)
    return flat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2a discovery curve for TTT-SDPO on one LCBv6 problem.")
    parser.add_argument("--problem_index", type=int, default=0, help="LCBv6 index (0=abc387_b).")
    parser.add_argument("--max_steps", type=int, default=15, help="TTT steps on the single problem.")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--thinking", action="store_true",
                        help="Enable Qwen3 thinking mode (default off; needs much higher max_new_tokens).")
    parser.add_argument("--eval_samples", type=int, default=8, help="Samples for pre/post eval.")
    parser.add_argument("--policy_loss_mode", type=str, default="hybrid",
                        choices=["hybrid", "distillation_only"],
                        help="hybrid adds GRPO policy term that reinforces successes.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Training sampling temp. Lower (0.7) curbs rambling -> more code.")
    parser.add_argument("--dynamic_feedback", action="store_true",
                        help="Append REAL evaluator feedback on the model's own greedy attempt "
                             "to privileged_context (Phase 2b) instead of static public-test hint only.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs/07_discovery_curve")
    parser.add_argument("--wandb_project", type=str, default="ttt-sdpo-thesis")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This script requires a GPU (expected Colab L4).")

    set_seed(args.seed)

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {device_name}")
    print(f"Total VRAM: {total_vram_gb:.2f} GB")
    print(f"Seed: {args.seed}")

    use_wandb = not args.no_wandb
    report_to = "wandb" if use_wandb else "none"
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=f"discovery-{args.model_name.split('/')[-1]}-idx{args.problem_index}-{args.max_steps}step-{args.policy_loss_mode}",
            config=vars(args),
        )

    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    lcb = load_lcbv6_split()
    row = lcb[args.problem_index]
    problem_id = row.get("question_id", row.get("problem_id", f"idx_{args.problem_index}"))
    difficulty = row.get("difficulty", "unknown")
    question_content = str(row.get("question_content", ""))
    print(f"\n=== PROBLEM idx {args.problem_index}: {problem_id} ({difficulty}) ===")

    tokenizer = _prepare_tokenizer(args.model_name, thinking=args.thinking)
    print(f"Thinking mode: {args.thinking}")
    lora_config = _build_lora_config(args.lora_r)

    total_start = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    # Fresh base model.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    rendered = tokenizer.apply_chat_template(
        _build_messages(question_content), add_generation_prompt=True, tokenize=False
    )
    print(f"Prompt preview: {rendered[:180].replace(chr(10), ' ')}")

    # ----- PRE-eval (base model, before any TTT) -----
    print(f"\n[PRE-eval] sampling {args.eval_samples} solutions with the base model ...")
    pre_eval = safe_evaluate_model(model, tokenizer, row, args.eval_samples, args.max_new_tokens, label="PRE")
    print(f"[PRE-eval] pass_rate={pre_eval['pass_rate']:.3f} "
          f"mean_score={pre_eval['mean_score']:.3f} "
          f"max_score={pre_eval['max_score']:.3f} "
          f"greedy={pre_eval['greedy_score']:.3f} scores={pre_eval['scores']}")

    privileged_context = build_privileged_context(row)
    if args.dynamic_feedback:
        dyn = build_dynamic_feedback(pre_eval.get("greedy_code", ""), row)
        if dyn:
            privileged_context = (privileged_context + "\n\n" + dyn).strip()
    print(f"Privileged context ({len(privileged_context)} chars): "
          f"{privileged_context[:400].replace(chr(10), ' ')}")
    # Use MESSAGES (conversational) format. The trainer renders the chat template
    # itself with chat_template_kwargs={"enable_thinking": False} (set in config) ->
    # official channel: thinking off AND a proper assistant turn. The pre-rendered
    # string approach made training rollouts continue the user text (" below:...")
    # instead of starting a fresh ```python answer like eval does.
    train_dataset = Dataset.from_dict(
        {"prompt": [_build_messages(question_content)], "privileged_context": [privileged_context]}
    )

    problem_dir = output_root / f"problem_{args.problem_index:02d}_{problem_id}"
    config = _build_sdpo_config(
        problem_output_dir=problem_dir,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        temperature=args.temperature,
        policy_loss_mode=args.policy_loss_mode,
        thinking=args.thinking,
        report_to=report_to,
    )

    reward_history: list[list[float]] = []

    def reward_fn(completions, prompts=None, **kwargs):
        if not reward_history:  # first step only: inspect what training actually generated
            if prompts is not None and len(prompts) > 0:
                ptxt = _extract_text(prompts[0])
                print(f"[debug] TRAINER PROMPT[0] len={len(ptxt)}; LAST 600 (the part right before generation):\n{repr(ptxt[-600:])}\n[debug]===")
            dbg = _extract_text(completions[0])
            print(f"[debug] step1 completion[0] len={len(dbg)} chars; first 400:\n{dbg[:400]}\n[debug] last 150:\n{dbg[-150:]}\n[debug]---")
        rewards = []
        for completion in completions:
            code = _extract_text(completion)
            try:
                out = evaluate_solution(code, row, split="train")
                rewards.append(float(out["score"]))
            except Exception as exc:
                print(f"Reward eval error: {exc}")
                rewards.append(0.0)
        reward_history.append(rewards)
        step_i = len(reward_history)
        mean_r = statistics.mean(rewards) if rewards else 0.0
        max_r = max(rewards) if rewards else 0.0
        print(f"[step {step_i}] Reward batch ({len(rewards)}): {rewards}  mean={mean_r:.3f}")
        if use_wandb:
            wandb.log({"curve/step": step_i, "curve/mean_reward": mean_r, "curve/max_reward": max_r})
        return rewards

    trainer_kwargs = {
        "model": model,
        "reward_funcs": reward_fn,
        "args": config,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
        "peft_config": lora_config,
    }
    trainer = SDPOTrainer(**_filter_supported_kwargs(SDPOTrainer, trainer_kwargs))

    # DIAGNOSTIC: dump the exact prompt the trainer will feed to generation.
    try:
        td = trainer.train_dataset
        row0 = td[0]
        print(f"[diag] trainer.train_dataset[0] keys: {list(row0.keys())}")
        pv = row0.get("prompt")
        print(f"[diag] prompt type={type(pv).__name__} len={len(pv) if hasattr(pv,'__len__') else '?'}")
        print(f"[diag] prompt FIRST 300:\n{repr(pv)[:300]}")
        print(f"[diag] prompt LAST 400 (directive + assistant turn should be here):\n{repr(pv)[-400:]}")
        tok_len = len(tokenizer(rendered)["input_ids"])
        print(f"[diag] rendered prompt TOKEN count = {tok_len}; "
              f"trainer.max_prompt_length = {getattr(trainer, 'max_prompt_length', '?')}"
              f"  (tokens > limit -> truncation -> rambling)")
    except Exception as exc:
        print(f"[diag] could not inspect train_dataset: {exc}")

    print(f"\n[TTT] training {args.max_steps} steps on {problem_id} ...")
    run_start = time.time()
    trainer.train()
    run_elapsed = time.time() - run_start

    flat_rewards = _flatten_rewards(reward_history)
    final_loss = _last_finite_loss(trainer)
    trainer.save_model(str(problem_dir))

    # ----- POST-eval (TTT'd model) -----
    print(f"\n[POST-eval] sampling {args.eval_samples} solutions with the TTT'd model ...")
    post_eval = safe_evaluate_model(trainer.model, tokenizer, row, args.eval_samples, args.max_new_tokens, label="POST")
    print(f"[POST-eval] pass_rate={post_eval['pass_rate']:.3f} "
          f"mean_score={post_eval['mean_score']:.3f} "
          f"max_score={post_eval['max_score']:.3f} "
          f"greedy={post_eval['greedy_score']:.3f} scores={post_eval['scores']}")

    peak_vram = torch.cuda.max_memory_allocated(0)
    step_means = [statistics.mean(rs) if rs else 0.0 for rs in reward_history]
    total_elapsed = time.time() - total_start

    print("\n=== DISCOVERY CURVE (mean reward per step) ===")
    print(" -> ".join(f"{m:.2f}" for m in step_means))

    print("\n=== EFFECTIVENESS (pre vs post TTT) ===")
    print(f"pass_rate : {pre_eval['pass_rate']:.3f}  ->  {post_eval['pass_rate']:.3f}  "
          f"(delta {post_eval['pass_rate'] - pre_eval['pass_rate']:+.3f})")
    print(f"mean_score: {pre_eval['mean_score']:.3f}  ->  {post_eval['mean_score']:.3f}  "
          f"(delta {post_eval['mean_score'] - pre_eval['mean_score']:+.3f})")
    print(f"greedy    : {pre_eval['greedy_score']:.3f}  ->  {post_eval['greedy_score']:.3f}")
    print(f"\nFinal loss: {final_loss:.6f}")
    print(f"Peak VRAM allocated: {peak_vram / (1024**3):.2f} GB")
    print(f"TTT runtime: {run_elapsed:.2f} s | Total runtime: {total_elapsed:.2f} s")

    improved = (
        post_eval["pass_rate"] > pre_eval["pass_rate"]
        or post_eval["mean_score"] > pre_eval["mean_score"]
    )
    print(f"\nEFFECTIVENESS VERDICT: {'IMPROVED' if improved else 'NO IMPROVEMENT'}")

    if use_wandb:
        wandb.summary["pre_pass_rate"] = pre_eval["pass_rate"]
        wandb.summary["post_pass_rate"] = post_eval["pass_rate"]
        wandb.summary["pre_mean_score"] = pre_eval["mean_score"]
        wandb.summary["post_mean_score"] = post_eval["mean_score"]
        wandb.summary["pass_rate_delta"] = post_eval["pass_rate"] - pre_eval["pass_rate"]
        wandb.summary["final_loss"] = final_loss
        wandb.summary["improved"] = improved
        wandb.finish()

    # Local backup.
    import json

    summary_path = output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "problem_id": problem_id,
                "difficulty": difficulty,
                "pre_eval": pre_eval,
                "post_eval": post_eval,
                "improved": improved,
                "step_mean_rewards": step_means,
                "all_rewards": reward_history,
                "final_loss": final_loss,
                "ttt_runtime_s": run_elapsed,
                "total_runtime_s": total_elapsed,
            },
            f,
            indent=2,
        )
    print(f"Summary written to: {summary_path}")

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Required because evaluator/compute_score uses multiprocessing.
    main()
