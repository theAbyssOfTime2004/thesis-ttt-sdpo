"""
06_feedback_path_test.py

Phase 1: open the SDPO feedback path (Path B).
- Same as 05 kickoff (Qwen2.5-1.5B + LoRA, multi-problem, per-problem TTT)
- include_environment_feedback=True + environment_feedback_only_without_solution=True
- privileged_context = public test cases (privileged hint the student does NOT get)
- Goal: problems with NO perfect rollout (abc387_b, abc387_f) should now get
  loss != 0 via the feedback path, not just abc387_a (which learns via Path A).
"""

from __future__ import annotations

import argparse
import gc
import inspect
import math
import os
import pathlib
import shutil
import statistics
import sys
import time
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import Dataset
from huggingface_hub import HfApi
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
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


def _prepare_tokenizer(model_name: str):
    # Some repos do not expose `additional_chat_templates/`; treat it as optional.
    original_list_repo_templates = hub_utils.list_repo_templates

    def _safe_list_repo_templates(*args, **kwargs):
        try:
            return original_list_repo_templates(*args, **kwargs)
        except Exception:
            return []

    hub_utils.list_repo_templates = _safe_list_repo_templates
    tokenization_utils_base.list_repo_templates = _safe_list_repo_templates
    return AutoTokenizer.from_pretrained(model_name)


def _build_messages(question_content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": question_content}]


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
        "temperature": 1.0,
        "max_completion_length": max_new_tokens,
        "generation_kwargs": {"max_new_tokens": max_new_tokens, "max_time": 30.0},
        "gradient_checkpointing": True,
        "distillation_topk": 20,
        "full_logit_distillation": True,
        "distillation_alpha": 1.0,
        "sdpo_policy_loss_mode": "distillation_only",
        # Phase 1: open Path B (feedback). Use feedback only when no perfect
        # rollout exists, so easy problems still learn via solution demo (Path A)
        # and hard problems learn via teacher feedback (Path B).
        "include_environment_feedback": True,
        "environment_feedback_only_without_solution": True,
        # L4 (Ada) supports bf16 natively: stable, no GradScaler needed.
        "bf16": True,
        "fp16": False,
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


def _copy_last_adapter(last_problem_dir: pathlib.Path, final_dir: pathlib.Path) -> tuple[pathlib.Path, int]:
    adapter_files = [p for p in last_problem_dir.iterdir() if p.is_file() and "adapter" in p.name.lower()]
    if not adapter_files:
        raise RuntimeError(f"No adapter files found in {last_problem_dir}")

    final_dir.mkdir(parents=True, exist_ok=True)
    for p in last_problem_dir.iterdir():
        if p.is_file():
            shutil.copy2(p, final_dir / p.name)

    total_adapter_size = sum(p.stat().st_size for p in adapter_files)
    largest = sorted(adapter_files, key=lambda p: p.stat().st_size, reverse=True)[0]
    return final_dir / largest.name, total_adapter_size


def _print_aggregate_table(rows: list[dict[str, Any]]) -> None:
    print("\n=== AGGREGATE SUMMARY ===")
    print("problem_id | n_steps | mean_reward | max_reward | variance | final_loss | status")
    for row in rows:
        if row["status"] == "ok":
            print(
                f"{row['problem_id']} | {row['n_steps']} | "
                f"{row['mean_reward']:.6f} | {row['max_reward']:.6f} | "
                f"{row['variance']:.6f} | {row['final_loss']:.6f} | ok"
            )
        else:
            print(f"{row['problem_id']} | 0 | - | - | - | - | error: {row['error']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab kickoff for Qwen1.5B SDPO on LCBv6.")
    parser.add_argument("--num_problems", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=2, help="Steps per problem")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="outputs/06_feedback_path")
    parser.add_argument("--hf_repo", type=str, default="MaiDang2004/ttt-sdpo-qwen15b-poc")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="ttt-sdpo-thesis")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This script requires a GPU (expected Colab L4).")

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {device_name}")
    print(f"Total VRAM: {total_vram_gb:.2f} GB")

    use_wandb = not args.no_wandb
    report_to = "wandb" if use_wandb else "none"
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=f"feedback-{args.model_name.split('/')[-1]}-{args.num_problems}prob",
            config=vars(args),
        )

    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    lcb = load_lcbv6_split()
    selected = [lcb[i] for i in range(min(args.num_problems, len(lcb)))]
    print(f"Loaded {len(selected)} problems (deterministic indices 0..{len(selected)-1}).")

    tokenizer = _prepare_tokenizer(args.model_name)
    lora_config = _build_lora_config(args.lora_r)

    aggregate: list[dict[str, Any]] = []
    last_success_problem_dir: pathlib.Path | None = None
    total_start = time.time()

    for idx, row in enumerate(selected):
        problem_id = row.get("question_id", row.get("problem_id", f"idx_{idx}"))
        difficulty = row.get("difficulty", "unknown")
        question_content = str(row.get("question_content", ""))

        print(f"\n=== PROBLEM {idx+1}/{len(selected)}: {problem_id} ({difficulty}) ===")

        if not question_content.strip():
            err = "empty question_content"
            print(f"Skip: {err}")
            aggregate.append({"problem_id": problem_id, "status": "error", "error": err})
            continue

        model = None
        trainer = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)

            # Per-problem TTT: fresh model each problem.
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            )

            messages = _build_messages(question_content)
            rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            print(f"Prompt preview: {rendered[:180].replace(chr(10), ' ')}")

            privileged_context = build_privileged_context(row)
            print(f"Privileged context ({len(privileged_context)} chars): "
                  f"{privileged_context[:160].replace(chr(10), ' ')}")
            train_dataset = Dataset.from_dict(
                {
                    "prompt": [messages],
                    "privileged_context": [privileged_context],
                }
            )

            problem_dir = output_root / f"problem_{idx:02d}_{problem_id}"
            config = _build_sdpo_config(
                problem_output_dir=problem_dir,
                num_generations=args.num_generations,
                max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
                report_to=report_to,
            )

            reward_history: list[list[float]] = []
            flat_reward_warning = False

            def reward_fn(completions, prompts=None, **kwargs):
                rewards = []
                for completion in completions:
                    code = _extract_text(completion)
                    try:
                        out = evaluate_solution(code, row, split="train")
                        rewards.append(float(out["score"]))
                    except Exception as exc:
                        print(f"Reward eval error: {exc}")
                        rewards.append(0.0)
                nonlocal flat_reward_warning
                if rewards and all(r == 0.0 for r in rewards):
                    flat_reward_warning = True
                reward_history.append(rewards)
                print(f"Reward batch ({len(rewards)}): {rewards}")
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

            run_start = time.time()
            trainer.train()
            run_elapsed = time.time() - run_start

            flat_rewards = _flatten_rewards(reward_history)
            if not flat_rewards:
                raise RuntimeError("No rewards recorded from reward function.")

            final_loss = _last_finite_loss(trainer)

            # Save per-problem adapter.
            trainer.save_model(str(problem_dir))
            last_success_problem_dir = problem_dir

            peak_vram = torch.cuda.max_memory_allocated(0)
            reward_min = min(flat_rewards)
            reward_max = max(flat_rewards)
            reward_mean = statistics.mean(flat_rewards)
            reward_var = statistics.pvariance(flat_rewards) if len(flat_rewards) > 1 else 0.0

            print(f"All rewards by step: {reward_history}")
            print(
                f"Reward stats: min={reward_min:.6f}, max={reward_max:.6f}, "
                f"mean={reward_mean:.6f}, variance={reward_var:.6f}"
            )
            print(f"Flat reward warning (all-zero any step): {flat_reward_warning}")
            print(f"Final loss: {final_loss:.6f}")
            print(f"Peak VRAM allocated: {peak_vram / (1024**3):.2f} GB")
            print(f"Problem runtime: {run_elapsed:.2f} s")

            if use_wandb:
                wandb.log(
                    {
                        "problem_idx": idx,
                        "problem/mean_reward": reward_mean,
                        "problem/max_reward": reward_max,
                        "problem/variance": reward_var,
                        "problem/final_loss": final_loss,
                        "problem/peak_vram_gb": peak_vram / (1024**3),
                        "problem/runtime_s": run_elapsed,
                    }
                )

            aggregate.append(
                {
                    "problem_id": problem_id,
                    "status": "ok",
                    "difficulty": difficulty,
                    "n_steps": args.max_steps,
                    "all_rewards": reward_history,
                    "min_reward": reward_min,
                    "max_reward": reward_max,
                    "mean_reward": reward_mean,
                    "variance": reward_var,
                    "flat_reward_warning": flat_reward_warning,
                    "final_loss": final_loss,
                    "peak_vram_bytes": peak_vram,
                }
            )
        except Exception as exc:
            print(f"Problem {problem_id} failed: {exc}")
            aggregate.append(
                {
                    "problem_id": problem_id,
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue
        finally:
            # Free VRAM before next per-problem run to avoid accumulation/OOM.
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()

    _print_aggregate_table(aggregate)

    ok_rows = [r for r in aggregate if r.get("status") == "ok"]
    solvable_count = sum(1 for r in ok_rows if r["max_reward"] > 0.0)
    useful_signal_count = sum(1 for r in ok_rows if r["variance"] > 0.0)
    total_elapsed = time.time() - total_start

    print(f"\nSolvable count (any reward > 0): {solvable_count}/{len(ok_rows)}")
    print(f"Useful SDPO signal count (variance > 0): {useful_signal_count}/{len(ok_rows)}")
    print(f"Total runtime: {total_elapsed:.2f} s")

    # Local backup of all per-problem results (independent of W&B).
    import json

    summary_path = output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "solvable_count": solvable_count,
                "useful_signal_count": useful_signal_count,
                "n_ok": len(ok_rows),
                "total_runtime_s": total_elapsed,
                "problems": aggregate,
            },
            f,
            indent=2,
        )
    print(f"Summary written to: {summary_path}")

    if use_wandb:
        wandb.summary["solvable_count"] = solvable_count
        wandb.summary["useful_signal_count"] = useful_signal_count
        wandb.summary["n_ok"] = len(ok_rows)
        wandb.finish()

    if last_success_problem_dir is None:
        print("No successful problem run; skipping final adapter export and hub push.")
        return

    final_dir = output_root / "final"
    adapter_main, adapter_total = _copy_last_adapter(last_success_problem_dir, final_dir)
    print(f"Final adapter copied from last successful problem to: {final_dir}")
    print(f"Main adapter file: {adapter_main}")
    print(f"Total adapter size bytes: {adapter_total}")

    if args.push_to_hub:
        api = HfApi()
        api.create_repo(repo_id=args.hf_repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=args.hf_repo,
            repo_type="model",
            folder_path=str(final_dir),
            path_in_repo=".",
        )
        print(f"Pushed adapter folder to Hugging Face Hub: {args.hf_repo}")


if __name__ == "__main__":
    # Required because evaluator/compute_score uses multiprocessing.
    main()
