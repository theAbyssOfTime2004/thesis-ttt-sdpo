"""
04_lcbv6_single_problem.py

End-to-end POC:
- Load 1 real LCBv6 problem
- Run 1 SDPO step with Qwen2.5-0.5B + LoRA
- Use evaluator.evaluate_solution as reward function
"""

from __future__ import annotations

import inspect
import math
import os
import pathlib
import statistics
import sys
import time
from typing import Any

import peft.import_utils as peft_import_utils
import peft.tuners.lora.model as peft_lora_model
import torch
from datasets import Dataset
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
    # Some repos don't expose `additional_chat_templates/`; treat it as optional.
    original_list_repo_templates = hub_utils.list_repo_templates

    def _safe_list_repo_templates(*args, **kwargs):
        try:
            return original_list_repo_templates(*args, **kwargs)
        except Exception:
            return []

    hub_utils.list_repo_templates = _safe_list_repo_templates
    tokenization_utils_base.list_repo_templates = _safe_list_repo_templates
    return AutoTokenizer.from_pretrained(model_name)


def _build_prompt(tokenizer, question_content: str) -> tuple[list[dict[str, str]], str]:
    messages = [{"role": "user", "content": question_content}]
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    return messages, rendered


def _make_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )


def _make_config() -> SDPOConfig:
    requested = {
        "output_dir": "outputs/04_single_problem",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,  # 1 * 4 -> generation_batch_size=4 (divisible by num_generations=4)
        "num_generations": 4,
        "max_steps": 1,
        "learning_rate": 1e-5,
        "temperature": 1.0,
        "max_completion_length": 512,
        "generation_kwargs": {"max_new_tokens": 512, "max_time": 20.0},
        "distillation_topk": 20,
        "full_logit_distillation": True,
        "distillation_alpha": 1.0,
        "sdpo_policy_loss_mode": "distillation_only",
        "bf16": False,
        "fp16": False,
        "report_to": "none",
    }
    supported = _filter_supported_kwargs(SDPOConfig, requested)
    return SDPOConfig(**supported)


def _assert_finite_loss(trainer: SDPOTrainer) -> float:
    loss_keys = ("loss", "train_loss")
    loss_values = []
    for log in getattr(trainer.state, "log_history", []):
        for key in loss_keys:
            if key in log:
                value = float(log[key])
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite {key} detected: {value}")
                loss_values.append(value)
    if not loss_values:
        raise RuntimeError("No loss values found in trainer.state.log_history")
    return loss_values[-1]


def _save_and_measure_adapter(trainer: SDPOTrainer, out_dir: pathlib.Path) -> tuple[pathlib.Path, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))

    adapter_files = [p for p in out_dir.iterdir() if p.is_file() and "adapter" in p.name.lower()]
    if not adapter_files:
        raise RuntimeError(f"No adapter files found in {out_dir}")

    total_adapter_bytes = sum(p.stat().st_size for p in adapter_files)
    # Use the main adapter file path for concise summary.
    adapter_main = sorted(adapter_files, key=lambda p: p.stat().st_size, reverse=True)[0]
    return adapter_main, total_adapter_bytes


def main() -> None:
    os.environ.setdefault("WANDB_DISABLED", "true")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a CUDA GPU.")

    start_total = time.time()
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # Load one real LCBv6 problem.
    lcb = load_lcbv6_split()
    raw_row = lcb[0]
    problem_id = raw_row.get("question_id", raw_row.get("problem_id", "unknown"))
    question_content = str(raw_row.get("question_content", ""))
    if not question_content.strip():
        raise RuntimeError("Selected problem has empty question_content")

    # Build prompt via chat template (single-turn for Week 1).
    tokenizer = _prepare_tokenizer(model_name)
    messages, rendered_prompt = _build_prompt(tokenizer, question_content)
    print(f"Loaded problem_id={problem_id}")
    print(f"Prompt preview: {rendered_prompt[:200].replace(chr(10), ' ')}")

    # Build a one-row dataset for SDPOTrainer.
    train_dataset = Dataset.from_dict(
        {
            "prompt": [messages],
            "privileged_context": [""],
        }
    )

    # Local env may have bitsandbytes/triton mismatch; force plain torch LoRA path.
    peft_import_utils.is_bnb_available = lambda: False
    peft_import_utils.is_bnb_4bit_available = lambda: False
    peft_lora_model.is_bnb_available = lambda: False
    peft_lora_model.is_bnb_4bit_available = lambda: False

    config = _make_config()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    lora_config = _make_lora_config()

    reward_trace: dict[str, Any] = {"history": [], "last": []}

    def reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        for completion in completions:
            code = _extract_text(completion)
            try:
                result = evaluate_solution(code, raw_row, split="train")
                rewards.append(float(result["score"]))
            except Exception as exc:
                # Keep training resilient to per-completion evaluator failures/timeouts.
                print(f"Reward eval error: {exc}")
                rewards.append(0.0)
        reward_trace["last"] = rewards
        reward_trace["history"].append(rewards)
        print(f"Reward batch ({len(rewards)}): {rewards}")
        return rewards

    trainer_requested = {
        "model": model,
        "reward_funcs": reward_fn,
        "args": config,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
        "peft_config": lora_config,
    }
    trainer_supported = _filter_supported_kwargs(SDPOTrainer, trainer_requested)
    trainer = SDPOTrainer(**trainer_supported)

    train_start = time.time()
    trainer.train()
    train_elapsed = time.time() - train_start

    latest_rewards = reward_trace["last"] if reward_trace["last"] else []
    assert len(latest_rewards) >= 1, "No rewards captured from reward function"
    final_loss = _assert_finite_loss(trainer)

    out_dir = pathlib.Path("outputs/04_single_problem/final")
    adapter_path, adapter_size = _save_and_measure_adapter(trainer, out_dir)
    total_elapsed = time.time() - start_total

    # Reward statistics: log-only, do not fail on flat rewards.
    reward_min = min(latest_rewards)
    reward_max = max(latest_rewards)
    reward_mean = statistics.mean(latest_rewards)
    reward_var = statistics.pvariance(latest_rewards) if len(latest_rewards) > 1 else 0.0

    print("\n=== SUMMARY ===")
    print(f"Final loss: {final_loss}")
    print(f"All rewards ({len(latest_rewards)}): {latest_rewards}")
    print(
        "Reward stats: "
        f"min={reward_min:.6f}, max={reward_max:.6f}, "
        f"mean={reward_mean:.6f}, variance={reward_var:.6f}"
    )
    print(f"Adapter saved: {adapter_path}")
    print(f"Adapter size bytes: {adapter_size}")
    print(f"Train runtime sec: {train_elapsed:.2f}")
    print(f"Total runtime sec: {total_elapsed:.2f}")


if __name__ == "__main__":
    # Required for multiprocessing in compute_score path.
    main()
