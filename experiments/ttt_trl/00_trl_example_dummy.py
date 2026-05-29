"""
00_trl_example_dummy.py

Smoke test TRL SDPOTrainer with dummy dataset.
Verifies:
- SDPOTrainer can be instantiated
- 2 training steps can run
- loss values are finite
- checkpoint can be reloaded

No LCBv6 and no LoRA yet.
"""

from __future__ import annotations

import inspect
import math
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import tokenization_utils_base
from transformers.utils import hub as hub_utils
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer


def _filter_supported_kwargs(cls_or_fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by the target constructor/function."""
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


def dummy_reward_func(prompts, completions, **kwargs):
    """Reward = 1.0 if completion contains any digit, else 0.0."""
    rewards = []
    for comp in completions:
        text = _extract_text(comp)
        rewards.append(1.0 if any(c.isdigit() for c in text) else 0.0)
    return rewards


def build_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": "Solve 2+2."}],
                [{"role": "user", "content": "Solve 3+3."}],
                [{"role": "user", "content": "Solve 4+4."}],
                [{"role": "user", "content": "Solve 5+5."}],
            ],
            "privileged_context": [
                "Your earlier answer used the wrong format.",
                "Hint: think step by step.",
                "Hint: the answer is a single number.",
                "Hint: addition is commutative.",
            ],
        }
    )


def make_config() -> SDPOConfig:
    requested = {
        "output_dir": "outputs/00_dummy",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_generations": 4,
        "max_steps": 2,
        "learning_rate": 1e-5,
        "distillation_topk": 20,
        "full_logit_distillation": True,
        "distillation_alpha": 1.0,
        "include_environment_feedback": True,
        "sdpo_policy_loss_mode": "distillation_only",
        "bf16": False,
        "fp16": False,
        "report_to": "none",
    }
    supported = _filter_supported_kwargs(SDPOConfig, requested)
    return SDPOConfig(**supported)


def make_trainer(dataset: Dataset, config: SDPOConfig) -> SDPOTrainer:
    # Work around TRL 1.4.0 passing `dtype` (not `torch_dtype`) when model is provided as a string.
    # Loading the model ourselves avoids that path and is compatible with Qwen2.
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
    )
    # Some HF repos do not provide `additional_chat_templates/`; newer transformers may
    # raise on that missing folder. For this smoke test, treat it as optional.
    original_list_repo_templates = hub_utils.list_repo_templates

    def _safe_list_repo_templates(*args, **kwargs):
        try:
            return original_list_repo_templates(*args, **kwargs)
        except Exception:
            return []

    hub_utils.list_repo_templates = _safe_list_repo_templates
    tokenization_utils_base.list_repo_templates = _safe_list_repo_templates
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    requested = {
        "model": model,
        "reward_funcs": dummy_reward_func,
        "args": config,
        "train_dataset": dataset,
        "processing_class": tokenizer,
    }
    supported = _filter_supported_kwargs(SDPOTrainer, requested)
    return SDPOTrainer(**supported)


def assert_finite_losses(trainer: SDPOTrainer) -> None:
    loss_keys = ("loss", "train_loss")
    loss_values = []
    for log in getattr(trainer.state, "log_history", []):
        for key in loss_keys:
            if key in log:
                value = float(log[key])
                loss_values.append(value)
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite {key} detected: {value}")

    if not loss_values:
        raise RuntimeError("No loss values found in trainer.state.log_history")

    print(f"Collected finite losses: {loss_values}")


def main() -> None:
    os.environ.setdefault("WANDB_DISABLED", "true")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a CUDA GPU.")

    dataset = build_dataset()
    config = make_config()

    print("Initializing SDPOTrainer...")
    trainer = make_trainer(dataset=dataset, config=config)

    print("Starting training for 2 steps...")
    trainer.train()
    assert_finite_losses(trainer)

    final_dir = Path("outputs/00_dummy/final")
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))

    print("Reloading checkpoint with AutoModelForCausalLM...")
    _ = AutoModelForCausalLM.from_pretrained(str(final_dir))

    print("OK: 2 steps trained, finite loss observed, checkpoint saved and reloaded.")


if __name__ == "__main__":
    main()
