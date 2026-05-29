"""
01_trl_lora_dummy.py

Smoke test TRL SDPOTrainer with LoRA on dummy dataset.
Verifies:
- SDPOTrainer can be instantiated with peft_config
- 2 training steps can run
- loss values are finite
- LoRA adapter files are saved (not full model checkpoint)
- adapter can be reloaded with PeftModel.from_pretrained
"""

from __future__ import annotations

import inspect
import math
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
import peft.import_utils as peft_import_utils
import peft.tuners.lora.model as peft_lora_model
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
        "output_dir": "outputs/01_lora_dummy",
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


def make_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )


def make_trainer(dataset: Dataset, config: SDPOConfig) -> SDPOTrainer:
    # Local env may have bitsandbytes/triton mismatch; force PEFT to use plain torch LoRA modules.
    peft_import_utils.is_bnb_available = lambda: False
    peft_import_utils.is_bnb_4bit_available = lambda: False
    peft_lora_model.is_bnb_available = lambda: False
    peft_lora_model.is_bnb_4bit_available = lambda: False

    # Work around TRL 1.4.0 passing `dtype` (not `torch_dtype`) when model is provided as a string.
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

    lora_config = make_lora_config()
    requested = {
        "model": model,
        "reward_funcs": dummy_reward_func,
        "args": config,
        "train_dataset": dataset,
        "processing_class": tokenizer,
        "peft_config": lora_config,
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


def verify_lora_artifacts(final_dir: Path) -> None:
    files = sorted(p for p in final_dir.iterdir() if p.is_file())
    if not files:
        raise RuntimeError(f"No files found in {final_dir}")

    print("\nSaved files:")
    adapter_files = []
    for path in files:
        size = path.stat().st_size
        print(f"- {path.name}: {size} bytes")
        if "adapter" in path.name.lower():
            adapter_files.append(path)

    if not adapter_files:
        raise RuntimeError("No adapter file found (expected at least one filename containing 'adapter').")

    adapter_total = sum(p.stat().st_size for p in adapter_files)
    print(f"Total adapter size: {adapter_total} bytes")
    if adapter_total >= 50 * 1024 * 1024:
        raise RuntimeError(f"Adapter size too large: {adapter_total} bytes (expected < 50 MB).")


def verify_adapter_reload(final_dir: Path) -> None:
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
    )
    _ = PeftModel.from_pretrained(base_model, str(final_dir))
    print("Adapter reload check: OK")


def main() -> None:
    os.environ.setdefault("WANDB_DISABLED", "true")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a CUDA GPU.")

    dataset = build_dataset()
    config = make_config()

    print("Initializing SDPOTrainer with LoRA...")
    trainer = make_trainer(dataset=dataset, config=config)

    print("Starting training for 2 steps...")
    trainer.train()
    assert_finite_losses(trainer)

    final_dir = Path("outputs/01_lora_dummy/final")
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))

    verify_lora_artifacts(final_dir)
    verify_adapter_reload(final_dir)

    print("OK: LoRA SDPO dummy run completed, finite loss observed, adapter saved and reloaded.")


if __name__ == "__main__":
    main()
