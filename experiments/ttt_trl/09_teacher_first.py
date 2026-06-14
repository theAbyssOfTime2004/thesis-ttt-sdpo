"""
09_teacher_first.py

Teacher-first, judge-filtered test-time SDPO (P0: code only).

Differs from 07 (student-first SDPOTrainer baseline):
- CUSTOM training loop (SDPOTrainer hard-codes student-first rollout).
- Trajectories to distill are TEACHER-generated then JUDGE-filtered
  (correct AND independent), not the student's own rollouts.
- Teacher & student SHARE weights (self-distillation):
    student = model(prompt_only)
    teacher = model(prompt + feedback + few-shot exemplars)
  Only the student forward carries gradient; teacher forward is stopgrad.

Spec: syn_teacher_first_impl_spec.md  (decisions in section 4 resolved:
  * reference_mode flag {best_in_batch,none}, default best_in_batch
  * KL reuse: trl SelfDistillationMixin._compute_divergence (reverse KL, top-k))
"""

from __future__ import annotations

import argparse
import difflib
import gc
import importlib
import json
import math
import os
import pathlib
import re
import statistics
import sys
import time
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import Dataset  # noqa: F401  (kept for parity / potential dataset use)
from peft import get_peft_model
from transformers import AutoModelForCausalLM, set_seed
from trl.experimental.sdpo.loss_utils import compute_topk_self_distillation_loss

# Ensure repo root is importable when this file is executed directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# Module 07 cannot be imported with normal syntax (name starts with a digit),
# so reuse its helpers via importlib instead of copy-pasting.
_m07 = importlib.import_module("experiments.ttt_trl.07_discovery_curve")

_prepare_tokenizer = _m07._prepare_tokenizer
_build_messages = _m07._build_messages
build_privileged_context = _m07.build_privileged_context
build_dynamic_feedback = _m07.build_dynamic_feedback
safe_evaluate_model = _m07.safe_evaluate_model
_build_lora_config = _m07._build_lora_config
_extract_text = _m07._extract_text
REPROMPT_TEMPLATES = _m07.REPROMPT_TEMPLATES

from experiments.ttt_trl.evaluator import evaluate_solution, load_lcbv6_split


# --------------------------------------------------------------------------- #
# Similarity (judge): normalized-code difflib ratio. This is a knob, not final.
# --------------------------------------------------------------------------- #
def _normalize_code(code: str) -> str:
    """Strip comments + collapse whitespace so similarity tracks structure, not formatting."""
    lines = []
    for raw in code.splitlines():
        line = raw.split("#", 1)[0]  # drop trailing/standalone comments (approx)
        line = line.strip()
        if line:
            lines.append(line)
    joined = "\n".join(lines)
    return re.sub(r"\s+", " ", joined).strip()


def similarity(a: str, b: str) -> float:
    na, nb = _normalize_code(a), _normalize_code(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


# --------------------------------------------------------------------------- #
# Teacher prompt assembly (reprompt-preset aware + few-shot exemplars).
# --------------------------------------------------------------------------- #
def _apply_reprompt_preset(preset: str, feedback_raw: str) -> tuple[str, str]:
    """
    Map a 07 reprompt-template preset onto the teacher prompt for the custom loop.

    Returns (formatted_feedback, trailing_instruction):
      - formatted_feedback: feedback_template applied to the raw feedback text.
      - trailing_instruction: the preset's reprompt_template with empty slots
        ({prompt}{solution}{feedback}) -> keeps the RQ1 instruction-framing
        dimension alive even though we don't use SDPOTrainer.
    """
    tpl = REPROMPT_TEMPLATES[preset]
    fb_tpl = tpl["feedback_template"]
    has_placeholder = "{feedback_raw}" in fb_tpl
    if feedback_raw or not has_placeholder:
        formatted_fb = fb_tpl.format(feedback_raw=feedback_raw or "").strip()
    else:
        formatted_fb = ""
    trailing = tpl["reprompt_template"].format(prompt="", solution="", feedback="").strip()
    return formatted_fb, trailing


def _build_fewshot_block(
    good_pool: list[dict],
    bad_pool: list[dict],
    option: str,
    max_fewshot: int,
) -> str:
    """good_pool/bad_pool items are dicts {"code": str, "score": float}."""
    blocks: list[str] = []
    goods = good_pool[:max_fewshot]
    if goods:
        blocks.append("Here are correct, independent example solutions:")
        for i, ex in enumerate(goods):
            blocks.append(f"Correct example {i + 1}:\n```python\n{ex['code']}\n```")

    if option == "good_bad":
        bads = bad_pool[:max_fewshot]
        if bads:
            blocks.append("Here are INCORRECT or copied attempts to avoid:")
            for i, ex in enumerate(bads):
                blocks.append(f"Bad example {i + 1} (do not imitate):\n```python\n{ex['code']}\n```")

    return "\n\n".join(blocks)


def _build_teacher_messages(
    question_content: str,
    feedback_text: str,
    good_pool: list[dict],
    bad_pool: list[dict],
    option: str,
    max_fewshot: int,
    reprompt_preset: str,
) -> list[dict[str, str]]:
    formatted_fb, trailing = _apply_reprompt_preset(reprompt_preset, feedback_text)
    fewshot_block = _build_fewshot_block(good_pool, bad_pool, option, max_fewshot)

    parts = [question_content]
    if formatted_fb:
        parts.append(formatted_fb)
    if fewshot_block:
        parts.append(fewshot_block)  # few-shot BEFORE the directive (spec 2a)
    if trailing:
        parts.append(trailing)
    teacher_user = "\n\n".join(p for p in parts if p)
    # _build_messages appends the standard "Respond with ONLY ... ```python ...```" directive.
    return _build_messages(teacher_user)


@torch.no_grad()
def teacher_generate(
    model,
    tokenizer,
    question_content: str,
    feedback_text: str,
    good_pool: list[dict],
    bad_pool: list[dict],
    option: str,
    n: int,
    temperature: float,
    max_new_tokens: int,
    max_fewshot: int,
    reprompt_preset: str,
    max_prompt_length: int,
    verbose: bool = False,
) -> tuple[list[str], str]:
    """Sample N teacher completions from prompt + feedback + few-shot exemplars."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    messages = _build_teacher_messages(
        question_content, feedback_text, good_pool, bad_pool, option, max_fewshot, reprompt_preset
    )
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]

    if verbose:
        print(f"[teacher] prompt tokens={prompt_tokens} vs max_prompt_length={max_prompt_length}"
              f"{'  [WARN: exceeds]' if prompt_tokens > max_prompt_length else ''}")
        print("[teacher] FULL PROMPT BELOW >>>>>>>>>>")
        print(prompt_text)
        print("[teacher] <<<<<<<<<< END PROMPT")

    completions: list[str] = []
    if n > 0:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            num_return_sequences=n,
            use_cache=True,
            pad_token_id=pad_id,
        )
        for i in range(out.shape[0]):
            completions.append(tokenizer.decode(out[i, prompt_tokens:], skip_special_tokens=True))

    if was_training:
        model.train()
    return completions, prompt_text


# --------------------------------------------------------------------------- #
# Judge / filter.
# --------------------------------------------------------------------------- #
def get_reference_text(
    mode: str,
    row: dict,
    batch_trajectories: list[tuple[str, float]],
) -> tuple[str | None, int | None]:
    """
    Dispatch the similarity reference (spec 4.1). Returns (reference_text, best_idx).

    - "best_in_batch" (code P0): proxy = highest-scoring trajectory in THIS batch.
      best_idx is returned so the proxy is never flagged as a copy of itself.
    - "ground_truth" (math P2): row's reference solution (empty for LCBv6 code rows).
    - "none": similarity disabled (None, None).

    `batch_trajectories` is a list of (code, score).
    """
    if mode == "none":
        return None, None
    if mode == "ground_truth":
        for key in ("ground_truth", "solution", "reference_solution", "canonical_solution", "answer"):
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                return val, None
        return "", None
    # best_in_batch
    if not batch_trajectories:
        return None, None
    best_idx = max(range(len(batch_trajectories)), key=lambda i: batch_trajectories[i][1])
    return batch_trajectories[best_idx][0], best_idx


def filter_trajectories(
    trajectories: list[str],
    row: dict,
    sim_threshold: float,
    reference_mode: str,
) -> tuple[list[dict], list[dict], dict]:
    """
    good := correct AND independent ; bad := copy of reference (or wrong in 'none' mode).

    Fallback (spec): if similarity removes every good trajectory but at least one
    correct (score>=1.0) trajectory exists, keep the best-scoring correct one so the
    good pool never collapses to empty purely due to the copy filter.
    """
    scored: list[tuple[str, float]] = []
    for t in trajectories:
        code = _extract_text(t)
        try:
            s = float(evaluate_solution(code, row, split="train")["score"])
        except Exception as exc:  # noqa: BLE001
            print(f"[filter] eval error: {exc}")
            s = 0.0
        scored.append((code, s))

    ref, best_idx = get_reference_text(reference_mode, row, scored)

    good: list[dict] = []
    bad: list[dict] = []
    sims: list[float] = []
    for i, (code, s) in enumerate(scored):
        if reference_mode == "none" or not ref:
            sim = 0.0
        elif i == best_idx:
            sim = 0.0  # the proxy reference is not a "copy" of itself
        else:
            sim = similarity(code, ref)
        sims.append(sim)

        if reference_mode == "none" or not ref:
            if s >= 1.0:
                good.append({"code": code, "score": s})
            elif s <= 0.0:
                bad.append({"code": code, "score": s})
        else:
            if s >= 1.0 and sim < sim_threshold:
                good.append({"code": code, "score": s})
            elif sim >= sim_threshold:
                bad.append({"code": code, "score": s})

    # Fallback: similarity wiped all good, but a correct trajectory exists -> keep best.
    fallback_used = False
    if not good:
        correct = [(c, s) for c, s in scored if s >= 1.0]
        if correct:
            best_code, best_score = max(correct, key=lambda cs: cs[1])
            good.append({"code": best_code, "score": best_score})
            bad = [b for b in bad if b["code"] != best_code]
            fallback_used = True

    stats = {
        "n_good": len(good),
        "n_bad": len(bad),
        "n_total": len(scored),
        "mean_sim": statistics.mean(sims) if sims else 0.0,
        "scores": [s for _, s in scored],
        "fallback_used": fallback_used,
    }
    return good, bad, stats


# --------------------------------------------------------------------------- #
# Core KL distillation step (token-aligned, top-k reverse KL, stopgrad teacher).
# --------------------------------------------------------------------------- #
def teacher_first_step(
    model,
    tokenizer,
    student_messages: list[dict[str, str]],
    teacher_messages: list[dict[str, str]],
    y_good_list: list[str],
    optimizer,
    kl_topk: int,
    kl_alpha: float,
    verbose: bool = False,
) -> float:
    """
    For each y_good, align its token positions on BOTH the student prompt
    (prompt-only) and the teacher prompt (prompt+feedback+few-shot), then distill
    teacher -> student over those positions with top-k (reverse) KL.

    The student/teacher prefixes have DIFFERENT lengths; we append the SAME
    completion token ids to each prefix and slice logits at [prefix-1 : prefix-1+L]
    (the positions that PREDICT the completion tokens). This is the off-by-one
    danger zone (spec 2c/7) -> asserts below guard it.
    """
    device = next(model.parameters()).device

    student_prefix_ids = tokenizer.apply_chat_template(
        student_messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    teacher_prefix_ids = tokenizer.apply_chat_template(
        teacher_messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    s_prefix = student_prefix_ids.shape[1]
    t_prefix = teacher_prefix_ids.shape[1]
    if verbose:
        print(f"[kl] student_prefix_len={s_prefix} teacher_prefix_len={t_prefix}")

    optimizer.zero_grad()
    n = len(y_good_list)
    loss_sum = 0.0
    contributing = 0

    for y in y_good_list:
        comp_ids = tokenizer(y, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        length = comp_ids.shape[1]
        if length == 0:
            continue

        student_input = torch.cat([student_prefix_ids, comp_ids], dim=1)
        teacher_input = torch.cat([teacher_prefix_ids, comp_ids], dim=1)

        # Alignment guard (spec 7): the appended y_good tokens must be identical on
        # both sides (same comp_ids), so prefix-len slicing aligns the same targets.
        assert torch.equal(
            student_input[:, s_prefix:], teacher_input[:, t_prefix:]
        ), "y_good tokens differ across student/teacher sides"

        # Student forward: gradient ON.
        student_logits = model(input_ids=student_input, use_cache=False).logits
        s_slice = student_logits[:, s_prefix - 1 : s_prefix - 1 + length, :]

        # Teacher forward: stopgrad (same shared weights, no grad).
        with torch.no_grad():
            teacher_logits = model(input_ids=teacher_input, use_cache=False).logits
            t_slice = teacher_logits[:, t_prefix - 1 : t_prefix - 1 + length, :]

        # Both slices must cover exactly the L y_good token positions.
        assert s_slice.shape[1] == length, f"student slice {s_slice.shape[1]} != {length}"
        assert t_slice.shape[1] == length, f"teacher slice {t_slice.shape[1]} != {length}"

        # Top-k reverse-KL via the SAME loss fn SDPOTrainer (07) uses -> teacher-first
        # arm stays apples-to-apples with the student-first baseline. Takes RAW logits,
        # does top-k extraction + renorm + tail bucket internally, returns per-token loss.
        # alpha=1.0 -> reverse KL = KL(teacher||student) (lasgroup LCBv6 config).
        per_token = compute_topk_self_distillation_loss(
            s_slice, t_slice,
            distillation_topk=kl_topk,
            distillation_alpha=kl_alpha,
            distillation_add_tail=True,
        )  # [1, L]
        item_loss = per_token.mean()

        if verbose and contributing == 0:
            kl_val = item_loss.item()
            print(f"[kl-sanity] L={length} s_slice={tuple(s_slice.shape)} "
                  f"t_slice={tuple(t_slice.shape)} topk={k} KL(token-mean)={kl_val:.6f}")
            assert kl_val > 0, f"KL should be > 0 on step 1 (got {kl_val}); check alignment/sign"

        (item_loss / n).backward()
        loss_sum += item_loss.item()
        contributing += 1

        del student_logits, teacher_logits, s_slice, t_slice

    if contributing == 0:
        optimizer.zero_grad()
        return 0.0

    optimizer.step()
    optimizer.zero_grad()
    return loss_sum / contributing


# --------------------------------------------------------------------------- #
# Pool maintenance.
# --------------------------------------------------------------------------- #
def _update_pool(pool: list[dict], new_items: list[dict], cap: int) -> list[dict]:
    """Prepend new items (freshest first), dedup by code, cap by highest score."""
    seen = set()
    merged = new_items + pool
    out = []
    for item in merged:
        key = item["code"]
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return out[:cap]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-first judge-filtered TTT-SDPO (P0: code).")
    parser.add_argument("--problem_index", type=int, default=23, help="LCBv6 index (P0 frontier=23).")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--teacher_n", type=int, default=10, help="Teacher samples per step.")
    parser.add_argument("--teacher_temperature", type=float, default=1.0)
    parser.add_argument("--sim_threshold", type=float, default=0.9)
    parser.add_argument("--fewshot_option", type=str, default="good_only",
                        choices=["good_only", "good_bad"])
    parser.add_argument("--max_fewshot", type=int, default=3)
    parser.add_argument("--reference_mode", type=str, default="best_in_batch",
                        choices=["best_in_batch", "ground_truth", "none"],
                        help="Similarity reference for the copy-filter (spec 4.1): "
                             "best_in_batch=code P0, ground_truth=math P2, none=ablation.")
    parser.add_argument("--kl_topk", type=int, default=20)
    parser.add_argument("--kl_alpha", type=float, default=1.0, help="1.0 = reverse KL (lasgroup LCBv6).")
    parser.add_argument("--reprompt_template", type=str, default="T2_standard",
                        choices=sorted(REPROMPT_TEMPLATES),
                        help="Reprompt-template preset used to frame teacher feedback/instruction.")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--eval_samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=4096,
                        help="Logging-only token budget reference for the teacher prompt.")
    parser.add_argument("--pool_cap", type=int, default=8)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs/09_teacher_first")
    parser.add_argument("--wandb_project", type=str, default="ttt-sdpo-thesis")
    parser.add_argument("--no_wandb", action="store_true")
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
    print(f"Seed: {args.seed} | reference_mode={args.reference_mode} | fewshot={args.fewshot_option}")

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=f"teacherfirst-{args.model_name.split('/')[-1]}-idx{args.problem_index}"
                 f"-{args.max_steps}step-{args.fewshot_option}-{args.reprompt_template}",
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
    print(f"[row] keys = {list(row.keys())}")

    tokenizer = _prepare_tokenizer(args.model_name, thinking=args.thinking)
    lora_config = _build_lora_config(args.lora_r)

    total_start = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )

    # ----- PRE-eval (student-only) -----
    print(f"\n[PRE-eval] sampling {args.eval_samples} student solutions ...")
    pre_eval = safe_evaluate_model(model, tokenizer, row, args.eval_samples, args.max_new_tokens, label="PRE")
    print(f"[PRE-eval] pass_rate={pre_eval['pass_rate']:.3f} mean={pre_eval['mean_score']:.3f} "
          f"max={pre_eval['max_score']:.3f} greedy={pre_eval['greedy_score']:.3f}")

    base_hint = build_privileged_context(row)
    student_messages = _build_messages(question_content)

    good_pool: list[dict] = []
    bad_pool: list[dict] = []
    step_records: list[dict] = []
    step_mean_rewards: list[float] = []

    print(f"\n[TTT] teacher-first on {problem_id} for {args.max_steps} steps ...")
    run_start = time.time()

    for step in range(args.max_steps):
        verbose = step == 0

        # Dynamic feedback from the student's CURRENT greedy attempt (reuse 07).
        greedy_eval = safe_evaluate_model(model, tokenizer, row, 0, args.max_new_tokens, label=f"STEP{step}")
        dyn = build_dynamic_feedback(greedy_eval.get("greedy_code", ""), row)
        feedback_text = "\n\n".join(p for p in [base_hint, dyn] if p).strip()

        trajectories, teacher_prompt = teacher_generate(
            model, tokenizer, question_content, feedback_text,
            good_pool, bad_pool, args.fewshot_option, args.teacher_n,
            args.teacher_temperature, args.max_new_tokens, args.max_fewshot,
            args.reprompt_template, args.max_prompt_length, verbose=verbose,
        )

        good, bad, stats = filter_trajectories(
            trajectories, row, args.sim_threshold, args.reference_mode,
        )
        good_pool = _update_pool(good_pool, good, args.pool_cap)
        bad_pool = _update_pool(bad_pool, bad, args.pool_cap)

        if good:
            teacher_messages = _build_teacher_messages(
                question_content, feedback_text, good_pool, bad_pool,
                args.fewshot_option, args.max_fewshot, args.reprompt_template,
            )
            model.train()
            loss = teacher_first_step(
                model, tokenizer, student_messages, teacher_messages,
                [g["code"] for g in good], optimizer, args.kl_topk, args.kl_alpha,
                verbose=verbose,
            )
        else:
            loss = 0.0
            print(f"[step {step + 1}] no good trajectory (cold-start/collapse signal)")

        batch_mean_reward = statistics.mean(stats["scores"]) if stats["scores"] else 0.0
        step_mean_rewards.append(batch_mean_reward)
        rec = {
            "step": step + 1,
            "n_good": stats["n_good"],
            "n_bad": stats["n_bad"],
            "n_total": stats["n_total"],
            "mean_sim": stats["mean_sim"],
            "loss": loss,
            "good_pool_size": len(good_pool),
            "batch_mean_reward": batch_mean_reward,
            "batch_max_reward": max(stats["scores"]) if stats["scores"] else 0.0,
        }
        step_records.append(rec)
        print(f"[step {step + 1}] n_good={rec['n_good']} n_bad={rec['n_bad']} "
              f"n_total={rec['n_total']} mean_sim={rec['mean_sim']:.3f} loss={loss:.4f} "
              f"good_pool={rec['good_pool_size']} batch_mean_r={batch_mean_reward:.3f}")
        if use_wandb:
            wandb.log({f"curve/{k}": v for k, v in rec.items()})

    run_elapsed = time.time() - run_start

    # ----- POST-eval (student-only) -----
    print(f"\n[POST-eval] sampling {args.eval_samples} student solutions ...")
    post_eval = safe_evaluate_model(model, tokenizer, row, args.eval_samples, args.max_new_tokens, label="POST")
    print(f"[POST-eval] pass_rate={post_eval['pass_rate']:.3f} mean={post_eval['mean_score']:.3f} "
          f"max={post_eval['max_score']:.3f} greedy={post_eval['greedy_score']:.3f}")

    peak_vram = torch.cuda.max_memory_allocated(0)
    total_elapsed = time.time() - total_start

    print("\n=== DISCOVERY CURVE (batch mean reward per step) ===")
    print(" -> ".join(f"{m:.2f}" for m in step_mean_rewards))

    print("\n=== EFFECTIVENESS (pre vs post TTT, student-only) ===")
    print(f"pass_rate : {pre_eval['pass_rate']:.3f} -> {post_eval['pass_rate']:.3f} "
          f"(delta {post_eval['pass_rate'] - pre_eval['pass_rate']:+.3f})")
    print(f"mean_score: {pre_eval['mean_score']:.3f} -> {post_eval['mean_score']:.3f} "
          f"(delta {post_eval['mean_score'] - pre_eval['mean_score']:+.3f})")
    print(f"greedy    : {pre_eval['greedy_score']:.3f} -> {post_eval['greedy_score']:.3f}")
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
        wandb.summary["pass_rate_delta"] = post_eval["pass_rate"] - pre_eval["pass_rate"]
        wandb.summary["improved"] = improved
        wandb.finish()

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
                "step_records": step_records,
                "step_mean_rewards": step_mean_rewards,
                "ttt_runtime_s": run_elapsed,
                "total_runtime_s": total_elapsed,
            },
            f,
            indent=2,
        )
    print(f"Summary written to: {summary_path}")

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
