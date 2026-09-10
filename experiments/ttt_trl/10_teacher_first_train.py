"""
10_teacher_first_train.py

Teacher-first TRAIN-TIME SDPO, code + math (spec: SPEC_10_teacher_first_train.md).

Difference from 09 (test-time):
- 09 resets optimizer/LoRA per problem and evaluates PRE/POST on the SAME
  problem (test-time discovery regime).
- This script keeps ONE LoRA + ONE optimizer alive across every training
  problem, and evaluates PRE/POST once each on a HELD-OUT set the model never
  trained on (generalization regime, the paper's main setting).

Everything else is imported from 09/07 rather than reimplemented: teacher
generation, judge filtering, dynamic feedback, evaluation, LoRA setup, and the
KL step itself (which uses the same trl loss fn SDPOTrainer uses).

Two additions this script owns (spec 2b/2c/2d):
- Trust-region teacher regularization (paper 2.3): the distillation target is
  the log-prob interpolation (1-a)*log q_theta_ref + a*log q_theta, where BOTH
  terms are TEACHERS (same feedback context), differing only in weights.
  Under LoRA, q_theta_ref = same context with the adapter DISABLED, so there is
  no second weight copy -- only one extra no-grad forward.
- Skip accounting per arm (2d): a problem can produce no gradient because the
  student already solved it, or because the teacher produced no valid target.
  The latter is a TEACHER-FIRST-ONLY drop path, so the two arms can end up
  trained on different amounts of data -- that is a confound, not just a
  reporting detail. Counted here, per epoch, and written to summary.json.

Loss is pure distillation (verified against verl: dp_actor.py takes an
if/else, and compute_self_distillation_loss has no `advantages` parameter --
adv_estimator=grpo only disables the critic at the pipeline level).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import json
import os
import pathlib
import statistics
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed,
)
from trl.experimental.sdpo.loss_utils import compute_topk_self_distillation_loss

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# 07/09 start with a digit -> import via importlib, same pattern 09 uses for 07.
_m07 = importlib.import_module("experiments.ttt_trl.07_discovery_curve")
_m09 = importlib.import_module("experiments.ttt_trl.09_teacher_first")

_prepare_tokenizer = _m07._prepare_tokenizer
_build_messages = _m07._build_messages
build_dynamic_feedback = _m07.build_dynamic_feedback
safe_evaluate_model = _m07.safe_evaluate_model
_apply_lora = _m07._apply_lora
REPROMPT_TEMPLATES = _m07.REPROMPT_TEMPLATES

teacher_generate = _m09.teacher_generate
filter_trajectories = _m09.filter_trajectories
teacher_first_step = _m09.teacher_first_step
_build_teacher_messages = _m09._build_teacher_messages
_update_pool = _m09._update_pool
_build_provider_chain = _m09._build_provider_chain
_provider_has_key = _m09._provider_has_key

from experiments.ttt_trl.domains import Domain, get_domain
from experiments.ttt_trl.evaluator import load_aime_split


# --------------------------------------------------------------------------- #
# Generation speed. Measured 2026-09-04: generation is ~93% of training
# wall-clock (teacher_generate 73% + greedy eval 20%; verification only 7%), and
# one batched teacher_generate call took 107s against a ~8s memory-bandwidth
# floor -- i.e. ~13x off, all of it HF `generate` overhead, not physics.
# Both helpers below are applied from THIS script only; 07/09 are left untouched
# so the thesis runs stay reproducible.
# --------------------------------------------------------------------------- #
class ClosingFenceCriteria(StoppingCriteria):
    """
    Stop a sequence once its SECOND ``` fence has been emitted, i.e. the code
    block has been closed.

    Do NOT do this with `stop_strings=["\\n```"]`: the rendered prompt ends in
    newlines, so the very first ``` the model emits already makes the sequence
    end with "\\n```" and generation stops after 3 characters. That regression
    was observed on 2026-09-04 (every completion became a bare fence, every
    score went to 0.00, and the run got 60x "faster" by generating nothing).

    Counting is incremental: only the newest token is decoded each step and
    appended to a short tail buffer, so this is O(1) per sequence per step
    rather than re-decoding the whole growing sequence.
    """

    def __init__(self, tokenizer, n_sequences: int, fence: str = "```"):
        self.tokenizer = tokenizer
        self.fence = fence
        self.counts = [0] * n_sequences
        self.tails = [""] * n_sequences

    def __call__(self, input_ids, scores, **kwargs):
        done = []
        for i in range(input_ids.shape[0]):
            idx = i if i < len(self.counts) else len(self.counts) - 1
            piece = self.tokenizer.decode(input_ids[i, -1:], skip_special_tokens=True)
            self.tails[idx] = (self.tails[idx] + piece)[-8:]
            if self.fence in self.tails[idx]:
                self.counts[idx] += 1
                self.tails[idx] = ""  # consume, so one fence is counted once
            done.append(self.counts[idx] >= 2)
        return torch.tensor(done, device=input_ids.device)


def install_stop_at_code_fence(model, tokenizer, domain: Domain) -> bool:
    """
    Make generation stop at the closing ``` fence instead of always running to
    max_new_tokens. Correct code solutions in the logs are 74-539 chars (~30-150
    tokens) while the budget is 1024-2048, so nearly all decode time is spent on
    text that the ```python extractor discards anyway -- stopping early cannot
    change which code is extracted, only how long we wait for it.

    Implemented by wrapping the bound `model.generate`, because the actual
    generate calls live inside 07/09 and we do not want to edit those.
    CODE ONLY: math/aime need the full reasoning chain before \\boxed{}, so
    there is no safe early stop there.
    """
    if domain.name != "code":
        return False
    original_generate = model.generate

    def generate_with_stop(*args, **kwargs):
        if "stopping_criteria" not in kwargs:
            ids = kwargs.get("input_ids")
            if ids is None and args:
                ids = args[0]
            n_seq = kwargs.get("num_return_sequences", 1) or 1
            if ids is not None:
                n_seq *= ids.shape[0]
            kwargs["stopping_criteria"] = StoppingCriteriaList(
                [ClosingFenceCriteria(tokenizer, n_seq)]
            )
        return original_generate(*args, **kwargs)

    model.generate = generate_with_stop
    return True


@contextlib.contextmanager
def merged_for_generation(model, enabled: bool):
    """
    Temporarily fold the LoRA adapter into the base weights so decode does not
    pay for an extra unfused pair of matmuls in every linear layer.

    ON by default. Measured on Qwen3-4B + LoRA r=32, A100, 2026-09-09:

        adapter active            9.6s / 100 tokens
        LoRA merged               5.4s / 100 tokens   -> 1.78x
        base model, no wrapper    9.6s / 100 tokens

    (The third number is not a control: `get_base_model()` only strips the outer
    PeftModel wrapper, the nn.Linear layers underneath are still lora.Linear, so
    it still pays for the adapter matmuls. The cost is the ~500 extra unfused
    tiny matmuls per decode step, not the wrapper.)

    Drift check, 80 merge/unmerge cycles (= one 40-problem run) with non-zero
    lora_B: relative drift 3.3e-4, i.e. an order of magnitude BELOW bf16's own
    ~0.4% per-element precision, and far below what a gradient step moves. Safe.
    Note the first drift test returned exactly 0.0 because a freshly initialised
    LoRA has B = 0, so the delta was zero and nothing was actually exercised --
    perturb lora_B before trusting such a test.

    Even merged, decode is ~54ms/step against a ~4ms bandwidth floor; that
    remaining ~13x is the HF generate loop itself and only an inference engine
    (vLLM) would remove it.
    """
    if enabled:
        model.merge_adapter()
    try:
        yield
    finally:
        if enabled:
            model.unmerge_adapter()


# --------------------------------------------------------------------------- #
# Student-first arm = the paper's own SDPO, as the baseline to beat.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def student_generate(
    model,
    tokenizer,
    student_messages: list[dict[str, str]],
    n: int,
    temperature: float,
    max_new_tokens: int,
    top_p: float = 1.0,
    top_k: int = 0,
) -> list[str]:
    """
    Sample n rollouts from the PLAIN student prompt -- no feedback, no few-shot.

    Mirror of 09.teacher_generate, minus the privileged context. The asymmetry is
    the whole point of the comparison: the teacher-first arm lets the teacher see
    feedback + exemplars before producing the trajectory to distil, the
    student-first arm distils the student's own unaided attempts.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    prompt_text = tokenizer.apply_chat_template(
        student_messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]

    sample_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "num_return_sequences": n,
        "use_cache": True,
        "pad_token_id": pad_id,
    }
    if top_k > 0:
        sample_kwargs["top_k"] = top_k

    out = model.generate(**inputs, **sample_kwargs)
    completions = [
        tokenizer.decode(out[i, prompt_tokens:], skip_special_tokens=True)
        for i in range(out.shape[0])
    ]
    if was_training:
        model.train()
    return completions


def select_student_targets(
    rollouts: list[str],
    row: dict,
    domain: Domain,
    dont_reprompt_on_self_success: bool,
) -> tuple[list[str], str, list[float]]:
    """
    Pick which of the student's own rollouts get distilled, following the paper.

    Returns (targets, best_successful_rollout, all_scores).

    Two things differ from the teacher-first arm and they matter for §2d:
      * A rollout does NOT have to be correct to be a target. The paper's whole
        point is learning from FAILED attempts: the teacher sees the feedback and
        retrospectively fixes them, and the student distils toward that. So this
        arm produces a gradient on almost every problem, while teacher-first
        needs the teacher to actually solve it (measured yield ~5%).
      * `dont_reprompt_on_self_success` drops rollouts that already passed --
        there is nothing to fix in those.
    Consequence: the two arms will see very different amounts of data. That is a
    property of the mechanisms, not a bug, but it must be reported (and the
    comparison made at equal gradient-step count if the gap is large).
    """
    scored: list[tuple[str, float]] = []
    for code in rollouts:
        try:
            s = float(domain.evaluate(code, row, split="train")["score"])
        except Exception as exc:
            print(f"[student-first] score error: {exc}")
            s = 0.0
        scored.append((code, s))

    successes = [c for c, s in scored if s >= 1.0]
    best_success = successes[0] if successes else ""
    if dont_reprompt_on_self_success:
        targets = [c for c, s in scored if s < 1.0]
    else:
        targets = [c for c, _ in scored]
    return targets, best_success, [s for _, s in scored]


# --------------------------------------------------------------------------- #
# Dataset split.
# --------------------------------------------------------------------------- #
def split_rows(rows, seed: int, n_train: int, n_holdout: int) -> tuple[list[dict], list[dict]]:
    """
    Deterministic seeded shuffle, then slice. Train and holdout are disjoint by
    construction (holdout starts where train ends); main() asserts on ids too.

    NOTE (spec 3): no frontier filtering by default. The thesis's test-time runs
    deliberately cherry-picked problems with model pass-rate in (0,1); doing that
    here would weaken the generalization claim, so the split is taken as-is.
    """
    shuffled = rows.shuffle(seed=seed) if hasattr(rows, "shuffle") else list(rows)
    total = len(shuffled)
    n_train = min(n_train, total)
    train = [shuffled[i] for i in range(n_train)]
    holdout_end = min(total, n_train + n_holdout) if n_holdout > 0 else total
    holdout = [shuffled[i] for i in range(n_train, holdout_end)]
    return train, holdout


# --------------------------------------------------------------------------- #
# Held-out evaluation (aggregate over many problems, unlike 09's single row).
# --------------------------------------------------------------------------- #
def evaluate_set(model, tokenizer, rows: list[dict], domain: Domain, args, label: str) -> dict:
    """Run safe_evaluate_model per row and aggregate. Keeps per-row detail."""
    # Seed identically for every eval pass. With unchanged weights this makes PRE
    # and POST bit-identical, so a nonzero delta is attributable to training rather
    # than sampling noise (spec 7). The 2026-09-04 smoke test reported IMPROVED on a
    # run with ZERO gradient steps because this was missing -- if that ever recurs,
    # it is a regression, not a result.
    set_seed(args.seed)
    per_row = []
    t0 = time.time()
    # One merge for the whole eval pass rather than one per problem: same speedup,
    # far fewer bf16 merge/unmerge round-trips.
    merge = bool(getattr(args, "merge_lora_for_generation", 0))
    for i, row in enumerate(rows):
        with merged_for_generation(model, merge):
            ev = safe_evaluate_model(
                model, tokenizer, row, args.eval_samples, args.max_new_tokens,
                label=f"{label}-{i}", domain=domain, model_name=args.model_name,
                thinking=args.thinking, top_p=args.top_p, top_k=args.top_k,
            )
        per_row.append({
            "problem_id": str(domain.problem_id(row)),
            "pass_rate": ev["pass_rate"],
            "mean_score": ev["mean_score"],
            "greedy_score": ev["greedy_score"],
        })
        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i + 1}/{len(rows)} evaluated ...")

    agg = {
        "n_problems": len(per_row),
        "pass_rate": statistics.mean([r["pass_rate"] for r in per_row]) if per_row else 0.0,
        "mean_score": statistics.mean([r["mean_score"] for r in per_row]) if per_row else 0.0,
        "greedy_score": statistics.mean([r["greedy_score"] for r in per_row]) if per_row else 0.0,
        "solved_at_greedy": sum(1 for r in per_row if r["greedy_score"] >= 1.0),
        "eval_runtime_s": time.time() - t0,
        "per_row": per_row,
    }
    print(f"[{label}] n={agg['n_problems']} pass_rate={agg['pass_rate']:.3f} "
          f"mean={agg['mean_score']:.3f} greedy={agg['greedy_score']:.3f} "
          f"({agg['eval_runtime_s']:.0f}s)")
    return agg


# --------------------------------------------------------------------------- #
# Trust-region teacher (spec 2b).
# --------------------------------------------------------------------------- #
def teacher_first_step_trust_region(
    model,
    tokenizer,
    student_messages: list[dict[str, str]],
    teacher_messages: list[dict[str, str]],
    y_good_list: list[str],
    optimizer,
    kl_topk: int,
    kl_alpha: float,
    tr_alpha: float,
    verbose: bool = False,
) -> float:
    """
    Same as 09.teacher_first_step, except the teacher distribution is the
    trust-region interpolation of the INITIAL teacher and the CURRENT teacher:

        log q_teacher = (1 - tr_alpha) * log q_theta_ref + tr_alpha * log q_theta

    Both terms run on the TEACHER prefix (feedback + few-shot). q_theta_ref is
    the same forward with the LoRA adapter disabled -- NOT the initial model on
    the student prompt, which would be the Chen et al. (2025c) variant the SDPO
    authors report underperforms (spec 2b.2, paper App. B.2).

    lerp on logits == lerp on log-probs after renormalization: the two differ by
    a per-position constant that softmax/top-k renorm cancels.
    """
    device = next(model.parameters()).device

    student_prefix_text = tokenizer.apply_chat_template(
        student_messages, add_generation_prompt=True, tokenize=False
    )
    teacher_prefix_text = tokenizer.apply_chat_template(
        teacher_messages, add_generation_prompt=True, tokenize=False
    )
    student_prefix_ids = tokenizer(student_prefix_text, return_tensors="pt").input_ids.to(device)
    teacher_prefix_ids = tokenizer(teacher_prefix_text, return_tensors="pt").input_ids.to(device)
    s_prefix = student_prefix_ids.shape[1]
    t_prefix = teacher_prefix_ids.shape[1]
    if verbose:
        print(f"[kl-tr] student_prefix_len={s_prefix} teacher_prefix_len={t_prefix} "
              f"tr_alpha={tr_alpha}")
        print(f"[kl-tr] ref forward uses the TEACHER prefix ({t_prefix} tokens) with the "
              f"adapter disabled -> q_theta_ref, not pi_theta_ref")

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
        assert torch.equal(
            student_input[:, s_prefix:], teacher_input[:, t_prefix:]
        ), "y_good tokens differ across student/teacher sides"

        student_logits = model(input_ids=student_input, use_cache=False).logits
        s_slice = student_logits[:, s_prefix - 1 : s_prefix - 1 + length, :]

        with torch.no_grad():
            # Two teacher forwards. Slice and free each full-sequence logits tensor
            # BEFORE running the next one: at Qwen3 vocab (~152k) a [1, L, V] bf16
            # tensor is ~300 MB at L=1024, and holding all of them at once is what
            # pushes an L4 over the edge.
            # q_theta : current weights, teacher context.
            t_cur_full = model(input_ids=teacher_input, use_cache=False).logits
            t_cur = t_cur_full[:, t_prefix - 1 : t_prefix - 1 + length, :].clone()
            del t_cur_full
            # q_theta_ref : initial weights (adapter off), SAME teacher context.
            with model.disable_adapter():
                t_ref_full = model(input_ids=teacher_input, use_cache=False).logits
                t_ref = t_ref_full[:, t_prefix - 1 : t_prefix - 1 + length, :].clone()
                del t_ref_full
            t_slice = torch.lerp(t_ref, t_cur, tr_alpha)

        assert s_slice.shape[1] == length, f"student slice {s_slice.shape[1]} != {length}"
        assert t_slice.shape[1] == length, f"teacher slice {t_slice.shape[1]} != {length}"

        per_token = compute_topk_self_distillation_loss(
            s_slice, t_slice,
            distillation_topk=kl_topk,
            distillation_alpha=kl_alpha,
            distillation_add_tail=True,
        )
        item_loss = per_token.mean()

        if verbose and contributing == 0:
            d_cur = (t_slice - t_cur).abs().mean().item()
            d_ref = (t_slice - t_ref).abs().mean().item()
            print(f"[kl-tr-sanity] L={length} KL={item_loss.item():.6f} "
                  f"dist_to_cur={d_cur:.4f} dist_to_ref={d_ref:.4f} "
                  f"(small tr_alpha -> teacher must sit CLOSER to ref)")

        (item_loss / n).backward()
        loss_sum += item_loss.item()
        contributing += 1

        del student_logits, t_cur, t_ref, s_slice, t_slice

    if contributing == 0:
        optimizer.zero_grad()
        return 0.0

    optimizer.step()
    optimizer.zero_grad()
    return loss_sum / contributing


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher-first TRAIN-TIME SDPO (code or math).")
    # Domain / data
    p.add_argument("--domain", type=str, required=True, choices=["code", "math"],
                   help="REQUIRED. code=LCBv6 (headline), math=MATH-500 (generalization).")
    p.add_argument("--arm", type=str, default="teacher_first",
                   choices=["teacher_first", "student_first"],
                   help="teacher_first = trajectories generated by the teacher (feedback + "
                        "few-shot) then judge-filtered. student_first = the paper's own SDPO "
                        "baseline: distil the student's own rollouts, including failed ones. "
                        "Identical KL step, teacher context and accounting either way -- the "
                        "arm changes only where the distilled trajectories come from.")
    p.add_argument("--n_train_problems", type=int, default=100)
    p.add_argument("--n_holdout_problems", type=int, default=0,
                   help="0 = use every remaining row after the train slice.")
    p.add_argument("--n_eval_problems", type=int, default=40,
                   help="Cap on how many held-out rows are actually evaluated PRE/POST "
                        "(full holdout x eval_samples gets expensive fast). 0 = all.")
    p.add_argument("--eval_ood", action="store_true",
                   help="math only: also evaluate AIME-2026 as an OOD set.")
    p.add_argument("--frontier_scan_samples", type=int, default=4,
                   help="Samples per problem used by --filter_frontier to measure pass_rate. "
                        "Must be >0: frontier means the model solves it SOMETIMES.")
    p.add_argument("--filter_frontier", action="store_true",
                   help="Restrict the TRAIN split to problems with greedy score in (0,1). "
                        "OFF by default on purpose -- see spec 3.")
    # Model / optim (defaults mirror the paper's LCBv6 train-time run)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-6,
                   help="1e-6 = lasgroup run_sdpo.sh LCBv6 value (09 used 1e-5 for test-time).")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps_per_problem", type=int, default=1)
    p.add_argument("--dont_reprompt_on_self_success", type=int, default=1,
                   help="1 = skip a problem the student already solves greedily (paper default).")
    # Teacher-first mechanism (reused from 09)
    p.add_argument("--teacher_n", type=int, default=10)
    p.add_argument("--teacher_temperature", type=float, default=1.0)
    p.add_argument("--fewshot_option", type=str, default="good_only",
                   choices=["good_only", "good_bad"])
    p.add_argument("--max_fewshot", type=int, default=3)
    p.add_argument("--pool_mode", type=str, default="reset", choices=["reset", "persistent"],
                   help="reset = clear few-shot pools per problem (parity with 09). "
                        "persistent = carry across problems (code-only ablation, spec 4b.1).")
    p.add_argument("--pool_cap", type=int, default=8)
    p.add_argument("--reference_mode", type=str, default="best_in_batch",
                   choices=["best_in_batch", "ground_truth", "none"])
    p.add_argument("--sim_threshold", type=float, default=0.9)
    p.add_argument("--reprompt_template", type=str, default="T2_standard",
                   choices=sorted(REPROMPT_TEMPLATES))
    # Distillation / teacher regularization
    p.add_argument("--kl_topk", type=int, default=20)
    p.add_argument("--kl_alpha", type=float, default=1.0, help="1.0 = reverse KL (lasgroup LCBv6).")
    p.add_argument("--teacher_reg", type=str, default="trust_region",
                   choices=["none", "trust_region"],
                   help="none == trust_region at alpha 1.0 == 09 behavior (delegates to 09).")
    p.add_argument("--teacher_reg_alpha", type=float, default=0.01,
                   help="Trust-region alpha, paper value 0.01. NOT an EMA rate: it is applied "
                        "fresh each step and never accumulates (spec 2b.5).")
    # Judge
    p.add_argument("--judge", type=str, default="difflib", choices=["difflib", "llm"])
    p.add_argument("--judge_provider", type=str, default="gemini",
                   choices=["gemini", "groq", "openrouter", "zai"])
    p.add_argument("--judge_fallback", type=str, nargs="*",
                   default=["gemini", "groq", "openrouter"],
                   choices=["gemini", "groq", "openrouter", "zai"])
    p.add_argument("--judge_model", type=str, default="gemini-2.5-flash")
    p.add_argument("--judge_groq_model", type=str, default="llama-3.3-70b-versatile")
    p.add_argument("--judge_openrouter_model", type=str,
                   default="meta-llama/llama-3.3-70b-instruct:free")
    p.add_argument("--judge_zai_model", type=str, default="glm-4.5-flash")
    # Generation / eval
    p.add_argument("--eval_samples", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_prompt_length", type=int, default=4096)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--stop_at_code_fence", type=int, default=0,
                   help="1 = stop generation at the closing ``` fence (code domain only). "
                        "Cannot change which code is extracted, only how long decode runs. "
                        "Set 0 to reproduce the pre-2026-09-04 timing behavior.")
    p.add_argument("--merge_lora_for_generation", type=int, default=1,
                   help="1 = fold the LoRA adapter into base weights around each generation "
                        "call. Faster decode, but merge/unmerge does not round-trip exactly "
                        "in bf16, so drift accumulates in the trained weights. Leave 0 unless "
                        "benchmarking.")
    # Run plumbing
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_every_n", type=int, default=25,
                   help="Checkpoint adapter + progress every N training problems so a run can "
                        "resume across Modal invocations (spec 0b decision gate).")
    p.add_argument("--output_dir", type=str, default="outputs/10_teacher_first_train")
    p.add_argument("--save_adapter_path", type=str, default="")
    p.add_argument("--wandb_project", type=str, default="ttt-sdpo-thesis")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This script requires a GPU.")
    if args.domain == "math" and args.judge != "llm":
        print("[WARN] math domain normally requires --judge llm (difflib cannot judge math "
              "derivations); continuing because it was explicitly requested.")

    set_seed(args.seed)
    domain = get_domain(args.domain)

    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    print(f"Arm: {args.arm}")
    print(f"Domain: {args.domain} | seed={args.seed} | teacher_reg={args.teacher_reg}"
          f"(alpha={args.teacher_reg_alpha}) | pool_mode={args.pool_mode} | lr={args.learning_rate}")

    judge_provider_models = {
        "gemini": args.judge_model,
        "groq": args.judge_groq_model,
        "openrouter": args.judge_openrouter_model,
        "zai": args.judge_zai_model,
    }
    judge_primary_model = judge_provider_models[args.judge_provider]
    if args.judge == "llm":
        chain = _build_provider_chain(
            args.judge_provider, judge_primary_model, args.judge_fallback, judge_provider_models
        )
        usable = [f"{p}:{m}" for p, m in chain if _provider_has_key(p)]
        print(f"Judge: llm | usable providers: {usable or 'NONE (will fall back to difflib)'}")

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"{args.arm}-{args.model_name.split('/')[-1]}-{args.domain}"
                 f"-n{args.n_train_problems}-{args.teacher_reg}{args.teacher_reg_alpha}"
                 f"-{args.pool_mode}-seed{args.seed}",
            config=vars(args),
        )

    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    rows = domain.load_split()
    print(f"[data] {domain.name} split has {len(rows)} rows total")
    train_rows, holdout_rows = split_rows(
        rows, args.seed, args.n_train_problems, args.n_holdout_problems
    )
    train_ids = {str(domain.problem_id(r)) for r in train_rows}
    holdout_ids = {str(domain.problem_id(r)) for r in holdout_rows}
    overlap = train_ids & holdout_ids
    assert not overlap, f"train/holdout overlap on {len(overlap)} ids: {sorted(overlap)[:5]}"
    eval_rows = holdout_rows if args.n_eval_problems <= 0 else holdout_rows[: args.n_eval_problems]
    print(f"[data] train={len(train_rows)} holdout={len(holdout_rows)} "
          f"(evaluating {len(eval_rows)} of them)")

    ood_rows = []
    if args.eval_ood and args.domain == "math":
        ood_rows = list(load_aime_split())
        if args.n_eval_problems > 0:
            ood_rows = ood_rows[: args.n_eval_problems]
        print(f"[data] OOD (AIME-2026): {len(ood_rows)} rows")

    # ---------------- model ----------------
    tokenizer = _prepare_tokenizer(args.model_name, thinking=args.thinking)
    total_start = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model = _apply_lora(model, args.lora_r, model_name=args.model_name)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # ONE optimizer for the whole run -- this is what makes it train-time.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )

    if args.stop_at_code_fence:
        if install_stop_at_code_fence(model, tokenizer, domain):
            print("[speed] generation stops at the closing ``` fence")
        else:
            print(f"[speed] stop-at-fence not applied (domain={domain.name}, code only)")
    if args.merge_lora_for_generation:
        print("[speed] LoRA merged around generation (measured 1.8x end-to-end; "
              "weight drift over 80 merge/unmerge cycles is 3.3e-4 relative, well under "
              "bf16 precision)")

    # ---------------- PRE-eval ----------------
    print(f"\n=== PRE-eval on {len(eval_rows)} held-out problems ===")
    pre_eval = evaluate_set(model, tokenizer, eval_rows, domain, args, "PRE")
    pre_ood = evaluate_set(model, tokenizer, ood_rows, domain, args, "PRE-OOD") if ood_rows else None

    # ---------------- optional frontier filter on TRAIN ----------------
    if args.filter_frontier:
        # Frontier must be measured as PASS RATE over samples (the model solves the
        # problem SOMETIMES), not as the dense greedy score. A greedy score of 0.42
        # only means partial test-case credit and is compatible with pass@k == 0,
        # i.e. the model never actually solves it -- and since filter_trajectories
        # requires score >= 1.0, such problems can never yield a distillation
        # target. Screening on greedy score (the 2026-09-04 bug) selected exactly
        # the problems least able to produce a gradient: 0/3 fired.
        print(f"\n[filter_frontier] screening train split on pass_rate over "
              f"{args.frontier_scan_samples} samples (strictly in (0,1)) ...")
        kept = []
        for row in train_rows:
            ev = safe_evaluate_model(
                model, tokenizer, row, args.frontier_scan_samples, args.max_new_tokens,
                label="SCAN", domain=domain, model_name=args.model_name,
                thinking=args.thinking, top_p=args.top_p, top_k=args.top_k,
            )
            pid = str(domain.problem_id(row))
            keep = 0.0 < ev["pass_rate"] < 1.0
            print(f"  [scan] {pid}: pass_rate={ev['pass_rate']:.2f} "
                  f"greedy={ev['greedy_score']:.2f} -> {'keep' if keep else 'drop'}")
            if keep:
                kept.append(row)
        print(f"[filter_frontier] kept {len(kept)}/{len(train_rows)}")
        train_rows = kept

    # ---------------- train loop ----------------
    judge_cache: dict[str, dict] = {}
    good_pool: list[dict] = []
    bad_pool: list[dict] = []
    problem_records: list[dict] = []
    epoch_stats: list[dict] = []
    all_reward_samples: list[float] = []
    n_grad_steps = 0
    # Where the wall-clock actually goes. Generation (GPU, autoregressive) and
    # verification (CPU, runs the generated program against test cases and can sit
    # on timeouts for non-terminating code) are very different costs, and which one
    # dominates decides whether a bigger model is affordable.
    timing = {"greedy_eval": 0.0, "teacher_generate": 0.0, "filter_judge": 0.0, "grad_step": 0.0}
    run_start = time.time()

    for epoch in range(args.epochs):
        ep = {
            "epoch": epoch + 1,
            "n_problems": 0,
            "n_with_gradient": 0,
            "n_skipped_already_solved": 0,
            "n_skipped_no_good_trajectory": 0,
            "n_good_total": 0,
            "judge_fallbacks": 0,
            "loss_sum": 0.0,
        }
        print(f"\n=== EPOCH {epoch + 1}/{args.epochs} over {len(train_rows)} problems ===")

        for p_idx, row in enumerate(train_rows):
            verbose = (epoch == 0 and p_idx == 0)
            problem_id = str(domain.problem_id(row) or f"idx_{p_idx}")
            question_content = domain.problem_text(row)
            ep["n_problems"] += 1

            if args.pool_mode == "reset":
                good_pool, bad_pool = [], []

            # Student's current greedy attempt: gives dynamic feedback AND decides
            # the dont_reprompt_on_self_success skip.
            _t = time.time()
            with merged_for_generation(model, bool(args.merge_lora_for_generation)):
                greedy_eval = safe_evaluate_model(
                    model, tokenizer, row, 0, args.max_new_tokens,
                    label=f"E{epoch}P{p_idx}", domain=domain,
                    model_name=args.model_name, thinking=args.thinking,
                    top_p=args.top_p, top_k=args.top_k,
                )
            timing["greedy_eval"] += time.time() - _t
            greedy_score = greedy_eval.get("greedy_score", 0.0)

            if args.dont_reprompt_on_self_success and greedy_score >= 1.0:
                ep["n_skipped_already_solved"] += 1
                problem_records.append({
                    "epoch": epoch + 1, "problem_id": problem_id, "outcome": "skip_already_solved",
                    "greedy_score": greedy_score, "loss": 0.0, "n_good": 0,
                })
                print(f"[e{epoch + 1} p{p_idx + 1}/{len(train_rows)}] {problem_id}: "
                      f"already solved (greedy={greedy_score:.2f}) -> skip")
                continue

            base_hint = domain.privileged_context(row)
            dyn = build_dynamic_feedback(greedy_eval.get("greedy_code", ""), row, domain=domain)
            feedback_text = "\n\n".join(x for x in [base_hint, dyn] if x).strip()
            student_messages = _build_messages(
                question_content, domain, args.model_name, args.thinking
            )

            step_loss = 0.0
            stats: dict = {}
            n_good_here = 0
            for _ in range(args.steps_per_problem):
                # The two arms differ in exactly one thing: where the trajectories
                # being distilled come from. Everything after this block -- the
                # teacher context, the KL step, the accounting -- is shared, so the
                # comparison isolates the mechanism.
                if args.arm == "teacher_first":
                    _t = time.time()
                    with merged_for_generation(model, bool(args.merge_lora_for_generation)):
                        trajectories, _teacher_prompt = teacher_generate(
                            model, tokenizer, question_content, feedback_text,
                            good_pool, bad_pool, args.fewshot_option, args.teacher_n,
                            args.teacher_temperature, args.max_new_tokens, args.max_fewshot,
                            args.reprompt_template, args.max_prompt_length, verbose=verbose,
                            domain=domain, model_name=args.model_name, thinking=args.thinking,
                            top_p=args.top_p, top_k=args.top_k,
                        )
                    timing["teacher_generate"] += time.time() - _t
                    _t = time.time()
                    good, bad, stats = filter_trajectories(
                        trajectories, row, args.sim_threshold, args.reference_mode,
                        judge=args.judge, judge_model=judge_primary_model,
                        judge_cache=judge_cache,
                        problem_id=problem_id, problem_text=question_content,
                        judge_provider=args.judge_provider,
                        judge_fallback_providers=args.judge_fallback,
                        judge_provider_models=judge_provider_models,
                        domain=domain,
                    )
                    timing["filter_judge"] += time.time() - _t
                    all_reward_samples.extend(stats.get("scores", []))
                    ep["judge_fallbacks"] += stats.get("judge_fallbacks", 0)
                    good_pool = _update_pool(good_pool, good, args.pool_cap)
                    bad_pool = _update_pool(bad_pool, bad, args.pool_cap)
                    y_targets = [g["code"] for g in good]
                    demo_good, demo_bad = good_pool, bad_pool
                    if not y_targets:
                        # Retry with fresh teacher samples rather than abandoning the
                        # problem: teacher generation is stochastic, so another attempt
                        # can succeed where this one did not. (Was `break`, which made
                        # --steps_per_problem a no-op.)
                        continue
                else:  # student_first == the paper's own SDPO, the baseline
                    _t = time.time()
                    with merged_for_generation(model, bool(args.merge_lora_for_generation)):
                        rollouts = student_generate(
                            model, tokenizer, student_messages, args.teacher_n,
                            args.teacher_temperature, args.max_new_tokens,
                            top_p=args.top_p, top_k=args.top_k,
                        )
                    timing["teacher_generate"] += time.time() - _t
                    _t = time.time()
                    y_targets, best_success, scores = select_student_targets(
                        rollouts, row, domain, bool(args.dont_reprompt_on_self_success)
                    )
                    timing["filter_judge"] += time.time() - _t
                    all_reward_samples.extend(scores)
                    stats = {
                        "scores": scores,
                        "n_good": len(y_targets),
                        "n_bad": len(rollouts) - len(y_targets),
                        "n_total": len(rollouts),
                        "mean_sim": 0.0,
                    }
                    # Path A: a successful sibling rollout becomes the demonstration
                    # in the teacher's context. Path B (environment feedback) is
                    # already inside feedback_text. No few-shot exemplar pools here --
                    # the paper's reprompt template has {solution} and {feedback}
                    # slots only.
                    demo_good = [{"code": best_success, "score": 1.0}] if best_success else []
                    demo_bad = []
                    if not y_targets:
                        continue

                teacher_messages = _build_teacher_messages(
                    question_content, feedback_text, demo_good, demo_bad,
                    args.fewshot_option, args.max_fewshot, args.reprompt_template, domain,
                    args.model_name, args.thinking,
                )
                model.train()
                _t = time.time()
                if args.teacher_reg == "none" or args.teacher_reg_alpha >= 1.0:
                    # Delegate to 09 verbatim -> bit-for-bit parity with the thesis runs.
                    step_loss = teacher_first_step(
                        model, tokenizer, student_messages, teacher_messages,
                        y_targets, optimizer, args.kl_topk, args.kl_alpha, verbose=verbose,
                    )
                else:
                    step_loss = teacher_first_step_trust_region(
                        model, tokenizer, student_messages, teacher_messages,
                        y_targets, optimizer, args.kl_topk, args.kl_alpha,
                        args.teacher_reg_alpha, verbose=verbose,
                    )
                timing["grad_step"] += time.time() - _t
                n_good_here += len(y_targets)
                n_grad_steps += 1

            if n_good_here == 0:
                ep["n_skipped_no_good_trajectory"] += 1
                outcome = "skip_no_good_trajectory"
            else:
                ep["n_with_gradient"] += 1
                ep["n_good_total"] += n_good_here
                ep["loss_sum"] += step_loss
                outcome = "gradient"

            rec = {
                "epoch": epoch + 1,
                "problem_id": problem_id,
                "outcome": outcome,
                "greedy_score": greedy_score,
                "loss": step_loss,
                "n_good": n_good_here,
                "batch_mean_reward": (
                    statistics.mean(stats["scores"]) if stats.get("scores") else 0.0
                ),
            }
            problem_records.append(rec)
            print(f"[e{epoch + 1} p{p_idx + 1}/{len(train_rows)}] {problem_id}: {outcome} "
                  f"greedy={greedy_score:.2f} n_good={n_good_here} loss={step_loss:.4f}")
            if use_wandb:
                wandb.log({f"train/{k}": v for k, v in rec.items() if isinstance(v, (int, float))})

            if args.save_every_n and (p_idx + 1) % args.save_every_n == 0:
                ckpt = output_root / f"adapter_e{epoch + 1}_p{p_idx + 1}"
                model.save_pretrained(str(ckpt))
                with open(output_root / "progress.json", "w", encoding="utf-8") as f:
                    json.dump({"epoch": epoch + 1, "problem_index": p_idx + 1,
                               "n_grad_steps": n_grad_steps}, f, indent=2)
                print(f"  [ckpt] adapter + progress saved -> {ckpt}")

        ep["effective_problems_seen"] = ep["n_with_gradient"]
        ep["mean_loss"] = ep["loss_sum"] / ep["n_with_gradient"] if ep["n_with_gradient"] else 0.0
        epoch_stats.append(ep)
        print(f"\n[epoch {epoch + 1} summary] gradient={ep['n_with_gradient']} "
              f"skip_solved={ep['n_skipped_already_solved']} "
              f"skip_no_good={ep['n_skipped_no_good_trajectory']} "
              f"mean_loss={ep['mean_loss']:.4f}")
        if use_wandb:
            wandb.log({f"epoch/{k}": v for k, v in ep.items() if isinstance(v, (int, float))})

    run_elapsed = time.time() - run_start

    # ---------------- POST-eval (same rows, same seed) ----------------
    print(f"\n=== POST-eval on the SAME {len(eval_rows)} held-out problems ===")
    post_eval = evaluate_set(model, tokenizer, eval_rows, domain, args, "POST")
    post_ood = evaluate_set(model, tokenizer, ood_rows, domain, args, "POST-OOD") if ood_rows else None

    # ---------------- report ----------------
    n_total = sum(e["n_problems"] for e in epoch_stats)
    n_grad = sum(e["n_with_gradient"] for e in epoch_stats)
    n_solved = sum(e["n_skipped_already_solved"] for e in epoch_stats)
    n_nogood = sum(e["n_skipped_no_good_trajectory"] for e in epoch_stats)

    print("\n=== EFFECTIVE TRAINING SET (spec 2d -- quote THIS, not the nominal split) ===")
    print(f"Of {n_total} problem-visits, {n_grad} produced at least one gradient step; "
          f"{n_solved} skipped (already solved), {n_nogood} skipped (no valid distillation target).")
    print(f"Total gradient steps: {n_grad_steps}")
    for e in epoch_stats:
        print(f"  epoch {e['epoch']}: gradient={e['n_with_gradient']} "
              f"solved={e['n_skipped_already_solved']} no_good={e['n_skipped_no_good_trajectory']}")

    print("\n=== GENERALIZATION (held-out, PRE vs POST) ===")
    print(f"pass_rate : {pre_eval['pass_rate']:.3f} -> {post_eval['pass_rate']:.3f} "
          f"(delta {post_eval['pass_rate'] - pre_eval['pass_rate']:+.3f})")
    print(f"mean_score: {pre_eval['mean_score']:.3f} -> {post_eval['mean_score']:.3f} "
          f"(delta {post_eval['mean_score'] - pre_eval['mean_score']:+.3f})")
    print(f"greedy    : {pre_eval['greedy_score']:.3f} -> {post_eval['greedy_score']:.3f}")
    if pre_ood and post_ood:
        print("\n=== OOD (AIME-2026) -- reported separately, never averaged in ===")
        print(f"pass_rate : {pre_ood['pass_rate']:.3f} -> {post_ood['pass_rate']:.3f} "
              f"(delta {post_ood['pass_rate'] - pre_ood['pass_rate']:+.3f})")

    print("\n=== WHERE THE TIME WENT (training loop only) ===")
    tracked = sum(timing.values())
    for k, v in sorted(timing.items(), key=lambda kv: -kv[1]):
        share = 100 * v / run_elapsed if run_elapsed else 0.0
        print(f"  {k:<18} {v:8.0f}s  ({share:4.1f}% of training wall-clock)")
    print(f"  {'untracked':<18} {run_elapsed - tracked:8.0f}s  "
          f"({100 * (run_elapsed - tracked) / run_elapsed if run_elapsed else 0:4.1f}%)")
    print("  note: teacher_generate is GPU-bound (autoregressive decode); filter_judge "
          "is mostly CPU (running generated programs against test cases, incl. timeouts).")

    peak_vram = torch.cuda.max_memory_allocated(0)
    total_elapsed = time.time() - total_start
    print(f"\nPeak VRAM: {peak_vram / 1024**3:.2f} GB | train {run_elapsed:.0f}s "
          f"| total {total_elapsed:.0f}s")

    delta_pass = post_eval["pass_rate"] - pre_eval["pass_rate"]
    delta_mean = post_eval["mean_score"] - pre_eval["mean_score"]
    if n_grad_steps == 0:
        # Nothing was trained: any delta is eval noise. Report the noise floor
        # instead of a verdict -- never let this path print IMPROVED.
        improved = False
        print(f"VERDICT: NO TRAINING OCCURRED (0 gradient steps). "
              f"pass delta {delta_pass:+.3f}, mean delta {delta_mean:+.3f} is pure eval "
              f"noise; with seeded eval these should now be exactly 0.000.")
    else:
        improved = delta_pass > 0 or delta_mean > 0
        print(f"VERDICT: {'IMPROVED' if improved else 'NO IMPROVEMENT'} on held-out "
              f"({n_grad_steps} gradient steps)")

    if use_wandb:
        wandb.summary.update({
            "pre_pass_rate": pre_eval["pass_rate"],
            "post_pass_rate": post_eval["pass_rate"],
            "pass_rate_delta": post_eval["pass_rate"] - pre_eval["pass_rate"],
            "effective_problems_seen": n_grad,
            "n_grad_steps": n_grad_steps,
            "improved": improved,
        })
        wandb.finish()

    if args.save_adapter_path:
        model.save_pretrained(args.save_adapter_path)
        print(f"Adapter saved -> {args.save_adapter_path}")

    summary_path = output_root / f"summary_{args.arm}_{args.domain}_seed{args.seed}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "split": {
                "n_rows_total": len(rows),
                "train_ids": sorted(train_ids),
                "holdout_ids": sorted(holdout_ids),
                "evaluated_ids": [r["problem_id"] for r in pre_eval["per_row"]],
            },
            "effective_training": {
                "n_problem_visits": n_total,
                "n_with_gradient": n_grad,
                "n_skipped_already_solved": n_solved,
                "n_skipped_no_good_trajectory": n_nogood,
                "n_grad_steps": n_grad_steps,
            },
            "epoch_stats": epoch_stats,
            "problem_records": problem_records,
            "reward_histogram_samples": all_reward_samples,
            "pre_eval": pre_eval,
            "post_eval": post_eval,
            "pre_ood": pre_ood,
            "post_ood": post_ood,
            "improved": improved,
            "timing_breakdown_s": timing,
            "train_runtime_s": run_elapsed,
            "total_runtime_s": total_elapsed,
            "peak_vram_gb": peak_vram / 1024**3,
        }, f, indent=2)
    print(f"Summary written to: {summary_path}")

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
