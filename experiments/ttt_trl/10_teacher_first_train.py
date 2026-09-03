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
from transformers import AutoModelForCausalLM, set_seed
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
    per_row = []
    t0 = time.time()
    for i, row in enumerate(rows):
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
    p.add_argument("--n_train_problems", type=int, default=100)
    p.add_argument("--n_holdout_problems", type=int, default=0,
                   help="0 = use every remaining row after the train slice.")
    p.add_argument("--n_eval_problems", type=int, default=40,
                   help="Cap on how many held-out rows are actually evaluated PRE/POST "
                        "(full holdout x eval_samples gets expensive fast). 0 = all.")
    p.add_argument("--eval_ood", action="store_true",
                   help="math only: also evaluate AIME-2026 as an OOD set.")
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
            name=f"tf-train-{args.model_name.split('/')[-1]}-{args.domain}"
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

    # ---------------- PRE-eval ----------------
    print(f"\n=== PRE-eval on {len(eval_rows)} held-out problems ===")
    pre_eval = evaluate_set(model, tokenizer, eval_rows, domain, args, "PRE")
    pre_ood = evaluate_set(model, tokenizer, ood_rows, domain, args, "PRE-OOD") if ood_rows else None

    # ---------------- optional frontier filter on TRAIN ----------------
    if args.filter_frontier:
        print("\n[filter_frontier] screening train split (greedy score strictly in (0,1)) ...")
        kept = []
        for row in train_rows:
            ev = safe_evaluate_model(
                model, tokenizer, row, 0, args.max_new_tokens, label="SCAN",
                domain=domain, model_name=args.model_name, thinking=args.thinking,
                top_p=args.top_p, top_k=args.top_k,
            )
            if 0.0 < ev["greedy_score"] < 1.0:
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
            greedy_eval = safe_evaluate_model(
                model, tokenizer, row, 0, args.max_new_tokens,
                label=f"E{epoch}P{p_idx}", domain=domain,
                model_name=args.model_name, thinking=args.thinking,
                top_p=args.top_p, top_k=args.top_k,
            )
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
                trajectories, _teacher_prompt = teacher_generate(
                    model, tokenizer, question_content, feedback_text,
                    good_pool, bad_pool, args.fewshot_option, args.teacher_n,
                    args.teacher_temperature, args.max_new_tokens, args.max_fewshot,
                    args.reprompt_template, args.max_prompt_length, verbose=verbose,
                    domain=domain, model_name=args.model_name, thinking=args.thinking,
                    top_p=args.top_p, top_k=args.top_k,
                )
                good, bad, stats = filter_trajectories(
                    trajectories, row, args.sim_threshold, args.reference_mode,
                    judge=args.judge, judge_model=judge_primary_model, judge_cache=judge_cache,
                    problem_id=problem_id, problem_text=question_content,
                    judge_provider=args.judge_provider,
                    judge_fallback_providers=args.judge_fallback,
                    judge_provider_models=judge_provider_models,
                    domain=domain,
                )
                all_reward_samples.extend(stats.get("scores", []))
                ep["judge_fallbacks"] += stats.get("judge_fallbacks", 0)
                good_pool = _update_pool(good_pool, good, args.pool_cap)
                bad_pool = _update_pool(bad_pool, bad, args.pool_cap)

                if not good:
                    break

                teacher_messages = _build_teacher_messages(
                    question_content, feedback_text, good_pool, bad_pool,
                    args.fewshot_option, args.max_fewshot, args.reprompt_template, domain,
                    args.model_name, args.thinking,
                )
                model.train()
                y_good = [g["code"] for g in good]
                if args.teacher_reg == "none" or args.teacher_reg_alpha >= 1.0:
                    # Delegate to 09 verbatim -> bit-for-bit parity with the thesis runs.
                    step_loss = teacher_first_step(
                        model, tokenizer, student_messages, teacher_messages,
                        y_good, optimizer, args.kl_topk, args.kl_alpha, verbose=verbose,
                    )
                else:
                    step_loss = teacher_first_step_trust_region(
                        model, tokenizer, student_messages, teacher_messages,
                        y_good, optimizer, args.kl_topk, args.kl_alpha,
                        args.teacher_reg_alpha, verbose=verbose,
                    )
                n_good_here += len(good)
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

    peak_vram = torch.cuda.max_memory_allocated(0)
    total_elapsed = time.time() - total_start
    print(f"\nPeak VRAM: {peak_vram / 1024**3:.2f} GB | train {run_elapsed:.0f}s "
          f"| total {total_elapsed:.0f}s")

    improved = (
        post_eval["pass_rate"] > pre_eval["pass_rate"]
        or post_eval["mean_score"] > pre_eval["mean_score"]
    )
    print(f"VERDICT: {'IMPROVED' if improved else 'NO IMPROVEMENT'} on held-out")

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

    summary_path = output_root / f"summary_{args.domain}_seed{args.seed}.json"
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
