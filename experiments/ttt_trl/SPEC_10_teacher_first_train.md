# Implementation Spec — Teacher-First Train-Time SDPO on MATH (P0: pilot)

Spec to hand to Cursor (or self-implement), Claude reviews after. Extends the
validated test-time mechanism in `09_teacher_first.py` (see `con_teacher_first_judge`
in the thesis wiki for method background) into a **train-time** loop: one
accumulated LoRA checkpoint trained across many MATH problems, then evaluated
for generalization. Separate from the thesis (advisor's ad-hoc ask, not thesis
scope) — do not touch `10_Projects/14_Thesis_SDPO/` for this.

## 0. Goal & architecture decision

Advisor wants: run teacher-first SDPO across a whole math dataset (not a single
frontier problem), expecting the trained checkpoint to show **better metrics**
on math benchmarks afterward — no escape-zero framing needed, just aggregate
improvement.

**Decision: extend `09_teacher_first.py`, do NOT port into the verl repo.**
Reasons: verl's `run_sdpo.sh` assumes a Slurm HPC cluster (`sbatch`,
`--environment=sdpo` container, 4 GPU/node) that a rented single/few-GPU box
can't reproduce directly; the loss/EMA-teacher internals there
(`compute_self_distillation_loss`, `TrustRegionTeacher`) are reusable in theory
but the actual gap — teacher-generate + judge-filter — doesn't exist in verl at
all and would have to be built fresh inside its distributed rollout worker.
`09_teacher_first.py` already has that gap filled, tested, and validated (core
result, 2026-06-16). Cheaper to extend it into a training loop than to
re-implement teacher-first inside a heavier unfamiliar framework.

**Key difference from test-time (09):** 09 resets state per problem (fresh
optimizer/LoRA per TTT run, PRE/POST eval on the SAME problem). This script
keeps **one LoRA + one optimizer alive across all training problems** — no
reset between problems. PRE-eval and POST-eval run once each, on a **held-out**
set the model never trained on.

## 1. Reuse from `09_teacher_first.py` (import, don't copy-paste)

- `teacher_generate`, `filter_trajectories`, `teacher_first_step`, `llm_judge`,
  `_cached_llm_judge`, `JudgeUnavailable`, `_build_provider_chain`,
  `get_reference_text`, `_update_pool`, `_build_teacher_messages`,
  `_build_fewshot_block`, `build_dynamic_feedback` (from 07 via domain dispatch),
  `_build_lora_config`, `_prepare_tokenizer`, `safe_evaluate_model`.
- `domains.py` dispatch (`Domain`, `get_domain`) — already has `--domain math`
  wired to `MathDomain` (MATH-500), and `--judge llm` already implemented with
  the derive-vs-copy prompt (`_JUDGE_PROMPT_MATH`) validated in the math pilot.
  **Nothing new needed here except the fix in step 2.**

## 2. Fix required — `MathDomain.privileged_context` (domains.py:156-158)

Currently:
```python
def privileged_context(self, row: dict) -> str:
    # Leak regime: teacher always sees the reference answer (LLM judge catches copy).
    return f"Reference: the correct final answer is {row.get('answer', '')}."
```
`HuggingFaceH4/MATH-500` rows already carry a `solution` field (full worked
derivation) per `load_math_split()`'s own docstring — it's just not being used.
This is the exact failure mode the math pilot found (teacher only had the
answer, couldn't derive, 2/2 no-escape). Change to expose the worked solution:
```python
def privileged_context(self, row: dict) -> str:
    solution = row.get("solution", "")
    if solution:
        return f"Reference solution (do not copy verbatim):\n{solution}"
    return f"Reference: the correct final answer is {row.get('answer', '')}."
```
Keep the LLM judge (`_JUDGE_PROMPT_MATH`, derive-vs-copy) as the guard against
verbatim copying — same as math pilot, now with a stronger/more useful teacher.

## 3. Dataset split — reuse MATH-500, no new loader needed

`load_math_split()` already returns `HuggingFaceH4/MATH-500` (500 rows). Split
in the new script (deterministic, seeded shuffle):
- **Train subset**: first ~100 rows (pilot scale, per user decision) after
  shuffle(seed=args.seed).
- **Held-out eval (in-distribution)**: remaining ~400 rows, same source —
  keeps "improvement" claim honest since it's genuinely unseen.
- **OOD eval**: `load_aime_split()` (AIME 2026, already wired) — reuse
  directly, no new loader.
- **GSM8K (optional, if time allows)**: no loader exists yet in `evaluator.py`.
  Would need a small `load_gsm8k_split()` (openai/gsm8k, extract answer after
  `####`, mirroring `process_gsm8k` in the verl repo's `data/utils/math.py` for
  the extraction logic only — do not import from `verl/` directly, keep the
  TRL script dependency-free of the verl package). Treat as stretch goal, not
  a blocker for the pilot.

## 4. New file: `10_teacher_first_train.py`

### 4a. Main loop structure (replaces 09's single-problem `main()`)

```
load model + LoRA + optimizer (ONE state for the whole run — no re-init per problem)
train_rows, holdout_rows = split_math500(seed)
ood_rows = load_aime_split()

PRE-eval: safe_evaluate_model on holdout_rows (aggregate pass_rate) + ood_rows
judge_cache = {}; good_pool = []; bad_pool = []   # pools can persist or reset per-problem (see open decision)

for row in train_rows:
    question_content = domain.problem_text(row)
    greedy = safe_evaluate_model(model, ..., row, label=f"TRAIN-{problem_id}")
    feedback = build_dynamic_feedback(greedy, row, domain=domain)
    traj = teacher_generate(model, ..., feedback, good_pool, bad_pool, option, N, temp, domain=domain)
    good, bad, stats = filter_trajectories(traj, row, reference_text=get_reference_text(row, domain),
                                           judge="llm", domain=domain, ...)
    if good:
        teacher_first_step(model, ..., good, optimizer, kl_topk, kl_alpha)
    log per-problem stats -> W&B (n_good/n_bad, loss, running eval-holdout every K problems if cheap)

POST-eval: safe_evaluate_model on holdout_rows + ood_rows (SAME sets as PRE)
print PRE vs POST deltas; write summary.json + save LoRA adapter
```

### 4b. Open decisions — surface to user, do not silently assume

1. **Pool persistence across problems**: does `good_pool`/`bad_pool` (used for
   few-shot exemplars in `teacher_generate`) reset per problem (like 09,
   exemplars only relevant to the current problem) or persist/accumulate
   cross-problem? **Recommend: reset per problem** — exemplars are
   problem-specific (a good MATH-500 solution to problem A is meaningless as a
   few-shot for problem B), unlike code where similar patterns can transfer.
   Cursor should implement reset-per-problem and flag if this seems wrong.
2. **Steps per problem**: 09 does `max_steps` KL steps per problem before
   moving on (test-time regime). For train-time across ~100 problems, doing
   many steps per problem risks per-problem overfitting before ever seeing the
   next one. **Recommend: 1 distillation step per problem** (one pass through
   train_rows = one epoch), optionally loop for `--epochs` > 1 if pilot shows
   underfitting. Flag as a CLI arg (`--steps_per_problem`, default 1).
3. **Checkpointing eval cadence**: running full holdout eval after every
   problem is expensive (400+30 problems × generation). Recommend eval only at
   PRE and POST (not per-problem) for the pilot; add `--eval_every_n` as an
   optional stretch if budget allows watching the curve mid-run.
4. **LoRA r / hyperparams**: reuse 09's defaults (`Qwen/Qwen3-4B`, LoRA r=32,
   AdamW, `kl_topk=20`, `kl_alpha=1.0`, `teacher_n=10`, `sim_threshold` N/A for
   math since judge="llm" is required for math, not difflib) unless the pilot
   shows instability.

## 5. CLI args (extend 09's style)

`--domain math` (fixed for this script) `--model_name (default Qwen/Qwen3-4B)
--n_train_problems (100) --seed --steps_per_problem (1) --epochs (1)
--teacher_n (10) --teacher_temperature (1.0) --fewshot_option (good_only)
--max_fewshot (3) --kl_topk (20) --kl_alpha (1.0) --judge llm --judge_provider
--reprompt_template --eval_samples --max_new_tokens --wandb_project --no_wandb
--save_adapter_path`

## 6. Logging (required)

- Per-problem: `n_good/n_bad/n_total`, mean judge verdict distribution
  (is_copy rate — watch this number, math pilot saw ~75% copy on too-hard
  problems; MATH-500 train subset should be easier so expect lower copy rate),
  loss, running problem index.
- PRE/POST aggregate: pass_rate on holdout MATH-500 subset AND on AIME2026,
  reported separately (in-distribution vs OOD) — do not average them together.
- Save `summary.json` with both eval sets' PRE/POST numbers + the exact row
  indices used for train/holdout split (reproducibility).

## 7. Review checklist (Claude will check after implementation)

- `MathDomain.privileged_context` fix applied correctly, judge still guards
  against copy (checked via `_JUDGE_PROMPT_MATH` still invoked with `judge=llm`
  forced for math, not silently falling back to difflib).
- Optimizer/LoRA state genuinely persists across the problem loop (no
  accidental re-init hiding inside a helper reused from 09).
- Held-out set has zero overlap with train subset (assert on `unique_id`).
- Token-alignment in `teacher_first_step` still correct when problem changes
  between calls (prefix lengths change every problem, not just every step).
- PRE/POST eval use identical held-out rows and identical sampling (seed) so
  deltas are attributable to training, not eval noise.

## Links
- `09_teacher_first.py` — mechanism being extended
- `domains.py` — `MathDomain`, fix target
- Thesis wiki `con_teacher_first_judge`, `syn_teacher_first_impl_spec`,
  `syn_math_pilot` (background only — this experiment is NOT filed there)
