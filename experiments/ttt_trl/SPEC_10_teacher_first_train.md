# Implementation Spec — Teacher-First Train-Time SDPO on CODE + MATH (P0: pilot)

Spec to hand to Cursor (or self-implement), Claude reviews after. Extends the
validated test-time mechanism in `09_teacher_first.py` (see `con_teacher_first_judge`
in the thesis wiki for method background) into a **train-time** loop: one
accumulated LoRA checkpoint trained across many problems, then evaluated
for generalization.

**2026-09-03 scope change**: originally advisor's ad-hoc MATH-only ask,
separate from the thesis. Now upgraded to a **paper goal**: CODE (LCBv6) is
the headline domain (directly extends the defended thesis's test-time core
result — same model, same teacher-first mechanism, same problems pool — into
train-time, which is the literature's main comparison point and gives far
more statistical power than the thesis's n=2-3 test-time problems). MATH is
kept as a **generalization-across-domain** arm (this is the part advisor
originally asked for). One script, `--domain {code,math}`, serves both — do
not fork into two scripts. Target venue: COLM 2027 (~March 2027) / ICLR 2027
RSI-successor workshop (~Jan-Feb 2027) as an earlier checkpoint. See
`project_thesis_paper_scaleup` in the assistant's memory for full context.
Still fine to read/cite `10_Projects/14_Thesis_SDPO/` wiki for method
background, but this new code lives here in the fork, not in the vault.

## 0. Goal & architecture decision

Two goals, same mechanism, same script:
- **CODE (headline)**: teacher-first SDPO across a train split of LCBv6 (not
  a single frontier problem as in the thesis), evaluated for pass@k
  improvement on a held-out LCBv6 split — the train-time analog of the
  thesis's test-time core result (TF ≥ SF, escape-zero). This is the paper's
  main claim.
- **MATH (generalization)**: same recipe on MATH-500, held-out MATH-500 +
  AIME2026 OOD — advisor's original ask, now doubling as evidence the
  teacher-first mechanism isn't code-specific. No escape-zero framing needed
  here, just aggregate pass-rate improvement.

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

## 0b. Calibration test — run BEFORE committing to the full pilot, PER DOMAIN

Calibrate CODE and MATH separately — they have very different generation
profiles (CODE: thinking OFF, short completions, per thesis's regression-
critical `_CODE_DIRECTIVE`; MATH: thinking ON, long reasoning chains) so one
calibration does not predict the other's wall-clock, at all.

**CODE**: the thesis's own test-time runs (`09`, `07`) already give real
per-step/per-generation timing on Qwen3-4B + LCBv6 on Colab L4 / A100 (see
`project_ttt_pipeline_milestone` memory / `syn_core_result.md`) — reuse those
numbers as the calibration baseline instead of a fresh calibration pass;
train-time changes the *loop* (many problems, no reset) not the per-generation
cost, so the existing numbers should extrapolate directly. Do a short (5-10
problem) smoke run of the new train-time loop regardless, to catch loop-level
bugs (state leaking across problems, held-out contamination) — not to
re-measure per-generation timing.

**MATH**: full-scope settings (thinking ON — recommended by 09's own `--help`
text for math/aime; `--judge llm`; `--eval_samples 8`; `--teacher_n 10`) have
no timing data yet for **Qwen3-4B on MATH-500** specifically. The one
reference point available (math pilot, Gemma-4-E4B, AIME2026,
`max_new_tokens=16384`) is a different model on a different (harder,
longer-reasoning) benchmark, so it is not a safe basis for budgeting GPU-hours
here — could be off by 5x in either direction. Run the calibration pass below
for MATH.

**Before launching the MATH 100-problem training run**, run a calibration pass:
- 5 problems from the MATH-500 train subset (not held-out — throwaway, just
  for timing).
- **Same full settings as the real run**: `--thinking`, `--judge llm`,
  `--teacher_n 10`, `--eval_samples 8`. Do not simplify these for the
  calibration — the whole point is to measure the real config, not a cheaper
  proxy that won't predict it.
- Start `--max_new_tokens` at 2048 (MATH-500 is generally easier/shorter than
  AIME2026; 16384 was specifically needed there to avoid truncating
  `\boxed{}` on long AIME chains — MATH-500 likely doesn't need that much, but
  confirm: log how many completions in the calibration batch hit the token
  cap without producing a `\boxed{}` answer, and bump the cap if that rate is
  non-trivial, e.g. >10%).
- Log wall-clock per generation (teacher_generate batch, greedy eval, and one
  eval_samples pass) separately, plus total judge-call latency (network-bound,
  separate from GPU time).
- From this, extrapolate total GPU-hours for the full run: training
  (100 × 11 generations) + eval (430 × 8 × 2 generations) ≈ 8,000 generations
  total — multiply by the measured per-generation time from calibration.
- **Decision gate**: only proceed to the full run once this extrapolation is
  in hand. If it implies >8h wall-clock (current `modal_run.py` timeout,
  `28800` sec), restructure the training loop to checkpoint LoRA + progress
  state every N problems so the job can resume across multiple Modal
  invocations, rather than requiring one unbroken run.

## 1. Reuse from `09_teacher_first.py` (import, don't copy-paste)

- `teacher_generate`, `filter_trajectories`, `teacher_first_step`, `llm_judge`,
  `_cached_llm_judge`, `JudgeUnavailable`, `_build_provider_chain`,
  `get_reference_text`, `_update_pool`, `_build_teacher_messages`,
  `_build_fewshot_block`, `build_dynamic_feedback` (from 07 via domain dispatch),
  `_build_lora_config`, `_prepare_tokenizer`, `safe_evaluate_model`.
- `domains.py` dispatch (`Domain`, `get_domain`) already has both `--domain
  code` (`CodeDomain`, wraps `load_lcbv6_split()` + `evaluate_solution`,
  byte-for-byte the thesis's own test-time path) and `--domain math`
  (`MathDomain`, MATH-500), plus `--judge llm` with the derive-vs-copy prompt
  (`_JUDGE_PROMPT_MATH`) validated in the math pilot. **CODE needs no fix**
  (see step 2) — **MATH needs the fix in step 2**.

## 2. Fix required — `MathDomain.privileged_context` (domains.py:156-158) — MATH only

`CodeDomain.privileged_context` (→ `build_privileged_context`, public test
cases) is already correct — it's the exact function the thesis's core result
ran on. No change needed for CODE. The fix below is MATH-only.

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

## 2b. Teacher regularization — trust-region (DECIDED 2026-09-03)

09 currently has **no** teacher weight-regularization: its teacher is the
current weights on a richer context. In the paper's terms that is exactly
**trust-region with α = 1.0** — the knob exists implicitly, pinned at 1.0.
So adding it generalizes the validated mechanism rather than replacing it,
and the existing thesis results remain the α=1.0 point on the new axis.

**Formula (Hübotter et al. §2.3)**: the teacher used for distillation is an
interpolation *in log-prob space* of the **initial teacher** and the
**current teacher**:

```
log q_teacher  =  (1 − α) · log q_θref  +  α · log q_θ
```

Implementation notes — get these right, they are easy to get subtly wrong:

1. **Both terms are TEACHERS.** `q_θref` and `q_θ` are BOTH conditioned on the
   feedback/few-shot context `f`. They differ only in *weights* (initial vs
   current), never in context. Concretely: run the **teacher context** twice —
   once with the LoRA adapter **disabled** (`with model.disable_adapter():`)
   → `q_θref`, once with it enabled → `q_θ`. Then lerp.
2. **Do NOT use the student prompt for the reference term.** Taking the
   reference as the initial model on the *plain student prompt* (`π_θref`
   instead of `q_θref`) is the Chen et al. (2025c) variant, which the SDPO
   authors explicitly report (App. B.2) they "observe to underperform" their
   formulation. This is the single most likely implementation slip here.
3. **lerp on logits is exact, not an approximation.** `torch.lerp(ref_logits,
   student_logits, α)` followed by softmax equals the log-prob formula above:
   `log q = z − logsumexp(z)`, so the two differ by
   `(1−α)logZ_ref + α logZ_θ`, a constant along the vocab axis that softmax
   cancels. Both give the normalized geometric mean of the two distributions.
   → reuse lasgroup's `TrustRegionTeacher` approach verbatim
   (`verl/workers/actor/dp_actor.py:51-64`), no correction needed.
4. **Memory vs runtime trade** (paper, §2.3 footnote): EMA costs extra GPU
   memory for θ′ but no runtime; trust-region costs an extra log-prob
   computation but **no extra memory**. Under LoRA the trust-region side is
   even cheaper than in the paper's full-FT setting: the reference is just
   "adapter off", so there is no second weight copy at all — the only cost is
   one extra no-grad forward on the (long) teacher context. Budget this in
   calibration.
5. **α does NOT mean the same thing as in EMA — do not copy the number
   blindly.** Trust-region α is applied fresh every step and never
   accumulates: α=0.01 means the teacher stays 99% the *initial* teacher
   forever. EMA α=0.01 accumulates: after ~90 steps the teacher has drifted
   roughly `1 − 0.99^90 ≈ 60%` toward the student. Same constant, opposite
   regimes. (The 60% figure is our own arithmetic, not a paper claim.)
   → Trust-region at the paper's α=0.01 sits very close to the **fixed
   teacher** that Kim et al. recommend, while EMA at 0.01 drifts — the two
   modes bracket exactly the axis those two papers disagree on. Worth an
   ablation over α ∈ {0.01, 0.5, 1.0}, where 1.0 reproduces the thesis's
   existing test-time behavior.

**Fidelity caveat to document**: `experiments/rich_feedback/run_sdpo.sh` does
not override `teacher_regularization`, so the paper's LCBv6 train-time run
used the default **`ema` at rate 0.01**, not trust-region. Choosing
trust-region is a deviation from that specific run — but stays within the
paper's own supported option set (both modes ship in `dp_actor.py`), and is
justified by the LoRA memory/implementation argument above. If exact parity
with the published LCBv6 run is later demanded, add EMA behind the same
`--teacher_reg {none,trust_region,ema}` switch.

## 2c. Loss composition — pure distillation (DECIDED 2026-09-03)

**Distillation-only, no policy-gradient term** (TRL equivalent:
`distillation_weight = 1.0`). This is already what `teacher_first_step` does —
pure top-k reverse KL, no GRPO term — so no code change, and it keeps the
teacher-first arm and the student-first baseline structurally identical
apart from where the distilled completion comes from.

**Fidelity: VERIFIED from source (2026-09-03) — distill-only IS the original.**
`algorithm.adv_estimator=grpo` in `run_sdpo.sh` only disables the critic at
the pipeline level; the SDPO branch ignores advantages entirely. Evidence:
- `dp_actor.py` is a mutually-exclusive if/else — the `self_distillation_enabled`
  branch calls `compute_self_distillation_loss`; only the `else` branch calls
  `policy_loss_fn(advantages=advantages)`.
- `advantages` appears exactly 3× in the file, and the call site passing it
  into the loss (line 859) sits inside that `else` branch.
- `compute_self_distillation_loss` (`core_algos.py:1085`) has no `advantages`
  parameter at all.
- `policy_loss = pg_loss` — direct assignment, no `+`, no λ.
- Neither `sdpo.yaml` nor `actor.yaml` exposes any mixing coefficient.
→ Keeping distillation-only **matches the original**; this is parity, not a
deviation. Safe to state as such in the paper.

## 2d. Skip accounting — a CONFOUND, not just reporting hygiene

Pure distill + `dont_reprompt_on_self_success=True` means a problem the
student already solves produces **no gradient step**. On a deliberately
unfiltered LCBv6 split (§3) that bucket is non-trivial. But the more
important issue is that **the two arms do not lose problems the same way**:

| Reason for no gradient | Student-first | Teacher-first |
|---|---|---|
| All rollouts already correct → `dont_reprompt_on_self_success` | yes | yes |
| All rollouts wrong, feedback ON | still learns via Path B | — |
| Teacher produced no valid `y_good` | — | **yes (TF-only drop path)** |

Teacher-first has an extra way to fall through that student-first does not.
If TF skips 30% of problems while SF skips 10%, the two arms were **not
trained on the same amount of data**, and any outcome difference is partly a
data-quantity artifact — structurally the same confound as the unmatched
generation budget in the thesis's test-time results (`syn_core_result.md`
§Limitations). So this is a **variable to control**, not just a number to
report honestly.

**Mitigation**: log `effective_problems_seen` per arm; if the two diverge
materially, compare the arms at **equal gradient-step count** rather than
equal epochs (and say which basis is used).

### Required logging schema

Beyond the three headline counts (`n_problems_with_gradient_step`,
`n_skipped_already_solved`, `n_skipped_no_good_trajectory`):

1. **Break the counts down per epoch, not just run totals.** The
   "already solved" bucket should grow as the model improves → the effective
   training signal decays over time. That decay curve is itself a reportable
   finding ("effective training signal decays as the model improves"), not
   just diagnostics.
2. **Fraction of the batch that produced gradient, not only per-problem
   counts** — within a single problem some rollouts can be skipped while
   others contribute.
3. **Reward histogram.** With `success_reward_threshold = 1.0` (kept as-is,
   deliberately not lowered to 0.5 — see the thesis decision log), how many
   problems have `has_solution` depends strongly on that threshold. Logging
   the reward distribution is what makes the threshold choice defensible
   later instead of arbitrary.
4. **TF-specific**: `n_good` per step, and a **counter for fallback
   activations**. Fallback can silently mask judge failure (the math `idx8`
   case). At test-time scale that was catchable by reading logs by hand; at
   train-time scale it must be a counter.

### Paper phrasing for the effective-N number

> "Of the N problems in the training set, M produced at least one gradient
> step; K were skipped because the student already solved them, and L
> because no valid distillation target was available. All learning-rate and
> step budgets below refer to M."

If M turns out to be much smaller than N, that is not an embarrassment —
it *supports* the thesis's own methodology: it is direct evidence that
frontier filtering is necessary, and it explains why the test-time
experiments selected problems by model pass-rate rather than by contest
difficulty label. Honest and argumentative at the same time.

(Historical note: an early thesis run diagnosed "distillation-only means a
successful rollout does not reinforce itself" as a possible cause of a
null result — see `project_ttt_pipeline_milestone` memory, Phase 2a
2026-05-31. That concern is largely defused in the teacher-first setup,
where the distillation target is a *verified-correct teacher trajectory*, so
there is a direct mechanism raising the probability of correct solutions.
Still worth keeping in mind if training goes flat.)

## 3. Dataset split — no new loader needed for either domain

Both `load_lcbv6_split()` and `load_math_split()` return one flat split (no
existing train/held-out division) — same pattern, split in the new script via
deterministic seeded shuffle.

**MATH**:
- **Train subset**: first ~100 rows (pilot scale) after `shuffle(seed=args.seed)`.
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

**CODE**:
- `load_lcbv6_split()` returns LCBv6 `test6` (~175 rows per earlier thesis
  notes — **print `len(dataset)` first and confirm**, do not assume). This
  pool is much smaller than MATH-500's 500, so split sizes need to be picked
  to leave both a usable train set and enough held-out rows for a
  statistically meaningful pass@k delta — e.g. ~100 train / ~75 held-out as a
  starting point, adjust once the real count is known.
- **Open decision (surface to user): filter to frontier or use the raw
  split?** The thesis's test-time work deliberately cherry-picked "frontier"
  problems (model pass-rate strictly between 0 and 1) to guarantee headroom
  to learn on. Train-time here should probably **NOT** cherry-pick the same
  way for the *train* set — a fair train-time story trains on whatever's in
  the split (including some the model already solves at pass=1 and some it
  never solves), same as how the original paper's own train-time regime
  works, and is a stronger generalization claim than "we only trained on
  problems we knew had headroom." Held-out set should also be unfiltered for
  the same reason. Flag `--filter_frontier` as an optional CLI flag (default
  off) rather than silently baking cherry-picking in.
- No OOD split defined yet for code (no second code benchmark wired in
  `evaluator.py`). Stretch goal, not a pilot blocker — in-distribution
  held-out LCBv6 is the primary metric.

## 4. New file: `10_teacher_first_train.py`

### 4a. Main loop structure (replaces 09's single-problem `main()`)

```
load model + LoRA + optimizer (ONE state for the whole run — no re-init per problem)
domain = get_domain(args.domain)   # "code" or "math"
train_rows, holdout_rows = split_domain(domain, seed, n_train=args.n_train_problems)
ood_rows = load_aime_split() if domain.name == "math" else None   # code has no OOD split yet

PRE-eval: safe_evaluate_model on holdout_rows (aggregate pass_rate) + ood_rows (if any)
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
   cross-problem? **MATH: reset per problem** — exemplars are problem-specific
   (a good MATH-500 solution to problem A is meaningless as a few-shot for
   problem B). **CODE: worth trying persistent pool as a real ablation**
   (`--pool_mode {reset,persistent}`, default `reset` for parity with the
   test-time thesis mechanism, but run `persistent` too if time allows) —
   common algorithmic idioms (two-pointer, DP table, sliding window) may
   transfer usefully as few-shot exemplars across *different* LCBv6 problems
   in a way a MATH-500 solution never would; this is a free extra finding if
   it works, not a blocker if it doesn't. Implement the flag either way,
   default to reset, flag to user if the ablation seems worth prioritizing.
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

`--domain {code,math}` (**required, no default** — was wrongly fixed to math
in the original draft) `--model_name (default Qwen/Qwen3-4B)
--n_train_problems (100) --seed --steps_per_problem (1) --epochs (1)
--teacher_n (10) --teacher_temperature (1.0) --fewshot_option (good_only)
--pool_mode {reset,persistent} (default reset) --max_fewshot (3) --kl_topk
(20) --kl_alpha (1.0) --teacher_reg {none,trust_region} (default
trust_region; `none` == trust_region at alpha 1.0 == 09's current behavior,
kept as an explicit alias for reproducing the thesis runs)
--teacher_reg_alpha (0.01, paper's value; see §2b — NOT interchangeable with
an EMA rate) --judge {llm,difflib} (llm required for math; either ok
for code, matching thesis's judge-invariance finding) --judge_provider
--reprompt_template --eval_samples --max_new_tokens --filter_frontier
(flag, default off, code only) --wandb_project --no_wandb --save_adapter_path`

## 6. Logging (required)

- Per-problem: `n_good/n_bad/n_total`, mean judge verdict distribution
  (is_copy rate — watch this number, math pilot saw ~75% copy on too-hard
  problems; MATH-500 train subset should be easier so expect lower copy rate),
  loss, running problem index.
- PRE/POST aggregate, **logged separately per domain, never averaged
  together**: MATH → pass_rate on holdout MATH-500 subset AND on AIME2026
  (in-distribution vs OOD); CODE → pass_rate (pass@k) on holdout LCBv6 subset
  (no OOD split yet, see §3).
- Save `summary.json` per run with domain, eval set PRE/POST numbers, and the
  exact row indices/ids used for train/holdout split (reproducibility) —
  `unique_id` for math, `question_id`/`problem_id` for code (see
  `Domain.problem_id`).

## 7. Review checklist (Claude will check after implementation)

- `MathDomain.privileged_context` fix applied correctly (MATH runs only),
  judge still guards against copy (checked via `_JUDGE_PROMPT_MATH` still
  invoked with `judge=llm` forced for math, not silently falling back to
  difflib). Not applicable to CODE runs — confirm no fix was needed/attempted
  there.
- Optimizer/LoRA state genuinely persists across the problem loop (no
  accidental re-init hiding inside a helper reused from 09) — check for both
  domains, not just the one first tested.
- **Trust-region reference is `q_θref`, not `π_θref`** (§2b.2): assert the
  adapter-disabled forward runs on the **teacher** message list (feedback +
  few-shot), NOT the student prompt. Cheap check: log both prefix token
  lengths — they must be equal, since it is the same context with the adapter
  toggled. If the ref prefix is shorter, it is running on the student prompt
  and the run is the Chen et al. variant, not SDPO's.
- **Skip accounting implemented per §2d, and compared ACROSS ARMS before any
  outcome claim.** The TF-only "no valid `y_good`" drop path means the arms
  can see different amounts of data; check `effective_problems_seen` for both
  arms first, and if they diverge materially, report the comparison at equal
  gradient-step count (stating that basis) rather than equal epochs.
- Trust-region actually applied: with `--teacher_reg_alpha 1.0` the loss must
  match the `none` path bit-for-bit (sanity that lerp is wired correctly);
  with alpha 0.01 the teacher logits must sit far closer to the
  adapter-disabled forward than to the current-weights forward.
- Held-out set has zero overlap with train subset (assert on `unique_id` for
  math, `problem_id` for code).
- Token-alignment in `teacher_first_step` still correct when problem changes
  between calls (prefix lengths change every problem, not just every step) —
  code prompts (full `question_content`) can be much longer/more variable in
  length than math problems, worth an explicit length-distribution check.
- PRE/POST eval use identical held-out rows and identical sampling (seed) so
  deltas are attributable to training, not eval noise.
- If `--filter_frontier` was left off for code (the recommended default),
  confirm the held-out set legitimately contains a mix of pass=0/mixed/pass=1
  problems — a held-out set that's accidentally all-frontier or all-trivial
  would misrepresent the generalization claim.

## Links
- `09_teacher_first.py` — mechanism being extended
- `domains.py` — `CodeDomain` (no fix needed), `MathDomain` (fix target)
- Thesis wiki `con_teacher_first_judge`, `syn_teacher_first_impl_spec`,
  `syn_core_result` (CODE test-time result this train-time run extends),
  `syn_math_pilot` (background for the MATH arm)
- Assistant memory `project_thesis_paper_scaleup` — why this now targets a
  paper (COLM 2027 / ICLR 2027 workshop), not just an advisor side-ask
