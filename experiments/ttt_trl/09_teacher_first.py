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
  * reference_mode flag {best_in_batch,ground_truth,none}, default best_in_batch
  * KL reuse: trl==1.6.0 trl.experimental.sdpo.loss_utils
    .compute_topk_self_distillation_loss (reverse KL alpha=1.0, top-k, add_tail)
    -- the SAME function SDPOTrainer uses, so the two arms stay comparable.)

TRL PIN: trl==1.6.0 (colab/requirements_colab.txt). In 1.6.0 the divergence
helpers are module-level functions in trl.experimental.sdpo.loss_utils. There is
NO trl.experimental.self_distillation / SelfDistillationMixin in 1.6.0 -- do not
import from that path (it crashes on Colab).
"""

from __future__ import annotations

import argparse
import difflib
import gc
import hashlib
import importlib
import json
import math
import os
import pathlib
import random
import re
import statistics
import sys
import time
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import Dataset  # noqa: F401  (kept for parity / potential dataset use)
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
_apply_lora = _m07._apply_lora
_extract_text = _m07._extract_text
REPROMPT_TEMPLATES = _m07.REPROMPT_TEMPLATES

from experiments.ttt_trl.domains import Domain, get_domain


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
    domain: Domain | None = None,
    model_name: str = "",
    thinking: bool = False,
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
    # _build_messages appends the domain directive (code = ```python``` block;
    # math = \boxed{} final answer) and Gemma-4 <|think|> system when needed.
    return _build_messages(teacher_user, domain, model_name, thinking)


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
    domain: Domain | None = None,
    model_name: str = "",
    thinking: bool = False,
    top_p: float = 1.0,
    top_k: int = 0,
) -> tuple[list[str], str]:
    """Sample N teacher completions from prompt + feedback + few-shot exemplars."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    messages = _build_teacher_messages(
        question_content, feedback_text, good_pool, bad_pool, option, max_fewshot,
        reprompt_preset, domain, model_name, thinking,
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
        for i in range(out.shape[0]):
            completions.append(tokenizer.decode(out[i, prompt_tokens:], skip_special_tokens=True))

    if was_training:
        model.train()
    return completions, prompt_text


# --------------------------------------------------------------------------- #
# Optional LLM judge. Alternative to the difflib similarity step ONLY -- the
# verifier (evaluate_solution) still decides correctness. The judge runs only on
# verifier-correct trajectories and decides, among them, good (independent +
# clear reasoning) vs bad (copy of reference / poor).
#
# Providers are tried as a CHAIN so a quota-exhausted primary falls through to a
# free backup, and finally to difflib -- the run never crashes:
#   - gemini     : google-genai SDK. key GEMINI_API_KEY / GOOGLE_API_KEY.
#   - groq       : OpenAI-compatible REST (raw HTTP). key GROQ_API_KEY.
#   - openrouter : OpenAI-compatible REST (raw HTTP). key OPENROUTER_API_KEY.
# groq/openrouter use only stdlib urllib -> no extra pip dependency.
# --------------------------------------------------------------------------- #
class JudgeUnavailable(Exception):
    """Raised when the LLM judge cannot return a verdict -> caller falls back to difflib."""


class _ProviderError(Exception):
    """Normalized provider error carrying an HTTP-ish status code (or None)."""

    def __init__(self, code: int | None, message: str):
        self.code = code
        super().__init__(message)


_GENAI_CLIENT = None  # lazily created, reused across calls (one gemini client per run)

# Default model per provider (used for fallback providers, and for the primary
# provider when its model is left empty).
_PROVIDER_DEFAULT_MODEL = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    "zai": "glm-4.5-flash",
}

# Env var(s) holding each provider's API key (first non-empty wins).
_PROVIDER_ENV = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "zai": ("ZAI_API_KEY",),
}

# OpenAI-compatible chat-completions endpoints (gemini uses its own SDK).
_OPENAI_COMPAT_URL = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "zai": "https://api.z.ai/api/paas/v4/chat/completions",
}


def _provider_key(provider: str) -> str | None:
    for env in _PROVIDER_ENV.get(provider, ()):
        val = os.environ.get(env)
        if val:
            return val
    return None


def _provider_has_key(provider: str) -> bool:
    return _provider_key(provider) is not None


_JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_copy": {"type": "BOOLEAN"},
        "reasoning_quality": {"type": "INTEGER"},
    },
    "required": ["is_copy", "reasoning_quality"],
}

_JUDGE_PROMPT = """You are judging a CANDIDATE solution to a competitive-programming problem.

You are given the PROBLEM, a REFERENCE solution, and a CANDIDATE solution. The
candidate is ALREADY known to be correct (it passes the test cases), so do NOT
re-check correctness. Decide only two things:

  1. is_copy: Is the candidate essentially a copy of the reference solution --
     i.e. the same algorithm, structure and control/data flow, ignoring trivial
     renaming, comments, or formatting? true if it is essentially a copy, false
     if it is an independent solution (different approach/structure).
  2. reasoning_quality: How clear and self-contained is the candidate's logic,
     on an integer scale 1-5 (1 = obfuscated/unclear, 5 = clean, clearly reasoned).

PROBLEM:
{problem}

REFERENCE SOLUTION:
```python
{reference}
```

CANDIDATE SOLUTION:
```python
{candidate}
```

Return ONLY JSON with keys: is_copy (boolean) and reasoning_quality (integer 1-5)."""


# MATH judge: string/code similarity is meaningless for short numeric answers.
# We are in the LEAK regime — the candidate was shown the correct answer via
# feedback. The judge must decide whether the candidate genuinely DERIVED the
# answer (step-by-step reasoning, typically inside its <think> block) or just
# COPIED/stated the leaked answer with no real derivation.
#   is_copy=true  -> memorized the leaked answer  -> verdict "bad"
#   is_copy=false -> derived it via real reasoning -> verdict "good"
# The CANDIDATE text MUST include the candidate's <think> block (thinking-on),
# since that is where the derivation lives.
_JUDGE_PROMPT_MATH = """You are judging a CANDIDATE solution to a math problem.

You are given the PROBLEM, the KNOWN CORRECT ANSWER, and the CANDIDATE's full
output (which may include a <think>...</think> reasoning block followed by a final
\\boxed{{answer}}). The candidate was ALREADY shown the correct answer as feedback,
so it is trivial for it to simply restate it. Decide only two things:

  1. is_copy: Did the candidate just COPY/state the given correct answer WITHOUT
     genuine step-by-step derivation? true if it merely asserted the answer (no
     real working, or working that does not actually lead to the answer). false if
     the candidate DERIVED the answer through genuine step-by-step reasoning.
  2. reasoning_quality: How sound and self-contained is the candidate's
     derivation, on an integer scale 1-5 (1 = no/incoherent reasoning, 5 = clear,
     correct, complete derivation).

PROBLEM:
{problem}

KNOWN CORRECT ANSWER:
{reference}

CANDIDATE OUTPUT (including any <think> reasoning):
{candidate}

Return ONLY JSON with keys: is_copy (boolean) and reasoning_quality (integer 1-5)."""


def _get_genai_client():
    """Create (once) and return a google-genai client. Reads the API key from env."""
    global _GENAI_CLIENT
    if _GENAI_CLIENT is None:
        from google import genai  # imported lazily so --judge difflib needs no SDK

        _GENAI_CLIENT = genai.Client()  # picks up GEMINI_API_KEY / GOOGLE_API_KEY
    return _GENAI_CLIENT


def _parse_judge_json(raw: str) -> dict:
    """Parse the judge's JSON, tolerating code fences / surrounding prose."""
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)  # first {...} block
        if match:
            return json.loads(match.group(0))
        raise


def _call_gemini(model: str, prompt: str) -> str:
    """Gemini via google-genai. Returns raw JSON text. Raises _ProviderError."""
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types

    try:
        client = _get_genai_client()
    except Exception as exc:  # noqa: BLE001  (missing key / bad SDK)
        raise _ProviderError(None, f"gemini client unavailable: {exc}") from exc

    config = genai_types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=_JUDGE_SCHEMA,
    )
    try:
        resp = client.models.generate_content(model=model, contents=prompt, config=config)
    except genai_errors.APIError as exc:
        raise _ProviderError(getattr(exc, "code", None), str(exc)) from exc
    return resp.text or ""


def _call_openai_compat(provider: str, model: str, prompt: str) -> str:
    """Groq/OpenRouter via OpenAI-compatible REST (stdlib urllib). Raises _ProviderError."""
    import urllib.error
    import urllib.request

    key = _provider_key(provider)
    if not key:
        raise _ProviderError(None, f"{provider}: no API key in env")
    url = _OPENAI_COMPAT_URL[provider]
    payload_body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    if provider == "zai":
        # z.ai rejects response_format={"type":"json_object"} (HTTP 400). The judge
        # prompt already enforces "Return ONLY JSON", so drop the field for z.ai.
        payload_body.pop("response_format", None)
    body = json.dumps(payload_body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        # urllib's default UA ("Python-urllib/..") is Cloudflare-blocked by Groq (403/1010).
        "User-Agent": "ttt-sdpo-judge/1.0",
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/sdpo/ttt"  # OpenRouter attribution (optional)
        headers["X-Title"] = "ttt-sdpo-judge"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:200] if hasattr(exc, "read") else ""
        raise _ProviderError(exc.code, f"{provider} HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise _ProviderError(None, f"{provider} network error: {exc}") from exc
    try:
        return payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise _ProviderError(None, f"{provider} unexpected response shape: {exc}") from exc


def _call_provider(provider: str, model: str, prompt: str) -> str:
    if provider == "gemini":
        return _call_gemini(model, prompt)
    if provider in _OPENAI_COMPAT_URL:
        return _call_openai_compat(provider, model, prompt)
    raise _ProviderError(None, f"unknown provider {provider!r}")


def _build_provider_chain(
    provider: str,
    model: str,
    fallback_providers: list[str] | None,
    provider_models: dict[str, str] | None,
) -> list[tuple[str, str]]:
    """Primary first, then each fallback provider that actually has an API key."""
    primary_model = model or _PROVIDER_DEFAULT_MODEL.get(provider, model)
    chain: list[tuple[str, str]] = [(provider, primary_model)]
    seen = {provider}
    for prov in fallback_providers or []:
        if prov in seen or not _provider_has_key(prov):
            continue
        seen.add(prov)
        mdl = (provider_models or {}).get(prov) or _PROVIDER_DEFAULT_MODEL.get(prov, primary_model)
        chain.append((prov, mdl))
    return chain


def llm_judge(
    problem_text: str,
    reference_code: str,
    candidate_code: str,
    model: str,
    provider: str = "gemini",
    fallback_providers: list[str] | None = None,
    provider_models: dict[str, str] | None = None,
    max_retries: int = 4,
    base_delay: float = 2.0,
    domain: str = "code",
) -> dict:
    """
    Ask an LLM whether `candidate_code` is a copy of `reference_code` and how clear
    its reasoning is. temperature=0, structured JSON output.

    Tries providers as a chain: `provider` (primary) first, then any of
    `fallback_providers` that have an API key set. Within a provider, retries on
    429/503 with exponential backoff only when it's the LAST provider in the chain
    (otherwise a 429 immediately switches to the next provider, so a quota-exhausted
    primary doesn't waste time waiting). Raises JudgeUnavailable if the whole chain
    fails -> caller falls back to difflib (the run never crashes).

    Returns {"is_copy": bool, "reasoning_quality": int(1-5), "verdict": "good"|"bad",
             "raw": <raw json str>, "provider": str, "model": str}.
    verdict good := (not is_copy) and reasoning_quality >= 3.
    """
    # MATH: derive-vs-copy judge (reference = known correct answer, candidate =
    # full output incl <think>). CODE: copy-of-reference-solution judge.
    template = _JUDGE_PROMPT_MATH if domain in ("math", "aime") else _JUDGE_PROMPT
    prompt = template.format(
        problem=problem_text[:8000],  # cap problem text to keep tokens bounded
        reference=reference_code,
        candidate=candidate_code,
    )
    chain = _build_provider_chain(provider, model, fallback_providers, provider_models)

    last_exc: Exception | None = None
    for ci, (prov, mdl) in enumerate(chain):
        is_last = ci == len(chain) - 1
        for attempt in range(max_retries):
            try:
                raw = _call_provider(prov, mdl, prompt)
                data = _parse_judge_json(raw)
                is_copy = bool(data["is_copy"])
                rq = int(data["reasoning_quality"])
                verdict = "good" if (not is_copy and rq >= 3) else "bad"
                return {
                    "is_copy": is_copy,
                    "reasoning_quality": rq,
                    "verdict": verdict,
                    "raw": raw,
                    "provider": prov,
                    "model": mdl,
                }
            except _ProviderError as exc:
                last_exc = exc
                # Wait-and-retry only if it's worth it: last provider in the chain
                # (no alternative), or a transient 503. A 429 on a non-last provider
                # switches immediately to the next provider.
                retryable = exc.code in (429, 503)
                if retryable and (is_last or exc.code == 503) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"[llm-judge] {prov} {exc.code}; retry {attempt + 1}/{max_retries} "
                          f"in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                print(f"[llm-judge] {prov} failed ({exc}); "
                      f"{'switching provider' if not is_last else 'chain exhausted'}")
                break
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                last_exc = exc  # malformed/empty JSON -> try next provider
                print(f"[llm-judge] {prov} bad JSON ({type(exc).__name__}); "
                      f"{'switching provider' if not is_last else 'chain exhausted'}")
                break

    raise JudgeUnavailable(f"all judge providers failed (last: {type(last_exc).__name__}: {last_exc})")


def _cached_llm_judge(
    cache: dict,
    problem_id: str,
    problem_text: str,
    reference_code: str,
    candidate_code: str,
    model: str,
    provider: str = "gemini",
    fallback_providers: list[str] | None = None,
    provider_models: dict[str, str] | None = None,
    domain: str = "code",
) -> dict:
    """
    Judge once per (problem_id, candidate_code). The TTT loop collapses to near-
    identical trajectories, so caching keeps real API calls to ~10-20 per run.
    Only successful judgments are cached. Logs every fresh judgment for repro.
    """
    key = hashlib.sha256(f"{problem_id}\x00{candidate_code}".encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]
    result = llm_judge(
        problem_text, reference_code, candidate_code, model,
        provider=provider, fallback_providers=fallback_providers,
        provider_models=provider_models, domain=domain,
    )
    cache[key] = result
    print(f"[llm-judge] problem_id={problem_id} provider={result['provider']} "
          f"verdict={result['verdict']} raw={result['raw']}")
    return result


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
    judge: str = "difflib",
    judge_model: str | None = None,
    judge_cache: dict | None = None,
    problem_id: str = "",
    problem_text: str = "",
    judge_provider: str = "gemini",
    judge_fallback_providers: list[str] | None = None,
    judge_provider_models: dict[str, str] | None = None,
    domain: Domain | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """
    good := correct AND independent ; bad := copy of reference (or wrong in 'none' mode).

    Independence test (the good/bad decision among verifier-correct trajectories):
      - judge="difflib" (default): normalized-code similarity vs the reference.
      - judge="llm": Gemini decides copy/independence + reasoning quality, run ONLY
        on verifier-correct trajectories; incorrect trajectories are ignored. The
        difflib similarity is still computed (for logging + per-trajectory fallback
        if the API fails). Either way the verifier alone decides correctness.

    Fallback (spec): if the independence filter removes every good trajectory but at
    least one correct (score>=1.0) trajectory exists, keep the best-scoring correct
    one so the good pool never collapses to empty purely due to the copy filter.
    """
    if judge_cache is None:
        judge_cache = {}
    if domain is None:
        domain = get_domain("code")
    scored: list[tuple[str, float]] = []
    for t in trajectories:
        code = _extract_text(t)
        try:
            s = float(domain.evaluate(code, row, split="train")["score"])
        except Exception as exc:  # noqa: BLE001
            print(f"[filter] eval error: {exc}")
            s = 0.0
        scored.append((code, s))

    ref, best_idx = get_reference_text(reference_mode, row, scored)

    good: list[dict] = []
    bad: list[dict] = []
    sims: list[float] = []
    judge_calls = 0
    judge_fallbacks = 0
    use_llm = judge == "llm" and reference_mode != "none" and bool(ref)
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
        elif use_llm:
            # Verifier decides correctness; LLM judge decides good/bad ONLY among
            # the correct ones. Incorrect trajectories are ignored (neither). The
            # reference trajectory itself is independent by construction.
            if s >= 1.0:
                if i == best_idx:
                    good.append({"code": code, "score": s})
                else:
                    try:
                        verdict = _cached_llm_judge(
                            judge_cache, problem_id, problem_text, ref, code, judge_model,
                            provider=judge_provider,
                            fallback_providers=judge_fallback_providers,
                            provider_models=judge_provider_models,
                            domain=domain.name,
                        )["verdict"]
                        judge_calls += 1
                        (good if verdict == "good" else bad).append({"code": code, "score": s})
                    except JudgeUnavailable as exc:
                        judge_fallbacks += 1
                        print(f"[llm-judge] fallback to difflib for one trajectory: {exc}")
                        if sim < sim_threshold:
                            good.append({"code": code, "score": s})
                        else:
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
        "judge": judge,
        "judge_calls": judge_calls,
        "judge_fallbacks": judge_fallbacks,
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

    # NOTE: apply_chat_template(return_tensors="pt") returns a BatchEncoding (dict)
    # in this transformers version, not a bare tensor -> .shape fails. Render to
    # text then tokenize, SAME pattern as teacher_generate so prefix tokenization
    # stays consistent between generation and the KL step.
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
                  f"t_slice={tuple(t_slice.shape)} topk={kl_topk} KL(token-mean)={kl_val:.6f}")
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
    parser = argparse.ArgumentParser(description="Teacher-first judge-filtered TTT-SDPO (code or math).")
    parser.add_argument("--domain", type=str, default="code", choices=["code", "math", "aime"],
                        help="Task domain: code (LCBv6), math (MATH-500), or aime (AIME 2026). "
                             "For math/aime, recommend --reference_mode ground_truth and --thinking.")
    parser.add_argument("--problem_index", type=int, default=23, help="LCBv6 index (P0 frontier=23).")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--teacher_n", type=int, default=10, help="Teacher samples per step.")
    parser.add_argument("--teacher_temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Sampling top_p (Gemma-4 card recommends 0.95; default 1.0 = Qwen behavior).")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Sampling top_k (0 = omit; Gemma-4 card recommends 64).")
    parser.add_argument("--sim_threshold", type=float, default=0.9)
    parser.add_argument("--fewshot_option", type=str, default="good_only",
                        choices=["good_only", "good_bad"])
    parser.add_argument("--max_fewshot", type=int, default=3)
    parser.add_argument("--reference_mode", type=str, default="best_in_batch",
                        choices=["best_in_batch", "ground_truth", "none"],
                        help="Similarity reference for the copy-filter (spec 4.1): "
                             "best_in_batch=code P0, ground_truth=math P2, none=ablation.")
    parser.add_argument("--judge", type=str, default="difflib", choices=["difflib", "llm"],
                        help="Independence judge among verifier-correct trajectories: "
                             "difflib (default, current behavior) or llm.")
    parser.add_argument("--judge_provider", type=str, default="gemini",
                        choices=["gemini", "groq", "openrouter", "zai"],
                        help="Primary LLM-judge provider for --judge llm.")
    parser.add_argument("--judge_fallback", type=str, nargs="*",
                        default=["gemini", "groq", "openrouter"],
                        choices=["gemini", "groq", "openrouter", "zai"],
                        help="Fallback providers tried (in order) if the primary is "
                             "quota-exhausted/unavailable; only those with an API key are used. "
                             "Final fallback is always difflib.")
    parser.add_argument("--judge_model", type=str, default="gemini-2.5-flash",
                        help="Gemini model for --judge llm (free-tier flash default).")
    parser.add_argument("--judge_groq_model", type=str, default="llama-3.3-70b-versatile",
                        help="Groq model (when groq is used as primary or fallback).")
    parser.add_argument("--judge_openrouter_model", type=str,
                        default="meta-llama/llama-3.3-70b-instruct:free",
                        help="OpenRouter model (when openrouter is used as primary or fallback).")
    parser.add_argument("--judge_zai_model", type=str, default="glm-4.5-flash",
                        help="z.ai/GLM model when zai is primary or fallback.")
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

    domain = get_domain(args.domain)

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {device_name}")
    print(f"Total VRAM: {total_vram_gb:.2f} GB")
    print(f"Domain: {args.domain} | Seed: {args.seed} | reference_mode={args.reference_mode} "
          f"| fewshot={args.fewshot_option}")

    # Resolve the LLM-judge provider chain once (primary + fallbacks that have keys).
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
        print(f"Judge: llm | provider chain: {[f'{p}:{m}' for p, m in chain]}")
        if usable:
            print(f"[llm-judge] usable providers (key present): {usable}")
        else:
            print("[llm-judge][WARN] no provider has an API key set -- judge calls will "
                  "fall back to difflib. Set one of: GEMINI_API_KEY / GROQ_API_KEY / "
                  "OPENROUTER_API_KEY.")
    else:
        print(f"Judge: {args.judge}")

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=f"teacherfirst-{args.model_name.split('/')[-1]}-{args.domain}-idx{args.problem_index}"
                 f"-{args.max_steps}step-{args.fewshot_option}-{args.judge}-{args.reprompt_template}",
            config=vars(args),
        )

    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    lcb = domain.load_split()
    row = lcb[args.problem_index]
    problem_id = domain.problem_id(row) or f"idx_{args.problem_index}"
    difficulty = domain.difficulty(row)
    question_content = domain.problem_text(row)
    print(f"\n=== PROBLEM idx {args.problem_index}: {problem_id} ({difficulty}) ===")
    print(f"[row] keys = {list(row.keys())}")

    tokenizer = _prepare_tokenizer(args.model_name, thinking=args.thinking)

    total_start = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model = _apply_lora(model, args.lora_r, model_name=args.model_name)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )

    # ----- PRE-eval (student-only) -----
    print(f"\n[PRE-eval] sampling {args.eval_samples} student solutions ...")
    pre_eval = safe_evaluate_model(
        model, tokenizer, row, args.eval_samples, args.max_new_tokens,
        label="PRE", domain=domain, model_name=args.model_name, thinking=args.thinking,
        top_p=args.top_p, top_k=args.top_k,
    )
    print(f"[PRE-eval] pass_rate={pre_eval['pass_rate']:.3f} mean={pre_eval['mean_score']:.3f} "
          f"max={pre_eval['max_score']:.3f} greedy={pre_eval['greedy_score']:.3f}")

    base_hint = domain.privileged_context(row)
    student_messages = _build_messages(question_content, domain, args.model_name, args.thinking)

    good_pool: list[dict] = []
    bad_pool: list[dict] = []
    judge_cache: dict[str, dict] = {}  # (problem_id, candidate_code) -> judgment
    step_records: list[dict] = []
    step_mean_rewards: list[float] = []

    print(f"\n[TTT] teacher-first on {problem_id} for {args.max_steps} steps ...")
    run_start = time.time()

    for step in range(args.max_steps):
        verbose = step == 0

        # Dynamic feedback from the student's CURRENT greedy attempt (reuse 07).
        greedy_eval = safe_evaluate_model(
            model, tokenizer, row, 0, args.max_new_tokens,
            label=f"STEP{step}", domain=domain,
            model_name=args.model_name, thinking=args.thinking,
            top_p=args.top_p, top_k=args.top_k,
        )
        dyn = build_dynamic_feedback(greedy_eval.get("greedy_code", ""), row, domain=domain)
        feedback_text = "\n\n".join(p for p in [base_hint, dyn] if p).strip()

        trajectories, teacher_prompt = teacher_generate(
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
            problem_id=str(problem_id), problem_text=question_content,
            judge_provider=args.judge_provider,
            judge_fallback_providers=args.judge_fallback,
            judge_provider_models=judge_provider_models,
            domain=domain,
        )
        good_pool = _update_pool(good_pool, good, args.pool_cap)
        bad_pool = _update_pool(bad_pool, bad, args.pool_cap)

        if good:
            teacher_messages = _build_teacher_messages(
                question_content, feedback_text, good_pool, bad_pool,
                args.fewshot_option, args.max_fewshot, args.reprompt_template, domain,
                args.model_name, args.thinking,
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
            "judge_calls": stats.get("judge_calls", 0),
            "judge_fallbacks": stats.get("judge_fallbacks", 0),
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
    post_eval = safe_evaluate_model(
        model, tokenizer, row, args.eval_samples, args.max_new_tokens,
        label="POST", domain=domain, model_name=args.model_name, thinking=args.thinking,
        top_p=args.top_p, top_k=args.top_k,
    )
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
