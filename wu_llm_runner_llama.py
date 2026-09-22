"""
Wu et al. — LLM Runner cho Llama 4 Scout qua OpenRouter

Config:
  - Model:    meta-llama/llama-4-scout
  - Env:      OPENROUTER_API_KEY (chung voi Qwen/Claude)
  - Khong thinking mode
  - Khong suffix dac biet
  - Pricing (thang 9/2026): ~$0.08/M input, $0.30/M output
"""

import os
import time
import json
import threading
from pathlib import Path
from openai import OpenAI

from wu_input_formatter import annotate_bug_lines, build_prompt, to_zero_indexed

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL           = "https://openrouter.ai/api/v1"
MODEL_ID           = "meta-llama/llama-4-scout"
FILE_SLUG          = "llama4_scout"

PRICE_INPUT_PER_M  = 0.08
PRICE_OUTPUT_PER_M = 0.30

N_PATCHES     = 5
TEMPERATURE   = 0.8

# Adaptive max_tokens strategy:
#   - Bat dau MAX_TOKENS_SMALL (1500 = du cho ~90% patch typical)
#   - Neu 3 candidate lien tiep bi truncate (finish_reason='length') hoac
#     khong co code fence → escalate len MAX_TOKENS_LARGE
#   - Reset lai SMALL sau khi mot candidate thanh cong
MAX_TOKENS_SMALL      = 1500
MAX_TOKENS_LARGE      = 4096
ESCALATE_AFTER_BAD    = 3

MAX_RETRIES   = 3
RETRY_DELAY   = 2

EXTRA_BODY    = {}
EXTRA_HEADERS = {
    "HTTP-Referer": "https://github.com/patcheval/wu-adapter",
    "X-Title":      "PatchEval Wu Adapter (Llama 4 Scout)",
}
PROMPT_SUFFIX = ""

SYSTEM_PROMPT = (
    "You are a senior security engineer specialized in patching CVEs. "
    "You output only source code inside a single fenced code block."
)

_STATS_LOCK = threading.Lock()
_STATS = {"input_tokens": 0, "output_tokens": 0, "attempts": 0, "latency_s": 0.0}
_STATS_FILE = None


def set_stats_file(path):
    global _STATS_FILE
    _STATS_FILE = Path(path) if path else None


def _record_usage(input_tokens, output_tokens, elapsed):
    with _STATS_LOCK:
        _STATS["input_tokens"] += input_tokens
        _STATS["output_tokens"] += output_tokens
        _STATS["attempts"] += 1
        _STATS["latency_s"] += elapsed


def flush_stats():
    if _STATS_FILE is None:
        return
    stats = dict(_STATS)
    stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]
    stats["cost_usd"] = (
        stats["input_tokens"] / 1_000_000 * PRICE_INPUT_PER_M +
        stats["output_tokens"] / 1_000_000 * PRICE_OUTPUT_PER_M
    )
    stats["model_id"] = MODEL_ID
    with open(_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n[STATS] input={stats['input_tokens']:,} output={stats['output_tokens']:,} "
          f"attempts={stats['attempts']} latency={stats['latency_s']:.1f}s "
          f"cost=${stats['cost_usd']:.6f}")
    print(f"        Saved: {_STATS_FILE}")


def get_client():
    if not OPENROUTER_API_KEY:
        raise SystemExit(
            "OPENROUTER_API_KEY chua duoc set. "
            "PowerShell: $env:OPENROUTER_API_KEY='sk-or-v1-...'"
        )
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=BASE_URL,
        default_headers=EXTRA_HEADERS,
    )


def _call_llm(client, prompt, max_tokens=MAX_TOKENS_SMALL):
    """Goi API. Tra ve (content, finish_reason).
    finish_reason == 'length' nghia la bi truncate — can tang max_tokens."""
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt + PROMPT_SUFFIX},
                ],
                extra_body=EXTRA_BODY,
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
            )
            elapsed = time.perf_counter() - t0

            usage = getattr(resp, "usage", None)
            if usage:
                _record_usage(
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    elapsed=elapsed,
                )

            choice = resp.choices[0]
            content = choice.message.content
            finish_reason = getattr(choice, "finish_reason", "stop") or "stop"
            if content is None:
                raise RuntimeError("empty content")
            return content.strip(), finish_reason
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    [retry {attempt+1}/{MAX_RETRIES}] {e}")
                time.sleep(RETRY_DELAY)
            else:
                raise


def generate_candidates_llama(
    cve_id, cve_desc, cwe_list, buggy_snippet, buggy_lines_1idx,
    file_path, lang, n_patches=N_PATCHES, verbose=False,
    max_tokens_start=None,
):
    """Sinh N candidate patches cho 1 CVE bang Llama 4 Scout.

    Adaptive max_tokens:
      - Bat dau max_tokens_start (default MAX_TOKENS_SMALL)
      - Neu ESCALATE_AFTER_BAD candidate lien tiep bi truncate hoac khong co
        code fence → tang len MAX_TOKENS_LARGE
      - Pass max_tokens_start=MAX_TOKENS_LARGE de skip adaptive va dung LARGE
        tu dau (dung khi retry cho CVE co snippet dai)
    """
    if max_tokens_start is None:
        max_tokens_start = MAX_TOKENS_SMALL

    client = get_client()
    buggy_0idx = to_zero_indexed(buggy_lines_1idx)
    annotated  = annotate_bug_lines(buggy_snippet, buggy_0idx, lang)
    prompt     = build_prompt(cve_desc, cwe_list, annotated, lang, file_path)

    if verbose:
        print(f"  [{cve_id}] model={MODEL_ID} prompt_len~{len(prompt)//4} tokens, "
              f"generating {n_patches} (max_tokens starts at {max_tokens_start})...")

    raw_outputs = []
    seen = set()
    attempts = 0
    consecutive_bad = 0
    current_max_tokens = max_tokens_start
    escalated = (max_tokens_start >= MAX_TOKENS_LARGE)  # neu da start LARGE thi khong escalate nua

    while len(raw_outputs) < n_patches and attempts < n_patches * 3:
        attempts += 1
        try:
            out, finish_reason = _call_llm(client, prompt, max_tokens=current_max_tokens)
        except Exception as e:
            print(f"    [FAIL] LLM call: {e}")
            break

        # Detect "bad" output: truncated OR no code fence
        has_code_fence = "```" in out
        is_bad = (finish_reason == "length") or (not has_code_fence)

        if is_bad:
            consecutive_bad += 1
            if verbose:
                reason = "truncated" if finish_reason == "length" else "no-code-fence"
                print(f"    attempt {attempts}: BAD ({reason}, consecutive_bad={consecutive_bad})")

            # Escalate max_tokens neu qua nguong bad
            if (consecutive_bad >= ESCALATE_AFTER_BAD
                    and not escalated
                    and current_max_tokens < MAX_TOKENS_LARGE):
                print(f"    [ESCALATE {cve_id}] max_tokens {current_max_tokens} → {MAX_TOKENS_LARGE} "
                      f"sau {consecutive_bad} bad candidate lien tiep")
                current_max_tokens = MAX_TOKENS_LARGE
                escalated = True
                consecutive_bad = 0
            continue

        # Good output → reset counter, dedup, save
        consecutive_bad = 0
        key = hash(out)
        if key in seen:
            if verbose:
                print(f"    attempt {attempts}: DUPLICATE, skip")
            continue
        seen.add(key)
        raw_outputs.append(out)
        if verbose:
            first_line = out.splitlines()[0] if out.splitlines() else "<empty>"
            print(f"    candidate {len(raw_outputs)}: {first_line[:80]}")

    if verbose and escalated:
        print(f"    [note] {cve_id} da escalate max_tokens → xet lai buggy snippet dai co the")
    return raw_outputs
