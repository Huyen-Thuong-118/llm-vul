"""
Wu et al. (ISSTA 2023) — LLM Runner cho Qwen3-14B qua OpenRouter

TACH RIENG khoi wu_llm_runner.py (DeepSeek). File nay hardcode moi thu cho
Qwen3-14B, khong dinh dang qua model family de tranh conflict.

Config:
  - Provider: OpenRouter (base_url = https://openrouter.ai/api/v1)
  - Model:    qwen/qwen3-14b (non-thinking variant)
  - Env:      OPENROUTER_API_KEY
  - Thinking DISABLE 2 lop:
      1) extra_body = {"reasoning": {"enabled": False}}  (OpenRouter unified param)
      2) suffix "/no_think" vao user message              (Qwen3 native command)
"""

import os
import time
import json
import threading
from pathlib import Path
from openai import OpenAI

from wu_input_formatter import annotate_bug_lines, build_prompt, to_zero_indexed

# -- Hardcoded config for Qwen3-14B ------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL           = "https://openrouter.ai/api/v1"
MODEL_ID           = "qwen/qwen3-14b"
FILE_SLUG          = "qwen3_14b"     # dung trong ten file/dir output

# Pricing OpenRouter Qwen3-14B (USD per 1M tokens) — cap nhat neu thay doi
PRICE_INPUT_PER_M  = 0.06
PRICE_OUTPUT_PER_M = 0.24

N_PATCHES     = 5
TEMPERATURE   = 0.8
MAX_TOKENS    = 2048
MAX_RETRIES   = 3
RETRY_DELAY   = 2

EXTRA_BODY    = {"reasoning": {"enabled": False}}
EXTRA_HEADERS = {
    "HTTP-Referer": "https://github.com/patcheval/wu-adapter",
    "X-Title":      "PatchEval Wu Adapter (Qwen3-14B)",
}
PROMPT_SUFFIX = "\n\n/no_think"

SYSTEM_PROMPT = (
    "You are a senior security engineer specialized in patching CVEs. "
    "You output only source code inside a single fenced code block."
)


# -- Usage tracking (aggregate qua ca run) -----------------------------------
_STATS_LOCK = threading.Lock()
_STATS = {
    "input_tokens":  0,
    "output_tokens": 0,
    "attempts":      0,
    "latency_s":     0.0,
}
_STATS_FILE = None  # set boi eval runner


def set_stats_file(path):
    """Eval runner goi ham nay de flush stats sau khi xong."""
    global _STATS_FILE
    _STATS_FILE = Path(path) if path else None


def _record_usage(input_tokens: int, output_tokens: int, elapsed: float):
    with _STATS_LOCK:
        _STATS["input_tokens"] += input_tokens
        _STATS["output_tokens"] += output_tokens
        _STATS["attempts"] += 1
        _STATS["latency_s"] += elapsed


def flush_stats():
    """Ghi stats ra file JSON. Goi sau khi stage_generate xong."""
    if _STATS_FILE is None:
        return
    stats = dict(_STATS)
    stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]
    stats["cost_usd"] = (
        stats["input_tokens"]  / 1_000_000 * PRICE_INPUT_PER_M +
        stats["output_tokens"] / 1_000_000 * PRICE_OUTPUT_PER_M
    )
    stats["model_id"] = MODEL_ID
    with open(_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n[STATS] input={stats['input_tokens']:,} output={stats['output_tokens']:,} "
          f"attempts={stats['attempts']} latency={stats['latency_s']:.1f}s "
          f"cost=${stats['cost_usd']:.6f}")
    print(f"        Saved: {_STATS_FILE}")


# -- Client ------------------------------------------------------------------
def get_client() -> OpenAI:
    if not OPENROUTER_API_KEY:
        raise SystemExit(
            "OPENROUTER_API_KEY chua duoc set. "
            "PowerShell: $env:OPENROUTER_API_KEY='sk-or-v1-...' truoc khi chay."
        )
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=BASE_URL,
        default_headers=EXTRA_HEADERS,
    )


def _call_llm(client: OpenAI, prompt: str) -> str:
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
                max_tokens=MAX_TOKENS,
            )
            elapsed = time.perf_counter() - t0

            # Track usage
            usage = getattr(resp, "usage", None)
            if usage:
                _record_usage(
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    elapsed=elapsed,
                )

            content = resp.choices[0].message.content
            if content is None:
                raise RuntimeError("empty content (thinking mode leaked?)")
            return content.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    [retry {attempt+1}/{MAX_RETRIES}] {e}")
                time.sleep(RETRY_DELAY)
            else:
                raise


def generate_candidates_qwen(
    cve_id: str,
    cve_desc: str,
    cwe_list,
    buggy_snippet: str,
    buggy_lines_1idx,
    file_path: str,
    lang: str,
    n_patches: int = N_PATCHES,
    verbose: bool = False,
):
    """Sinh N candidate patches cho 1 CVE bang Qwen3-14B. Tra ve list raw outputs."""
    client = get_client()
    buggy_0idx = to_zero_indexed(buggy_lines_1idx)
    annotated  = annotate_bug_lines(buggy_snippet, buggy_0idx, lang)
    prompt     = build_prompt(cve_desc, cwe_list, annotated, lang, file_path)

    if verbose:
        print(f"  [{cve_id}] model={MODEL_ID} prompt_len~{len(prompt)//4} tokens, "
              f"generating {n_patches}...")

    raw_outputs = []
    seen = set()
    attempts = 0
    while len(raw_outputs) < n_patches and attempts < n_patches * 3:
        attempts += 1
        try:
            out = _call_llm(client, prompt)
        except Exception as e:
            print(f"    [FAIL] LLM call: {e}")
            break
        key = hash(out)
        if key in seen:
            continue
        seen.add(key)
        raw_outputs.append(out)
        if verbose:
            first_line = out.splitlines()[0] if out.splitlines() else "<empty>"
            print(f"    candidate {len(raw_outputs)}: {first_line[:80]}")

    return raw_outputs


if __name__ == "__main__":
    buggy = (
        "def read_file(user_path):\n"
        "    full_path = \"/data/\" + user_path\n"
        "    return open(full_path).read()\n"
    )
    outs = generate_candidates_qwen(
        cve_id="TEST-0001",
        cve_desc="Path traversal via unchecked user input in file loader.",
        cwe_list=["CWE-22"],
        buggy_snippet=buggy,
        buggy_lines_1idx=[2],
        file_path="app/loader.py",
        lang="python",
        n_patches=2,
        verbose=True,
    )
    for i, o in enumerate(outs, 1):
        print(f"\n=== Candidate {i} ===\n{o}")
