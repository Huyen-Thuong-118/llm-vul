"""
Wu et al. (ISSTA 2023) — LLM Runner (PatchEval-adapted)

Gọi DeepSeek V4-Flash với prompt Wu et al.-style, trả về N candidate patches
cho một CVE. Cấu hình quan trọng:
  - `"thinking":{"type":"disabled"}` (DeepSeek V4-Flash bật thinking mặc định,
    `reasoning_effort="none"` không đủ — đã confirm trong ways-of-working).
  - `temperature=0.8`: Wu et al. cần diversity giữa các candidate. Baseline
    PatchEval dùng temp=0 cho reproducibility, KHÔNG áp dụng ở đây.
  - `n_patches=5`: Wu paper dùng 10, ta dùng 5 để tiết kiệm quota + Docker time.
    Plausible@5 vẫn là metric hợp lệ.
"""

import os
import time
from openai import OpenAI

from wu_input_formatter import annotate_bug_lines, build_prompt, to_zero_indexed

# ── Config ────────────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL     = "deepseek-chat"
N_PATCHES         = 5
TEMPERATURE       = 0.8
MAX_TOKENS        = 2048
MAX_RETRIES       = 3
RETRY_DELAY       = 2

SYSTEM_PROMPT = (
    "You are a senior security engineer specialized in patching CVEs. "
    "You output only source code inside a single fenced code block."
)


# ── Client ────────────────────────────────────────────────────────────────────
def get_client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise SystemExit("DEEPSEEK_API_KEY chua duoc set.")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def _call_llm(client: OpenAI, prompt: str, model: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                extra_body={"thinking": {"type": "disabled"}},
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    [retry {attempt+1}/{MAX_RETRIES}] {e}")
                time.sleep(RETRY_DELAY)
            else:
                raise


def generate_candidates(
    cve_id: str,
    cve_desc: str,
    cwe_list: list[str],
    buggy_snippet: str,
    buggy_lines_1idx: list[int],
    file_path: str,
    lang: str,
    model: str = DEFAULT_MODEL,
    n_patches: int = N_PATCHES,
    verbose: bool = False,
) -> list[str]:
    """
    Sinh N candidate patches cho 1 CVE (Wu et al. plausible@k setting).

    Args:
        cve_id           : CVE identifier để log
        cve_desc         : description string từ dataset
        cwe_list         : list of CWE ids từ dataset (list(cwe_info.keys()))
        buggy_snippet    : `vul_func[0].snippet`
        buggy_lines_1idx : `vul_func[0].vul_localization[0].patch_lines`
                           (1-indexed relative to snippet)
        file_path        : `vul_func[0].file_path` — cần cho unified diff sau này
        lang             : "Python" | "Go" | "JavaScript" (case-insensitive)

    Returns:
        List[str] các RAW LLM outputs (mỗi cái là 1 code fence chứa fixed
        snippet). Chưa parse, chưa build diff — để wu_eval_runner tự xử.
    """
    client = get_client()
    buggy_0idx = to_zero_indexed(buggy_lines_1idx)
    annotated  = annotate_bug_lines(buggy_snippet, buggy_0idx, lang)
    prompt     = build_prompt(cve_desc, cwe_list, annotated, lang, file_path)

    if verbose:
        print(f"  [{cve_id}] prompt tokens ~= {len(prompt)//4}, generating {n_patches}...")

    raw_outputs = []
    seen = set()
    attempts = 0
    while len(raw_outputs) < n_patches and attempts < n_patches * 3:
        attempts += 1
        try:
            out = _call_llm(client, prompt, model)
        except Exception as e:
            print(f"    [FAIL] LLM call: {e}")
            break
        # Dedup theo hash để tránh lãng phí Docker run trên candidate trùng
        key = hash(out)
        if key in seen:
            continue
        seen.add(key)
        raw_outputs.append(out)
        if verbose:
            first_line = out.splitlines()[0] if out.splitlines() else "<empty>"
            print(f"    candidate {len(raw_outputs)}: {first_line[:80]}")

    return raw_outputs


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not DEEPSEEK_API_KEY:
        print("Set DEEPSEEK_API_KEY truoc khi chay smoke test.")
        raise SystemExit(1)

    buggy = (
        "def read_file(user_path):\n"
        "    full_path = \"/data/\" + user_path\n"
        "    return open(full_path).read()\n"
    )
    outs = generate_candidates(
        cve_id="TEST-0001",
        cve_desc="Path traversal via unchecked user input in file loader.",
        cwe_list=["CWE-22"],
        buggy_snippet=buggy,
        buggy_lines_1idx=[2],  # dòng thứ 2 = "full_path = ..."
        file_path="app/loader.py",
        lang="python",
        n_patches=3,
        verbose=True,
    )
    for i, o in enumerate(outs, 1):
        print(f"\n=== Candidate {i} ===\n{o}")
