"""
Wu et al. (ISSTA 2023) — Input Format Adapter (PatchEval-adapted)

Chuyển buggy snippet sang định dạng comment BUG/FIXED (Codex-style của Wu et al.)
để đưa vào prompt cho LLM. Adapt cho Python / Go / JavaScript vì PatchEval
không có Java (Java là dataset gốc của paper — Vul4J/VJBench, nằm ở
`wu_method/VJBench-trans/`, KHÔNG dùng chung `run_evaluation.py`).

Khác với artifact gốc của Wu et al. (Codex completion), ở đây DeepSeek là chat
model → ta yêu cầu LLM trả lại NGUYÊN snippet đã fix (full function), không
phải chỉ vài dòng thay thế. Điều này khớp với cách baseline PatchEval
(`patcheval_toolkit.py::_gen_build_unified_diff`) đang xử lý và tránh được lỗi
reconstruction line-by-line.
"""

LANG_COMMENT = {
    "python":     "#",
    "go":         "//",
    "javascript": "//",
    "js":         "//",
}


def to_zero_indexed(patch_lines: list[int]) -> list[int]:
    """PatchEval lưu `patch_lines` 1-indexed relative to `snippet`.
    Convert về 0-indexed cho Python list."""
    return [max(0, int(l) - 1) for l in patch_lines]


def annotate_bug_lines(snippet: str, buggy_lines_0idx: list[int], lang: str) -> str:
    """
    Chèn comment `<c> BUG:` phía trên các dòng lỗi (0-indexed) để LLM thấy
    Wu et al.-style hint. Không xóa dòng gốc — chỉ đánh dấu.

    Ví dụ với Python (buggy_lines_0idx=[2]):
        def parse_path(user_input):
            base = "/var/data/"
        # BUG:
            path = base + user_input
            with open(path) as f:
                return f.read()
    """
    c = LANG_COMMENT.get(lang.lower(), "//")
    lines = snippet.splitlines()
    out = []
    marked = set(buggy_lines_0idx)

    for i, line in enumerate(lines):
        if i in marked:
            # đánh dấu bằng comment trên cùng level indentation với dòng buggy
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}{c} BUG (vulnerable line — needs fix):")
        out.append(line)

    return "\n".join(out)


def build_prompt(cve_desc: str, cwe_list: list[str], annotated_snippet: str,
                 lang: str, file_path: str) -> str:
    """
    Wu et al. Codex-style prompt, adapted cho chat LLM:
    - đưa CVE description + CWE làm context vulnerability semantics
    - đưa full snippet đã annotate BUG
    - yêu cầu trả lại FULL FIXED SNIPPET (không phải chỉ vài dòng)

    Trả về prompt string để feed vào chat completion.
    """
    cwe_str = ", ".join(cwe_list) if cwe_list else "N/A"
    lang_pretty = lang.capitalize() if lang.lower() != "javascript" else "JavaScript"

    return f"""You are a security patch expert. Fix the vulnerability marked with `BUG` comment.

CVE description:
{cve_desc}

CWE: {cwe_str}
Language: {lang_pretty}
File: {file_path}

Vulnerable snippet (BUG comment marks the vulnerable line):
```{lang.lower()}
{annotated_snippet}
```

Output ONLY the full fixed snippet (same function/scope), wrapped in a single \
```{lang.lower()} code block. Requirements:
- Preserve exact indentation and function signature.
- Do NOT include the `# BUG` / `// BUG` comment in your output.
- Do NOT add explanation, markdown headers, or trailing prose — code fence only.
- Fix must address the root cause, not just mask the symptom.
"""


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_py = (
        "def parse_path(user_input):\n"
        "    base = \"/var/data/\"\n"
        "    path = base + user_input\n"
        "    with open(path) as f:\n"
        "        return f.read()\n"
    )
    annotated = annotate_bug_lines(sample_py, buggy_lines_0idx=[2], lang="python")
    print("=== Annotated snippet ===")
    print(annotated)
    print()
    print("=== 1-indexed → 0-indexed conversion ===")
    print("patch_lines=[3, 4]  ->", to_zero_indexed([3, 4]))
