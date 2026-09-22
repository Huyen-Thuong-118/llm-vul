"""
Test 1 CVE cu the voi 1 LLM runner (mac dinh Claude Sonnet 5).

Chay:
    # Test CVE dai nhat JS (CVE-2020-26237, highlight.js, 845 lines)
    python wu_method\\test_one_cve.py --cve CVE-2020-26237 --runner claude

    # Test voi Qwen de compare
    python wu_method\\test_one_cve.py --cve CVE-2020-26237 --runner qwen

    # Sinh 3 candidate thay vi 1
    python wu_method\\test_one_cve.py --cve CVE-2020-26237 --runner claude --n-patches 3

Output: in ra prompt gui LLM, candidate patches, unified diff, tokens/cost/latency.
Khong dung Docker — chi test khau sinh patch.
"""

import argparse
import json
import sys
from pathlib import Path

WU_DIR         = Path(__file__).resolve().parent
PATCHEVAL_ROOT = WU_DIR.parent
sys.path.insert(0, str(PATCHEVAL_ROOT))
sys.path.insert(0, str(WU_DIR))

PATCHEVAL_JSON = PATCHEVAL_ROOT / "patcheval" / "datasets" / "patcheval_verified.json"

from patcheval_toolkit import (
    _gen_extract_code_block,
    _gen_normalize_trailing_blank,
    _gen_reindent_to_match,
    _gen_ensure_matching_trailing_blank,
    _gen_build_unified_diff,
)


def load_runner(runner_name: str):
    """Import runner theo ten. Tra ve (module, generate_fn)."""
    if runner_name == "claude":
        import wu_llm_runner_claude as mod
        return mod, mod.generate_candidates_claude
    if runner_name == "qwen":
        import wu_llm_runner_qwen as mod
        return mod, mod.generate_candidates_qwen
    if runner_name == "llama":
        import wu_llm_runner_llama as mod
        return mod, mod.generate_candidates_llama
    raise ValueError(f"Unknown runner: {runner_name}. Choices: claude, qwen, llama")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cve", required=True, help="CVE ID (vd: CVE-2020-26237)")
    p.add_argument("--runner", default="claude", choices=["claude", "qwen", "llama"])
    p.add_argument("--n-patches", type=int, default=1,
                   help="So candidate sinh. Default 1 de test nhanh.")
    p.add_argument("--show-patch", action="store_true",
                   help="In full patch content ra terminal.")
    args = p.parse_args()

    # Load dataset
    with open(PATCHEVAL_JSON, encoding="utf-8") as f:
        data = json.load(f)

    cve = next((d for d in data if d["cve_id"] == args.cve), None)
    if cve is None:
        print(f"[FAIL] Khong tim thay {args.cve} trong dataset")
        available = [d["cve_id"] for d in data[:5]]
        print(f"Vi du CVE co san: {available}")
        return

    vul       = cve["vul_func"][0]
    buggy     = vul["snippet"]
    file_path = vul["file_path"]
    start_line = vul.get("start_line", 1)
    patch_lines = vul.get("vul_localization", [{}])[0].get("patch_lines", [])
    lang      = cve["programing_language"].lower()

    print(f"=== {args.cve} ===")
    print(f"  Language: {cve['programing_language']}")
    print(f"  File:     {file_path}")
    print(f"  Lines:    {len(buggy.splitlines())} total, buggy at {patch_lines} (1-idx)")
    print(f"  Chars:    {len(buggy):,}")
    print(f"  CWE:      {list(cve.get('cwe_info', {}).keys())}")
    print(f"  Runner:   {args.runner}")
    print()

    # Load runner + set stats file
    mod, gen_fn = load_runner(args.runner)
    stats_file = PATCHEVAL_ROOT / f"test_stats_{args.cve}_{args.runner}.json"
    mod.set_stats_file(stats_file)

    print(f"Calling {mod.MODEL_ID} to sinh {args.n_patches} candidate(s)...")
    print()

    raw_outputs = gen_fn(
        cve_id=args.cve,
        cve_desc=cve.get("cve_description", ""),
        cwe_list=list(cve.get("cwe_info", {}).keys()),
        buggy_snippet=buggy,
        buggy_lines_1idx=patch_lines,
        file_path=file_path,
        lang=lang,
        n_patches=args.n_patches,
        verbose=True,
    )

    print()
    print(f"=== Ket qua: {len(raw_outputs)} candidate(s) ===")

    for i, raw in enumerate(raw_outputs, 1):
        print(f"\n--- Candidate {i} raw output (first 300 chars) ---")
        print(raw[:300])
        if len(raw) > 300:
            print(f"... (+{len(raw)-300} chars)")

        # Build unified diff
        buggy_norm = _gen_normalize_trailing_blank(buggy)
        new_code = _gen_extract_code_block(raw)
        new_code = _gen_reindent_to_match(buggy_norm, new_code)
        new_code = _gen_ensure_matching_trailing_blank(buggy_norm, new_code)
        diff = _gen_build_unified_diff(file_path, buggy_norm, new_code,
                                        start_line=start_line)

        print(f"\n--- Candidate {i} unified diff (preview) ---")
        diff_lines = diff.splitlines()
        # In ~15 dong dau + het
        preview = diff_lines[:15]
        if len(diff_lines) > 15:
            preview.append(f"... ({len(diff_lines)-15} more lines)")
        for line in preview:
            print(line)

        if args.show_patch:
            print(f"\n--- Candidate {i} FULL diff ---")
            print(diff)

    # Flush stats
    mod.flush_stats()
    print(f"\n=== Test done ===")


if __name__ == "__main__":
    main()
