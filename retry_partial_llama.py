"""
Retry sinh candidate cho nhung CVE chua du n_patches, dung MAX_TOKENS_LARGE
tu dau (khong adaptive).

Logic:
  1. Quet wu_patches_<lang>_llama4_scout_r{1..N}.jsonl → dem so candidate/CVE
  2. Loc CVE co count < n_patches
  3. Voi moi CVE thieu:
     - Load metadata tu dataset
     - Sinh (n_patches - existing_count) candidate voi max_tokens=LARGE
     - Append vao rank files (rank tiep theo con thieu)

Chay:
    python wu_method\\retry_partial_llama.py --lang javascript --n-patches 5

Options:
    --limit N          Chi retry N CVE dau (test truoc)
    --workers N        Parallel workers (default 4)
    --verbose          Log chi tiet moi call
    --dry-run          Chi in danh sach CVE se retry, khong goi API
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from wu_llm_runner_llama import (
    generate_candidates_llama, MODEL_ID, FILE_SLUG,
    MAX_TOKENS_LARGE, set_stats_file, flush_stats,
)


LANG_MAP = {"python": "Python", "go": "Go", "javascript": "JavaScript", "js": "JavaScript"}


def patch_jsonl_path(lang, rank):
    return PATCHEVAL_ROOT / f"wu_patches_{lang.lower()}_{FILE_SLUG}_r{rank}.jsonl"


def count_candidates_per_cve(lang, n_patches):
    """Tra ve dict {cve_id: {'count': int, 'existing_ranks': set([1,2,3])}}"""
    result = {}
    for r in range(1, n_patches + 1):
        path = patch_jsonl_path(lang, r)
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cve = rec["cve"]
                if cve not in result:
                    result[cve] = {"count": 0, "existing_ranks": set()}
                result[cve]["count"] += 1
                result[cve]["existing_ranks"].add(r)
    return result


def process_one_cve(cve, canonical_lang, needed, missing_ranks, rank_files,
                    file_lock, progress, verbose):
    """Retry 1 CVE — sinh `needed` candidate voi MAX_TOKENS_LARGE."""
    cve_id    = cve["cve_id"]
    vul_entry = cve["vul_func"][0]
    buggy     = _gen_normalize_trailing_blank(vul_entry["snippet"])
    file_path = vul_entry["file_path"]
    start_line = vul_entry.get("start_line", 1)
    locs      = vul_entry.get("vul_localization", [])
    patch_lines = locs[0]["patch_lines"] if locs else [1]

    t0 = time.time()
    try:
        raw_outputs = generate_candidates_llama(
            cve_id=cve_id,
            cve_desc=cve.get("cve_description", ""),
            cwe_list=list(cve.get("cwe_info", {}).keys()),
            buggy_snippet=vul_entry["snippet"],
            buggy_lines_1idx=patch_lines,
            file_path=file_path,
            lang=canonical_lang.lower(),
            n_patches=needed,
            verbose=verbose,
            max_tokens_start=MAX_TOKENS_LARGE,  # FORCE LARGE
        )
    except Exception as e:
        return (cve_id, 0, f"LLM failed: {e}")

    # Append vao missing ranks theo thu tu
    written = 0
    with file_lock:
        sorted_missing = sorted(missing_ranks)
        for raw, r in zip(raw_outputs, sorted_missing):
            new_code = _gen_extract_code_block(raw)
            new_code = _gen_reindent_to_match(buggy, new_code)
            new_code = _gen_ensure_matching_trailing_blank(buggy, new_code)
            fix_patch = _gen_build_unified_diff(
                file_path, buggy, new_code, start_line=start_line,
            )
            record = {
                "cve":       cve_id,
                "fix_patch": fix_patch,
                "language":  canonical_lang,
                "model":     MODEL_ID,
                "wu_rank":   r,
                "retry":     True,
                "max_tokens": MAX_TOKENS_LARGE,
            }
            with open(rank_files[r - 1], "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

        progress["done"] += 1
        elapsed = round(time.time() - t0, 1)
        pct = 100 * progress["done"] / progress["total"]
        print(f"[{progress['done']:3d}/{progress['total']} ({pct:5.1f}%)] "
              f"{cve_id}  +{written} candidate (ranks {sorted(missing_ranks)})  {elapsed}s")

    return (cve_id, written, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="javascript",
                   choices=["python", "go", "javascript", "js"])
    p.add_argument("--n-patches", type=int, default=5,
                   help="Target so candidate/CVE (default 5).")
    p.add_argument("--limit", type=int, default=0,
                   help="Chi retry N CVE dau (0 = full).")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers (default 4).")
    p.add_argument("--dry-run", action="store_true",
                   help="Chi in danh sach CVE thieu, khong goi API.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    canonical_lang = LANG_MAP[args.lang.lower()]

    print(f"[SCAN] wu_patches_{args.lang}_{FILE_SLUG}_r*.jsonl...")
    counts = count_candidates_per_cve(args.lang, args.n_patches)
    print(f"  {len(counts)} CVE co it nhat 1 candidate")

    # Filter partial
    partial = {cve: info for cve, info in counts.items()
               if info["count"] < args.n_patches}
    print(f"  {len(partial)} CVE co < {args.n_patches} candidate → can retry")
    print()

    if not partial:
        print("[OK] Tat ca CVE da du candidate. Khong can retry.")
        return

    # Load dataset
    with open(PATCHEVAL_JSON, encoding="utf-8") as f:
        data = json.load(f)
    cves_by_id = {d["cve_id"]: d for d in data if d["programing_language"] == canonical_lang}

    # Build todo list
    todo = []
    for cve_id, info in partial.items():
        if cve_id not in cves_by_id:
            print(f"[SKIP] {cve_id} khong co trong dataset {canonical_lang}")
            continue
        needed = args.n_patches - info["count"]
        missing_ranks = set(range(1, args.n_patches + 1)) - info["existing_ranks"]
        todo.append({
            "cve":            cves_by_id[cve_id],
            "needed":         needed,
            "missing_ranks":  missing_ranks,
            "existing":       info["count"],
        })

    # Sort by missing count desc (uu tien CVE thieu nhieu nhat)
    todo.sort(key=lambda t: -t["needed"])
    if args.limit:
        todo = todo[:args.limit]

    print(f"Retry plan (top 20):")
    print(f"{'CVE':<25} {'Existing':>8} {'Needed':>7} {'Missing ranks'}")
    for t in todo[:20]:
        cve_id = t["cve"]["cve_id"]
        print(f"{cve_id:<25} {t['existing']:>8} {t['needed']:>7} {sorted(t['missing_ranks'])}")
    if len(todo) > 20:
        print(f"... (+{len(todo)-20} nua)")
    print()

    if args.dry_run:
        print(f"[DRY-RUN] {len(todo)} CVE se duoc retry voi max_tokens={MAX_TOKENS_LARGE}. "
              f"Total candidates can sinh: {sum(t['needed'] for t in todo)}")
        return

    # Setup stats tracking (append to existing stats)
    stats_file = PATCHEVAL_ROOT / f"wu_stats_{args.lang}_{FILE_SLUG}_retry.json"
    set_stats_file(stats_file)

    rank_files = [patch_jsonl_path(args.lang, r) for r in range(1, args.n_patches + 1)]
    file_lock = threading.Lock()
    progress = {"done": 0, "total": len(todo)}
    t_start = time.time()

    print(f"[RETRY] workers={args.workers}, max_tokens=LARGE ({MAX_TOKENS_LARGE}) tu dau")
    print()

    if args.workers <= 1:
        for t in todo:
            process_one_cve(t["cve"], canonical_lang, t["needed"], t["missing_ranks"],
                            rank_files, file_lock, progress, args.verbose)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(process_one_cve, t["cve"], canonical_lang, t["needed"],
                          t["missing_ranks"], rank_files, file_lock, progress, args.verbose)
                for t in todo
            ]
            for fut in as_completed(futures):
                cve_id, written, err = fut.result()
                if err:
                    print(f"  [FAIL] {cve_id}: {err}")

    total_elapsed = round(time.time() - t_start, 1)
    print(f"\n[RETRY DONE] {len(todo)} CVE xu ly xong trong {total_elapsed}s")

    flush_stats()

    # Verify final counts
    print()
    print("Verification — final candidate counts:")
    final = count_candidates_per_cve(args.lang, args.n_patches)
    still_partial = [cve for cve, info in final.items() if info["count"] < args.n_patches]
    if still_partial:
        print(f"  Van con {len(still_partial)} CVE thieu candidate (probably dedup exhausted):")
        for cve in still_partial[:10]:
            print(f"    {cve}: {final[cve]['count']}/{args.n_patches}")
    else:
        print(f"  All {len(final)} CVE da du {args.n_patches} candidate")


if __name__ == "__main__":
    main()
