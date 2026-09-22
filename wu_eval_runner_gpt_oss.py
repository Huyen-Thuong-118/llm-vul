r"""
Wu et al. (ISSTA 2023) — Eval Runner cho GPT-OSS 20B

Song hanh voi wu_eval_runner.py (DeepSeek). File nay HARDCODE GPT-OSS 20B —
khong cai gi doi voi output cua DeepSeek. Naming tach hoan toan:

    wu_patches_<lang>_qwen3_14b_r{r}.jsonl
    evaluation_output/wu_<lang>_qwen3_14b_r{r}/
    wu_report_<lang>_qwen3_14b.json

Chay (PowerShell, thu muc goc PatchEval, env da activate):

    $env:PYTHONUTF8="1"
    $env:PYTHONIOENCODING="utf-8"
    $env:OPENROUTER_API_KEY="sk-or-v1-..."

    # Smoke test 3 CVE JS
    python wu_method\wu_eval_runner_qwen.py --lang javascript --limit 3 --n-patches 3 --stage all --verbose

    # Full 77 CVE JS × 5 candidates
    python wu_method\wu_eval_runner_qwen.py --lang javascript --n-patches 5 --stage generate
    python wu_method\wu_eval_runner_qwen.py --lang javascript --n-patches 5 --stage validate --batch-size 3 --min-free-gb 8
    python wu_method\wu_eval_runner_qwen.py --lang javascript --n-patches 5 --stage report
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# -- Path setup --------------------------------------------------------------
WU_DIR         = Path(__file__).resolve().parent
PATCHEVAL_ROOT = WU_DIR.parent
sys.path.insert(0, str(PATCHEVAL_ROOT))
sys.path.insert(0, str(WU_DIR))

PATCHEVAL_JSON = PATCHEVAL_ROOT / "patcheval" / "datasets" / "patcheval_verified.json"
EVAL_DIR       = PATCHEVAL_ROOT / "patcheval" / "evaluation"

from patcheval_toolkit import (
    _gen_extract_code_block,
    _gen_normalize_trailing_blank,
    _gen_reindent_to_match,
    _gen_ensure_matching_trailing_blank,
    _gen_build_unified_diff,
)
from wu_llm_runner_gpt_oss import (
    generate_candidates_gpt_oss, MODEL_ID, FILE_SLUG,
    set_stats_file, flush_stats,
)

LANG_MAP = {"python": "Python", "go": "Go", "javascript": "JavaScript", "js": "JavaScript"}


def patch_jsonl_path(lang: str, rank: int) -> Path:
    return PATCHEVAL_ROOT / f"wu_patches_{lang.lower()}_{FILE_SLUG}_r{rank}.jsonl"


def rank_output_name(lang: str, rank: int) -> str:
    return f"wu_{lang.lower()}_{FILE_SLUG}_r{rank}"


def report_json_path(lang: str) -> Path:
    return PATCHEVAL_ROOT / f"wu_report_{lang.lower()}_{FILE_SLUG}.json"


def stats_json_path(lang: str) -> Path:
    return PATCHEVAL_ROOT / f"wu_stats_{lang.lower()}_{FILE_SLUG}.json"


# -- Stage 1: generate -------------------------------------------------------
def _process_one_cve(cve: dict, canonical_lang: str, n_patches: int,
                     rank_files, done_per_rank, file_lock: threading.Lock,
                     progress: dict, verbose: bool):
    """Xu ly 1 CVE (5 candidate tuan tu ben trong). Ghi ra rank files voi lock."""
    cve_id    = cve["cve_id"]
    vul_entry = cve["vul_func"][0]
    buggy     = _gen_normalize_trailing_blank(vul_entry["snippet"])
    file_path = vul_entry["file_path"]
    start_line = vul_entry.get("start_line", 1)
    locs      = vul_entry.get("vul_localization", [])
    patch_lines = locs[0]["patch_lines"] if locs else [1]

    t0 = time.time()
    try:
        raw_outputs = generate_candidates_gpt_oss(
            cve_id=cve_id,
            cve_desc=cve.get("cve_description", ""),
            cwe_list=list(cve.get("cwe_info", {}).keys()),
            buggy_snippet=vul_entry["snippet"],
            buggy_lines_1idx=patch_lines,
            file_path=file_path,
            lang=canonical_lang.lower(),
            n_patches=n_patches,
            verbose=verbose,
        )
    except Exception as e:
        return (cve_id, 0, f"LLM failed: {e}", 0.0)

    # Build diff + ghi ra file voi lock
    written = 0
    with file_lock:
        for r, raw in enumerate(raw_outputs, 1):
            if cve_id in done_per_rank[r - 1]:
                continue
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
            }
            with open(rank_files[r - 1], "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            done_per_rank[r - 1].add(cve_id)
            written += 1

        progress["done"] += 1
        elapsed = round(time.time() - t0, 1)
        pct = 100 * progress["done"] / progress["total"]
        print(f"[{progress['done']:3d}/{progress['total']} ({pct:5.1f}%)] "
              f"{cve_id}  {written}/{n_patches} candidates  {elapsed}s")

    return (cve_id, written, None, time.time() - t0)


def stage_generate(lang: str, n_patches: int, limit: int, workers: int, verbose: bool):
    canonical_lang = LANG_MAP[lang.lower()]

    # Enable usage tracking → luu ra stats file
    set_stats_file(stats_json_path(lang))

    with open(PATCHEVAL_JSON, encoding="utf-8") as f:
        data = json.load(f)
    cves = [d for d in data if d["programing_language"] == canonical_lang]
    if limit:
        cves = cves[:limit]

    rank_files = [patch_jsonl_path(lang, r) for r in range(1, n_patches + 1)]
    done_per_rank = []
    for rf in rank_files:
        s = set()
        if rf.exists():
            with open(rf, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        s.add(json.loads(line)["cve"])
        done_per_rank.append(s)

    fully_done = set.intersection(*done_per_rank) if done_per_rank else set()
    todo = [c for c in cves if c["cve_id"] not in fully_done]
    print(f"[GENERATE gpt-oss-20b] lang={canonical_lang} total={len(cves)} "
          f"resume-skip={len(fully_done)} todo={len(todo)} workers={workers}")

    if not todo:
        print("  Khong co CVE nao can sinh. Da xong.")
        return

    file_lock = threading.Lock()
    progress = {"done": 0, "total": len(todo)}
    t_start = time.time()

    if workers <= 1:
        # Fallback tuan tu — de debug hoac tranh rate limit
        for cve in todo:
            _process_one_cve(cve, canonical_lang, n_patches,
                             rank_files, done_per_rank, file_lock, progress, verbose)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_process_one_cve, cve, canonical_lang, n_patches,
                          rank_files, done_per_rank, file_lock, progress, verbose)
                for cve in todo
            ]
            for fut in as_completed(futures):
                cve_id, written, err, _ = fut.result()
                if err:
                    print(f"  [FAIL] {cve_id}: {err}")

    total_elapsed = round(time.time() - t_start, 1)
    print(f"\n[GENERATE DONE] {len(todo)} CVE xu ly xong trong {total_elapsed}s "
          f"({total_elapsed/max(len(todo),1):.1f}s/CVE trung binh)")

    # Flush usage stats ra file JSON
    flush_stats()


# -- Stage 2: validate -------------------------------------------------------
def _docker_preflight() -> bool:
    try:
        r = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0 or not r.stdout.strip():
            print("[PREFLIGHT FAIL] Docker daemon khong phan hoi.")
            print(f"  stdout: {r.stdout.strip()}")
            print(f"  stderr: {r.stderr.strip()}")
            return False
        print(f"[PREFLIGHT OK] Docker server version = {r.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("[PREFLIGHT FAIL] Khong tim thay lenh 'docker' trong PATH.")
        return False
    except subprocess.TimeoutExpired:
        print("[PREFLIGHT FAIL] 'docker version' timeout.")
        return False


def _cves_with_logs(out_dir: Path) -> set:
    done = set()
    logs_dir = out_dir / "logs"
    if not logs_dir.exists():
        return done
    for cve_dir in logs_dir.iterdir():
        if not cve_dir.is_dir():
            continue
        if (cve_dir / "success_output.log").exists() or (cve_dir / "error_output.log").exists():
            done.add(cve_dir.name)
    return done


def _cves_passed_in_rank(lang: str, rank: int) -> set:
    """CVE nao co success_output.log o 1 rank cu the (dung cho early exit)."""
    out_dir = EVAL_DIR / "evaluation_output" / rank_output_name(lang, rank)
    logs_dir = out_dir / "logs"
    passed = set()
    if not logs_dir.exists():
        return passed
    for cve_dir in logs_dir.iterdir():
        if cve_dir.is_dir() and (cve_dir / "success_output.log").exists():
            passed.add(cve_dir.name)
    return passed


def _load_records(jsonl_path: Path):
    out = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _print_disk_usage():
    print("  --- docker system df ---")
    subprocess.run(["docker", "system", "df"], check=False)
    try:
        import shutil as _sh
        total, used, free = _sh.disk_usage(str(PATCHEVAL_ROOT))
        gb = 1024 ** 3
        print(f"  --- host disk ({PATCHEVAL_ROOT.drive or PATCHEVAL_ROOT}): "
              f"free = {free/gb:.1f} GB / total {total/gb:.1f} GB ---")
    except Exception:
        pass


def stage_validate(lang: str, n_patches: int, batch_size: int, prune_every_batch: bool,
                   max_workers: int, force: bool = False, min_free_gb: float = 5.0,
                   early_exit: bool = True):
    if not _docker_preflight():
        print("\n[ABORT] Docker Desktop chua chay hoac khong the ket noi.")
        return

    tmp_batch_dir = PATCHEVAL_ROOT / ".wu_batches"
    tmp_batch_dir.mkdir(exist_ok=True)

    # Tich luy pass CVE tu cac rank truoc do — early exit skip nhung CVE nay
    already_passed_across_ranks: set = set()

    for r in range(1, n_patches + 1):
        # Cap nhat pass status tu rank r-1 (neu co)
        if r > 1 and early_exit:
            just_passed = _cves_passed_in_rank(lang, r - 1)
            already_passed_across_ranks |= just_passed

        patch_jsonl = patch_jsonl_path(lang, r)
        if not patch_jsonl.exists():
            print(f"[SKIP rank {r}] {patch_jsonl.name} chua ton tai — chay generate truoc.")
            continue

        out_name = rank_output_name(lang, r)
        out_dir  = EVAL_DIR / "evaluation_output" / out_name

        if force and out_dir.exists():
            import shutil
            print(f"[FORCE rank {r}] xoa {out_name}/")
            shutil.rmtree(out_dir, ignore_errors=True)

        all_records = _load_records(patch_jsonl)
        done = _cves_with_logs(out_dir)

        # Filter todo: bo CVE da chay xong (resume) + CVE da pass o rank truoc (early exit)
        skipped_early = 0
        if early_exit:
            filtered = []
            for rec in all_records:
                if rec["cve"] in done:
                    continue
                if rec["cve"] in already_passed_across_ranks:
                    skipped_early += 1
                    continue
                filtered.append(rec)
            todo = filtered
        else:
            todo = [rec for rec in all_records if rec["cve"] not in done]

        if not todo:
            reason = "tat ca da co log" if not skipped_early else f"tat ca da co log ({skipped_early} early-exit)"
            print(f"[SKIP rank {r}] {reason}.")
            continue

        print(f"\n=== RANK {r}/{n_patches} (gpt-oss-20b) — total={len(all_records)} "
              f"resume-done={len(done)} early-exit-skip={skipped_early} todo={len(todo)} ===")

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        n_batches = (len(todo) + batch_size - 1) // batch_size
        for b in range(n_batches):
            chunk = todo[b * batch_size : (b + 1) * batch_size]
            batch_file = tmp_batch_dir / f"{lang.lower()}_{FILE_SLUG}_r{r}_b{b+1}.jsonl"
            with open(batch_file, "w", encoding="utf-8", newline="\n") as f:
                for rec in chunk:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            cve_list = ", ".join(rec["cve"] for rec in chunk[:3])
            if len(chunk) > 3:
                cve_list += f", ... (+{len(chunk)-3})"
            print(f"\n  --- rank {r} batch {b+1}/{n_batches}  ({len(chunk)} CVE: {cve_list}) ---")

            cmd = [
                sys.executable, "run_evaluation.py",
                "--output",      out_name,
                "--patch_file",  str(batch_file.resolve()),
                "--max_workers", str(max_workers),
            ]
            result = subprocess.run(cmd, cwd=str(EVAL_DIR), env=env)
            if result.returncode != 0:
                print(f"    [WARN] batch exit code {result.returncode}")

            try:
                batch_file.unlink()
            except Exception:
                pass

            batch_done = _cves_with_logs(out_dir)
            new_logs = batch_done - done
            if not new_logs:
                print(f"    [FAIL] batch khong ghi log CVE nao. Dung lai.")
                return
            done = batch_done

            if prune_every_batch:
                print(f"    --- docker system prune -af ---")
                subprocess.run(["docker", "system", "prune", "-af"], check=False)
            _print_disk_usage()

            try:
                import shutil as _sh
                free_gb = _sh.disk_usage(str(PATCHEVAL_ROOT)).free / (1024 ** 3)
                if free_gb < min_free_gb:
                    print(f"\n[HALT] Con {free_gb:.1f} GB free < nguong {min_free_gb} GB.")
                    print(f"       Chay 'wsl --shutdown' + 'diskpart compact vdisk' de shrink docker_data.vhdx")
                    print(f"       roi resume bang: python wu_method\\wu_eval_runner_qwen.py "
                          f"--lang {lang} --n-patches {n_patches} --stage validate")
                    return
            except Exception:
                pass

    try:
        tmp_batch_dir.rmdir()
    except OSError:
        pass


# -- Stage 3: report ---------------------------------------------------------
def stage_report(lang: str, n_patches: int):
    passed_per_rank = []
    all_cves = set()

    for r in range(1, n_patches + 1):
        out_dir = EVAL_DIR / "evaluation_output" / rank_output_name(lang, r)
        logs_dir = out_dir / "logs"
        passed = set()
        if not logs_dir.exists():
            print(f"[rank {r}] khong co logs — bo qua")
            passed_per_rank.append(passed)
            continue
        for cve_dir in logs_dir.iterdir():
            if not cve_dir.is_dir():
                continue
            all_cves.add(cve_dir.name)
            if (cve_dir / "success_output.log").exists():
                passed.add(cve_dir.name)
        passed_per_rank.append(passed)
        print(f"[rank {r}]  passed = {len(passed):3d}  ({', '.join(sorted(passed)[:5])}"
              f"{'...' if len(passed) > 5 else ''})")

    cumulative = set()
    print(f"\n=== Plausible@k (Wu et al. metric) — lang={lang} model={MODEL_ID} ===")
    print(f"{'k':>3}  {'newly_passed':>13}  {'cumulative':>10}  {'plausible@k':>11}")
    for r, passed in enumerate(passed_per_rank, 1):
        new = passed - cumulative
        cumulative |= passed
        rate = (len(cumulative) / max(len(all_cves), 1)) * 100
        print(f"{r:>3}  {len(new):>13}  {len(cumulative):>10}  {rate:>10.2f}%")

    report = {
        "lang": lang,
        "model_id": MODEL_ID,
        "n_patches": n_patches,
        "total_cves_run": len(all_cves),
        "plausible_at_k": [
            {
                "k": r + 1,
                "passed_cves_cumulative": sorted(set().union(*passed_per_rank[: r + 1])),
                "count": len(set().union(*passed_per_rank[: r + 1])),
            }
            for r in range(len(passed_per_rank))
        ],
    }
    out_file = report_json_path(lang)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved -> {out_file}")


# -- CLI ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Wu et al. pipeline cho GPT-OSS 20B qua OpenRouter")
    p.add_argument("--lang", required=True, choices=["python", "go", "javascript", "js"])
    p.add_argument("--stage", default="all", choices=["all", "generate", "validate", "report"])
    p.add_argument("--n-patches", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--no-prune", action="store_true")
    p.add_argument("--min-free-gb", type=float, default=5.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-early-exit", action="store_true",
                   help="TAT early exit. Mac dinh: skip CVE da pass o rank truoc "
                        "(giam 30-40%% Docker run). Chi tat neu can data day du cho analysis.")
    p.add_argument("--workers", type=int, default=4,
                   help="Song song bao nhieu CVE cung luc trong stage generate. "
                        "1 = tuan tu. Default 4. Tang len 8-12 neu OpenRouter tier cao.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.stage in ("all", "generate"):
        stage_generate(args.lang, args.n_patches, args.limit, args.workers, args.verbose)
    if args.stage in ("all", "validate"):
        stage_validate(
            args.lang, args.n_patches,
            batch_size=args.batch_size,
            prune_every_batch=not args.no_prune,
            max_workers=args.max_workers,
            force=args.force,
            min_free_gb=args.min_free_gb,
            early_exit=not args.no_early_exit,
        )
    if args.stage in ("all", "report"):
        stage_report(args.lang, args.n_patches)


if __name__ == "__main__":
    main()
