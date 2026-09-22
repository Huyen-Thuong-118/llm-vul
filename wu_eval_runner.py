r"""
Wu et al. (ISSTA 2023) — Main Evaluation Runner (PatchEval-adapted)

QUY TRÌNH (giống hệt cách baseline + feedback-loop của PatchEval đang chạy,
chỉ khác PROMPT + N candidates + metric):

  Bước 1  Load `patcheval_verified.json`, lọc theo language (Python/Go/JS).
  Bước 2  Với mỗi CVE, gọi DeepSeek N lần (temperature=0.8) để sinh N candidate
          fixed snippets — RAW LLM outputs, chưa build diff.
  Bước 3  Với mỗi candidate rank r ∈ 1..N, dùng lại
             _gen_extract_code_block + _gen_reindent_to_match +
             _gen_ensure_matching_trailing_blank + _gen_build_unified_diff
          từ `patcheval_toolkit.py` để build ra unified diff đúng format
          `[{"cve","fix_patch","language"}]` — CHÍNH LÀ format `run_evaluation.py`
          nhận.
  Bước 4  Ghi ra `wu_patches_<lang>_deepseek_r{1..N}.jsonl` (một file / rank),
          NGANG hàng với `python_patches_deepseek_r*.jsonl` ở gốc project.
  Bước 5  Với mỗi rank, gọi `run_evaluation.py --output wu_<lang>_r{r}
          --patch_file wu_patches_<lang>_deepseek_r{r}.jsonl` (giống hệt
          patcheval_pipeline.py step_run() và feedback_loop_retry.py).
  Bước 6  Đọc `evaluation_output/wu_<lang>_r{r}/logs/<cve>/success_output.log`
          để biết CVE nào pass ở rank r. Plausible@k = union CVE pass qua r=1..k.

TẠI SAO KHÔNG GỌI DOCKER TRỰC TIẾP MÀ QUA run_evaluation.py:
  run_evaluation.py đã handle: mount patch vào /workspace/fix.patch, chạy
  fix-run.sh (git apply + PoC + regression test), phân loại error type
  (apply_fail / compilation_fail / validation_fail / Repair Success), log riêng
  từng CVE. Reimplement lại là chuốc bug. Bug 0/10 hiện tại chính là do đã cố
  gọi tay và bỏ qua fix-run.sh của image.

CÁCH CHẠY (ở thư mục gốc PatchEval/, PowerShell, env đã activate):

  $env:PYTHONUTF8="1"
  $env:PYTHONIOENCODING="utf-8"
  $env:DEEPSEEK_API_KEY="sk-..."

  # Smoke test 5 CVE JS truoc, chi sinh 3 candidates:
  python wu_method\wu_eval_runner.py --lang javascript --limit 5 --n-patches 3 `
      --stage all

  # Full run 1 ngon ngu (Python 70 CVE, 5 candidates, plausible@5):
  python wu_method\wu_eval_runner.py --lang python --n-patches 5 --stage all `
      --prune-every 5

  # Chi sinh patch (khong Docker) — de re-use lai patch da sinh:
  python wu_method\wu_eval_runner.py --lang go --stage generate --n-patches 5

  # Chi chay Docker validation tren patch da sinh san:
  python wu_method\wu_eval_runner.py --lang go --stage validate

  # Tinh lai plausible@k tu logs (khong chay gi):
  python wu_method\wu_eval_runner.py --lang go --stage report
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict

# ── Path setup ────────────────────────────────────────────────────────────────
WU_DIR         = Path(__file__).resolve().parent
PATCHEVAL_ROOT = WU_DIR.parent
sys.path.insert(0, str(PATCHEVAL_ROOT))   # import patcheval_toolkit
sys.path.insert(0, str(WU_DIR))           # import wu_llm_runner

PATCHEVAL_JSON = PATCHEVAL_ROOT / "patcheval" / "datasets" / "patcheval_verified.json"
EVAL_DIR       = PATCHEVAL_ROOT / "patcheval" / "evaluation"

# ── Import project helpers (proven-working từ deepseek2 run) ─────────────────
from patcheval_toolkit import (
    _gen_extract_code_block,
    _gen_normalize_trailing_blank,
    _gen_reindent_to_match,
    _gen_ensure_matching_trailing_blank,
    _gen_build_unified_diff,
)
from wu_llm_runner import generate_candidates

LANG_MAP = {  # PatchEval canonical spelling
    "python":     "Python",
    "go":         "Go",
    "javascript": "JavaScript",
    "js":         "JavaScript",
}


# ── Stage 1: generate ────────────────────────────────────────────────────────
def stage_generate(lang: str, n_patches: int, limit: int, verbose: bool):
    """Sinh N candidate patch per CVE, ghi ra wu_patches_<lang>_deepseek_r{r}.jsonl."""
    canonical_lang = LANG_MAP[lang.lower()]

    with open(PATCHEVAL_JSON, encoding="utf-8") as f:
        data = json.load(f)
    cves = [d for d in data if d["programing_language"] == canonical_lang]
    if limit:
        cves = cves[:limit]

    # Resume: đọc các file rank đã có, skip CVE nào đã đủ N candidate.
    rank_files = [
        PATCHEVAL_ROOT / f"wu_patches_{lang.lower()}_deepseek_r{r}.jsonl"
        for r in range(1, n_patches + 1)
    ]
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

    # CVE nào đã có mặt trong TẤT CẢ N rank files thì bỏ qua
    fully_done = set.intersection(*done_per_rank) if done_per_rank else set()
    todo = [c for c in cves if c["cve_id"] not in fully_done]
    print(f"[GENERATE] lang={canonical_lang} total={len(cves)} "
          f"resume-skip={len(fully_done)} todo={len(todo)}")

    for i, cve in enumerate(todo, 1):
        cve_id    = cve["cve_id"]
        vul_entry = cve["vul_func"][0]
        buggy     = _gen_normalize_trailing_blank(vul_entry["snippet"])
        file_path = vul_entry["file_path"]
        start_line = vul_entry.get("start_line", 1)
        locs      = vul_entry.get("vul_localization", [])
        patch_lines = locs[0]["patch_lines"] if locs else [1]

        print(f"[{i}/{len(todo)}] {cve_id}  file={file_path}  "
              f"patch_lines(1idx)={patch_lines}")

        t0 = time.time()
        try:
            raw_outputs = generate_candidates(
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
            print(f"  [SKIP] LLM failed: {e}")
            continue

        # Build diff & append vào đúng rank file
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
                "cve":      cve_id,
                "fix_patch": fix_patch,
                "language":  canonical_lang,
                "model":     "deepseek-chat",
                "wu_rank":   r,
            }
            with open(rank_files[r - 1], "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        elapsed = round(time.time() - t0, 1)
        print(f"  [OK] {len(raw_outputs)}/{n_patches} candidates in {elapsed}s")


# ── Stage 2: validate ────────────────────────────────────────────────────────
def _docker_preflight() -> bool:
    """Xác nhận Docker daemon đang chạy trước khi gọi run_evaluation.py.
    Nếu không, run_evaluation.py sẽ ném DockerException, KHÔNG ghi log CVE nào,
    nhưng vẫn ghi summary.json với 0 pass → dễ nhầm là 'đã validate'."""
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
        print("[PREFLIGHT FAIL] 'docker version' timeout — Docker Desktop dang start hoac treo.")
        return False


def _rank_has_real_logs(out_dir: Path) -> bool:
    """Rank được coi là 'đã chạy thật' khi trong logs/ có ít nhất 1 file
    success_output.log hoặc error_output.log. summary.json một mình KHÔNG đủ
    — run_evaluation.py vẫn ghi summary.json dù toàn bộ CVE fail ở tầng Docker."""
    logs_dir = out_dir / "logs"
    if not logs_dir.exists():
        return False
    for cve_dir in logs_dir.iterdir():
        if not cve_dir.is_dir():
            continue
        if (cve_dir / "success_output.log").exists() or (cve_dir / "error_output.log").exists():
            return True
    return False


def _cves_with_logs(out_dir: Path) -> set[str]:
    """CVE nào đã có success_output.log hoặc error_output.log → coi như xong."""
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


def _load_records(jsonl_path: Path) -> list[dict]:
    out = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _print_disk_usage():
    """In docker df + host disk cua thu muc PatchEval — de thay space thuc su
    duoc giai phong hay khong (docker prune tra ve khoang trong ben trong vhdx
    nhung host disk chi thay khi 'diskpart compact vdisk' — xem help)."""
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
                   max_workers: int, force: bool = False, min_free_gb: float = 5.0):
    """Chạy run_evaluation.py trên mỗi rank file, output vào evaluation_output/wu_<lang>_r{r}.

    Batch nhỏ trong 1 rank + prune sau mỗi batch để chống hết ổ. Mỗi CVE đã có
    success/error log sẽ tự động skip → resume an toàn khi Docker chết giữa chừng
    hoặc Ctrl+C.
    """
    if not _docker_preflight():
        print("\n[ABORT] Docker Desktop chua chay hoac khong the ket noi.")
        print("        Mo Docker Desktop, doi den khi status = Running, roi chay lai lenh.")
        print("        Neu Docker dang chay ma van fail: 'wsl --shutdown' roi start lai Docker Desktop.")
        return

    tmp_batch_dir = PATCHEVAL_ROOT / ".wu_batches"
    tmp_batch_dir.mkdir(exist_ok=True)

    for r in range(1, n_patches + 1):
        patch_jsonl = PATCHEVAL_ROOT / f"wu_patches_{lang.lower()}_deepseek_r{r}.jsonl"
        if not patch_jsonl.exists():
            print(f"[SKIP rank {r}] {patch_jsonl.name} chua ton tai — chay stage generate truoc.")
            continue

        out_name = f"wu_{lang.lower()}_r{r}"
        out_dir  = EVAL_DIR / "evaluation_output" / out_name

        if force and out_dir.exists():
            import shutil
            print(f"[FORCE rank {r}] --force → xoa {out_name}/")
            shutil.rmtree(out_dir, ignore_errors=True)

        all_records = _load_records(patch_jsonl)
        done = _cves_with_logs(out_dir)
        todo = [rec for rec in all_records if rec["cve"] not in done]
        if not todo:
            print(f"[SKIP rank {r}] tat ca {len(all_records)} CVE da co log — done.")
            continue

        print(f"\n=== RANK {r}/{n_patches} — total={len(all_records)} "
              f"resume-skip={len(done)} todo={len(todo)} ===")

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        n_batches = (len(todo) + batch_size - 1) // batch_size
        for b in range(n_batches):
            chunk = todo[b * batch_size : (b + 1) * batch_size]
            batch_file = tmp_batch_dir / f"{lang.lower()}_r{r}_b{b+1}.jsonl"
            with open(batch_file, "w", encoding="utf-8", newline="\n") as f:
                for rec in chunk:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            cve_list = ", ".join(rec["cve"] for rec in chunk[:3])
            if len(chunk) > 3:
                cve_list += f", ... (+{len(chunk)-3})"
            print(f"\n  --- rank {r} batch {b+1}/{n_batches}  ({len(chunk)} CVE: {cve_list}) ---")

            cmd = [
                sys.executable, "run_evaluation.py",
                "--output",      out_name,       # cung 1 output dir → logs go don ve day
                "--patch_file",  str(batch_file.resolve()),
                "--max_workers", str(max_workers),
            ]
            result = subprocess.run(cmd, cwd=str(EVAL_DIR), env=env)
            if result.returncode != 0:
                print(f"    [WARN] batch exit code {result.returncode}")

            # Xoa batch file tam
            try:
                batch_file.unlink()
            except Exception:
                pass

            # Sanity: batch phai ghi it nhat 1 log CVE (dau hieu Docker con song)
            batch_done = _cves_with_logs(out_dir)
            new_logs = batch_done - done
            if not new_logs:
                print(f"    [FAIL] batch khong ghi log CVE nao. Docker co the chet. Dung lai.")
                print(f"           Kiem tra: {out_dir}/run_evaluation.log")
                return
            done = batch_done

            if prune_every_batch:
                print(f"    --- docker system prune -af ---")
                subprocess.run(["docker", "system", "prune", "-af"], check=False)
            _print_disk_usage()

            # Guard chong het o dia hoan toan
            try:
                import shutil as _sh
                free_gb = _sh.disk_usage(str(PATCHEVAL_ROOT)).free / (1024 ** 3)
                if free_gb < min_free_gb:
                    print(f"\n[HALT] Con {free_gb:.1f} GB free < nguong {min_free_gb} GB.")
                    print(f"       Chay 'wsl --shutdown' + 'diskpart compact vdisk' de shrink docker_data.vhdx")
                    print(f"       roi resume bang: python wu_method\\wu_eval_runner.py --lang {lang} "
                          f"--n-patches {n_patches} --stage validate")
                    return
            except Exception:
                pass

    # Dep .wu_batches neu rong
    try:
        tmp_batch_dir.rmdir()
    except OSError:
        pass


# ── Stage 3: report (plausible@k) ─────────────────────────────────────────────
def stage_report(lang: str, n_patches: int):
    """Tính plausible@1..N từ success_output.log trên từng rank."""
    passed_per_rank: list[set[str]] = []
    all_cves: set[str] = set()

    for r in range(1, n_patches + 1):
        out_dir = EVAL_DIR / "evaluation_output" / f"wu_{lang.lower()}_r{r}"
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
    print("\n=== Plausible@k (Wu et al. metric) — lang={} ===".format(lang))
    print(f"{'k':>3}  {'newly_passed':>13}  {'cumulative':>10}  {'plausible@k':>11}")
    for r, passed in enumerate(passed_per_rank, 1):
        new = passed - cumulative
        cumulative |= passed
        rate = (len(cumulative) / max(len(all_cves), 1)) * 100
        print(f"{r:>3}  {len(new):>13}  {len(cumulative):>10}  {rate:>10.2f}%")

    # Ghi JSON để merge vào báo cáo NCKH sau
    report = {
        "lang": lang,
        "n_patches": n_patches,
        "total_cves_run": len(all_cves),
        "plausible_at_k": [
            {
                "k": r + 1,
                "passed_cves_cumulative": sorted(
                    set().union(*passed_per_rank[: r + 1])
                ),
                "count": len(set().union(*passed_per_rank[: r + 1])),
            }
            for r in range(len(passed_per_rank))
        ],
    }
    out_file = PATCHEVAL_ROOT / f"wu_report_{lang.lower()}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved -> {out_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True,
                   choices=["python", "go", "javascript", "js"])
    p.add_argument("--stage", default="all",
                   choices=["all", "generate", "validate", "report"])
    p.add_argument("--n-patches", type=int, default=5,
                   help="Wu et al. paper dung 10, mac dinh 5.")
    p.add_argument("--limit", type=int, default=0,
                   help="Chi lay N CVE dau (0 = full).")
    p.add_argument("--max-workers", type=int, default=1,
                   help="run_evaluation.py --max_workers (1 an toan tren WSL2).")
    p.add_argument("--batch-size", type=int, default=5,
                   help="So CVE moi batch trong 1 rank. Prune Docker sau moi batch. Default 5.")
    p.add_argument("--no-prune", action="store_true",
                   help="Tat 'docker system prune -af' sau moi batch (khong khuyen — chi khi debug).")
    p.add_argument("--min-free-gb", type=float, default=5.0,
                   help="Dung lai neu o dia con < N GB (default 5). Dat 0 de tat guard.")
    p.add_argument("--force", action="store_true",
                   help="Xoa evaluation_output/wu_<lang>_r*/ va chay lai stage validate tu dau.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    lang = "js" if args.lang == "javascript" else args.lang  # short form for filenames
    lang = args.lang.lower().replace("js", "javascript") if args.lang == "js" else args.lang.lower()

    if args.stage in ("all", "generate"):
        stage_generate(args.lang, args.n_patches, args.limit, args.verbose)
    if args.stage in ("all", "validate"):
        stage_validate(
            args.lang, args.n_patches,
            batch_size=args.batch_size,
            prune_every_batch=not args.no_prune,
            max_workers=args.max_workers,
            force=args.force,
            min_free_gb=args.min_free_gb,
        )
    if args.stage in ("all", "report"):
        stage_report(args.lang, args.n_patches)


if __name__ == "__main__":
    main()
