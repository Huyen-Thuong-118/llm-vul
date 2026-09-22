"""Chạy file này 1 lần để patch wu_eval_runner.py cho đúng với format JSON của PatchEval."""
from pathlib import Path

target = Path(__file__).parent / "wu_eval_runner.py"
content = target.read_text(encoding="utf-8")

old = """        buggy    = cve.get("vul_func", "")

        if not buggy:
            print(f"[{idx}/{len(cves)}] SKIP {cve_id} — no buggy_code")
            continue

        buggy_lines = get_buggy_lines(cve)"""

new = """        vul_func = cve.get("vul_func", [])
        if not vul_func:
            print(f"[{idx}/{len(cves)}] SKIP {cve_id} — no vul_func")
            continue
        vul_entry   = vul_func[0]
        buggy       = vul_entry.get("snippet", "")
        locs        = vul_entry.get("vul_localization", [])
        buggy_lines = locs[0]["patch_lines"] if locs else [0]"""

if old in content:
    target.write_text(content.replace(old, new), encoding="utf-8")
    print("Patched OK")
else:
    print("Pattern not found — có thể đã được patch rồi, hoặc file bị thay đổi.")
    print("Tìm dòng chứa 'vul_func' trong file:")
    for i, line in enumerate(content.splitlines(), 1):
        if "vul_func" in line or "buggy_code" in line or "buggy_lines" in line:
            print(f"  L{i}: {line}")
