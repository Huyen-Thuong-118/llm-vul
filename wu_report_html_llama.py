r"""
Wu et al. — HTML report generator cho Llama 4 Scout (hardcoded).
Song hanh voi wu_report_html.py (DeepSeek). Doc paths co suffix _qwen3_14b_.

Chay:
    python wu_method\wu_report_html_qwen.py --lang javascript --n-patches 3

Output: wu_report_<lang>_qwen3_14b.html o goc PatchEval.
r"""

import argparse
import html
import json
import re
import sys
from pathlib import Path
from datetime import datetime

WU_DIR         = Path(__file__).resolve().parent
PATCHEVAL_ROOT = WU_DIR.parent
PATCHEVAL_JSON = PATCHEVAL_ROOT / "patcheval" / "datasets" / "patcheval_verified.json"
EVAL_OUT       = PATCHEVAL_ROOT / "patcheval" / "evaluation" / "evaluation_output"

sys.path.insert(0, str(WU_DIR))
from wu_llm_runner_llama import MODEL_ID, FILE_SLUG

LANG_CANONICAL = {"python": "Python", "go": "Go", "javascript": "JavaScript", "js": "JavaScript"}


# -- User's taxonomy ---------------------------------------------------------
TAXONOMY = {
    "A1": ("A1. Loi bien dich",
           "Model sinh code sai cu phap / sai kieu / tham chieu ham khong ton tai."),
    "A2": ("A2. Pha vo chuc nang hien co",
           "Patch bien dich duoc nhung lam hong hanh vi dung ngoai vung lo hong."),
    "A3": ("A3. Lo hong con nhung co tin hieu",
           "Patch qua duoc kiem thu chuc nang nhung PoC/ASAN van kich hoat duoc."),
    "B1": ("B1. Chon sai chien luoc sua",
           "Code hop le, logic trong hop ly, nhung ve mat bao mat sai huong hoan toan."),
    "B4": ("B4. Chien luoc dung — chi tiet sai",
           "Huong tiep can dung nhung cai dat sai chi tiet."),
    "C7": ("C7. Khong xac nhan duoc",
           "Patch logic nhat quan voi ban sua goc nhung khac cau truc, "
           "hoac bo kiem thu / Docker co van de."),
    "INFRA": ("Infrastructure noise (KHONG phai reasoning fail)",
              "PATCH_APPLY_FAILED — diff khong ap dung duoc vao repo."),
    "UNKNOWN": ("Chua phan loai — can review thu cong",
                "Log khong khop pattern nao. Xem log excerpt ben duoi."),
}


def _classify_error_from_log(log: str, language: str) -> str:
    if not log:
        return "unknown"
    if "patch does not apply" in log:
        return "apply_fail"
    if language == "Python":
        if "SyntaxError" in log or "IndentationError" in log:
            return "compilation_fail"
        return "validation_fail"
    if language == "JavaScript":
        if "SyntaxError" in log or "TypeError" in log:
            return "compilation_fail"
        return "validation_fail"
    if language == "Go":
        if re.search(r'^.*\.go:\d+:\d+: ', log, re.MULTILINE) and not re.search(r'panic:', log):
            return "compilation_fail"
        return "validation_fail"
    return "unknown"


def classify_failure(error_type: str, log_content: str) -> str:
    if error_type == "apply_fail":
        return "INFRA"
    if error_type == "compilation_fail":
        return "A1"
    if error_type == "validation_fail":
        low = log_content.lower()
        poc_hit = any(k in low for k in ("poc", "exploit", "vuln", "still vulnerable"))
        reg_hit = any(k in low for k in ("regression", "existing test", "test failed",
                                          "assertionerror", "expect(", "should"))
        if poc_hit and not reg_hit:
            return "A3"
        if reg_hit and not poc_hit:
            return "A2"
        if reg_hit and poc_hit:
            return "B1"
        return "B4"
    return "UNKNOWN"


# -- Load data ---------------------------------------------------------------
def load_dataset(canonical_lang: str) -> dict:
    with open(PATCHEVAL_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return {d["cve_id"]: d for d in data if d["programing_language"] == canonical_lang}


def load_patches(lang: str, n_patches: int) -> dict:
    out = {}
    for r in range(1, n_patches + 1):
        path = PATCHEVAL_ROOT / f"wu_patches_{lang.lower()}_{FILE_SLUG}_r{r}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out.setdefault(rec["cve"], {})[r] = rec.get("fix_patch", "")
    return out


def load_rank_results(lang: str, n_patches: int) -> dict:
    """Doc TRUC TIEP tu logs/<cve>/{success,error}_output.log (summary.json bi
    ghi de qua cac batch nen khong tin duoc)."""
    canonical_lang = LANG_CANONICAL[lang.lower()]
    results = {}
    for r in range(1, n_patches + 1):
        out_dir = EVAL_OUT / f"wu_{lang.lower()}_{FILE_SLUG}_r{r}"
        rank = {"passed": set(), "failed": {}, "error_log": {}}
        logs_dir = out_dir / "logs"
        if not logs_dir.exists():
            results[r] = rank
            continue
        for cve_dir in logs_dir.iterdir():
            if not cve_dir.is_dir():
                continue
            cve = cve_dir.name
            success_log = cve_dir / "success_output.log"
            error_log = cve_dir / "error_output.log"
            if success_log.exists():
                rank["passed"].add(cve)
                continue
            if error_log.exists():
                try:
                    content = error_log.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    content = ""
                rank["error_log"][cve] = content
                rank["failed"][cve] = _classify_error_from_log(content, canonical_lang)
        results[r] = rank
    return results


def build_cve_status(dataset: dict, patches: dict, results: dict, n_patches: int) -> list:
    rows = []
    for cve_id in sorted(dataset.keys()):
        meta = dataset[cve_id]
        status = {}
        fail_type = {}
        first_pass = None
        for r in range(1, n_patches + 1):
            if cve_id in results.get(r, {}).get("passed", set()):
                status[r] = "pass"
                if first_pass is None:
                    first_pass = r
            elif cve_id in results.get(r, {}).get("failed", {}):
                status[r] = "fail"
                fail_type[r] = results[r]["failed"][cve_id]
            else:
                status[r] = "not_run"

        if all(s == "not_run" for s in status.values()) and cve_id not in patches:
            continue

        tax_code = None
        if first_pass is None:
            for r in range(1, n_patches + 1):
                if status[r] == "fail":
                    log = results[r]["error_log"].get(cve_id, "")
                    tax_code = classify_failure(fail_type[r], log)
                    break

        rows.append({
            "cve":         cve_id,
            "meta":        meta,
            "status":      status,
            "fail_type":   fail_type,
            "tax_code":    tax_code,
            "first_pass_rank": first_pass,
            "patches":     patches.get(cve_id, {}),
            "error_logs":  {r: results[r]["error_log"].get(cve_id, "")
                            for r in range(1, n_patches + 1)},
        })
    return rows


# -- HTML rendering (giu nguyen tu ban DeepSeek) ------------------------------
CSS = """
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0;
       background: #0f172a; color: #e2e8f0; line-height: 1.55; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
h1 { font-size: 28px; margin: 0 0 8px; color: #f1f5f9; }
h2 { font-size: 22px; margin: 32px 0 12px; padding-bottom: 6px;
     border-bottom: 1px solid #334155; color: #f1f5f9; }
h3 { font-size: 17px; margin: 20px 0 8px; color: #cbd5e1; }
.meta { color: #94a3b8; font-size: 13px; margin-bottom: 24px; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 14px; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #334155; }
th { background: #1e293b; color: #cbd5e1; font-weight: 600; }
tr:hover td { background: #1e293b40; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
         font-size: 12px; font-weight: 600; }
.badge-pass { background: #16a34a; color: white; }
.badge-fail { background: #dc2626; color: white; }
.badge-none { background: #475569; color: #cbd5e1; }
.badge-tax  { background: #1e40af; color: white; }
.badge-infra{ background: #92400e; color: white; }
.badge-unk  { background: #6b7280; color: white; }
details { background: #1e293b; border-radius: 6px; margin: 10px 0;
          border: 1px solid #334155; }
details summary { padding: 12px 16px; cursor: pointer; font-weight: 600;
                  color: #f1f5f9; user-select: none; }
details summary:hover { background: #334155; }
details[open] summary { border-bottom: 1px solid #334155; }
.details-body { padding: 16px; }
pre { background: #020617; border: 1px solid #334155; border-radius: 4px;
      padding: 12px; overflow-x: auto; font-size: 12px; line-height: 1.45;
      color: #cbd5e1; margin: 8px 0; max-height: 500px; overflow-y: auto; }
.diff-add { background: #14532d40; color: #86efac; }
.diff-del { background: #7f1d1d40; color: #fca5a5; }
.diff-hdr { color: #93c5fd; }
.kv { display: grid; grid-template-columns: 160px 1fr; gap: 6px 16px;
      margin: 8px 0 16px; font-size: 14px; }
.kv .k { color: #94a3b8; }
.kv .v { color: #e2e8f0; word-break: break-all; }
.tax-box { background: #1e293b; border-left: 4px solid #f59e0b; padding: 12px 16px;
           margin: 12px 0; border-radius: 4px; }
.tax-box.tax-infra { border-left-color: #92400e; }
.tax-box.tax-pass  { border-left-color: #16a34a; }
.tax-title { font-weight: 700; margin-bottom: 4px; color: #fde68a; }
.tax-desc { color: #cbd5e1; font-size: 13px; }
.rank-row { display: flex; gap: 12px; margin: 8px 0; }
.rank-cell { flex: 1; background: #0f172a; border: 1px solid #334155;
             border-radius: 4px; padding: 10px; }
.rank-cell.pass { border-color: #16a34a; }
.rank-cell.fail { border-color: #dc2626; }
.rank-cell.none { border-color: #475569; opacity: 0.6; }
.filter { margin: 16px 0; }
.filter input { padding: 8px 12px; width: 300px; background: #1e293b;
                border: 1px solid #334155; color: #e2e8f0; border-radius: 4px; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
             gap: 12px; margin: 16px 0; }
.stat-card { background: #1e293b; padding: 16px; border-radius: 6px;
             border: 1px solid #334155; }
.stat-num { font-size: 26px; font-weight: 700; color: #f1f5f9; }
.stat-lbl { color: #94a3b8; font-size: 13px; margin-top: 4px; }
r"""

JS = """
document.addEventListener('DOMContentLoaded', () => {
  const filter = document.getElementById('cve-filter');
  if (filter) {
    filter.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase();
      document.querySelectorAll('details[data-cve]').forEach(d => {
        const cve = d.dataset.cve.toLowerCase();
        const tax = (d.dataset.tax || '').toLowerCase();
        d.style.display = (cve.includes(q) || tax.includes(q)) ? '' : 'none';
      });
    });
  }
});
r"""


def esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def render_diff(diff: str) -> str:
    if not diff:
        return '<pre>(khong co)</pre>'
    lines = []
    for line in diff.splitlines():
        e = esc(line)
        if line.startswith('+++') or line.startswith('---') or line.startswith('diff '):
            lines.append(f'<span class="diff-hdr">{e}</span>')
        elif line.startswith('@@'):
            lines.append(f'<span class="diff-hdr">{e}</span>')
        elif line.startswith('+'):
            lines.append(f'<span class="diff-add">{e}</span>')
        elif line.startswith('-'):
            lines.append(f'<span class="diff-del">{e}</span>')
        else:
            lines.append(e)
    return '<pre>' + '\n'.join(lines) + '</pre>'


def render_log_excerpt(log: str, max_lines: int = 40) -> str:
    if not log:
        return '<pre>(khong co log)</pre>'
    lines = log.splitlines()
    if len(lines) > max_lines:
        head = lines[:5]
        tail = lines[-(max_lines - 5):]
        excerpt = head + ['', f'... ({len(lines) - max_lines} dong bi luoc bo) ...', ''] + tail
    else:
        excerpt = lines
    return '<pre>' + esc('\n'.join(excerpt)) + '</pre>'


def render_status_badge(status: str) -> str:
    if status == "pass":
        return '<span class="badge badge-pass">PASS</span>'
    if status == "fail":
        return '<span class="badge badge-fail">FAIL</span>'
    return '<span class="badge badge-none">not run</span>'


def render_tax_badge(code: str) -> str:
    if not code:
        return ''
    if code == "INFRA":
        cls = "badge-infra"
    elif code == "UNKNOWN":
        cls = "badge-unk"
    else:
        cls = "badge-tax"
    return f'<span class="badge {cls}">{code}</span>'


def render_cve_detail(row: dict, n_patches: int) -> str:
    meta = row["meta"]
    cve_id = row["cve"]
    vul = meta["vul_func"][0]
    fix = meta["fix_func"][0] if meta.get("fix_func") else None
    file_path = vul.get("file_path", "?")
    start = vul.get("start_line", "?")
    end = vul.get("end_line", "?")
    patch_lines = vul.get("vul_localization", [{}])[0].get("patch_lines", [])
    cwes = list(meta.get("cwe_info", {}).keys())
    desc = meta.get("cve_description", "").strip() or "(khong co mo ta)"

    passed_ranks = [r for r, s in row["status"].items() if s == "pass"]
    if passed_ranks:
        head_status = f"PASS @rank {passed_ranks[0]}"
        if len(passed_ranks) > 1:
            head_status += f" (+ranks {', '.join(str(r) for r in passed_ranks[1:])})"
    else:
        head_status = "FAIL (moi rank)"

    tax_line = ""
    if row["tax_code"]:
        tt, td = TAXONOMY[row["tax_code"]]
        tax_cls = "tax-infra" if row["tax_code"] == "INFRA" else ""
        tax_line = f'''
        <div class="tax-box {tax_cls}">
          <div class="tax-title">Phan loai fail: {esc(tt)}</div>
          <div class="tax-desc">{esc(td)}</div>
        </div>'''
    elif row["first_pass_rank"]:
        tax_line = f'''
        <div class="tax-box tax-pass">
          <div class="tax-title">PASS — sua thanh cong o rank {row["first_pass_rank"]}</div>
          <div class="tax-desc">Xem patch ben duoi.</div>
        </div>'''

    kv = f'''
    <div class="kv">
      <div class="k">CWE</div>            <div class="v">{esc(', '.join(cwes) or 'N/A')}</div>
      <div class="k">File</div>           <div class="v">{esc(file_path)}</div>
      <div class="k">Line range</div>     <div class="v">{start} – {end}</div>
      <div class="k">Buggy line(s)</div>  <div class="v">{esc(', '.join(str(x) for x in patch_lines) or 'N/A')} (1-indexed trong snippet)</div>
      <div class="k">Mo ta CVE</div>      <div class="v">{esc(desc[:400])}{'...' if len(desc) > 400 else ''}</div>
    </div>
    '''

    rank_html_parts = []
    for r in range(1, n_patches + 1):
        st = row["status"].get(r, "not_run")
        cls = st if st in ("pass", "fail") else "none"
        cell_head = f'<div style="margin-bottom:6px;"><b>Rank {r}</b> {render_status_badge(st)}</div>'

        if st == "pass":
            fix_patch = row["patches"].get(r, "")
            cell_body = f'''<div style="font-size:13px;color:#94a3b8;margin-bottom:4px;">
                              CACH SUA (patch Llama 4 Scout sinh ra):
                            </div>
                            {render_diff(fix_patch)}'''
        elif st == "fail":
            err_type = row["fail_type"].get(r, "?")
            log = row["error_logs"].get(r, "")
            tax_code = classify_failure(err_type, log)
            tt, _ = TAXONOMY.get(tax_code, ("?", ""))
            cell_body = f'''<div style="font-size:13px;margin-bottom:6px;">
                              error_type: <code>{esc(err_type)}</code> → {render_tax_badge(tax_code)} {esc(tt)}
                            </div>
                            <details><summary style="font-size:12px;padding:6px 10px;">
                              Log excerpt ({len(log.splitlines()) if log else 0} dong)
                            </summary>
                            <div class="details-body">{render_log_excerpt(log)}</div>
                            </details>'''
        else:
            cell_body = '<div style="font-size:13px;color:#64748b;">(chua chay)</div>'

        rank_html_parts.append(
            f'<div class="rank-cell {cls}">{cell_head}{cell_body}</div>'
        )

    ref_snippet = ""
    if fix and fix.get("snippet"):
        ref_snippet = f'''
        <h3>Ground-truth fix (tham chieu)</h3>
        <details><summary>Xem snippet</summary>
        <div class="details-body">
          <div style="font-size:13px;color:#94a3b8;margin-bottom:6px;">
            File: <code>{esc(fix.get("file_path", file_path))}</code> —
            lines {fix.get("start_line", '?')} – {fix.get("end_line", '?')}
          </div>
          <pre>{esc(fix["snippet"])}</pre>
        </div>
        </details>'''

    body = f'''
    {tax_line}
    <h3>Metadata</h3>
    {kv}
    <h3>Ket qua tung rank</h3>
    <div class="rank-row">{"".join(rank_html_parts)}</div>
    {ref_snippet}
    '''

    tax_disp = render_tax_badge(row["tax_code"]) if row["tax_code"] else ""
    summary_title = f'{esc(cve_id)} — {head_status} {tax_disp}'
    if cwes:
        summary_title += f' <span style="color:#94a3b8;font-weight:400;font-size:13px;">[{esc(", ".join(cwes[:2]))}]</span>'

    tax_attr = row["tax_code"] or ""
    return f'''
    <details data-cve="{esc(cve_id)}" data-tax="{esc(tax_attr)}">
      <summary>{summary_title}</summary>
      <div class="details-body">{body}</div>
    </details>
    '''


def build_html(lang: str, n_patches: int, rows: list) -> str:
    total = len(rows)
    passed = [r for r in rows if r["first_pass_rank"] is not None]
    failed = [r for r in rows if r["first_pass_rank"] is None]

    cumulative = set()
    plausible_rows = []
    for k in range(1, n_patches + 1):
        newly = 0
        for r in rows:
            if r["first_pass_rank"] == k:
                newly += 1
                cumulative.add(r["cve"])
        rate = (len(cumulative) / total * 100) if total else 0
        plausible_rows.append((k, newly, len(cumulative), rate))

    tax_counts = {}
    for r in failed:
        code = r["tax_code"] or "UNKNOWN"
        tax_counts[code] = tax_counts.get(code, 0) + 1

    stat_cards = f'''
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-num">{total}</div><div class="stat-lbl">Total CVE</div></div>
      <div class="stat-card"><div class="stat-num" style="color:#86efac;">{len(passed)}</div><div class="stat-lbl">Pass o it nhat 1 rank</div></div>
      <div class="stat-card"><div class="stat-num" style="color:#fca5a5;">{len(failed)}</div><div class="stat-lbl">Fail moi rank</div></div>
      <div class="stat-card"><div class="stat-num">{n_patches}</div><div class="stat-lbl">Candidates / CVE</div></div>
    </div>
    '''

    plausible_html = '<table><tr><th>k</th><th>Newly passed</th><th>Cumulative</th><th>Plausible@k</th></tr>'
    for k, newly, cum, rate in plausible_rows:
        plausible_html += f'<tr><td>{k}</td><td>{newly}</td><td>{cum}</td><td><b>{rate:.2f}%</b></td></tr>'
    plausible_html += '</table>'

    tax_html = '<table><tr><th>Code</th><th>Ten</th><th>So CVE</th></tr>'
    for code, count in sorted(tax_counts.items(), key=lambda x: -x[1]):
        tt, _ = TAXONOMY.get(code, ("?", ""))
        tax_html += f'<tr><td>{render_tax_badge(code)}</td><td>{esc(tt)}</td><td>{count}</td></tr>'
    tax_html += '</table>'

    pass_details = "".join(render_cve_detail(r, n_patches) for r in passed)
    fail_details = "".join(render_cve_detail(r, n_patches) for r in failed)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    canonical = LANG_CANONICAL.get(lang.lower(), lang)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wu et al. — {canonical} — {MODEL_ID} — PatchEval</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
  <h1>Bao cao Wu et al. (ISSTA 2023) tren PatchEval — {canonical}</h1>
  <div class="meta">
    Sinh luc {now} · Model = <b>{esc(MODEL_ID)}</b> (OpenRouter) · N candidates / CVE = {n_patches} · temperature = 0.8
  </div>

  <h2>Tong quan</h2>
  {stat_cards}

  <h3>Plausible@k (Wu et al. metric)</h3>
  {plausible_html}

  <h3>Phan bo ly do fail (theo taxonomy A1–D3)</h3>
  {tax_html}

  <div class="filter">
    <input id="cve-filter" placeholder="Loc theo CVE ID hoac tax code (vi du: B1, INFRA, CVE-2020)..." />
  </div>

  <h2>CVE Pass ({len(passed)})</h2>
  {pass_details if pass_details else '<p style="color:#94a3b8;">Chua co CVE nao pass.</p>'}

  <h2>CVE Fail moi rank ({len(failed)})</h2>
  {fail_details if fail_details else '<p style="color:#94a3b8;">Khong co CVE fail.</p>'}
</div>
<script>{JS}</script>
</body>
</html>
r"""


# -- CLI ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="HTML report cho Llama 4 Scout run")
    p.add_argument("--lang", required=True, choices=["python", "go", "javascript", "js"])
    p.add_argument("--n-patches", type=int, default=3)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    canonical = LANG_CANONICAL[args.lang.lower()]
    print(f"[LOAD] dataset ({canonical})...")
    dataset = load_dataset(canonical)
    print(f"[LOAD] patches (llama-4-scout, n={args.n_patches})...")
    patches = load_patches(args.lang, args.n_patches)
    print(f"[LOAD] rank results...")
    results = load_rank_results(args.lang, args.n_patches)

    print(f"[BUILD] aggregating per-CVE status...")
    rows = build_cve_status(dataset, patches, results, args.n_patches)

    passed_count = sum(1 for r in rows if r["first_pass_rank"] is not None)
    print(f"  → {len(rows)} CVE tracked, {passed_count} pass, {len(rows) - passed_count} fail")

    print(f"[RENDER] HTML...")
    html_out = build_html(args.lang, args.n_patches, rows)

    out_path = Path(args.out) if args.out else PATCHEVAL_ROOT / f"wu_report_{args.lang.lower()}_{FILE_SLUG}.html"
    out_path.write_text(html_out, encoding="utf-8")
    print(f"\nBao cao HTML da luu: {out_path}")
    print(f"Mo bang browser: start {out_path.name}   (PowerShell tren Windows)")


if __name__ == "__main__":
    main()
