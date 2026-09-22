"""
Build HTML summary table so sanh nhieu method + model.

Doc file `summary_data.json` (user tu bao tri) va render bang HTML tuong tu
screenshot: Method | Model | PASS | Rate | Input | Output | Total tokens |
            Cost | LLM latency | Attempts.

Chay:
    python wu_method\\build_summary_table.py

Format `summary_data.json`:
{
  "title": "So sanh method + model tren PatchEval JavaScript (77 CVE)",
  "runs": [
    {
      "method":         "LLM4CVE",
      "model":          "DeepSeek V4 Flash",
      "pass":           37,
      "total":          77,
      "input_tokens":   249768,
      "output_tokens":  1859269,
      "cost_usd":       0.30714904,
      "latency_s":      27041.40,
      "attempts":       214
    },
    ...
  ]
}

Cot nao khong co du lieu (vd Qwen chua track tokens) thi de null → render "-".
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

WU_DIR         = Path(__file__).resolve().parent
PATCHEVAL_ROOT = WU_DIR.parent
DEFAULT_DATA   = PATCHEVAL_ROOT / "summary_data.json"


CSS = """
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
       background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }
.container { max-width: 1400px; margin: 0 auto; }
h1 { color: #f1f5f9; margin: 0 0 8px; font-size: 24px; }
.meta { color: #94a3b8; font-size: 13px; margin-bottom: 24px; }
table { width: 100%; border-collapse: collapse; background: #1e293b;
        border-radius: 8px; overflow: hidden; }
th { background: #0f172a; color: #cbd5e1; padding: 14px 16px; text-align: left;
     font-weight: 600; font-size: 13px; text-transform: none;
     border-bottom: 2px solid #334155; }
td { padding: 16px; color: #e2e8f0; font-size: 14px;
     border-bottom: 1px solid #334155; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #0f172a80; }
.method { font-weight: 600; color: #f1f5f9; }
.model { color: #cbd5e1; font-size: 13px; }
.pass { font-weight: 700; color: #f1f5f9; }
.rate { font-weight: 700; color: #86efac; }
.rate-low { color: #fca5a5; }
.rate-mid { color: #fde68a; }
.rate-high { color: #86efac; }
.num { font-family: "SF Mono", Menlo, Consolas, monospace; color: #cbd5e1; }
.cost { font-weight: 600; }
.cost-low { color: #86efac; }
.cost-mid { color: #fde68a; }
.cost-high { color: #fca5a5; }
.latency { font-family: "SF Mono", Menlo, Consolas, monospace; }
.na { color: #64748b; font-style: italic; }
.legend { color: #94a3b8; font-size: 12px; margin-top: 16px; }
.legend code { background: #1e293b; padding: 2px 6px; border-radius: 3px; }
"""


def fmt_num(n):
    if n is None:
        return '<span class="na">-</span>'
    if isinstance(n, int):
        return f'<span class="num">{n:,}</span>'
    return f'<span class="num">{n:,.2f}</span>'


def fmt_pass(p, t):
    if p is None or t is None:
        return '<span class="na">-</span>'
    return f'<span class="pass">{p}/{t}</span>'


def fmt_rate(p, t):
    if p is None or t is None:
        return '<span class="na">-</span>'
    rate = 100 * p / t
    cls = "rate-high" if rate >= 50 else "rate-mid" if rate >= 30 else "rate-low"
    return f'<span class="rate {cls}">{rate:.2f}%</span>'


def fmt_cost(c):
    if c is None:
        return '<span class="na">-</span>'
    cls = "cost-low" if c < 0.15 else "cost-mid" if c < 0.5 else "cost-high"
    return f'<span class="cost {cls}">${c:.8f}</span>'


def fmt_latency(s):
    if s is None:
        return '<span class="na">-</span>'
    return f'<span class="latency">{s:,.2f} s</span>'


def build_html(data: dict) -> str:
    title = data.get("title", "So sanh method + model")
    runs = data.get("runs", [])
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    rows = []
    for run in runs:
        rows.append(f"""
        <tr>
          <td class="method">{run.get("method", "?")}</td>
          <td class="model">{run.get("model", "?")}</td>
          <td>{fmt_pass(run.get("pass"), run.get("total"))}</td>
          <td>{fmt_rate(run.get("pass"), run.get("total"))}</td>
          <td>{fmt_num(run.get("input_tokens"))}</td>
          <td>{fmt_num(run.get("output_tokens"))}</td>
          <td>{fmt_num(run.get("total_tokens") or
                       (run.get("input_tokens", 0) + run.get("output_tokens", 0)
                        if run.get("input_tokens") and run.get("output_tokens") else None))}</td>
          <td>{fmt_cost(run.get("cost_usd"))}</td>
          <td>{fmt_latency(run.get("latency_s"))}</td>
          <td>{fmt_num(run.get("attempts"))}</td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <div class="meta">Sinh luc {now}</div>

  <table>
    <thead>
      <tr>
        <th>Method</th>
        <th>Model</th>
        <th>PASS</th>
        <th>Rate</th>
        <th>Input</th>
        <th>Output</th>
        <th>Total tokens</th>
        <th>Cost</th>
        <th>LLM latency</th>
        <th>Attempts</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>

  <div class="legend">
    <b>Chu thich cot:</b>
    <ul>
      <li><code>PASS</code>: so CVE pass / tong CVE danh gia</li>
      <li><code>Rate</code>: pass rate — <span style="color:#86efac;">xanh &ge;50%</span>,
          <span style="color:#fde68a;">vang 30-50%</span>,
          <span style="color:#fca5a5;">do &lt;30%</span></li>
      <li><code>Input/Output/Total tokens</code>: tong token API dung ca run</li>
      <li><code>Cost</code>: chi phi USD ca run (tinh theo pricing tai thoi diem chay)</li>
      <li><code>LLM latency</code>: tong thoi gian goi LLM (khong tinh Docker validate)</li>
      <li><code>Attempts</code>: tong so lan goi API (bao gom retries va dedup rejects)</li>
      <li>Cot <span class="na">-</span> = chua co du lieu, can retrofit tracking khi chay lai</li>
    </ul>
  </div>
</div>
</body>
</html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                   help=f"Path file JSON data. Default: {DEFAULT_DATA.name}")
    p.add_argument("--out", type=str, default=None,
                   help="Path file HTML output. Default: summary_table.html")
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[FAIL] Khong tim thay {data_path}")
        print(f"Tao mau {DEFAULT_DATA.name} truoc:")
        print(SAMPLE_DATA_TEMPLATE)
        return

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    html = build_html(data)
    out_path = Path(args.out) if args.out else PATCHEVAL_ROOT / "summary_table.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] Bang HTML luu: {out_path}")
    print(f"Mo bang browser: start {out_path.name}")


SAMPLE_DATA_TEMPLATE = '''
{
  "title": "So sanh method + model tren PatchEval JavaScript (77 CVE)",
  "runs": [
    {
      "method": "LLM4CVE",
      "model": "DeepSeek V4 Flash",
      "pass": 37,
      "total": 77,
      "input_tokens": null,
      "output_tokens": null,
      "cost_usd": null,
      "latency_s": null,
      "attempts": null
    },
    {
      "method": "LLM4CVE",
      "model": "Qwen3-14B",
      "pass": 23,
      "total": 77,
      "input_tokens": null,
      "output_tokens": null,
      "cost_usd": null,
      "latency_s": null,
      "attempts": null
    }
  ]
}
'''


if __name__ == "__main__":
    main()
