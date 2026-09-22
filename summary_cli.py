"""
In bang so sanh nhieu model trong CLI (khong HTML).

Auto-discover:
  - Report:  wu_report_<lang>*.json
  - Stats:   wu_stats_<lang>*.json

Chay:
    python wu_method\\summary_cli.py
    python wu_method\\summary_cli.py --lang javascript
    python wu_method\\summary_cli.py --lang javascript --method LLM4CVE
"""

import argparse
import json
import re
import sys
from pathlib import Path

WU_DIR         = Path(__file__).resolve().parent
PATCHEVAL_ROOT = WU_DIR.parent


# Map slug → display name. Them model moi vao day khi can.
MODEL_DISPLAY = {
    "":                 "DeepSeek V4 Flash",   # file cu khong slug
    "deepseek":         "DeepSeek V4 Flash",
    "qwen3_14b":        "Qwen3-14B",
    "claude_sonnet_5":  "Claude Sonnet 5",
    "llama4_scout":     "Llama 4 Scout",
    "gpt_oss_20b":      "GPT-OSS 20B",
}


# ANSI color codes
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    GRAY   = "\033[90m"


def color_rate(rate: float) -> str:
    if rate >= 40:  return f"{C.GREEN}{rate:6.2f}%{C.RESET}"
    if rate >= 25:  return f"{C.YELLOW}{rate:6.2f}%{C.RESET}"
    return f"{C.RED}{rate:6.2f}%{C.RESET}"


def color_cost(cost: float) -> str:
    if cost < 0.15:  return f"{C.GREEN}${cost:.4f}{C.RESET}"
    if cost < 0.50:  return f"{C.YELLOW}${cost:.4f}{C.RESET}"
    return f"{C.RED}${cost:.4f}{C.RESET}"


def na(width: int = 8) -> str:
    return f"{C.GRAY}{'—':^{width}}{C.RESET}"


def fmt_int(n, width: int = 8) -> str:
    if n is None:
        return na(width)
    return f"{n:>{width},}"


def fmt_time(s, width: int = 10) -> str:
    if s is None:
        return na(width)
    return f"{s:>{width-2},.1f} s"


def discover_runs(lang: str) -> list[dict]:
    """Quet file wu_report_<lang>*.json + match voi wu_stats_<lang>*.json."""
    pattern_report = re.compile(rf"^wu_report_{lang}(?:_(.+))?\.json$")
    runs = []
    for path in PATCHEVAL_ROOT.iterdir():
        if not path.is_file():
            continue
        m = pattern_report.match(path.name)
        if not m:
            continue
        slug = m.group(1) or ""
        model_name = MODEL_DISPLAY.get(slug, slug or "?")

        # Doc report
        try:
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        except Exception as e:
            print(f"[WARN] Cannot read {path.name}: {e}")
            continue

        total = report.get("total_cves_run", 0)
        plaus = report.get("plausible_at_k", [])
        pass_count = plaus[-1]["count"] if plaus else 0
        best_k     = plaus[-1]["k"] if plaus else "?"

        # Tim stats file tuong ung
        stats_name = f"wu_stats_{lang}_{slug}.json" if slug else f"wu_stats_{lang}.json"
        stats_path = PATCHEVAL_ROOT / stats_name
        stats = {}
        if stats_path.exists():
            try:
                with open(stats_path, encoding="utf-8") as f:
                    stats = json.load(f)
            except Exception:
                pass

        runs.append({
            "slug":          slug,
            "model":         model_name,
            "pass":          pass_count,
            "total":         total,
            "best_k":        best_k,
            "input_tokens":  stats.get("input_tokens"),
            "output_tokens": stats.get("output_tokens"),
            "total_tokens":  stats.get("total_tokens"),
            "cost_usd":      stats.get("cost_usd"),
            "latency_s":     stats.get("latency_s"),
            "attempts":      stats.get("attempts"),
            "has_stats":     bool(stats),
        })

    # Sort: model có stats hiện trước, DeepSeek trước Qwen trước Llama
    slug_order = ["", "deepseek", "claude_sonnet_5", "qwen3_14b", "llama4_scout"]
    runs.sort(key=lambda r: slug_order.index(r["slug"]) if r["slug"] in slug_order else 99)
    return runs


def print_table(runs: list[dict], lang: str, method_label: str):
    total_cves = runs[0]["total"] if runs else 0

    # Header line
    print()
    title = f"So sanh method + model tren PatchEval {lang.upper()} ({total_cves} CVE)"
    print(f"{C.BOLD}{C.CYAN}{title}{C.RESET}")
    print("=" * 128)

    # Header row
    header = (
        f"{C.BOLD}"
        f"{'Method':<9} {'Model':<20} "
        f"{'PASS':>7}  {'Rate':>8}  "
        f"{'Input':>9} {'Output':>9} {'Total':>9}  "
        f"{'Cost':>10}  {'Latency':>12}  {'Attempts':>9}"
        f"{C.RESET}"
    )
    print(header)
    print("-" * 128)

    for r in runs:
        pass_str = f"{r['pass']:>3}/{r['total']:<3}"
        rate = 100 * r["pass"] / max(r["total"], 1)
        rate_col = color_rate(rate)

        cost_col = color_cost(r["cost_usd"]) if r["cost_usd"] is not None else na(10)

        line = (
            f"{method_label:<9} {r['model']:<20} "
            f"{pass_str:>7}  {rate_col}  "
            f"{fmt_int(r['input_tokens'], 9)} "
            f"{fmt_int(r['output_tokens'], 9)} "
            f"{fmt_int(r['total_tokens'], 9)}  "
            f"{cost_col:>10}  "
            f"{fmt_time(r['latency_s'], 12)}  "
            f"{fmt_int(r['attempts'], 9)}"
        )
        print(line)

    print("=" * 128)
    print()

    # Legend
    print(f"{C.DIM}Chu thich:{C.RESET}")
    print(f"  Rate: {C.GREEN}xanh >=40%{C.RESET}  {C.YELLOW}vang 25-40%{C.RESET}  {C.RED}do <25%{C.RESET}")
    print(f"  Cost: {C.GREEN}xanh <$0.15{C.RESET}  {C.YELLOW}vang $0.15-$0.50{C.RESET}  {C.RED}do >$0.50{C.RESET}")
    print(f"  {C.GRAY}—{C.RESET} = chua co du lieu (run truoc khi track stats, hoac chua chay xong)")
    print()

    # Quick delta summary
    if len(runs) >= 2:
        best = max(runs, key=lambda r: r["pass"])
        best_rate = 100 * best["pass"] / max(best["total"], 1)
        print(f"{C.BOLD}Best pass rate:{C.RESET} {best['model']} @ {best_rate:.2f}% "
              f"({best['pass']}/{best['total']} @k={best['best_k']})")
        if all(r["cost_usd"] is not None for r in runs):
            cheapest = min(runs, key=lambda r: r["cost_usd"])
            print(f"{C.BOLD}Cheapest:{C.RESET}       {cheapest['model']} @ ${cheapest['cost_usd']:.4f}")
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="javascript",
                   choices=["python", "go", "javascript"])
    p.add_argument("--method", default="LLM4CVE",
                   help="Nhan Method o cot dau (default LLM4CVE).")
    p.add_argument("--no-color", action="store_true",
                   help="Tat ANSI color (dung neu terminal khong support).")
    args = p.parse_args()

    if args.no_color:
        for attr in dir(C):
            if not attr.startswith("_"):
                setattr(C, attr, "")

    runs = discover_runs(args.lang)
    if not runs:
        print(f"[FAIL] Khong tim thay wu_report_{args.lang}*.json nao trong {PATCHEVAL_ROOT}")
        print(f"       Chay stage report cho it nhat 1 model truoc.")
        return

    print_table(runs, args.lang, args.method)


if __name__ == "__main__":
    main()
