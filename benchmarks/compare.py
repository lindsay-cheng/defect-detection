"""compare two replay .md reports and print a regression delta table.

usage: compare.py <results_a.md> <results_b.md>
# ponytail: markdown parsing of our own stable replay.py output format. upgrade
# = metrics json sidecar (parse numbers instead of regex-scraping markdown).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import fmt_ci, wilson_interval

NUM = r"(-?\d+)"
CROSS_RE = re.compile(r"crossings detected:\s*(\d+)\s*/\s*expected:\s*(\d+)")
SWITCH_RE = re.compile(r"total class switches:\s*" + NUM)
NEAR_RE = re.compile(r"switches within .*?:\s*" + NUM)
VERDICT_RE = re.compile(
    r"OK-good=(\d+),\s*OK-defect=(\d+),\s*FALSE-POSITIVE=(\d+),\s*MISS=(\d+),\s*WRONG-TYPE=(\d+)"
)
STAB_RE = re.compile(
    r"(raw|stabilized)[^\n]*mean stability\s*([0-9.]+),\s*perfectly-stable\s*(\d+)/(\d+)"
)


def parse_report(path: str) -> dict:
    text = Path(path).read_text()
    out: dict = {
        "crossings": None,
        "expected": None,
        "switches": None,
        "near": None,
        "ok_good": 0,
        "ok_defect": 0,
        "false_positive": 0,
        "miss": 0,
        "wrong_type": 0,
        "stab_raw_mean": None,
        "stab_stable_mean": None,
        "stab_raw_perfect": None,
        "stab_stable_perfect": None,
    }
    m = CROSS_RE.search(text)
    if m:
        out["crossings"] = int(m.group(1))
        out["expected"] = int(m.group(2))
    m = SWITCH_RE.search(text)
    if m:
        out["switches"] = int(m.group(1))
    m = NEAR_RE.search(text)
    if m:
        out["near"] = int(m.group(1))
    m = VERDICT_RE.search(text)
    if m:
        out["ok_good"] = int(m.group(1))
        out["ok_defect"] = int(m.group(2))
        out["false_positive"] = int(m.group(3))
        out["miss"] = int(m.group(4))
        out["wrong_type"] = int(m.group(5))
    for sm in STAB_RE.finditer(text):
        mean = float(sm.group(2))
        pct = f"{int(sm.group(3))}/{int(sm.group(4))}"
        if sm.group(1) == "raw":
            out["stab_raw_mean"] = mean
            out["stab_raw_perfect"] = pct
        else:
            out["stab_stable_mean"] = mean
            out["stab_stable_perfect"] = pct
    return out


def correct_log_rate(r: dict) -> tuple[float, tuple[float, float]]:
    ok = r["ok_good"] + r["ok_defect"]
    exp = r["expected"] or 0
    rate = (ok / exp) if exp else 0.0
    ci = wilson_interval(ok, exp)
    return rate, ci


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    ra = parse_report(args.a)
    rb = parse_report(args.b)
    name_a = Path(args.a).stem
    name_b = Path(args.b).stem

    def row(label, va, vb, fmt="{:g}"):
        sa = fmt.format(va) if va is not None else "(absent)"
        sb = fmt.format(vb) if vb is not None else "(absent)"
        delta = ""
        try:
            delta = f"Δ={float(vb) - float(va):+g}"
        except (TypeError, ValueError):
            delta = ""
        return f"| {label} | {sa} | {sb} | {delta} |"

    rate_a, ci_a = correct_log_rate(ra)
    rate_b, ci_b = correct_log_rate(rb)
    overlap = not (ci_b[1] < ci_a[0] or ci_a[1] < ci_b[0])

    lines = []
    lines.append(f"# replay compare — `{name_a}` vs `{name_b}`")
    lines.append("")
    lines.append("| metric | A | B | delta |")
    lines.append("|---|---|---|---|")
    lines.append(row("crossings detected", ra["crossings"], rb["crossings"], "{:d}"))
    lines.append(row("expected (GT)", ra["expected"], rb["expected"], "{:d}"))
    lines.append(row("total switches", ra["switches"], rb["switches"], "{:d}"))
    lines.append(row("near-crossing switches", ra["near"], rb["near"], "{:d}"))
    lines.append(row("OK-good", ra["ok_good"], rb["ok_good"], "{:d}"))
    lines.append(row("OK-defect", ra["ok_defect"], rb["ok_defect"], "{:d}"))
    lines.append(row("FALSE-POSITIVE", ra["false_positive"], rb["false_positive"], "{:d}"))
    lines.append(row("MISS", ra["miss"], rb["miss"], "{:d}"))
    lines.append(row("WRONG-TYPE", ra["wrong_type"], rb["wrong_type"], "{:d}"))
    lines.append(row("correct-log rate", rate_a, rate_b, "{:.3f}"))
    lines.append(row("raw mean stability", ra["stab_raw_mean"], rb["stab_raw_mean"], "{:.3f}"))
    lines.append(
        row("stabilized mean stability", ra["stab_stable_mean"], rb["stab_stable_mean"], "{:.3f}")
    )
    lines.append(row("raw perfectly-stable", ra["stab_raw_perfect"], rb["stab_raw_perfect"], "{}"))
    lines.append(
        row(
            "stabilized perfectly-stable",
            ra["stab_stable_perfect"],
            rb["stab_stable_perfect"],
            "{}",
        )
    )
    lines.append("")
    lines.append(
        f"- correct-log rate CI overlap: A={fmt_ci(*ci_a)} B={fmt_ci(*ci_b)} -> "
        f"{'OVERLAP (no significant difference)' if overlap else 'DISJOINT (flags regression)'}"
    )
    lines.append("")
    out = "\n".join(lines)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
