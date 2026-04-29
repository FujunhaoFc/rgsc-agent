"""Diagnostic script: run section_parser on all 5 train_valid papers and report.

Outputs per-paper stats and a cross-paper summary table.

Usage:
    python eval/section_parser_diag.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

# Make project root importable when running this script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paper_observer.section_parser import parse_paper_md  # noqa: E402


PAPERS = ["AMUN", "Beyond-Ngram", "I0T", "INCLINE", "min-p"]


def _count_filtered(paper_path: Path) -> int:
    """Count heading-shaped lines that were filtered by rule D.

    A heading-shaped line is anything starting with '#' followed by a space.
    We compare that count against the parsed sections count.
    """
    with open(paper_path, "r", encoding="utf-8") as f:
        content = f.read()
    heading_lines = [
        line for line in content.splitlines() if re.match(r"^#{1,6}\s", line)
    ]
    parsed = parse_paper_md(str(paper_path))
    return max(0, len(heading_lines) - len(parsed))


def _file_total_lines(paper_path: Path) -> int:
    with open(paper_path, "r", encoding="utf-8") as f:
        return len(f.read().splitlines())


def diag_one(paper_name: str) -> dict:
    paper_path = PROJECT_ROOT / "data" / "train_valid" / paper_name / "paper.md"
    total_lines = _file_total_lines(paper_path)
    sections = parse_paper_md(str(paper_path))
    level_dist = Counter(s["level"] for s in sections)
    filtered = _count_filtered(paper_path)

    print(f"\n{'=' * 64}")
    print(f"Paper: {paper_name}  ({total_lines} lines)")
    print('=' * 64)

    print(f"Total sections: {len(sections)}")
    parts = [f"L{lvl}={level_dist.get(lvl, 0)}" for lvl in sorted(level_dist)]
    print(f"Level distribution: {', '.join(parts)}")

    print("\nFirst 5 sections:")
    for s in sections[:5]:
        print(f"  [{s['id']:<20}] L{s['level']}  {s['title'][:60]}")

    print("\nLast section:")
    if sections:
        s = sections[-1]
        print(f"  [{s['id']:<20}] L{s['level']}  {s['title'][:60]}")
        print(f"  line_start={s['line_start']}, line_end={s['line_end']}")

    last_end = sections[-1]["line_end"] if sections else 0
    coverage_ok = "✓" if last_end == total_lines else "✗"
    print(f"\nCoverage check:")
    print(f"  Last line_end: {last_end} / {total_lines}  {coverage_ok}")
    print(f"  Filtered titles (rule D): {filtered}")

    return {
        "paper": paper_name,
        "total_sections": len(sections),
        "level_dist": dict(level_dist),
        "last_line_end": last_end,
        "total_lines": total_lines,
        "filtered": filtered,
    }


def main():
    results = [diag_one(p) for p in PAPERS]

    print(f"\n{'=' * 64}")
    print("Cross-paper summary")
    print('=' * 64)
    header = f"{'Paper':<14} {'Sections':>9} {'L1':>4} {'L2':>4} {'L3':>4} {'L4+':>4} {'Filtered':>9} {'EOF':>5}"
    print(header)
    print("-" * len(header))
    for r in results:
        ld = r["level_dist"]
        l1 = ld.get(1, 0)
        l2 = ld.get(2, 0)
        l3 = ld.get(3, 0)
        l4plus = sum(v for k, v in ld.items() if k >= 4)
        eof_ok = "✓" if r["last_line_end"] == r["total_lines"] else "✗"
        print(
            f"{r['paper']:<14} {r['total_sections']:>9} "
            f"{l1:>4} {l2:>4} {l3:>4} {l4plus:>4} "
            f"{r['filtered']:>9} {eof_ok:>5}"
        )

    print()
    # Sanity flags
    issues = []
    for r in results:
        if r["total_sections"] < 10 or r["total_sections"] > 80:
            issues.append(
                f"{r['paper']}: section count {r['total_sections']} out of [10,80]"
            )
        if r["last_line_end"] != r["total_lines"]:
            issues.append(
                f"{r['paper']}: last line_end {r['last_line_end']} != "
                f"total {r['total_lines']}"
            )
    if issues:
        print("⚠ Issues detected:")
        for x in issues:
            print(f"  - {x}")
    else:
        print("✓ All papers passed basic sanity checks.")


if __name__ == "__main__":
    main()
