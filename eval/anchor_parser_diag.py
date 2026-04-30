"""Diagnostic: run anchor_parser on all 5 papers' rubrics.json.

Reports per-paper and cross-paper anchor extraction rate, broken down
by rubric type.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.rubric_normalizer.anchor_parser import (  # noqa: E402
    parse_anchors,
    has_any_anchor,
)


PAPERS = ["AMUN", "Beyond-Ngram", "I0T", "INCLINE", "min-p"]


def load_rubrics(paper: str):
    path = PROJECT_ROOT / "data" / "train_valid" / paper / "rubrics.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def diag_one(paper: str) -> dict:
    rubrics = load_rubrics(paper)
    total = len(rubrics)
    by_type: dict = defaultdict(lambda: {"total": 0, "with_anchor": 0})
    anchor_kind_counts = Counter()
    no_anchor_examples = []

    for r in rubrics:
        rtype = r.get("type", "Unknown")
        criteria = r.get("criteria", "")
        parsed = parse_anchors(criteria)

        by_type[rtype]["total"] += 1
        if has_any_anchor(parsed):
            by_type[rtype]["with_anchor"] += 1
            for kind in ("sections", "tables", "figures",
                         "algorithms", "equations"):
                if parsed[kind]:
                    anchor_kind_counts[kind] += 1
        elif len(no_anchor_examples) < 3:
            # Sample some no-anchor criteria for inspection
            no_anchor_examples.append((rtype, criteria[:90]))

    print(f"\n{'=' * 64}")
    print(f"Paper: {paper}  ({total} rubrics)")
    print('=' * 64)

    print(f"\nBy-type anchor extraction:")
    print(f"  {'Type':<22} {'Total':>6} {'WithAnchor':>11} {'Rate':>7}")
    overall_with = 0
    for rtype, stats in sorted(by_type.items()):
        rate = stats["with_anchor"] / stats["total"] if stats["total"] else 0
        print(f"  {rtype:<22} {stats['total']:>6} "
              f"{stats['with_anchor']:>11} {rate:>6.1%}")
        overall_with += stats["with_anchor"]
    overall_rate = overall_with / total if total else 0
    print(f"  {'-' * 50}")
    print(f"  {'OVERALL':<22} {total:>6} {overall_with:>11} {overall_rate:>6.1%}")

    print(f"\nAnchor kind frequencies:")
    for kind in ("sections", "tables", "figures", "algorithms", "equations"):
        print(f"  {kind:<12}: {anchor_kind_counts[kind]}")

    if no_anchor_examples:
        print(f"\nSample rubrics WITHOUT any anchor (first 3):")
        for rtype, snippet in no_anchor_examples:
            print(f"  [{rtype}] {snippet}...")

    return {
        "paper": paper,
        "total": total,
        "with_anchor": overall_with,
        "rate": overall_rate,
        "by_type": dict(by_type),
        "kind_counts": dict(anchor_kind_counts),
    }


def main():
    results = [diag_one(p) for p in PAPERS]

    print(f"\n{'=' * 64}")
    print("Cross-paper summary")
    print('=' * 64)
    print(f"{'Paper':<14} {'Total':>6} {'WithAnchor':>11} {'Rate':>7}")
    print("-" * 44)
    grand_total = 0
    grand_with = 0
    for r in results:
        print(f"{r['paper']:<14} {r['total']:>6} "
              f"{r['with_anchor']:>11} {r['rate']:>6.1%}")
        grand_total += r["total"]
        grand_with += r["with_anchor"]
    print("-" * 44)
    grand_rate = grand_with / grand_total if grand_total else 0
    print(f"{'TOTAL':<14} {grand_total:>6} {grand_with:>11} {grand_rate:>6.1%}")

    # Per-type aggregate (across all papers)
    type_agg: dict = defaultdict(lambda: {"total": 0, "with_anchor": 0})
    for r in results:
        for rtype, stats in r["by_type"].items():
            type_agg[rtype]["total"] += stats["total"]
            type_agg[rtype]["with_anchor"] += stats["with_anchor"]

    print(f"\nBy-type aggregate (5 papers combined):")
    print(f"  {'Type':<22} {'Total':>6} {'WithAnchor':>11} {'Rate':>7}")
    for rtype in sorted(type_agg):
        s = type_agg[rtype]
        rate = s["with_anchor"] / s["total"] if s["total"] else 0
        print(f"  {rtype:<22} {s['total']:>6} {s['with_anchor']:>11} {rate:>6.1%}")


if __name__ == "__main__":
    main()
