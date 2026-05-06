"""Diagnostic: extract entities from all 5 papers and report counts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paper_observer.section_parser import parse_paper_md  # noqa: E402
from pipeline.paper_observer.entity_extractor import extract_from_paper  # noqa: E402


PAPERS = ["AMUN", "Beyond-Ngram", "I0T", "INCLINE", "min-p"]


def diag_one(paper: str) -> dict:
    paper_path = PROJECT_ROOT / "data" / "train_valid" / paper / "paper.md"
    sections = parse_paper_md(str(paper_path))
    entities = extract_from_paper(str(paper_path), sections)

    print(f"\n{'=' * 64}")
    print(f"Paper: {paper}")
    print('=' * 64)

    summary = {}
    for kind in ("tables", "figures", "algorithms", "equations"):
        items = entities[kind]
        summary[kind] = len(items)
        print(f"\n{kind.upper()} ({len(items)}):")
        for e in items[:5]:
            cap = e["caption"]
            if len(cap) > 60:
                cap = cap[:57] + "..."
            sec = e["in_section"] or "(none)"
            print(f"  [{e['id']:<14}] line {e['first_mention_line']:>4} "
                  f"in {sec:<10}  caption={cap!r}")
        if len(items) > 5:
            print(f"  ... and {len(items) - 5} more")

    captioned = {
        kind: sum(1 for e in entities[kind] if e["caption"])
        for kind in ("tables", "figures", "algorithms")
    }
    print(f"\nCaptioned counts (out of total):")
    for kind in ("tables", "figures", "algorithms"):
        print(f"  {kind:<12}: {captioned[kind]} / {summary[kind]}")

    return {"paper": paper, **summary, "captioned": captioned}


def main():
    results = [diag_one(p) for p in PAPERS]

    print(f"\n{'=' * 64}")
    print("Cross-paper summary")
    print('=' * 64)
    print(f"{'Paper':<14} {'Tables':>7} {'Figures':>8} {'Algos':>6} {'Equations':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['paper']:<14} {r['tables']:>7} {r['figures']:>8} "
              f"{r['algorithms']:>6} {r['equations']:>10}")

    print(f"\nCaption recognition rate (table+figure+algorithm only):")
    print(f"{'Paper':<14} {'Captioned':>10} {'Total':>6} {'Rate':>7}")
    print("-" * 40)
    for r in results:
        cap_total = sum(r["captioned"].values())
        ent_total = r["tables"] + r["figures"] + r["algorithms"]
        rate = cap_total / ent_total if ent_total else 0
        print(f"{r['paper']:<14} {cap_total:>10} {ent_total:>6} {rate:>6.1%}")


if __name__ == "__main__":
    main()
