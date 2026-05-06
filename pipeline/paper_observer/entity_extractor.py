"""
Entity extractor: find Tables, Figures, Algorithms, Equations in paper.md.

Pure regex; no LLM calls.

For each entity type, produces a list of:
    {
        "id": "table-1",
        "label": "1",
        "caption": "...",           # may be empty if not recognized
        "first_mention_line": 412,  # 1-indexed
        "context_lines": [407, 432],
        "in_section": "sec-5"       # may be None if outside any section
    }

Reuses the regex patterns from anchor_parser to ensure that idents extracted
from paper.md and idents extracted from rubrics.json are mutually consistent.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


# Reuse the same patterns as anchor_parser, but anchored to paper text
# (case sensitivity matters less here; both forms appear).
_TABLE_RE = re.compile(r'\b(?:Table|Tab\.?)\s*(\d+[a-z]?)\b', re.IGNORECASE)
_FIGURE_RE = re.compile(r'\b(?:Figure|Fig\.?)\s*(\d+[a-z]?)\b', re.IGNORECASE)
_ALGORITHM_RE = re.compile(r'\b(?:Algorithm|Algorihm|Algo\.?)\s*(\d+)\b', re.IGNORECASE)
_EQUATION_RE = re.compile(r'\b(?:Equation|Eq\.?)\s*\(?(\d+)\)?\b', re.IGNORECASE)

# Caption recognition: a line that starts with "Table N:" / "Table N." or
# is bold-marked. We're permissive: try several patterns.
_CAPTION_PATTERNS = {
    'table': re.compile(r'^\s*(?:\*\*)?Table\s+(\d+[a-z]?)\s*[:.]\s*(.+?)(?:\*\*)?\s*$', re.IGNORECASE),
    'figure': re.compile(r'^\s*(?:\*\*)?Figure\s+(\d+[a-z]?)\s*[:.]\s*(.+?)(?:\*\*)?\s*$', re.IGNORECASE),
    'algorithm': re.compile(r'^\s*(?:\*\*)?Algorithm\s+(\d+)\s*[:.]?\s*(.+?)(?:\*\*)?\s*$', re.IGNORECASE),
}

# Context window around first mention
_CONTEXT_BEFORE = 5
_CONTEXT_AFTER = 20


@dataclass
class Entity:
    id: str
    label: str
    caption: str
    first_mention_line: int
    context_lines: List[int]    # [start, end], 1-indexed inclusive
    in_section: Optional[str]


def _find_section_for_line(line_no: int, sections: List[Dict]) -> Optional[str]:
    """Return the deepest-level section id containing `line_no`, or None."""
    candidates = [
        s for s in sections
        if s["line_start"] <= line_no <= s["line_end"]
    ]
    if not candidates:
        return None
    # Deepest section = highest level value
    return max(candidates, key=lambda s: s["level"])["id"]


def _find_caption(
    md_lines: List[str],
    first_mention_line: int,
    label: str,
    pattern_key: str,
) -> str:
    """Search for a caption line near the first mention.

    Looks at the first_mention line itself and the next 3 lines.
    Returns the caption text, or empty string if not found.
    """
    pattern = _CAPTION_PATTERNS.get(pattern_key)
    if pattern is None:
        return ""

    # 1-indexed → 0-indexed
    start = max(0, first_mention_line - 1)
    end = min(len(md_lines), first_mention_line + 3)

    for i in range(start, end):
        m = pattern.match(md_lines[i].rstrip('\n'))
        if m and m.group(1) == label:
            return m.group(2).strip()
    return ""


def _scan_entities(
    md_content: str,
    pattern: re.Pattern,
    kind: str,
    sections: List[Dict],
    has_caption: bool,
) -> List[Dict]:
    """Generic entity scanner.

    Tracks first-mention line for each unique label, then enriches with
    caption + context + section info.
    """
    md_lines = md_content.splitlines()
    first_mentions: Dict[str, int] = {}  # label -> line_no (1-indexed)

    for idx, line in enumerate(md_lines):
        for m in pattern.finditer(line):
            label = m.group(1)
            line_no = idx + 1
            if label not in first_mentions:
                first_mentions[label] = line_no

    # Order entities by their first mention line for a stable, document-order list
    ordered_labels = sorted(first_mentions.keys(), key=lambda lbl: first_mentions[lbl])

    entities: List[Entity] = []
    for label in ordered_labels:
        first_line = first_mentions[label]
        caption = ""
        if has_caption:
            caption = _find_caption(md_lines, first_line, label, kind)

        ctx_start = max(1, first_line - _CONTEXT_BEFORE)
        ctx_end = min(len(md_lines), first_line + _CONTEXT_AFTER)

        in_section = _find_section_for_line(first_line, sections) if sections else None

        entities.append(Entity(
            id=f"{kind}-{label}",
            label=label,
            caption=caption,
            first_mention_line=first_line,
            context_lines=[ctx_start, ctx_end],
            in_section=in_section,
        ))

    return [asdict(e) for e in entities]


def extract_entities(md_content: str, sections: Optional[List[Dict]] = None) -> Dict:
    """Extract Tables, Figures, Algorithms, Equations from paper.md content.

    `sections` is the output of section_parser.parse_sections; if provided,
    each entity gets an `in_section` field. Otherwise that field is None.
    """
    if sections is None:
        sections = []

    return {
        "tables": _scan_entities(md_content, _TABLE_RE, "table", sections, has_caption=True),
        "figures": _scan_entities(md_content, _FIGURE_RE, "figure", sections, has_caption=True),
        "algorithms": _scan_entities(md_content, _ALGORITHM_RE, "algorithm", sections, has_caption=True),
        "equations": _scan_entities(md_content, _EQUATION_RE, "equation", sections, has_caption=False),
    }


def extract_from_paper(paper_path: str, sections: Optional[List[Dict]] = None) -> Dict:
    """Convenience wrapper: extract entities from a paper.md file."""
    with open(paper_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return extract_entities(content, sections)


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from pipeline.paper_observer.section_parser import parse_paper_md  # noqa: E402

    amun = project_root / "data" / "train_valid" / "AMUN" / "paper.md"
    sections = parse_paper_md(str(amun))
    entities = extract_from_paper(str(amun), sections)

    for kind in ("tables", "figures", "algorithms", "equations"):
        print(f"\n{kind.upper()} ({len(entities[kind])}):")
        for e in entities[kind][:5]:
            cap = e["caption"][:50] + "..." if len(e["caption"]) > 50 else e["caption"]
            print(f"  [{e['id']}] line {e['first_mention_line']:>4} "
                  f"in {e['in_section']}  caption={cap!r}")
        if len(entities[kind]) > 5:
            print(f"  ... and {len(entities[kind]) - 5} more")
