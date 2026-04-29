"""
Section parser for paper.md files in NLPCC 2026 Task 11.

Parses the markdown content of a paper into a hierarchical list of sections.
Pure rule-based; no LLM calls.

The parser handles 5 cases (see rules A-E in module docstring of parse_sections):
  A. Numbered sections (e.g., "## 3.1 Notation")
  B. Appendix sections with letter prefix (e.g., "## A.1 Proof")
  C. Unnumbered top-level sections (e.g., "# Abstract")
  D. Filtering of polluted list-item-as-heading lines (min-p paper)
  E. line_end computed as last line before next same-or-higher-level section
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


# ----------------------------------------------------------------------------
# Regex patterns
# ----------------------------------------------------------------------------

# Rule A: numbered section, e.g.  "## 3.1. Notation"  or  "# 2 Method"
_NUMBERED_RE = re.compile(
    r'^(#{1,6})\s+'                  # one or more '#'
    r'(\d+(?:\.\d+)*)'               # numeric path: "3" or "3.1" or "3.1.1"
    r'\.?'                           # optional trailing dot
    r'\s+'                           # whitespace
    r'(.+?)\s*$'                     # title (non-greedy, strip trailing space)
)

# Rule B: appendix, e.g.  "## A.1 Proof"
_APPENDIX_RE = re.compile(
    r'^(#{1,6})\s+'
    r'([A-Z](?:\.\d+)*)'             # 'A' or 'A.1' or 'A.1.1'
    r'\.?'
    r'\s+'
    r'(.+?)\s*$'
)

# Rule C: unnumbered top-level (only level=1, '# Title')
_UNNUMBERED_TOP_RE = re.compile(r'^#\s+(.+?)\s*$')

# Rule D pollution patterns
_POLLUTION_LIST_LEADER = re.compile(r'^\s*(?:[-*•]|\d+\.)\s+')


# ----------------------------------------------------------------------------
# Section dataclass
# ----------------------------------------------------------------------------


@dataclass
class Section:
    id: str
    title: str
    level: int
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    parent: Optional[str]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert title text to a URL-safe slug for unnumbered sections."""
    s = text.lower().strip()
    # Drop anything that's not alnum / space / hyphen
    s = re.sub(r'[^\w\s-]', '', s)
    # Collapse whitespace runs to single hyphen
    s = re.sub(r'[\s_]+', '-', s)
    # Collapse multiple hyphens
    s = re.sub(r'-+', '-', s).strip('-')
    return s or 'untitled'


def _is_polluted_title(title: str) -> bool:
    """Rule D: detect list-item-as-heading pollution (min-p paper).

    Returns True if the title should be filtered out.
    """
    stripped = title.strip()

    # Empty after trimming
    if not stripped:
        return True

    # Starts with a list marker char
    if stripped[0] in '-*•':
        return True

    # Begins with a markdown list leader like "- foo" or "1. foo"
    if _POLLUTION_LIST_LEADER.match(stripped):
        return True

    # Too short (alphanumeric chars only)
    alnum_only = re.sub(r'\W+', '', stripped)
    if len(alnum_only) < 3:
        return True

    return False


def _classify_heading(line: str):
    """Try to classify a line as A / B / C heading.

    Returns a tuple (kind, level, ident, full_title) or None if not a heading.
      kind   : 'A' | 'B' | 'C'
      level  : int  (depth, NOT '#' count)
      ident  : numeric/letter id like '3.1' or 'A.1' or slug 'abstract'
      full_title : human-facing title incl. number prefix
    """
    # Check Rule A first (numbered) — covers most sections
    m = _NUMBERED_RE.match(line)
    if m:
        hashes, num_path, raw_title = m.groups()
        if _is_polluted_title(raw_title):
            return None
        depth = num_path.count('.') + 1                              # rule A
        full_title = f"{num_path} {raw_title}".strip()
        return ('A', depth, num_path, full_title)

    # Then Rule B (appendix letters)
    m = _APPENDIX_RE.match(line)
    if m:
        hashes, letter_path, raw_title = m.groups()
        if _is_polluted_title(raw_title):
            return None
        depth = letter_path.count('.') + 1                           # rule B
        full_title = f"{letter_path} {raw_title}".strip()
        return ('B', depth, letter_path, full_title)

    # Finally Rule C (unnumbered, level=1 only — '# Title')
    if line.startswith('# ') and not line.startswith('## '):
        m = _UNNUMBERED_TOP_RE.match(line)
        if m:
            raw_title = m.group(1)
            if _is_polluted_title(raw_title):
                return None
            slug = _slugify(raw_title)
            return ('C', 1, slug, raw_title)

    return None


# ----------------------------------------------------------------------------
# Main entry points
# ----------------------------------------------------------------------------


def parse_sections(md_content: str) -> List[Dict]:
    """Parse paper.md content into a list of section dicts.

    See module docstring for parsing rules. Returned list is in document order.
    """
    lines = md_content.splitlines(keepends=True)
    total_lines = len(lines)

    # First pass: collect heading entries with line_start + char_start
    raw_headings = []  # list of (line_no_1based, kind, level, ident, title, char_start)
    char_offset = 0
    for idx, line in enumerate(lines):
        # 1-indexed line number
        line_no = idx + 1
        stripped = line.rstrip('\n').rstrip('\r')
        if stripped.startswith('#'):
            classified = _classify_heading(stripped)
            if classified:
                kind, level, ident, title = classified
                raw_headings.append(
                    (line_no, kind, level, ident, title, char_offset)
                )
        char_offset += len(line)

    # Second pass: compute line_end / char_end / parent and build Sections.
    # Rule E: line_end = next same-or-higher-level heading's line_start - 1.
    sections: List[Section] = []
    parent_stack: List[Section] = []

    for i, (line_start, kind, level, ident, title, char_start) in enumerate(raw_headings):
        # Find the next heading whose level <= current level
        line_end = total_lines
        char_end = len(md_content)
        for j in range(i + 1, len(raw_headings)):
            next_line_start, _, next_level, _, _, next_char_start = raw_headings[j]
            if next_level <= level:
                line_end = next_line_start - 1
                char_end = next_char_start
                break

        # Parent: pop stack entries with level >= current level, then peek top.
        while parent_stack and parent_stack[-1].level >= level:
            parent_stack.pop()
        parent_id = parent_stack[-1].id if parent_stack else None

        # ID prefix: 'sec-' for all
        sec_id = f"sec-{ident}"

        sec = Section(
            id=sec_id,
            title=title,
            level=level,
            line_start=line_start,
            line_end=line_end,
            char_start=char_start,
            char_end=char_end,
            parent=parent_id,
        )
        sections.append(sec)
        parent_stack.append(sec)

    return [asdict(s) for s in sections]


def parse_paper_md(paper_path: str) -> List[Dict]:
    """Convenience wrapper: parse a paper.md file by path."""
    with open(paper_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return parse_sections(content)


# ----------------------------------------------------------------------------
# Demo: run on AMUN if invoked directly
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    amun_path = project_root / "data" / "train_valid" / "AMUN" / "paper.md"
    if not amun_path.exists():
        print(f"AMUN paper.md not found at {amun_path}", file=sys.stderr)
        sys.exit(1)

    sections = parse_paper_md(str(amun_path))
    print(f"Parsed {len(sections)} sections from AMUN\n")
    print("First 10:")
    for s in sections[:10]:
        print(f"  [{s['id']:<20}] L{s['level']}  {s['title'][:60]}")
    print(f"\nLast section:")
    if sections:
        s = sections[-1]
        print(f"  [{s['id']:<20}] L{s['level']}  {s['title'][:60]}")
        print(f"  line_start={s['line_start']}, line_end={s['line_end']}")
