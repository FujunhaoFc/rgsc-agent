"""
Section parser for paper.md files in NLPCC 2026 Task 11.

Parses the markdown content of a paper into a hierarchical list of sections.
Pure rule-based; no LLM calls.

Parsing rules (see implementation comments for "rule X" markers):

  A. Numbered sections, e.g. "## 3.1 Notation"
  B. Appendix sections with letter prefix, e.g. "## A.1 Proof"
  C. Unnumbered top-level sections, e.g. "# Abstract"
  D. Filter list-item-as-heading pollution (min-p case)
  E. line_end = last line before next same-or-higher-level section
  F. Document metadata zone (paper title, authors, affiliations) preceding
     the first true section is dropped from sections and recorded separately
  G. Unnumbered headings appearing AFTER the References / Bibliography
     section are filtered (Beyond-Ngram annotation form pollution)
  H. Unnumbered titles must contain at least one letter (defensive)

The parsed result has shape:
  {
      "sections": [Section, ...],
      "metadata_headings": [Section-shaped dict, ...]   # rule F + G drop list
  }
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------------
# Regex patterns
# ----------------------------------------------------------------------------

# Rule A: numbered section, e.g. "## 3.1. Notation"  or "# 2 Method"
_NUMBERED_RE = re.compile(
    r'^(#{1,6})\s+'                  # one or more '#'
    r'(\d+(?:\.\d+)*)'               # numeric path: "3" or "3.1" or "3.1.1"
    r'\.?'                           # optional trailing dot
    r'\s+'                           # whitespace
    r'(.+?)\s*$'                     # title (non-greedy, strip trailing space)
)

# Rule B: appendix, e.g. "## A.1 Proof"
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

# Rule F: known "true section entry" words (lowercased) used to detect
# the boundary between metadata and main body.
_TRUE_SECTION_ENTRY_WORDS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "preliminaries",
}

# Rule G: words that mark References / Bibliography
_REFERENCES_WORDS = {"references", "bibliography"}


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
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s_]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s or 'untitled'


def _is_polluted_title(title: str) -> bool:
    """Rule D: detect list-item-as-heading pollution (min-p paper)."""
    stripped = title.strip()
    if not stripped:
        return True
    if stripped[0] in '-*•':
        return True
    if _POLLUTION_LIST_LEADER.match(stripped):
        return True
    alnum_only = re.sub(r'\W+', '', stripped)
    if len(alnum_only) < 3:
        return True
    return False


def _has_letter(text: str) -> bool:
    """Rule H: title must contain at least one alphabetic char."""
    return bool(re.search(r'[A-Za-z]', text))


def _classify_heading(line: str):
    """Try to classify a line as A / B / C heading.

    Returns a tuple (kind, level, ident, full_title) or None if not a heading.
    """
    # Rule A first
    m = _NUMBERED_RE.match(line)
    if m:
        _hashes, num_path, raw_title = m.groups()
        if _is_polluted_title(raw_title):
            return None
        depth = num_path.count('.') + 1
        full_title = f"{num_path} {raw_title}".strip()
        return ('A', depth, num_path, full_title)

    # Rule B
    m = _APPENDIX_RE.match(line)
    if m:
        _hashes, letter_path, raw_title = m.groups()
        if _is_polluted_title(raw_title):
            return None
        depth = letter_path.count('.') + 1
        full_title = f"{letter_path} {raw_title}".strip()
        return ('B', depth, letter_path, full_title)

    # Rule C  (only "# Title", strict level=1)
    if line.startswith('# ') and not line.startswith('## '):
        m = _UNNUMBERED_TOP_RE.match(line)
        if m:
            raw_title = m.group(1)
            if _is_polluted_title(raw_title):
                return None
            if not _has_letter(raw_title):                              # rule H
                return None
            slug = _slugify(raw_title)
            return ('C', 1, slug, raw_title)

    return None


def _is_true_section_entry(kind: str, title: str) -> bool:
    """Rule F: does this heading mark the start of the main paper body?

    True if kind is A or B (numbered/appendix) or kind is C and the lowercased
    title matches a known section entry word.
    """
    if kind in ('A', 'B'):
        return True
    title_lc = title.strip().lower()
    if title_lc in _TRUE_SECTION_ENTRY_WORDS:
        return True
    # also accept "1 introduction"-style appearing as kind=C should never
    # happen, but allow common variants like "abstract" with trailing chars
    for w in _TRUE_SECTION_ENTRY_WORDS:
        if title_lc.startswith(w):
            return True
    return False


def _is_references_title(title: str) -> bool:
    """Rule G helper: is this title References / Bibliography?"""
    t = title.strip().lower()
    return any(w in t for w in _REFERENCES_WORDS)


# ----------------------------------------------------------------------------
# Main entry points
# ----------------------------------------------------------------------------


def parse_sections(md_content: str) -> List[Dict]:
    """Backward-compatible entry: returns just the sections list (drops metadata).

    Most existing callers (and tests) only want the sections array. For full
    output including metadata_headings, use parse_paper_full().
    """
    return parse_paper_full(md_content)["sections"]


def parse_paper_full(md_content: str) -> Dict:
    """Parse paper.md content. Returns dict with 'sections' and 'metadata_headings'.

    sections           : list of Section dicts that constitute the paper body.
    metadata_headings  : list of Section dicts dropped by rules F or G
                         (paper title, author lines, References-trailing
                         pollution, etc.). Useful for downstream debugging.
    """
    lines = md_content.splitlines(keepends=True)
    total_lines = len(lines)

    # --- Pass 1: scan all heading-shaped lines, keep classified ones --------
    raw_headings: List[Tuple[int, str, int, str, str, int]] = []
    # tuple: (line_no_1based, kind, level, ident, title, char_start)

    char_offset = 0
    for idx, line in enumerate(lines):
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

    # --- Pass 2: rule F — find boundary between metadata and body ----------
    first_body_idx = None
    for i, (_, kind, _, _, title, _) in enumerate(raw_headings):
        if _is_true_section_entry(kind, title):
            first_body_idx = i
            break

    # If we never found a true entry (degenerate paper), keep all as body.
    if first_body_idx is None:
        first_body_idx = 0

    metadata_raw = raw_headings[:first_body_idx]
    body_raw = raw_headings[first_body_idx:]

    # --- Pass 3: rule G — filter unnumbered headings AFTER References ------
    references_seen = False
    body_kept: List[Tuple[int, str, int, str, str, int]] = []
    body_dropped_by_G: List[Tuple[int, str, int, str, str, int]] = []

    for entry in body_raw:
        _line_no, kind, _level, _ident, title, _char = entry
        if references_seen and kind == 'C':
            body_dropped_by_G.append(entry)
            continue
        body_kept.append(entry)
        if _is_references_title(title):
            references_seen = True

    # --- Pass 4: build Sections for body_kept (rule E for line_end) --------
    # We need next-heading lookup against body_kept itself.
    sections: List[Section] = []
    parent_stack: List[Section] = []

    for i, entry in enumerate(body_kept):
        line_start, _kind, level, ident, title, char_start = entry

        # Rule E: line_end = (next same-or-higher-level entry line_start) - 1
        line_end = total_lines
        char_end = len(md_content)
        for j in range(i + 1, len(body_kept)):
            next_line_start, _, next_level, _, _, next_char_start = body_kept[j]
            if next_level <= level:
                line_end = next_line_start - 1
                char_end = next_char_start
                break

        # Parent computation
        while parent_stack and parent_stack[-1].level >= level:
            parent_stack.pop()
        parent_id = parent_stack[-1].id if parent_stack else None

        sec = Section(
            id=f"sec-{ident}",
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

    # --- Pass 5: build metadata_headings list (rule F + rule G drops) ------
    # For metadata entries we don't compute parent/line_end carefully — they
    # are diagnostic only. Provide minimal info.
    metadata_headings: List[Dict] = []
    for entry in metadata_raw + body_dropped_by_G:
        line_start, _kind, level, ident, title, char_start = entry
        metadata_headings.append({
            "id": f"meta-{ident}" if not ident.replace('.', '').isalnum() or len(ident) > 30
                  else f"meta-{ident}",
            "title": title,
            "level": level,
            "line_start": line_start,
            "char_start": char_start,
            "source_rule": "F" if entry in metadata_raw else "G",
        })

    return {
        "sections": [asdict(s) for s in sections],
        "metadata_headings": metadata_headings,
    }


def parse_paper_md(paper_path: str) -> List[Dict]:
    """Convenience wrapper: parse a paper.md file by path. Returns sections only."""
    with open(paper_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return parse_sections(content)


def parse_paper_md_full(paper_path: str) -> Dict:
    """Like parse_paper_md but returns the full {sections, metadata_headings} dict."""
    with open(paper_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return parse_paper_full(content)


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    amun_path = project_root / "data" / "train_valid" / "AMUN" / "paper.md"
    if not amun_path.exists():
        print(f"AMUN paper.md not found at {amun_path}", file=sys.stderr)
        sys.exit(1)

    full = parse_paper_md_full(str(amun_path))
    sections = full["sections"]
    metadata = full["metadata_headings"]

    print(f"AMUN  →  {len(sections)} sections, {len(metadata)} metadata-dropped\n")

    print("Metadata (rule F/G drops):")
    for m in metadata:
        print(f"  ({m['source_rule']}) L{m['level']}  {m['title'][:60]}  @line {m['line_start']}")

    print("\nFirst 10 body sections:")
    for s in sections[:10]:
        print(f"  [{s['id']:<14}] L{s['level']}  {s['title'][:60]}")

    print(f"\nLast section:")
    if sections:
        s = sections[-1]
        print(f"  [{s['id']:<14}] L{s['level']}  {s['title'][:60]}")
        print(f"  line_start={s['line_start']}, line_end={s['line_end']}")