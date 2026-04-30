"""
Anchor parser: extract structured references from rubric criteria text.

Pure regex + heuristics; no LLM calls.

Given a rubric criteria string like:
    "The agent has read section 3.1 to understand the notations,
     and verified Table 1 and Algorithm 2."

Produces:
    {
        "sections": ["3.1"],
        "tables": ["1"],
        "figures": [],
        "algorithms": ["2"],
        "equations": [],
        "raw_matches": [
            {"kind": "section",   "match": "section 3.1", "ident": "3.1"},
            {"kind": "table",     "match": "Table 1",     "ident": "1"},
            {"kind": "algorithm", "match": "Algorithm 2", "ident": "2"},
        ]
    }

Design notes:
- We accept both numeric (3.1) and letter (A.2) idents for sections.
- We tolerate common typos seen in real rubric data:
    "Algorihm" (missing 't') -> AMUN's rubrics.json has 5 such
- We do NOT cross-validate against a paper_state here; that is normalize.py's
  job. This module is purely about textual extraction.
"""

from __future__ import annotations

import re
from typing import Dict, List


# ----------------------------------------------------------------------------
# Regex patterns
#
# All patterns use word boundaries / context to avoid false positives like
# matching "Table 1" inside a URL or bibtex-key.
# ----------------------------------------------------------------------------

# section 3.1 / Section 3 / sec. 3.1 / sec 3 / §3.1
# Allow letter idents like A, A.1 too.
_SECTION_RE = re.compile(
    r'(?:(?<=\W)|(?<=^))(?:section|sec\.?|§)\s*'   # 前置：词首 或 非字母后
    r'([0-9A-Z](?:\.\d+)*)'                         # 编号
    r'(?=\W|$)',                                    # 后置：非字母 或 结尾
    re.IGNORECASE,
)

# Table 1 / Tab. 3 / Table 1a (we strip the trailing letter for ident)
_TABLE_RE = re.compile(
    r'\b(?:table|tab\.?)\s*'
    r'(\d+[a-z]?)'
    r'\b',
    re.IGNORECASE,
)

# Figure 1 / Fig. 3 / Figure 1a
_FIGURE_RE = re.compile(
    r'\b(?:figure|fig\.?)\s*'
    r'(\d+[a-z]?)'
    r'\b',
    re.IGNORECASE,
)

# Algorithm 1 / Algo. 2  +  tolerate "Algorihm" typo (AMUN rubrics)
_ALGORITHM_RE = re.compile(
    r'\b(?:algorithm|algorihm|algo\.?)\s*'   # 'algorihm' = common typo
    r'(\d+)'
    r'\b',
    re.IGNORECASE,
)

# equation (5) / Eq. (5) / eq 5 / equation 5
# Both forms: parenthesized and bare.
_EQUATION_RE = re.compile(
    r'\b(?:equation|eq\.?)\s*'
    r'\(?(\d+)\)?'
    r'\b',
    re.IGNORECASE,
)


# Mapping from regex to (kind label, target list key)
_PATTERNS = [
    ("section",   _SECTION_RE,   "sections"),
    ("table",     _TABLE_RE,     "tables"),
    ("figure",    _FIGURE_RE,    "figures"),
    ("algorithm", _ALGORITHM_RE, "algorithms"),
    ("equation",  _EQUATION_RE,  "equations"),
]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def parse_anchors(criteria: str) -> Dict:
    """Extract structured references from a rubric criteria string.

    Returns a dict with 5 ident lists + raw_matches diagnostic.
    Lists are deduplicated while preserving first-seen order.
    """
    result: Dict[str, List] = {
        "sections": [],
        "tables": [],
        "figures": [],
        "algorithms": [],
        "equations": [],
        "raw_matches": [],
    }
    seen_idents: Dict[str, set] = {
        "sections": set(),
        "tables": set(),
        "figures": set(),
        "algorithms": set(),
        "equations": set(),
    }

    for kind, pattern, list_key in _PATTERNS:
        for m in pattern.finditer(criteria):
            ident = m.group(1)
            # Normalize: strip optional trailing letter for tables/figures
            # but KEEP it in raw_matches for traceability.
            normalized = ident
            if kind in ("table", "figure"):
                # "1a" -> "1" only if you want to merge — but for rubric
                # cross-ref we keep "1a" as a distinct ident.
                pass

            if normalized not in seen_idents[list_key]:
                result[list_key].append(normalized)
                seen_idents[list_key].add(normalized)

            result["raw_matches"].append({
                "kind": kind,
                "match": m.group(0),
                "ident": normalized,
                "span": [m.start(), m.end()],
            })

    return result


def has_any_anchor(parsed: Dict) -> bool:
    """Convenience: True if at least one of the 5 ident lists is non-empty."""
    return any(parsed.get(k) for k in
               ("sections", "tables", "figures", "algorithms", "equations"))


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    samples = [
        "The agent has read section 3.1 to understand the notations.",
        "Verified Table 1 and Algorithm 2 against the paper.",
        "See equation (5) and Figure 3a for details.",
        "The agent identified the unlearning method described in §3.1 "
        "and the algorihm 1 in Section A.2.",  # tests typo + appendix
        "No anchors here at all, just plain English.",
    ]
    import json
    for s in samples:
        print(f"\nInput: {s}")
        parsed = parse_anchors(s)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
