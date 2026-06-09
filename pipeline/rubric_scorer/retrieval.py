"""Action filtering for rubric scoring.

Card 3: full per-type pre-filtering with keyword ranking and content truncation.
See docs/rubric_scorer_design.md Section 6 for design rationale.

All filtering is pure Python string matching — no external deps (no sklearn/nltk).
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Stop words for keyword extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "of", "for", "to", "in", "on", "and", "or",
    "has", "have", "agent", "this", "with", "that", "be", "it", "as",
    "by", "at", "from", "not", "but", "are", "was", "were", "been",
    "can", "will", "would", "should", "could", "may", "its",
})

_CONTENT_MAX_CHARS = 1500
_TRUNC_MARKER = "...[truncated]"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_path(action: Dict[str, Any]) -> str:
    """Extract path from an action, handling both top-level and arguments.path."""
    path = action.get("path", "")
    if not path:
        path = action.get("arguments", {}).get("path", "")
    return str(path)


def _get_tool(action: Dict[str, Any]) -> str:
    """Extract tool name, case-insensitive normalized."""
    return str(action.get("tool", "")).lower()


def _get_command(action: Dict[str, Any]) -> str:
    """Extract command string from an Execute action."""
    cmd = action.get("command", "")
    if not cmd:
        cmd = action.get("arguments", {}).get("command", "")
    return str(cmd)


# ---------------------------------------------------------------------------
# Step 1: Type-based filtering
# ---------------------------------------------------------------------------


def _filter_by_type(
    rubric_type: str, actions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Filter actions by rubric type using per-type rules (design doc §6.1).

    Returns actions matching the type-specific tool + path conditions.
    Order is preserved (original timestamp order).
    """
    rules = _RULES.get(rubric_type)
    if rules is None:
        # Unknown type — return empty (conservative)
        return []

    matched: List[Dict[str, Any]] = []
    for action in actions:
        tool = _get_tool(action)
        path = _get_path(action).lower()
        command = _get_command(action).lower()

        for rule_tool, path_patterns, extra_cond in rules:
            # Tool match (case-insensitive)
            if tool != rule_tool.lower():
                continue

            # Path match (substring, case-insensitive via lowered path)
            if path_patterns:
                if not any(p.lower() in path for p in path_patterns):
                    continue

            # Extra condition (e.g. Result Matching stdout check)
            if extra_cond and not extra_cond(action):
                continue

            matched.append(action)
            break  # Only match once per action

    return matched


def _lambda_exclude_paper_md(action: Dict[str, Any]) -> bool:
    """Reject actions whose path looks like paper.md (avoid false Plan Writing match)."""
    path = _get_path(action).lower()
    return "paper" not in path or ".py" in path


def _result_matching_extra(action: Dict[str, Any]) -> bool:
    """Extra check for Result Matching: Execute actions must reference metrics.

    For Execute: check that stdout or command contains metric-like keywords.
    For Read: always passes (already path-filtered to result/metric files).
    """
    tool = _get_tool(action)
    if tool in ("read", "exportrecord"):
        return True  # Path filter already handled metric/result files

    # Execute: check for metric keywords in stdout or command
    stdout = str(action.get("result", {}).get("stdout", "")).lower()
    command = _get_command(action).lower()
    # Look for numbers or common metric names
    metric_hints = (
        "accuracy", "acc", "loss", "score", "f1", "precision", "recall",
        "bleu", "rouge", "perplexity", "error", "%", "rate",
        "0.", "1.",  # numeric output
    )
    return any(h in stdout or h in command for h in metric_hints)


# Per-type rule table: (tool, path_keywords_or_None, extra_condition_or_None)
# path_keywords=None means no path filter (all paths match).
_RULES: Dict[str, List[tuple]] = {
    "Paper Observation": [
        ("read", ["paper.md", "paper"], None),
    ],
    "Plan Writing": [
        # Write/Edit to plan/design/.md files, but EXCLUDE paper.md
        ("write", ["plan", "design"], None),
        ("edit", ["plan", "design"], None),
        ("write", [".md"], _lambda_exclude_paper_md),
        ("edit", [".md"], _lambda_exclude_paper_md),
    ],
    "Code Implementation": [
        ("write", [".py"], None),
        ("edit", [".py"], None),
    ],
    "Command Execution": [
        ("execute", None, None),
    ],
    "Result Matching": [
        ("execute", None, _result_matching_extra),
        ("read", ["result", "metric", "output", ".json", ".csv"], None),
        ("exportrecord", None, None),
    ],
}


# ---------------------------------------------------------------------------
# Step 2: Keyword extraction and ranking
# ---------------------------------------------------------------------------


def _extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from rubric criteria text.

    Lowercase → split on non-alphanumeric → remove stop words and short tokens.
    """
    # Split on non-alphanumeric (keep letters, digits, hyphens for compound words)
    tokens = re.split(r"[^a-z0-9\-]+", text.lower())
    keywords: List[str] = []
    for token in tokens:
        token = token.strip("-")
        if len(token) <= 2:
            continue
        if token in _STOP_WORDS:
            continue
        keywords.append(token)
    return keywords


def _action_search_text(action: Dict[str, Any]) -> str:
    """Build a searchable string from an action for keyword overlap scoring.

    Includes path, command (for Execute), and relevant content fields.
    """
    tool = _get_tool(action)
    parts = [_get_path(action).lower()]

    if tool == "execute":
        parts.append(_get_command(action).lower())
        parts.append(str(action.get("result", {}).get("stdout", "")).lower())
    elif tool in ("write", "edit"):
        content = action.get("arguments", {}).get("content", "")
        parts.append(str(content).lower())
    elif tool == "read":
        content = action.get("result", {}).get("content", "")
        parts.append(str(content).lower())

    return " ".join(parts)


def _rank_by_keyword_overlap(
    rubric: Dict[str, Any], actions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Rank actions by keyword overlap with rubric.criteria.

    Keywords extracted from rubric.criteria. Each action gets a score = number
    of distinct keyword occurrences in the action's searchable text.
    Sorted descending by score; ties preserve original order (stable sort).
    """
    criteria = str(rubric.get("criteria", "") or "")
    keywords = _extract_keywords(criteria)

    if not keywords or not actions:
        return list(actions)  # No ranking possible, return original order

    # Compute scores
    scored: List[tuple] = []
    for idx, action in enumerate(actions):
        search_text = _action_search_text(action)
        score = sum(1 for kw in keywords if kw in search_text)
        scored.append((score, idx, action))

    # Sort: descending score, then ascending original index for stability
    scored.sort(key=lambda x: (-x[0], x[1]))

    return [item[2] for item in scored]


# ---------------------------------------------------------------------------
# Step 3: Truncation
# ---------------------------------------------------------------------------


def _truncate_content(text: str, max_chars: int) -> str:
    """Truncate a string to max_chars, appending truncation marker if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + _TRUNC_MARKER


def _truncate(
    ranked: List[Dict[str, Any]],
    max_actions: int,
    content_max_chars: int = _CONTENT_MAX_CHARS,
) -> List[Dict[str, Any]]:
    """Take top max_actions and truncate content fields to content_max_chars.

    Each action is deep-copied so the original list is not mutated.
    Truncation targets:
      - Write/Edit → arguments.content
      - Execute → result.stdout and result.stderr
      - Read → result.content

    Metadata fields (id, timestamp, duration_ms, tool) are preserved.
    """
    top = ranked[:max_actions]
    result: List[Dict[str, Any]] = []

    for action in top:
        ac = copy.deepcopy(action)
        tool = _get_tool(ac)

        if tool in ("write", "edit"):
            # Truncate arguments.content
            args = ac.get("arguments", {})
            if isinstance(args, dict) and "content" in args:
                args["content"] = _truncate_content(
                    str(args["content"]), content_max_chars
                )
        elif tool == "execute":
            # Truncate stdout and stderr
            res = ac.get("result", {})
            if isinstance(res, dict):
                if "stdout" in res:
                    res["stdout"] = _truncate_content(
                        str(res["stdout"]), content_max_chars
                    )
                if "stderr" in res:
                    res["stderr"] = _truncate_content(
                        str(res["stderr"]), content_max_chars
                    )
        elif tool == "read":
            # Truncate result.content
            res = ac.get("result", {})
            if isinstance(res, dict) and "content" in res:
                res["content"] = _truncate_content(
                    str(res["content"]), content_max_chars
                )

        result.append(ac)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_actions(
    rubric: Dict[str, Any],
    actions: List[Dict[str, Any]],
    max_actions: int = 15,
) -> List[Dict[str, Any]]:
    """Filter actions relevant to the given rubric, based on rubric.type.

    Three-step pipeline:
      1. _filter_by_type: keep only actions matching the rubric type's tool+path rules.
      2. _rank_by_keyword_overlap: sort by keyword match against rubric.criteria.
      3. _truncate: take top max_actions and truncate long content fields to 1500 chars.

    Args:
        rubric: dict with "type" and "criteria" keys (from rubrics.json).
        actions: list of MCP action dicts.
        max_actions: max number of actions to return (default 15).

    Returns:
        Filtered, ranked, truncated action list (deep-copied from input).
    """
    rubric_type = rubric.get("type", "")

    # Step 1: type-based filter
    filtered = _filter_by_type(rubric_type, actions)

    # Step 2: keyword overlap ranking
    ranked = _rank_by_keyword_overlap(rubric, filtered)

    # Step 3: top-N + content truncation
    return _truncate(ranked, max_actions, content_max_chars=_CONTENT_MAX_CHARS)
