"""
Coverage diagnostic: compute recall of derived checklist against official rubrics.

For each official rubric item, search the derived checklist for a matching
item using:
  1. Anchor-based exact match (free, for Paper Observation / Result Matching)
  2. LLM-judge semantic match (DeepSeek V4-Pro) for the rest

Reports per-paper and cross-paper recall, broken down by rubric type.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.rubric_normalizer.anchor_parser import parse_anchors  # noqa: E402


PAPERS = ["AMUN", "Beyond-Ngram", "I0T", "INCLINE", "min-p"]

# Types for which anchor matching is reliable (rubric pattern is location-based)
ANCHOR_RELIABLE_TYPES = {"Paper Observation", "Result Matching"}


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------


def _make_client() -> OpenAI:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set.")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")


# ---------------------------------------------------------------------------
# Matchers
# ---------------------------------------------------------------------------


def _rubric_anchors(rubric: Dict) -> set:
    """Extract anchor entity ids (e.g. 'table-1') from a rubric criteria."""
    criteria = rubric.get("criteria", "")
    parsed = parse_anchors(criteria)
    anchors = set()
    for t in parsed.get("tables", []):
        anchors.add(f"table-{t}")
    for f in parsed.get("figures", []):
        anchors.add(f"figure-{f}")
    for a in parsed.get("algorithms", []):
        anchors.add(f"algorithm-{a}")
    return anchors


def _anchor_match(rubric: Dict, derived_items: List[Dict]) -> Optional[Dict]:
    """If anchor matching is reliable for this rubric type and there's a
    single derived item with the same type AND a matching anchor, return it.

    Returns the matched derived item, or None if no anchor match.
    """
    rtype = rubric.get("type")
    if rtype not in ANCHOR_RELIABLE_TYPES:
        return None
    r_anchors = _rubric_anchors(rubric)
    if not r_anchors:
        return None
    # Find derived items with the same type that share at least one anchor
    matches = [
        d for d in derived_items
        if d["type"] == rtype and d.get("anchor") in r_anchors
    ]
    if matches:
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


_JUDGE_PROMPT = """You are evaluating whether a derived checklist item semantically matches an official rubric criterion. The derived item was produced from a paper's structured representation by a rule-based system. The official rubric criterion describes a check that an evaluator would apply to a research paper reproduction agent.

# Official rubric (target)

Type: {rubric_type}
Criterion: {rubric_criteria}

# Candidate derived items (same type)

{candidates}

# Task

Decide if any candidate semantically corresponds to the official rubric criterion. Two items semantically match when a reasonable evaluator would consider that completing the candidate's described action would satisfy the official rubric.

# Judging guidelines — read carefully

When deciding hit/partial/miss, prioritize SEMANTIC correspondence over literal wording match.

A candidate is a HIT if:
- It refers to the same procedure, computation, or evidence as the rubric.
- A reasonable evaluator would mark the rubric satisfied when the candidate's action is completed.
- Different naming, abbreviations, parameter values, or phrasing are acceptable as long as the underlying action is the same.

A candidate is a PARTIAL match if (give partial credit, do NOT mark as miss):
- The candidate describes the same general procedure but with a parameter difference (e.g. rubric says "PGD-5" and candidate says "PGD-50" — same algorithm, different step count).
- The candidate refers to the procedure indirectly (e.g. rubric says "Algorithm 1 is implemented to build the adversarial set" and a candidate describes constructing the adversarial set via PGD attack — same action, different abstraction level).
- The candidate covers part of a compound rubric (e.g. rubric says "all baselines (FT, GA, l1-Sparse, SalUn) are implemented" and individual candidates implement each baseline separately — the union of candidates would satisfy the rubric).
- The candidate's wording uses synonyms or restates the same action differently (e.g. "compute Average Gap" vs "calculate the average gap").

A candidate is a MISS only if:
- No candidate refers to the same procedure, computation, or evidence, even loosely.
- The rubric describes an entity (dataset, model, table, etc.) that no candidate mentions and that's required by the rubric.

When in doubt between miss and partial, choose PARTIAL.
When in doubt between partial and hit, choose HIT if the underlying action matches.

# Output

Output JSON only, no markdown fences, no commentary:
{{"verdict": "hit" | "partial" | "miss", "matched_id": "derived-XX-NNN" | null, "reason": "<= 15 words"}}

CRITICAL: The "reason" field must be 15 words or fewer. Be terse, not verbose.

Examples of acceptable reasons:
- "candidate derives-pw-007 directly implements PGD attack as required"
- "candidates mention PGD-50, rubric asks PGD-5 — same algorithm, different params"
- "no candidate mentions Flickr30k retrieval evaluation"
- "candidates implement each baseline individually, rubric asks union"

If verdict is "partial" because multiple candidates together cover the rubric, set matched_id to any one of them.
"""


def _format_candidates(candidates: List[Dict], max_items: int = 60) -> str:
    """Format candidate derived items for LLM judge prompt."""
    if len(candidates) > max_items:
        # If too many, take first max_items (deterministic, order-stable)
        candidates = candidates[:max_items]
    lines = []
    for c in candidates:
        desc = c["description"]
        if len(desc) > 200:
            desc = desc[:197] + "..."
        lines.append(f"- [{c['id']}] {desc}")
    return "\n".join(lines)


def _judge_with_llm(
    client: OpenAI,
    rubric: Dict,
    candidates: List[Dict],
) -> Tuple[Dict, int, int]:
    """Run LLM judge with retry on empty response. Returns (verdict_dict, input_tokens, output_tokens)."""
    import time
    
    if not candidates:
        return {"verdict": "miss", "matched_id": None, "reason": "no candidates of same type"}, 0, 0

    prompt = _JUDGE_PROMPT.format(
        rubric_type=rubric.get("type", ""),
        rubric_criteria=rubric.get("criteria", ""),
        candidates=_format_candidates(candidates),
    )

    total_in_tokens = 0
    total_out_tokens = 0
    text = ""

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1200,
            )
        except Exception as e:
            # Network / API error
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # 1s, 2s
                continue
            return {"verdict": "miss", "matched_id": None, "reason": f"API error: {e}"[:80]}, total_in_tokens, total_out_tokens

        if resp is None or not resp.choices:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            return {"verdict": "miss", "matched_id": None, "reason": "empty LLM response after retries"}, total_in_tokens, total_out_tokens

        text = (resp.choices[0].message.content or "").strip()
        if resp.usage:
            total_in_tokens += resp.usage.prompt_tokens
            total_out_tokens += resp.usage.completion_tokens

        if text:
            break  # Got non-empty response, exit retry loop
        
        # Empty content — retry
        if attempt < MAX_RETRIES - 1:
            time.sleep(1)

    if not text:
        return {"verdict": "miss", "matched_id": None, "reason": "all retries returned empty"}, total_in_tokens, total_out_tokens

    # Strip optional code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        verdict = json.loads(text)
        if "verdict" not in verdict:
            verdict = {"verdict": "miss", "matched_id": None, "reason": "malformed LLM verdict"}
    except json.JSONDecodeError:
        verdict = {"verdict": "miss", "matched_id": None, "reason": f"unparseable LLM output: {text[:80]}"}

    return verdict, total_in_tokens, total_out_tokens


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------


def diagnose_paper(paper: str, client: OpenAI, verbose: bool = True) -> Dict:
    """Run coverage diagnostic for one paper."""
    derived_path = PROJECT_ROOT / "outputs" / paper / "derived_checklist.json"
    rubric_path = PROJECT_ROOT / "data" / "train_valid" / paper / "rubrics.json"

    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    rubrics = json.loads(rubric_path.read_text(encoding="utf-8"))

    # Group derived by type for fast candidate lookup
    derived_by_type: Dict[str, List[Dict]] = defaultdict(list)
    for d in derived:
        derived_by_type[d["type"]].append(d)

    results: List[Dict] = []
    total_in_tokens = 0
    total_out_tokens = 0

    if verbose:
        print(f"\n=== Diagnosing {paper} ({len(rubrics)} rubrics) ===")

    for i, rubric in enumerate(rubrics):
        rtype = rubric.get("type", "Unknown")
        candidates = derived_by_type.get(rtype, [])

        # Try anchor match first
        anchor_hit = _anchor_match(rubric, derived)
        if anchor_hit is not None:
            verdict = {
                "verdict": "hit",
                "matched_id": anchor_hit["id"],
                "reason": "anchor match",
                "method": "anchor",
            }
        else:
            # Fallback to LLM judge
            verdict, in_t, out_t = _judge_with_llm(client, rubric, candidates)
            verdict["method"] = "llm"
            total_in_tokens += in_t
            total_out_tokens += out_t

        results.append({
            "rubric_idx": i,
            "rubric_type": rtype,
            "rubric_criteria": rubric.get("criteria", "")[:200],
            "verdict": verdict["verdict"],
            "matched_id": verdict.get("matched_id"),
            "method": verdict.get("method", "?"),
            "reason": verdict.get("reason", ""),
        })

        if verbose and (i + 1) % 10 == 0:
            n_hit = sum(1 for r in results if r["verdict"] == "hit")
            n_partial = sum(1 for r in results if r["verdict"] == "partial")
            print(f"  [{i + 1}/{len(rubrics)}] hits={n_hit}, partial={n_partial}")

    # Per-type aggregation
    by_type: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "hit": 0, "partial": 0, "miss": 0})
    for r in results:
        t = r["rubric_type"]
        by_type[t]["total"] += 1
        by_type[t][r["verdict"]] += 1

    paper_summary = {
        "paper": paper,
        "by_type": dict(by_type),
        "results": results,
        "llm_input_tokens": total_in_tokens,
        "llm_output_tokens": total_out_tokens,
    }

    return paper_summary


def print_summary(summaries: List[Dict]) -> None:
    """Print a cross-paper recall summary."""
    print(f"\n{'=' * 80}")
    print("Recall by Paper × Type")
    print('=' * 80)

    types = ["Paper Observation", "Plan Writing", "Code Implementation",
             "Command Execution", "Result Matching"]

    header = f"{'Paper':<14} " + " ".join(f"{t[:8]:>10}" for t in types) + f"  {'TOTAL':>8}"
    print(header)
    print('-' * len(header))

    grand_by_type = defaultdict(lambda: {"total": 0, "hit": 0, "partial": 0})

    for s in summaries:
        row = f"{s['paper']:<14} "
        paper_total = 0
        paper_recall_sum = 0
        for t in types:
            stats = s["by_type"].get(t, {"total": 0, "hit": 0, "partial": 0})
            hit = stats["hit"] + 0.5 * stats["partial"]
            tot = stats["total"]
            if tot > 0:
                recall = hit / tot
                row += f"{recall:>5.1%}({tot:>2}) "
            else:
                row += f"{'  -':>10} "
            grand_by_type[t]["total"] += stats["total"]
            grand_by_type[t]["hit"] += stats["hit"]
            grand_by_type[t]["partial"] += stats["partial"]
            paper_total += tot
            paper_recall_sum += hit
        overall_recall = (paper_recall_sum / paper_total) if paper_total else 0
        row += f"  {overall_recall:>6.1%}"
        print(row)

    print('-' * len(header))

    # Cross-paper aggregate
    agg_row = f"{'CROSS-AGG':<14} "
    agg_total = 0
    agg_hit = 0
    for t in types:
        s = grand_by_type[t]
        hit = s["hit"] + 0.5 * s["partial"]
        tot = s["total"]
        if tot > 0:
            recall = hit / tot
            agg_row += f"{recall:>5.1%}({tot:>2}) "
        else:
            agg_row += f"{'  -':>10} "
        agg_total += tot
        agg_hit += hit
    overall = agg_hit / agg_total if agg_total else 0
    agg_row += f"  {overall:>6.1%}"
    print(agg_row)

    print(f"\nTotal LLM tokens: "
          f"{sum(s['llm_input_tokens'] for s in summaries):,} in / "
          f"{sum(s['llm_output_tokens'] for s in summaries):,} out")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    client = _make_client()
    summaries = []

    for paper in PAPERS:
        s = diagnose_paper(paper, client)
        summaries.append(s)

        # Save per-paper detailed results
        out_path = PROJECT_ROOT / "outputs" / paper / "coverage_diagnostic.json"
        out_path.write_text(
            json.dumps(s, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  ✓ Wrote: {out_path}")

    # Save cross-paper summary
    agg_path = PROJECT_ROOT / "outputs" / "coverage_summary.json"
    agg_path.write_text(
        json.dumps([{k: v for k, v in s.items() if k != "results"}
                    for s in summaries], indent=2),
        encoding="utf-8",
    )

    print_summary(summaries)


if __name__ == "__main__":
    main()
