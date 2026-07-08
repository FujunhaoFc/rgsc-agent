"""
Rubric-Matching Evaluator for RGSC-Agent Self-Evaluation.

For each rubric criterion in a paper's rubrics.json, this script uses a
DeepSeek judge to determine whether the agent's evidence (actions.json,
plan.md, and src/ files in the workspace) satisfies the criterion.

Outputs a per-paper JSON report and an aggregate CSV table for the paper.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python rubric_evaluator.py \\
        --train_dir /root/train_valid_all \\
        --workspace_root /root/agent-workspace \\
        --output_dir /root/eval_results \\
        --papers AMUN I0T min-p \\
        --model deepseek-chat

Requires: openai (>=1.0.0) or requests; python 3.10+.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# Prompt design
# ============================================================================

JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating whether an AI agent has satisfied a specific rubric criterion during a scientific paper reproduction task.

The criterion is one of five types:
- Paper Observation: The agent read and understood specific sections of the paper.
- Plan Writing: The agent's plan.md covers a necessary component.
- Code Implementation: The agent's src/ files implement the required functionality.
- Command Execution: The agent's actions.json shows a successful shell command execution.
- Result Matching: The agent's results.md reports results consistent with the paper.

You will be given the criterion, its type, and relevant evidence extracted from the agent's outputs. Judge strictly based on the provided evidence.

Respond with a compact JSON object with exactly these fields:
  {"satisfied": true|false, "confidence": 0.0..1.0, "reasoning": "one sentence explaining evidence"}

Rules:
- "satisfied": true only if the evidence clearly demonstrates the criterion is met.
- "confidence": your certainty on a 0-1 scale.
- "reasoning": a single sentence pointing to specific evidence (or lack thereof).
- Do NOT reward mentioning the criterion; the agent must actually satisfy it.
- Command Execution: satisfied requires an actual execute_cmd action with exit_code=0.
- Result Matching: satisfied requires numerical/textual results in results.md matching the paper. NOT REPRODUCED marks count as NOT satisfied.
"""

USER_PROMPT_TEMPLATE = """CRITERION TYPE: {rubric_type}
SCORE WEIGHT: {score}
CRITERION: {criteria}
COMMENT (if any): {comment}

EVIDENCE FROM AGENT:
========================
{evidence}
========================

Judge whether the criterion is satisfied. Output the JSON only, no other text."""


# ============================================================================
# Evidence extraction
# ============================================================================


def _read(p: Path, max_bytes: int = 15000) -> str:
    """Read a text file with a size cap; returns '' if missing."""
    if not p.exists() or p.stat().st_size == 0:
        return ""
    text = p.read_text(errors="replace")
    if len(text) > max_bytes:
        head = text[: max_bytes // 2]
        tail = text[-max_bytes // 2 :]
        return f"{head}\n\n[...truncated {len(text) - max_bytes} chars...]\n\n{tail}"
    return text


def _actions_summary(actions_path: Path, max_actions: int = 40) -> str:
    """Compact summary of actions.json — one line per action."""
    if not actions_path.exists() or actions_path.stat().st_size == 0:
        return "(no actions.json)"
    try:
        actions = json.loads(actions_path.read_text())
    except Exception as e:
        return f"(actions.json parse error: {e})"

    lines = [f"Total actions: {len(actions)}"]
    for a in actions[:max_actions]:
        tool = a.get("tool", "?")
        args = a.get("arguments", {})
        result = a.get("result", {})
        succ = result.get("success", "?")
        if tool == "Read":
            path = args.get("path", "?")
            lines.append(f"  #{a.get('id')}: Read {path} (success={succ})")
        elif tool == "Write":
            path = args.get("path", "?")
            content = str(args.get("content", ""))[:80]
            lines.append(f"  #{a.get('id')}: Write {path} — first 80 chars: {content!r}")
        elif tool == "Execute":
            cmd = str(args.get("cmd", ""))[:120]
            exit_code = result.get("exit_code", "?")
            lines.append(f"  #{a.get('id')}: Execute {cmd!r} (exit={exit_code})")
        elif tool == "ExportRecord":
            lines.append(f"  #{a.get('id')}: ExportRecord")
        else:
            lines.append(f"  #{a.get('id')}: {tool}")
    if len(actions) > max_actions:
        lines.append(f"  ... and {len(actions) - max_actions} more actions")
    return "\n".join(lines)


def _src_summary(src_dir: Path, max_bytes: int = 8000) -> str:
    """Concatenated summary of src/*.py: file list + brief snippet per file."""
    if not src_dir.exists() or not src_dir.is_dir():
        return "(no src/)"
    py_files = sorted(src_dir.glob("**/*.py"))
    if not py_files:
        return "(no .py files in src/)"
    lines = [f"src/ contains {len(py_files)} .py files:"]
    total_bytes = 0
    for f in py_files:
        content = f.read_text(errors="replace") if f.exists() else ""
        lines.append(f"\n--- {f.relative_to(src_dir.parent)} ({len(content)} chars) ---")
        # Include function/class signatures + first 20 lines
        sig_lines = [
            l for l in content.split("\n")
            if l.strip().startswith(("def ", "class ", "async def ", "import ", "from "))
        ][:15]
        snippet = "\n".join(sig_lines)
        if not snippet:
            snippet = "\n".join(content.split("\n")[:20])
        lines.append(snippet)
        total_bytes += len(snippet)
        if total_bytes > max_bytes:
            lines.append(f"\n[... remaining {len(py_files) - py_files.index(f) - 1} files omitted]")
            break
    return "\n".join(lines)


def _build_evidence_for_criterion(
    workspace_dir: Path, rubric_type: str, criteria: str
) -> str:
    """Assemble evidence tailored to the rubric type."""
    parts: list[str] = []

    # Always include a brief actions summary
    parts.append("=== actions.json summary ===")
    parts.append(_actions_summary(workspace_dir / "log" / "actions.json"))

    if rubric_type == "Paper Observation":
        # Focus on Read actions + which paper.md chunks were read
        parts.append("\n=== plan.md (first 5000 chars) ===")
        parts.append(_read(workspace_dir / "plan.md", 5000))

    elif rubric_type == "Plan Writing":
        # Focus on plan.md
        parts.append("\n=== plan.md (full, up to 15000 chars) ===")
        parts.append(_read(workspace_dir / "plan.md", 15000))

    elif rubric_type == "Code Implementation":
        # Focus on src/
        parts.append("\n=== src/ summary ===")
        parts.append(_src_summary(workspace_dir / "src", 10000))
        # And plan.md briefly, since code implements plan
        parts.append("\n=== plan.md (first 3000 chars) ===")
        parts.append(_read(workspace_dir / "plan.md", 3000))

    elif rubric_type == "Command Execution":
        # Focus on Execute actions from actions.json
        actions_path = workspace_dir / "log" / "actions.json"
        parts.append("\n=== All Execute actions ===")
        if actions_path.exists() and actions_path.stat().st_size > 0:
            try:
                actions = json.loads(actions_path.read_text())
                execs = [a for a in actions if a.get("tool") == "Execute"]
                if not execs:
                    parts.append("(no Execute actions)")
                else:
                    for a in execs[:20]:
                        cmd = str(a.get("arguments", {}).get("cmd", ""))[:200]
                        result = a.get("result", {})
                        parts.append(
                            f"  #{a.get('id')}: {cmd!r} "
                            f"exit={result.get('exit_code','?')} "
                            f"stdout_head={str(result.get('stdout',''))[:150]!r}"
                        )
            except Exception as e:
                parts.append(f"(parse error: {e})")

    elif rubric_type == "Result Matching":
        # Focus on results.md
        parts.append("\n=== results.md (full) ===")
        parts.append(_read(workspace_dir / "results.md", 12000))

    return "\n".join(parts)


# ============================================================================
# Judge invocation
# ============================================================================


def _call_judge(client: OpenAI, model: str, user_msg: str, retries: int = 3) -> dict[str, Any]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=400,
            )
            raw = resp.choices[0].message.content.strip()
            # Extract JSON — strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            data = json.loads(raw)
            return {
                "satisfied": bool(data.get("satisfied", False)),
                "confidence": float(data.get("confidence", 0.5)),
                "reasoning": str(data.get("reasoning", "")),
                "raw": raw,
            }
        except Exception as e:
            if attempt == retries - 1:
                return {
                    "satisfied": None,
                    "confidence": 0.0,
                    "reasoning": f"JUDGE_ERROR: {e}",
                    "raw": "",
                }
            time.sleep(2 ** attempt)
    return {}


def _evaluate_criterion(
    client: OpenAI,
    model: str,
    workspace_dir: Path,
    rubric: dict[str, Any],
    idx: int,
    total: int,
) -> dict[str, Any]:
    evidence = _build_evidence_for_criterion(
        workspace_dir, rubric["type"], rubric["criteria"]
    )
    user_msg = USER_PROMPT_TEMPLATE.format(
        rubric_type=rubric["type"],
        score=rubric["score"],
        criteria=rubric["criteria"],
        comment=rubric.get("comment") or "(none)",
        evidence=evidence,
    )
    verdict = _call_judge(client, model, user_msg)
    return {
        "index": idx,
        "type": rubric["type"],
        "score": rubric["score"],
        "criteria": rubric["criteria"][:200],
        **verdict,
    }


# ============================================================================
# Per-paper evaluation
# ============================================================================


def evaluate_paper(
    train_dir: Path,
    workspace_root: Path,
    paper_name: str,
    output_dir: Path,
    client: OpenAI,
    model: str,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Evaluate all rubric criteria for a single paper."""
    # 1. Locate rubrics.json — search recursively under train_dir
    rubric_paths = list(train_dir.glob(f"**/{paper_name}/rubrics.json"))
    if not rubric_paths:
        return {"error": f"rubrics.json not found for {paper_name} under {train_dir}"}
    rubric_path = rubric_paths[0]
    rubrics = json.loads(rubric_path.read_text())

    # 2. Locate workspace
    ws = workspace_root / paper_name
    if not ws.exists():
        return {"error": f"workspace {ws} not found — has skim mode been run on {paper_name}?"}
    if not (ws / "log" / "actions.json").exists():
        return {"error": f"actions.json missing for {paper_name} — skim mode did not complete"}

    print(f"\n[{paper_name}] Evaluating {len(rubrics)} rubric criteria...")
    t0 = time.time()

    results: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_evaluate_criterion, client, model, ws, r, i, len(rubrics)): i
            for i, r in enumerate(rubrics)
        }
        for fut in cf.as_completed(futures):
            r = fut.result()
            results.append(r)
            if len(results) % 10 == 0:
                print(f"  [{paper_name}] {len(results)}/{len(rubrics)} judged...")
    results.sort(key=lambda x: x["index"])

    # 3. Aggregate
    by_type: dict[str, dict[str, float]] = {}
    for r in results:
        t = r["type"]
        by_type.setdefault(t, {"n_items": 0, "n_satisfied": 0, "score_total": 0, "score_earned": 0})
        by_type[t]["n_items"] += 1
        by_type[t]["score_total"] += r["score"]
        if r["satisfied"] is True:
            by_type[t]["n_satisfied"] += 1
            by_type[t]["score_earned"] += r["score"]

    total_score_possible = sum(bt["score_total"] for bt in by_type.values())
    total_score_earned = sum(bt["score_earned"] for bt in by_type.values())

    for t, bt in by_type.items():
        bt["item_recall"] = bt["n_satisfied"] / bt["n_items"] if bt["n_items"] else 0.0
        bt["score_recall"] = bt["score_earned"] / bt["score_total"] if bt["score_total"] else 0.0

    report = {
        "paper": paper_name,
        "total_items": len(rubrics),
        "total_score_possible": total_score_possible,
        "total_score_earned": total_score_earned,
        "overall_score_recall": total_score_earned / total_score_possible if total_score_possible else 0,
        "by_type": by_type,
        "elapsed_seconds": time.time() - t0,
        "per_criterion": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{paper_name}_eval.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        f"[{paper_name}] Done in {report['elapsed_seconds']:.1f}s — "
        f"overall recall: {report['overall_score_recall']:.1%} "
        f"({total_score_earned}/{total_score_possible} points). "
        f"Saved to {out_path}"
    )
    return report


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_dir", default="/root/train_valid_all",
                    help="root of training data (contains <domain>/<paper>/rubrics.json)")
    ap.add_argument("--workspace_root", default="/root/agent-workspace",
                    help="root of agent workspaces (contains <paper>/log/actions.json)")
    ap.add_argument("--output_dir", default="/root/eval_results",
                    help="where to write per-paper eval JSON reports")
    ap.add_argument("--papers", nargs="+",
                    help="specific paper names to evaluate (default: all in train_dir)")
    ap.add_argument("--model", default="deepseek-chat",
                    help="judge model (via OpenAI-compatible API)")
    ap.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--base_url", default="https://api.deepseek.com")
    ap.add_argument("--max_workers", type=int, default=4,
                    help="parallel judge calls per paper")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"ERROR: {args.api_key_env} not set in environment", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    train_dir = Path(args.train_dir)
    workspace_root = Path(args.workspace_root)
    output_dir = Path(args.output_dir)

    if args.papers:
        papers = args.papers
    else:
        papers = sorted({p.parent.name for p in train_dir.glob("**/rubrics.json")})
    print(f"Will evaluate {len(papers)} papers: {papers}")

    aggregate: dict[str, Any] = {"papers": []}
    for paper in papers:
        report = evaluate_paper(
            train_dir, workspace_root, paper, output_dir, client, args.model, args.max_workers
        )
        if "error" in report:
            print(f"[SKIP] {paper}: {report['error']}", file=sys.stderr)
            continue
        aggregate["papers"].append(report)

    # Cross-paper aggregate by rubric type
    if aggregate["papers"]:
        cross_type: dict[str, dict[str, float]] = {}
        for rep in aggregate["papers"]:
            for t, bt in rep["by_type"].items():
                cross_type.setdefault(
                    t, {"n_items": 0, "n_satisfied": 0, "score_total": 0, "score_earned": 0}
                )
                for k in ["n_items", "n_satisfied", "score_total", "score_earned"]:
                    cross_type[t][k] += bt[k]
        for t, bt in cross_type.items():
            bt["item_recall"] = bt["n_satisfied"] / bt["n_items"] if bt["n_items"] else 0.0
            bt["score_recall"] = bt["score_earned"] / bt["score_total"] if bt["score_total"] else 0.0
        aggregate["cross_paper_by_type"] = cross_type

        agg_path = output_dir / "aggregate.json"
        agg_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False))
        print(f"\n=== Aggregate ===")
        print(f"{'Type':<25} {'Items':>6} {'Satisfied':>10} {'Item%':>8} {'Score':>8} {'Earned':>8} {'Score%':>8}")
        print("-" * 80)
        for t in ["Paper Observation", "Plan Writing", "Code Implementation", "Command Execution", "Result Matching"]:
            if t not in cross_type:
                continue
            bt = cross_type[t]
            print(
                f"{t:<25} {bt['n_items']:>6} {bt['n_satisfied']:>10} "
                f"{bt['item_recall']:>7.1%} {bt['score_total']:>8} {bt['score_earned']:>8} "
                f"{bt['score_recall']:>7.1%}"
            )
        print(f"\nAggregate saved to {agg_path}")


if __name__ == "__main__":
    main()
