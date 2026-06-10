"""
Rubric Scorer main module — Phase 3.2 external evaluation tool.

Simulates NLPCC grading by comparing Agent actions.json + reproduced_repo
against ground truth rubrics.json. Card 4: real LLM judge (deepseek-v4-pro)
with 3-retry strategy copied from pipeline/verifier/verifier.py.

Module structure mirrors pipeline/verifier/verifier.py pattern.

Tech debt: LLM client (_make_client, _call_llm, _parse_json_loose) is
duplicated from verifier.py (and planner.py).  Extract to pipeline/common/llm.py
before Phase 3.3.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from jsonschema import Draft7Validator
from openai import OpenAI

# Import openai exception classes for API-level retry
try:
    import openai as _openai_mod
    _APITimeoutError = _openai_mod.APITimeoutError
    _APIConnectionError = _openai_mod.APIConnectionError
    _RateLimitError = _openai_mod.RateLimitError
except (AttributeError, ImportError):
    # Fallback for older openai SDK versions
    _APITimeoutError = Exception
    _APIConnectionError = Exception
    _RateLimitError = Exception

# Ensure project root is on sys.path for direct script execution (python scorer.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.rubric_scorer.retrieval import filter_actions  # noqa: E402
from pipeline.common.paths import find_rubrics_json  # noqa: E402

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SCHEMA_PATH = PROJECT_ROOT / "pipeline" / "schemas" / "rubric_score_report.schema.json"

# ---------------------------------------------------------------------------
# Type slug mapping for prompt file lookup
# ---------------------------------------------------------------------------

_TYPE_SLUG = {
    "Paper Observation": "paper_observation",
    "Plan Writing": "plan_writing",
    "Code Implementation": "code_implementation",
    "Command Execution": "command_execution",
    "Result Matching": "result_matching",
}


def _type_slug(rubric_type: str) -> str:
    """Map rubric type name to lowercase underscore slug for prompt filename."""
    return _TYPE_SLUG.get(rubric_type, rubric_type.lower().replace(" ", "_"))


# ---------------------------------------------------------------------------
# LLM client helpers (Tech debt: duplicated from verifier.py / planner.py.
# Extract to pipeline/common/llm.py before Phase 3.3.)
# ---------------------------------------------------------------------------


def _make_client() -> OpenAI:
    """Build an OpenAI client pointed at DeepSeek's OpenAI-compat endpoint."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set. Put it in .env or export it.")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )


def _model_name() -> str:
    return os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")


def _strip_code_fences(text: str) -> str:
    """If the LLM wrapped JSON in ```json ... ```, strip those fences."""
    text = text.strip()
    fence_re = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)
    m = fence_re.match(text)
    if m:
        return m.group(1).strip()
    return text


def _parse_json_loose(text: str) -> dict:
    """Parse JSON, with fallback: extract first {...} balanced block."""
    text = _strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            candidate = text[first_brace : last_brace + 1]
            return json.loads(candidate)
        raise


# ---------------------------------------------------------------------------
# API-level retry constants
# ---------------------------------------------------------------------------

_API_RETRY_MAX = 3
_API_RETRY_BACKOFF = [2, 5, 10]  # seconds — fixed schedule, not exponential


def _call_llm(
    client: OpenAI,
    prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 4000,
) -> Tuple[str, dict]:
    """Make a single LLM call with API-level retry for transient errors.

    Retries on APITimeoutError, APIConnectionError, RateLimitError, and
    custom RuntimeError for up to _API_RETRY_MAX attempts with fixed backoff.

    temperature=0.2 for stability — lower than Stage Planner (0.3)
    because verdict consistency is critical for evaluation.
    """
    last_exception: Optional[Exception] = None

    for attempt in range(_API_RETRY_MAX):
        try:
            messages = [{"role": "user", "content": prompt}]
            resp = client.chat.completions.create(
                model=_model_name(),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,  # 120s per call — prevent hanging
            )
            if resp is None or not resp.choices:
                raise RuntimeError(f"DeepSeek API returned empty response. resp={resp!r}")
            text = resp.choices[0].message.content or ""
            usage = {
                "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "model": resp.model,
            }
            return text, usage

        except (_APITimeoutError, _APIConnectionError, _RateLimitError, RuntimeError) as e:
            last_exception = e
            if attempt < _API_RETRY_MAX - 1:
                wait = _API_RETRY_BACKOFF[attempt]
                print(
                    f"  [api-retry] attempt {attempt + 1}/{_API_RETRY_MAX} failed "
                    f"({type(e).__name__}: {e}), waiting {wait}s"
                )
                time.sleep(wait)
            else:
                print(
                    f"  [api-retry] EXHAUSTED after {_API_RETRY_MAX} attempts: {e}"
                )
                raise

    # Should never reach here, but satisfy type checker
    assert last_exception is not None
    raise last_exception


# ---------------------------------------------------------------------------
# Debug logging (per-rubric)
# ---------------------------------------------------------------------------


def _save_rubric_debug(
    paper_id: str,
    rubric_idx: int,
    rubric_type: str,
    rubric_criteria: str,
    sent_prompt: str,
    raw_response: str,
    parsed_verdict: Optional[dict],
    usage: dict,
    warnings: List[str],
) -> None:
    """Save per-rubric LLM prompt + response to _debug/rubric_scorer/."""
    debug_dir = OUTPUTS_DIR / paper_id / "_debug" / "rubric_scorer"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"llm_response_rubric_{rubric_idx:03d}.json"

    debug_data = {
        "paper_id": paper_id,
        "rubric_idx": rubric_idx,
        "rubric_type": rubric_type,
        "rubric_criteria": rubric_criteria,
        "model": _model_name(),
        "temperature": 0.2,
        "max_tokens": 4000,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "sent_prompt": sent_prompt,
        "raw_response": raw_response,
        "parsed_verdict": parsed_verdict,
        "warnings": warnings,
    }
    debug_path.write_text(
        json.dumps(debug_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def compute_earned_score(verdict: str, rubric_score: float) -> float:
    """Map verdict to earned score.

    hit     → full rubric_score
    partial → 0.5 * rubric_score
    miss    → 0
    """
    if verdict == "hit":
        return rubric_score
    elif verdict == "partial":
        return 0.5 * rubric_score
    elif verdict == "miss":
        return 0.0
    raise ValueError(f"Unknown verdict '{verdict}'; expected hit, partial, or miss")


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _render_judge_prompt(
    rubric: Dict[str, Any],
    filtered_actions: List[Dict[str, Any]],
    repo_files: List[str],
) -> str:
    """Render the type-specific judge prompt with rubric + actions + repo info.

    Uses {criteria}, {comment}, {filtered_actions_json}, {repo_files_list}
    placeholders that the prompt templates expect.
    """
    slug = _type_slug(rubric.get("type", ""))
    prompt_path = PROMPT_DIR / f"judge_{slug}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    template = prompt_path.read_text(encoding="utf-8")

    prompt = template.replace("{criteria}", rubric.get("criteria", ""))
    prompt = prompt.replace("{comment}", rubric.get("comment") or "")
    prompt = prompt.replace(
        "{filtered_actions_json}",
        json.dumps(filtered_actions, indent=2, ensure_ascii=False),
    )
    prompt = prompt.replace(
        "{repo_files_list}",
        "\n".join(repo_files) if repo_files else "(empty repository)",
    )
    return prompt


# ---------------------------------------------------------------------------
# LLM judge with retry (Card 4 — real LLM)
# ---------------------------------------------------------------------------


def judge_rubric(
    rubric: Dict[str, Any],
    actions: List[Dict[str, Any]],
    repo_files: List[str],
    *,
    client: Optional[OpenAI] = None,
    paper_id: str = "",
    verbose: bool = False,
    dry_run: bool = False,
) -> dict:
    """Judge a single rubric against actions.json + repo files.

    Card 4: real LLM call with 3-retry strategy (mirrors verifier.py pattern).
    On dry_run=True, falls back to mock "hit" verdict without calling the API.

    Args:
        rubric: rubric dict with type, criteria, score.
        actions: full actions.json list.
        repo_files: list of relative file paths in the repo.
        client: OpenAI client (created internally if None and not dry_run).
        paper_id: for debug output.
        verbose: print per-rubric progress.
        dry_run: if True, skip LLM and return mock "hit" verdict.

    Returns:
        dict with verdict, confidence, reasoning, evidence.
    """
    rubric_idx = rubric.get("rubric_idx", 0)
    rubric_type = rubric.get("type", "")
    rubric_criteria = rubric.get("criteria", "")

    # Pre-filter actions
    filtered = filter_actions(rubric, actions)

    # Render prompt
    prompt = _render_judge_prompt(rubric, filtered, repo_files)

    if dry_run:
        if verbose:
            print(f"  [rubric {rubric_idx}] dry-run → hit")
        return {
            "verdict": "hit",
            "confidence": 0.95,
            "reasoning": "dry-run mock judgement",
            "evidence": "dry-run mock evidence",
        }

    # Real LLM path
    if client is None:
        client = _make_client()

    MAX_RETRIES = 3
    last_error: Optional[str] = None
    last_raw: str = ""
    last_prompt: str = prompt
    last_usage: dict = {}
    last_parsed: Optional[dict] = None
    debug_warnings: List[str] = []

    for attempt in range(1, MAX_RETRIES + 1):
        if verbose and attempt > 1:
            print(f"    [judge] rubric {rubric_idx} retry {attempt}/{MAX_RETRIES}")

        if last_error and attempt > 1:
            retry_prompt = (
                "Your previous response was malformed.\n\n"
                "=== Your previous output ===\n"
                f"{last_raw}\n\n"
                "=== Error ===\n"
                f"{last_error}\n\n"
                "Re-output the COMPLETE JSON object with ALL of these required fields. "
                "Missing any field will fail validation:\n\n"
                "{\n"
                '  "verdict": "hit" | "partial" | "miss",\n'
                '  "confidence": <number between 0.0 and 1.0>,\n'
                '  "reasoning": "<≤ 80 chars, English, cite specific evidence>",\n'
                '  "evidence": "<specific action index or file path>"\n'
                "}\n\n"
                "Do NOT just fix the error field — include EVERY field above. "
                "Output ONLY the JSON object. No preamble, no markdown fences, "
                "no comments. Now output the complete corrected JSON:"
            )
            combined_prompt = (
                prompt
                + "\n\n---\n\n# RETRY NOTICE\n\n"
                + retry_prompt
            )
            last_prompt = combined_prompt
            try:
                text, usage = _call_llm(client, combined_prompt)
            except Exception as e:
                last_error = f"API call failed: {type(e).__name__}: {e}"
                continue
        else:
            last_prompt = prompt
            try:
                text, usage = _call_llm(client, prompt)
            except Exception as e:
                last_error = f"API call failed: {type(e).__name__}: {e}"
                continue

        last_raw = text
        last_usage = usage

        # Parse JSON
        try:
            result = _parse_json_loose(text)
            last_parsed = result
        except json.JSONDecodeError as e:
            last_error = f"Failed to parse JSON: {e}\n(first 300 chars): {text[:300]}"
            continue

        # Validate required fields
        verdict = result.get("verdict", "")
        if verdict not in ("hit", "partial", "miss"):
            last_error = (
                f"Invalid verdict '{verdict}'; must be hit, partial, or miss"
            )
            continue

        confidence = result.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            last_error = (
                f"Invalid confidence {confidence!r}; must be a number between 0 and 1"
            )
            continue

        reasoning = result.get("reasoning", "")
        if not isinstance(reasoning, str) or len(reasoning) > 200:
            reasoning = (reasoning or "")[:200]

        evidence = result.get("evidence", "") or ""

        # Success — save debug and return
        if paper_id:
            _save_rubric_debug(
                paper_id, rubric_idx, rubric_type, rubric_criteria,
                last_prompt, last_raw, result, last_usage, debug_warnings,
            )

        return {
            "verdict": verdict,
            "confidence": float(confidence),
            "reasoning": reasoning,
            "evidence": str(evidence),
        }

    # All retries exhausted — fallback to miss (conservative)
    if paper_id:
        debug_warnings.append(
            f"LLM judge failed after {MAX_RETRIES} attempts: {last_error or 'unknown error'}"
        )
        _save_rubric_debug(
            paper_id, rubric_idx, rubric_type, rubric_criteria,
            last_prompt, last_raw, last_parsed, last_usage, debug_warnings,
        )

    return {
        "verdict": "miss",
        "confidence": 0.3,
        "reasoning": f"LLM judge produced malformed JSON after {MAX_RETRIES} retries; conservatively marked miss.",
        "evidence": "LLM judge unavailable (malformed JSON after retries)",
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(
    paper_id: str,
    rubrics: List[Dict[str, Any]],
    judgements: List[dict],
) -> dict:
    """Aggregate per-rubric judgements into a rubric_score_report.

    Args:
        paper_id: Paper identifier.
        rubrics: Raw rubrics.json list (each with criteria, type, score).
        judgements: List of per-rubric judge outputs (verdict, confidence, etc.),
            in the same order as rubrics.

    Returns:
        rubric_score_report dict matching the schema.
    """
    total_score = sum(r.get("score", 0) for r in rubrics)
    rubric_results: List[dict] = []
    estimated_score = 0.0

    # Accumulators for by_type breakdown
    by_type_accum: Dict[str, Dict[str, float]] = {
        t: {"total": 0.0, "estimated": 0.0} for t in _TYPE_SLUG
    }

    for i, (rubric, judgement) in enumerate(zip(rubrics, judgements)):
        r_type = rubric.get("type", "")
        r_score = rubric.get("score", 0)
        verdict = judgement.get("verdict", "miss")
        earned = compute_earned_score(verdict, r_score)

        rubric_results.append({
            "rubric_idx": i,
            "criteria": rubric.get("criteria", ""),
            "type": r_type,
            "rubric_score": r_score,
            "verdict": verdict,
            "earned_score": earned,
            "confidence": judgement.get("confidence", 0.0),
            "reasoning": judgement.get("reasoning", "")[:200],
            "evidence": judgement.get("evidence", ""),
        })

        estimated_score += earned

        if r_type in by_type_accum:
            by_type_accum[r_type]["total"] += r_score
            by_type_accum[r_type]["estimated"] += earned

    total_score_val = float(total_score)
    estimated_recall = estimated_score / total_score_val if total_score_val > 0 else 0.0

    by_type = {}
    for t_name in _TYPE_SLUG:
        acc = by_type_accum[t_name]
        by_type[t_name] = {
            "total": acc["total"],
            "estimated": acc["estimated"],
            "rate": acc["estimated"] / acc["total"] if acc["total"] > 0 else 0.0,
        }

    return {
        "paper_id": paper_id,
        "total_score": total_score_val,
        "estimated_score": estimated_score,
        "estimated_recall": estimated_recall,
        "by_type": by_type,
        "rubric_results": rubric_results,
        "scorer_model": _model_name(),
        "scorer_version": "v1",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _validate_report(report: dict) -> Optional[str]:
    """Validate report against rubric_score_report.schema.json. Return None if OK."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(report), key=lambda e: list(e.path))
    if not errors:
        return None
    msgs = []
    for e in errors[:5]:
        path = "/".join(str(p) for p in e.path) or "<root>"
        msgs.append(f"  - at '{path}': {e.message}")
    if len(errors) > 5:
        msgs.append(f"  ... and {len(errors) - 5} more")
    return "Report schema validation failed:\n" + "\n".join(msgs)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Resume support (check cached judgements in _debug/)
# ---------------------------------------------------------------------------


def _check_existing_judgement(paper_id: str, rubric_idx: int) -> Optional[dict]:
    """Return cached verdict if a valid _debug record exists, else None.

    Checks outputs/{paper_id}/_debug/rubric_scorer/llm_response_rubric_{idx:03d}.json.
    Returns the parsed_verdict dict (verdict, confidence, reasoning, evidence)
    if the record exists and contains a valid verdict.
    """
    debug_path = (
        OUTPUTS_DIR / paper_id / "_debug" / "rubric_scorer"
        / f"llm_response_rubric_{rubric_idx:03d}.json"
    )
    if not debug_path.exists():
        return None
    try:
        data = json.loads(debug_path.read_text(encoding="utf-8"))
        parsed = data.get("parsed_verdict")
        if isinstance(parsed, dict) and parsed.get("verdict") in ("hit", "partial", "miss"):
            return {
                "verdict": parsed["verdict"],
                "confidence": parsed.get("confidence", 0.0),
                "reasoning": parsed.get("reasoning", "") or "",
                "evidence": parsed.get("evidence", "") or "",
            }
    except (json.JSONDecodeError, KeyError, OSError):
        pass
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def score(
    paper_id: str,
    *,
    actions_path: Optional[str] = None,
    rubrics_path: Optional[str] = None,
    repo_dir: Optional[str] = None,
    client: Optional[OpenAI] = None,
    verbose: bool = True,
    dry_run: bool = False,
    resume: bool = True,
) -> dict:
    """Run rubric scoring for one paper.

    Args:
        paper_id: Paper identifier (e.g. "AMUN").
        actions_path: Path to actions.json. Defaults to outputs/{paper_id}/actions.json.
        rubrics_path: Path to rubrics.json. Defaults to first split containing
            rubrics.json (train_valid/ then test/).
        repo_dir: Path to reproduced_repo directory. Defaults to outputs/{paper_id}/reproduced_repo/.
        client: Optional OpenAI client. If None, creates one (real API calls).
                Inject a mock client in tests to avoid real LLM calls.
        verbose: Print progress to stdout.
        dry_run: If True, skip LLM and use mock "hit" verdicts.
        resume: If True (default), skip rubrics that already have valid cached
                verdicts in _debug/rubric_scorer/.  Use --no-resume to force re-run.

    Returns:
        rubric_score_report dict (also validated against schema).
    """
    # Resolve paths
    if rubrics_path is None:
        rubrics_path = str(find_rubrics_json(paper_id))
    if actions_path is None:
        actions_path = str(OUTPUTS_DIR / paper_id / "actions.json")
    if repo_dir is None:
        repo_dir = str(OUTPUTS_DIR / paper_id / "reproduced_repo")

    # Validate API key FIRST — before any file loading or I/O
    if not dry_run and client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not set. Set it in .env or export it, "
                "or use --dry-run for mock mode."
            )

    # Load rubrics
    rubrics_file = Path(rubrics_path)
    if not rubrics_file.exists():
        raise FileNotFoundError(f"Rubrics file not found: {rubrics_path}")
    rubrics: List[Dict[str, Any]] = json.loads(rubrics_file.read_text(encoding="utf-8"))

    # Load actions
    actions_file = Path(actions_path)
    if actions_file.exists():
        actions: List[Dict[str, Any]] = json.loads(actions_file.read_text(encoding="utf-8"))
    else:
        if verbose:
            print(f"  Actions file not found: {actions_path}; using empty list")
        actions = []

    # List repo files
    repo_path = Path(repo_dir)
    if repo_path.is_dir():
        repo_files = sorted(
            str(p.relative_to(repo_path)) for p in repo_path.rglob("*") if p.is_file()
        )
    else:
        if verbose:
            print(f"  Repo dir not found: {repo_dir}; using empty file list")
        repo_files = []

    if verbose:
        mode = "DRY-RUN" if dry_run else "LLM"
        print(f"[{paper_id}] {len(rubrics)} rubrics, {len(actions)} actions, "
              f"{len(repo_files)} repo files ({mode})")

    # Judge each rubric
    total_rubrics = len(rubrics)
    judgements: List[dict] = []
    accumulated_input_tokens = 0
    accumulated_output_tokens = 0
    skipped_count = 0
    cached_count = 0

    for i, rubric in enumerate(rubrics):
        rubric["rubric_idx"] = i  # Tag for judge_rubric logging

        # Resume: check if we already have a valid cached judgement
        if resume and not dry_run:
            cached = _check_existing_judgement(paper_id, i)
            if cached is not None:
                if verbose:
                    print(
                        f"  [{i + 1}/{total_rubrics}] {rubric.get('type', '')}: "
                        f"cached → {cached['verdict']}"
                    )
                judgements.append(cached)
                cached_count += 1
                continue

        # Normal LLM judge (or dry-run mock)
        try:
            judgement = judge_rubric(
                rubric, actions, repo_files,
                client=client, paper_id=paper_id, verbose=verbose, dry_run=dry_run,
            )
        except Exception as e:
            # Fallback: mark as miss and continue with remaining rubrics
            if verbose:
                print(
                    f"  [{i + 1}/{total_rubrics}] {rubric.get('type', '')}: "
                    f"API FAILED → skipped ({type(e).__name__}: {e})"
                )
            judgement = {
                "verdict": "miss",
                "confidence": 0.0,
                "reasoning": f"API failure after retries: {type(e).__name__}",
                "evidence": "skipped due to API error",
            }
            skipped_count += 1

        judgements.append(judgement)

        # Progress: every 5 rubrics (or first or last), print accumulated stats
        if verbose and ((i + 1) % 5 == 0 or i == 0 or i == total_rubrics - 1):
            # Estimate token accumulation from _debug files (approximate)
            hits_sofar = sum(1 for j in judgements if j.get("verdict") == "hit")
            partials_sofar = sum(1 for j in judgements if j.get("verdict") == "partial")
            misses_sofar = sum(1 for j in judgements if j.get("verdict") == "miss")
            printed_sofar = i + 1
            print(
                f"  [{printed_sofar}/{total_rubrics}] {hits_sofar}H/{partials_sofar}P/"
                f"{misses_sofar}M (+{cached_count}c {'+' + str(skipped_count) + 's ' if skipped_count else ''})"
            )

    # Aggregate
    report = aggregate(paper_id, rubrics, judgements)

    # Validate against schema
    err = _validate_report(report)
    if err:
        if verbose:
            print(f"  ⚠ {err}")
        # Continue anyway — schema violations are non-fatal warnings for now

    if verbose:
        hits = sum(1 for j in judgements if j["verdict"] == "hit")
        partials = sum(1 for j in judgements if j["verdict"] == "partial")
        misses = sum(1 for j in judgements if j["verdict"] == "miss")
        extras = []
        if cached_count:
            extras.append(f"{cached_count} cached")
        if skipped_count:
            extras.append(f"{skipped_count} skipped")
        extra_str = f" ({', '.join(extras)})" if extras else ""
        print(
            f"  [{paper_id}] done: {hits}H/{partials}P/{misses}M{extra_str}, "
            f"total={report['total_score']:.0f}, "
            f"estimated={report['estimated_score']:.1f}, "
            f"recall={report['estimated_recall']:.3f}"
        )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python scorer.py [--dry-run] [--no-resume] <paper_id> [paper_id ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    dry_run = False
    resume = True
    paper_ids = []

    for arg in sys.argv[1:]:
        if arg == "--dry-run":
            dry_run = True
        elif arg == "--no-resume":
            resume = False
        else:
            paper_ids.append(arg)

    if not paper_ids:
        print(
            "Usage: python scorer.py [--dry-run] [--no-resume] <paper_id> [paper_id ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    for paper_id in paper_ids:
        try:
            report = score(paper_id, dry_run=dry_run, resume=resume)
        except (FileNotFoundError, ValueError) as e:
            print(f"[SKIP] {paper_id}: {e}", file=sys.stderr)
            continue

        # Write report
        output_path = OUTPUTS_DIR / paper_id / "rubric_score_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"  Wrote: {output_path}")


if __name__ == "__main__":
    main()
