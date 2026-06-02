"""
Verifier: judge whether Executor results match paper expected_claims.

For each claim in paper_state.expected_claims, look up the corresponding
experiment in results.json, ask an LLM judge for a pass/partial/fail verdict,
and compute an overall reproducibility score.

Placeholder detection: when result data contains null values (table) or a
"[PLACEHOLDER" summary (figure), the claim is skipped without calling the LLM.
This avoids burning API credits on mock data.

LLM client pattern matches planner.py (not shared — both duplicate
llm_summarizer.py's pattern, which will be unified in Phase 3 late-stage).
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from jsonschema import Draft7Validator
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROMPT_DIR = PROJECT_ROOT / "pipeline" / "verifier" / "prompts"
RESULTS_SCHEMA_PATH = PROJECT_ROOT / "pipeline" / "schemas" / "results.schema.json"
REPORT_SCHEMA_PATH = PROJECT_ROOT / "pipeline" / "schemas" / "verification_report.schema.json"

PLACEHOLDER_MARKER = "[PLACEHOLDER"

# ---------------------------------------------------------------------------
# LLM client helpers (duplicated from planner.py; tech debt tracked in worklog)
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


def _call_llm(
    client: OpenAI,
    prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 2500,
) -> Tuple[str, dict]:
    """Make a single LLM call. Returns (text_response, usage_info).

    temperature=0.2 for stability — lower than Stage Planner (0.3)
    because verdict consistency is critical for evaluation.
    """
    messages = [{"role": "user", "content": prompt}]
    resp = client.chat.completions.create(
        model=_model_name(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
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


# ---------------------------------------------------------------------------
# Debug logging (per-claim, aligned with Stage Planner Step 5 format)
# ---------------------------------------------------------------------------


def _save_claim_debug(
    paper_id: str,
    claim_id: str,
    sent_prompt: str,
    raw_response: str,
    parsed_verdict: Optional[dict],
    usage: dict,
    warnings: List[str],
) -> None:
    """Save per-claim LLM prompt + response to _debug/ for inspection."""
    debug_dir = OUTPUTS_DIR / paper_id / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"llm_response_verifier_{claim_id}.json"

    debug_data = {
        "paper_id": paper_id,
        "claim_id": claim_id,
        "model": _model_name(),
        "temperature": 0.2,
        "max_tokens": 2500,
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
# Data loading
# ---------------------------------------------------------------------------


def load_paper_state(paper_id: str) -> dict:
    """Load paper_state.json for a paper."""
    path = OUTPUTS_DIR / paper_id / "paper_state.json"
    if not path.exists():
        raise FileNotFoundError(f"paper_state not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(paper_id: str) -> dict:
    """Load results.json (or mock_results.json) for a paper."""
    # Prefer mock_results.json during Phase 3.1; real results.json in Phase 3.2+
    for filename in ("results.json", "mock_results.json"):
        path = OUTPUTS_DIR / paper_id / filename
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"Neither results.json nor mock_results.json found in {OUTPUTS_DIR / paper_id}"
    )


# ---------------------------------------------------------------------------
# Evidence → experiment lookup
# ---------------------------------------------------------------------------


def find_exp_by_evidence(
    main_experiments: List[dict], evidence_id: str
) -> Optional[dict]:
    """Find the main_experiment entry whose evidence_in_paper matches evidence_id.

    evidence_id comes from claim.evidence (e.g. "table-1").
    Returns the matching experiment dict, or None if not found.
    """
    for exp in main_experiments:
        if exp.get("evidence_in_paper") == evidence_id:
            return exp
    return None


def _find_entity_caption(paper_state: dict, evidence_id: str) -> str:
    """Look up the caption of an entity (table/figure) by its id.

    Searches paper_state.entities.tables and .figures for a matching id.
    Returns the caption string, or "" if not found.
    """
    entities = paper_state.get("entities", {})
    for entity_type in ("tables", "figures"):
        for e in entities.get(entity_type, []):
            if e.get("id") == evidence_id:
                return e.get("caption", "") or ""
    return ""


# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------


def is_placeholder(result_data: Optional[dict]) -> bool:
    """Check whether result data is a placeholder (null values or placeholder summary).

    For table-type experiments: checks if any cell in values is null.
    For figure-type experiments: checks if summary contains the placeholder marker.
    Returns True if result_data is None (missing experiment key).
    """
    if result_data is None:
        return True

    # Figure type: has summary field
    if "summary" in result_data:
        summary = result_data.get("summary", "")
        if PLACEHOLDER_MARKER in summary:
            return True
        return False

    # Table type: has values field
    if "values" in result_data:
        values = result_data.get("values", [])
        if not values:
            return True
        for row in values:
            for cell in row:
                if cell is None:
                    return True
        return False

    # Unknown experiment shape — treat as placeholder to be safe
    return True


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


def _render_judge_prompt(
    claim: dict, result_data: dict, exp: dict, *, evidence_caption: str = ""
) -> str:
    """Render the claim_judge prompt with a single {INPUT_JSON} injection.

    The prompt template uses {INPUT_JSON} as the sole placeholder.
    We use .replace() instead of .format() because the template contains
    JSON examples with literal curly braces that would break .format().
    """
    template = (PROMPT_DIR / "claim_judge.txt").read_text(encoding="utf-8")

    input_obj = {
        "claim_text": claim.get("claim_text", ""),
        "verification_hint": claim.get("verification_hint") or "",
        "evidence_id": claim.get("evidence", ""),
        "evidence_caption": evidence_caption,
        "result_data": result_data,
    }
    input_json = json.dumps(input_obj, indent=2, ensure_ascii=False)

    return template.replace("{INPUT_JSON}", input_json)


def llm_judge_claim(
    client: OpenAI,
    claim: dict,
    result_data: dict,
    exp: dict,
    *,
    paper_id: str = "",
    evidence_caption: str = "",
    verbose: bool = False,
) -> dict:
    """Call LLM to judge a single claim against result data.

    Returns a claim_result dict with verdict, confidence, reasoning, etc.
    Retries up to 3 times on JSON parse / field validation failure.
    Saves per-claim debug to _debug/ on every attempt.
    """
    prompt = _render_judge_prompt(
        claim, result_data, exp, evidence_caption=evidence_caption
    )

    MAX_RETRIES = 3
    last_error: Optional[str] = None
    last_raw: str = ""
    last_prompt: str = prompt
    last_usage: dict = {}
    last_parsed: Optional[dict] = None
    debug_warnings: List[str] = []

    for attempt in range(1, MAX_RETRIES + 1):
        if verbose and attempt > 1:
            print(f"    [judge] retry {attempt}/{MAX_RETRIES} for {claim['id']}")

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
                '  "verdict": "pass" | "partial" | "fail",\n'
                '  "confidence": <number between 0.0 and 1.0>,\n'
                '  "reasoning": "<concise explanation, max 30 words>",\n'
                '  "claimed_value": "<what the claim asserts, with numbers>",\n'
                '  "observed_value": "<what result_data actually shows, with numbers>"\n'
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
            text, usage = _call_llm(client, combined_prompt)
        else:
            last_prompt = prompt
            text, usage = _call_llm(client, prompt)

        last_raw = text
        last_usage = usage

        try:
            result = _parse_json_loose(text)
            last_parsed = result
        except json.JSONDecodeError as e:
            last_error = f"Failed to parse JSON: {e}\n(first 300 chars): {text[:300]}"
            continue

        # Validate required fields
        verdict = result.get("verdict", "")
        if verdict not in ("pass", "partial", "fail"):
            last_error = (
                f"Invalid verdict '{verdict}'; must be pass, partial, or fail"
            )
            continue

        confidence = result.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            last_error = (
                f"Invalid confidence {confidence!r}; must be a number between 0 and 1"
            )
            continue

        reasoning = result.get("reasoning", "")
        if not isinstance(reasoning, str) or len(reasoning) > 300:
            reasoning = (reasoning or "")[:300]

        claimed_value = result.get("claimed_value", "") or ""
        observed_value = result.get("observed_value", "") or ""

        # Success — save debug and return
        if paper_id:
            _save_claim_debug(
                paper_id, claim["id"], last_prompt, last_raw,
                result, last_usage, debug_warnings,
            )

        return {
            "claim_id": claim["id"],
            "verdict": verdict,
            "confidence": float(confidence),
            "reasoning": reasoning,
            "claimed_value": str(claimed_value),
            "observed_value": str(observed_value),
        }

    # All retries exhausted — save debug with last state and fall back
    if paper_id:
        debug_warnings.append(
            f"LLM judge failed after {MAX_RETRIES} attempts: {last_error or 'unknown error'}"
        )
        _save_claim_debug(
            paper_id, claim["id"], last_prompt, last_raw,
            last_parsed, last_usage, debug_warnings,
        )

    return {
        "claim_id": claim["id"],
        "verdict": "fail",
        "confidence": 0.3,
        "reasoning": f"LLM judge produced malformed JSON after {MAX_RETRIES} retries; conservatively marked fail.",
        "claimed_value": claim.get("claim_text", "")[:200],
        "observed_value": "LLM judge unavailable (malformed JSON after retries)",
    }


# ---------------------------------------------------------------------------
# Overall score
# ---------------------------------------------------------------------------


def compute_overall_score(claim_results: List[dict]) -> Optional[float]:
    """Compute overall_score as mean of non-skipped verdicts.

    pass = 1.0, partial = 0.5, fail = 0.0.
    skipped claims are excluded from the mean.
    Returns None if all claims are skipped.
    """
    WEIGHTS = {"pass": 1.0, "partial": 0.5, "fail": 0.0}
    scored = [WEIGHTS[cr["verdict"]] for cr in claim_results if cr["verdict"] != "skipped"]
    if not scored:
        return None
    return sum(scored) / len(scored)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _validate_results(results: dict) -> Optional[str]:
    """Validate results against results.schema.json. Return None if OK."""
    schema = json.loads(RESULTS_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(results), key=lambda e: list(e.path))
    if not errors:
        return None
    msgs = []
    for e in errors[:5]:
        path = "/".join(str(p) for p in e.path) or "<root>"
        msgs.append(f"  - at '{path}': {e.message}")
    if len(errors) > 5:
        msgs.append(f"  ... and {len(errors) - 5} more")
    return "Results schema validation failed:\n" + "\n".join(msgs)


def _validate_report(report: dict) -> Optional[str]:
    """Validate verification_report against verification_report.schema.json. Return None if OK."""
    schema = json.loads(REPORT_SCHEMA_PATH.read_text(encoding="utf-8"))
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
# Main entry point
# ---------------------------------------------------------------------------


def verify(
    paper_id: str,
    *,
    client: Optional[OpenAI] = None,
    verbose: bool = True,
) -> dict:
    """Run verification for one paper.

    Args:
        paper_id: Paper identifier (e.g. "AMUN").
        client: Optional OpenAI client. If None, creates one (real API calls).
                Inject a mock client in tests to avoid real LLM calls.
        verbose: Print progress to stdout.

    Returns:
        verification_report dict (also validated against schema).
    """
    paper_state = load_paper_state(paper_id)
    results = load_results(paper_id)
    claims = paper_state.get("expected_claims", [])
    main_experiments = paper_state.get("main_experiments", [])

    if verbose:
        print(f"[{paper_id}] {len(claims)} claims, {len(results.get('experiments', {}))} experiments")

    # Validate results against schema (non-blocking warning)
    warnings: List[str] = []
    results_err = _validate_results(results)
    if results_err:
        warnings.append(results_err)
        if verbose:
            print(f"  ⚠ {results_err}")

    # Build evidence → experiment id lookup
    evidence_to_exp_id: Dict[str, str] = {}
    for exp in main_experiments:
        evidence_id = exp.get("evidence_in_paper", "")
        if evidence_id:
            evidence_to_exp_id[evidence_id] = exp["id"]

    claim_results: List[dict] = []

    for claim in claims:
        evidence_id = claim.get("evidence", "")  # e.g. "table-1"
        claim_id = claim.get("id", "")

        # Find the corresponding main_experiment
        exp = find_exp_by_evidence(main_experiments, evidence_id)
        if exp is None:
            warnings.append(
                f"claim '{claim_id}' evidence '{evidence_id}' not found in main_experiments"
            )
            claim_results.append({
                "claim_id": claim_id,
                "verdict": "skipped",
                "confidence": 0.0,
                "reasoning": f"evidence '{evidence_id}' not found in paper_state.main_experiments",
                "claimed_value": claim.get("claim_text", "")[:200],
                "observed_value": "N/A",
            })
            continue

        # Look up the experiment result
        exp_id = exp["id"]  # e.g. "exp-table1"
        result_data = results.get("experiments", {}).get(exp_id)

        # Check for placeholder data
        if is_placeholder(result_data):
            if verbose:
                label = "null values" if result_data and "values" in result_data else "placeholder"
                print(f"  [{claim_id}] skipped ({label})")
            claim_results.append({
                "claim_id": claim_id,
                "verdict": "skipped",
                "confidence": 0.0,
                "reasoning": "results data is placeholder (null)",
                "claimed_value": claim.get("claim_text", "")[:200],
                "observed_value": "placeholder (null data)",
            })
            continue

        # Real data — call LLM judge
        if client is None:
            client = _make_client()

        evidence_caption = _find_entity_caption(paper_state, evidence_id)

        if verbose:
            print(f"  [{claim_id}] judging...")
        verdict = llm_judge_claim(
            client, claim, result_data, exp,
            paper_id=paper_id,
            evidence_caption=evidence_caption,
            verbose=verbose,
        )
        claim_results.append(verdict)
        if verbose:
            print(f"    → {verdict['verdict']} (confidence={verdict['confidence']:.2f})")

    overall_score = compute_overall_score(claim_results)

    report = {
        "paper_id": paper_id,
        "overall_score": overall_score,
        "claim_results": claim_results,
        "verifier_model": _model_name(),
        "verifier_version": "v1",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "warnings": warnings,
    }

    # Validate report against schema
    report_err = _validate_report(report)
    if report_err:
        warnings.append(report_err)
        report["warnings"] = warnings
        if verbose:
            print(f"  ⚠ {report_err}")

    if verbose:
        skipped = sum(1 for cr in claim_results if cr["verdict"] == "skipped")
        passed = sum(1 for cr in claim_results if cr["verdict"] == "pass")
        partial = sum(1 for cr in claim_results if cr["verdict"] == "partial")
        failed = sum(1 for cr in claim_results if cr["verdict"] == "fail")
        score_str = f"{overall_score:.3f}" if overall_score is not None else "null (all skipped)"
        print(
            f"  [{paper_id}] done: {passed}P/{partial}Q/{failed}F/{skipped}S, "
            f"overall={score_str}"
        )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python verifier.py <paper_id> [paper_id ...]", file=sys.stderr)
        sys.exit(1)

    for paper_id in sys.argv[1:]:
        try:
            report = verify(paper_id)
        except FileNotFoundError as e:
            print(f"[SKIP] {paper_id}: {e}", file=sys.stderr)
            continue

        # Write report
        output_path = OUTPUTS_DIR / paper_id / "verification_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"  Wrote: {output_path}")


if __name__ == "__main__":
    main()
