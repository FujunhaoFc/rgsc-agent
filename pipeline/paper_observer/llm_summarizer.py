"""
LLM Summarizer: invoke DeepSeek V4 to extract structured paper_state from a paper.

Provides a single high-level entry: extract_skeleton(paper_id, paper_md, sections, entities)

Implementation notes:
- Uses OpenAI-compatible client pointed at DeepSeek's OpenAI endpoint.
- API key from env var DEEPSEEK_API_KEY (loaded from .env via python-dotenv).
- Retries up to 3 times on JSON parse failure or schema validation failure,
  passing the error back to the LLM each retry.
- Default model: deepseek-v4-pro (override with DEEPSEEK_MODEL env var).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from pipeline.common.paths import find_paper_md
from jsonschema import Draft7Validator, ValidationError
from openai import OpenAI


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = PROJECT_ROOT / "pipeline" / "paper_observer" / "prompts"
SCHEMA_PATH = PROJECT_ROOT / "pipeline" / "schemas" / "paper_state.schema.json"

# Fraction of paper.md to feed into Call 1.
CALL1_PAPER_FRACTION = 0.6

# Retry budget for schema/JSON failures.
MAX_RETRIES = 3


# ----------------------------------------------------------------------------
# Client setup
# ----------------------------------------------------------------------------


def _make_client() -> OpenAI:
    """Build an OpenAI client pointed at DeepSeek's OpenAI-compat endpoint."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Put it in .env or export it."
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )


def _model_name() -> str:
    return os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")


# ----------------------------------------------------------------------------
# Prompt rendering
# ----------------------------------------------------------------------------


def _format_sections_listing(sections: List[Dict]) -> str:
    """Render sections list as a compact text block for prompt injection."""
    lines = []
    for s in sections:
        indent = "  " * (s["level"] - 1)
        title = s["title"][:80]
        lines.append(f"{indent}{s['id']}  L{s['level']}  {title}")
    return "\n".join(lines) if lines else "(no sections)"


def _format_entities_listing(items: List[Dict]) -> str:
    """Render an entity list (tables/figures/etc) as compact text."""
    if not items:
        return "(none)"
    lines = []
    for e in items:
        cap = e.get("caption", "") or ""
        cap = cap[:80] + ("..." if len(cap) > 80 else "")
        sec = e.get("in_section") or "(no section)"
        line = f"  {e['id']}  in {sec}"
        if cap:
            line += f"  caption={cap!r}"
        lines.append(line)
    return "\n".join(lines)


def _truncate_paper_md(paper_md: str, fraction: float) -> str:
    """Take the first `fraction` of paper_md by character count."""
    cutoff = int(len(paper_md) * fraction)
    truncated = paper_md[:cutoff]
    # Try to end at a newline so the truncation is clean
    last_nl = truncated.rfind("\n")
    if last_nl > 0 and (len(truncated) - last_nl) < 200:
        truncated = truncated[: last_nl + 1]
    return truncated + "\n\n[...paper truncated for Call 1...]\n"


def render_skeleton_prompt(
    paper_id: str,
    paper_md: str,
    sections: List[Dict],
    entities: Dict,
) -> str:
    """Render the Call 1 skeleton extraction prompt with all placeholders filled."""
    template_path = PROMPT_DIR / "skeleton_extraction.txt"
    template = template_path.read_text(encoding="utf-8")

    rendered = (
        template
        .replace("{{paper_id}}", paper_id)
        .replace("{{sections_listing}}", _format_sections_listing(sections))
        .replace("{{tables_listing}}", _format_entities_listing(entities.get("tables", [])))
        .replace("{{figures_listing}}", _format_entities_listing(entities.get("figures", [])))
        .replace("{{algorithms_listing}}", _format_entities_listing(entities.get("algorithms", [])))
        .replace("{{paper_content}}", _truncate_paper_md(paper_md, CALL1_PAPER_FRACTION))
    )
    return rendered


# ----------------------------------------------------------------------------
# JSON parsing helpers
# ----------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """If the LLM wrapped JSON in ```json ... ```, strip those fences."""
    text = text.strip()
    fence_re = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)
    m = fence_re.match(text)
    if m:
        return m.group(1).strip()
    return text


def _parse_json_loose(text: str) -> dict:
    """Parse JSON, with one fallback: extract the first {...} balanced block."""
    text = _strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find first { and last } and try parsing that
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            candidate = text[first_brace : last_brace + 1]
            return json.loads(candidate)
        raise


# ----------------------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------------------


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validate_paper_state(state: dict) -> Optional[str]:
    """Validate against schema. Return None if OK, else an error message."""
    schema = _load_schema()
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(state), key=lambda e: list(e.path))
    if not errors:
        return None
    # Aggregate at most 5 errors into one message
    msgs = []
    for e in errors[:5]:
        path = "/".join(str(p) for p in e.path) or "<root>"
        msgs.append(f"  - at '{path}': {e.message}")
    if len(errors) > 5:
        msgs.append(f"  ... and {len(errors) - 5} more")
    return "Schema validation failed:\n" + "\n".join(msgs)


# ----------------------------------------------------------------------------
# LLM call
# ----------------------------------------------------------------------------


def _call_llm(
    client: OpenAI,
    prompt: str,
    *,
    extra_system: Optional[str] = None,
) -> Tuple[str, Dict]:
    """Make a single LLM call. Returns (text_response, usage_info)."""
    messages = []
    if extra_system:
        messages.append({"role": "system", "content": extra_system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=_model_name(),
        messages=messages,
        temperature=0.0,
        max_tokens=16000,
    )
    if resp is None or not resp.choices:
        raise RuntimeError(
            f"DeepSeek API returned empty response. "
            f"resp={resp!r}"
        )
    text = resp.choices[0].message.content or ""
    usage = {
        "input_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "output_tokens": resp.usage.completion_tokens if resp.usage else None,
        "model": resp.model,
    }
    return text, usage


# ----------------------------------------------------------------------------
# High-level entry
# ----------------------------------------------------------------------------


def extract_skeleton(
    paper_id: str,
    paper_md: str,
    sections: List[Dict],
    entities: Dict,
    *,
    verbose: bool = True,
) -> Tuple[Dict, List[Dict]]:
    """Run Call 1: extract skeleton paper_state for one paper.

    Returns (paper_state_dict, usage_log).

    paper_state_dict will have sections and entities injected back in
    (the LLM doesn't produce these — they come from rule-based modules).

    Raises RuntimeError if all retries exhausted.
    """
    client = _make_client()
    prompt = render_skeleton_prompt(paper_id, paper_md, sections, entities)

    if verbose:
        print(f"[skeleton] paper_id={paper_id}  prompt_chars={len(prompt):,}")

    usage_log: List[Dict] = []
    last_raw_output: str = ""
    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        if verbose:
            print(f"[skeleton] attempt {attempt}/{MAX_RETRIES}...")

        # On retry, give the LLM its own previous output + the validation
        # errors, and ask it to FIX those specific issues rather than
        # regenerate from scratch.
        if last_error and attempt > 1 and not last_error.startswith("Failed to parse"):
            retry_prompt = (
                "Your previous response had validation errors. "
                "Below is your previous JSON output and the errors. "
                "Output a CORRECTED full JSON object — keep all parts that "
                "were correct, fix only the errors. Output ONLY the JSON.\n\n"
                "=== Your previous output ===\n"
                f"{last_raw_output}\n\n"
                "=== Validation errors ===\n"
                f"{last_error}\n\n"
                "Now output the corrected full JSON:"
            )
            text, usage = _call_llm(client, retry_prompt)
        else:
            text, usage = _call_llm(client, prompt)
        usage_log.append({**usage, "attempt": attempt})
        last_raw_output = text
        if verbose:
            print(f"  → {usage['input_tokens']} in / {usage['output_tokens']} out tokens")

        # Try to parse JSON
        try:
            state = _parse_json_loose(text)
        except json.JSONDecodeError as e:
            last_error = f"Failed to parse output as JSON: {e}\n(first 300 chars of output): {text[:300]}"
            if verbose:
                print(f"  ✗ JSON parse failed: {e}")
            continue

        # Validate against schema
        err = _validate_paper_state(state)
        if err is None:
            # Success: inject regular fields and return
            state["sections"] = sections
            state["entities"] = entities
            if verbose:
                print(f"  ✓ schema valid on attempt {attempt}")
            return state, usage_log
        else:
            last_error = err
            if verbose:
                print(f"  ✗ schema invalid:")
                for line in err.splitlines()[:6]:
                    print(f"    {line}")
            continue

    raise RuntimeError(
        f"Failed to extract skeleton for {paper_id} after {MAX_RETRIES} attempts.\n"
        f"Last error:\n{last_error}"
    )

# ----------------------------------------------------------------------------
# Call 2: detail extraction
# ----------------------------------------------------------------------------


def _get_method_context(paper_md: str) -> Tuple[int, int, str]:
    """Return middle 50% of paper as method context for Call 2.

    Earlier versions tried to identify the method section by parsing
    titles. The heuristics needed paper-specific exceptions and never
    converged. The middle 50% of any reasonable ML paper reliably
    contains the method section, so we just return that — no parsing.
    """
    md_lines = paper_md.splitlines()
    n_lines = len(md_lines)
    line_start = max(1, n_lines // 4)
    line_end = (3 * n_lines) // 4
    return line_start, line_end, "\n".join(md_lines[line_start - 1 : line_end])


def render_detail_prompt(
    existing_state: Dict,
    paper_md: str,
    sections: List[Dict],
) -> Tuple[str, Dict]:
    """Render Call 2 (detail extraction) prompt."""
    template_path = PROMPT_DIR / "detail_extraction.txt"
    template = template_path.read_text(encoding="utf-8")

    state_for_llm = {k: v for k, v in existing_state.items() if k not in ("sections", "entities")}
    existing_state_json = json.dumps(state_for_llm, indent=2, ensure_ascii=False)

    line_start, line_end, method_section = _get_method_context(paper_md)

    rendered = (
        template
        .replace("{{existing_state_json}}", existing_state_json)
        .replace("{{method_line_start}}", str(line_start))
        .replace("{{method_line_end}}", str(line_end))
        .replace("{{method_section}}", method_section)
    )

    debug = {
        "method_line_start": line_start,
        "method_line_end": line_end,
        "method_section_chars": len(method_section),
    }
    return rendered, debug


def extract_detail(
    skeleton_state: Dict,
    paper_md: str,
    sections: List[Dict],
    *,
    verbose: bool = True,
) -> Tuple[Dict, List[Dict]]:
    """Run Call 2: fill atomic_steps and axes in an existing skeleton paper_state.

    Returns (updated_state, usage_log).
    """
    client = _make_client()
    prompt, debug = render_detail_prompt(skeleton_state, paper_md, sections)

    if verbose:
        print(f"[detail] method section: lines {debug['method_line_start']}-{debug['method_line_end']}, "
              f"{debug['method_section_chars']:,} chars")
        print(f"[detail] prompt_chars={len(prompt):,}")

    usage_log: List[Dict] = []
    last_error: Optional[str] = None
    last_raw_output: str = ""

    for attempt in range(1, MAX_RETRIES + 1):
        if verbose:
            print(f"[detail] attempt {attempt}/{MAX_RETRIES}...")

        if last_error and attempt > 1 and not last_error.startswith("Failed to parse"):
            retry_prompt = (
                "Your previous response had validation errors. "
                "Below is your previous JSON output and the errors. "
                "Output a CORRECTED full JSON object — keep all parts that "
                "were correct, fix only the errors. Output ONLY the JSON.\n\n"
                "=== Your previous output ===\n"
                f"{last_raw_output}\n\n"
                "=== Validation errors ===\n"
                f"{last_error}\n\n"
                "Now output the corrected full JSON:"
            )
            text, usage = _call_llm(client, retry_prompt)
        else:
            text, usage = _call_llm(client, prompt)

        usage_log.append({**usage, "attempt": attempt})
        last_raw_output = text
        if verbose:
            print(f"  → {usage['input_tokens']} in / {usage['output_tokens']} out tokens")

        try:
            state = _parse_json_loose(text)
        except json.JSONDecodeError as e:
            last_error = f"Failed to parse output as JSON: {e}\n(first 300 chars): {text[:300]}"
            if verbose:
                print(f"  ✗ JSON parse failed: {e}")
            continue

        # Re-inject sections and entities (LLM doesn't produce them)
        state["sections"] = skeleton_state["sections"]
        state["entities"] = skeleton_state["entities"]

        # Validate
        err = _validate_paper_state(state)

        # Extra Call-2-specific checks beyond schema
        extra_errors = []
        if not state.get("core_method", {}).get("atomic_steps"):
            extra_errors.append("core_method.atomic_steps must be a non-empty list (Call 2 must fill it)")
        for i, exp in enumerate(state.get("main_experiments", [])):
            if exp.get("axes") is None:
                extra_errors.append(f"main_experiments[{i}].axes is null; Call 2 must fill it (use empty arrays if unclear)")

        if err is None and not extra_errors:
            if verbose:
                print(f"  ✓ schema + Call-2 checks pass on attempt {attempt}")
            return state, usage_log

        combined = (err or "")
        if extra_errors:
            combined += "\nCall-2 specific:\n" + "\n".join(f"  - {e}" for e in extra_errors)
        last_error = combined
        if verbose:
            print(f"  ✗ validation issues:")
            for line in combined.splitlines()[:6]:
                print(f"    {line}")
        continue

    raise RuntimeError(
        f"Call 2 failed after {MAX_RETRIES} attempts.\n"
        f"Last error:\n{last_error}"
    )


# ----------------------------------------------------------------------------
# Combined driver: both Call 1 and Call 2
# ----------------------------------------------------------------------------


def extract_full_paper_state(
    paper_id: str,
    paper_md: str,
    sections: List[Dict],
    entities: Dict,
    *,
    verbose: bool = True,
) -> Tuple[Dict, Dict]:
    """Run Call 1 + Call 2 end-to-end. Returns (final_state, usage_breakdown)."""
    print(f"=== Call 1 (skeleton) ===")
    skeleton, call1_usage = extract_skeleton(
        paper_id, paper_md, sections, entities, verbose=verbose,
    )
    print(f"\n=== Call 2 (detail) ===")
    final, call2_usage = extract_detail(
        skeleton, paper_md, sections, verbose=verbose,
    )
    breakdown = {
        "call1_usage": call1_usage,
        "call2_usage": call2_usage,
    }
    return final, breakdown



# ----------------------------------------------------------------------------
# Driver: run on AMUN
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline.paper_observer.section_parser import parse_paper_md
    from pipeline.paper_observer.entity_extractor import extract_from_paper

    args = sys.argv[1:]
    paper_name = args[0] if args else "AMUN"
    mode = args[1] if len(args) > 1 else "full"   # "skeleton", "detail", "full"

    try:
        paper_path = find_paper_md(paper_name)
    except FileNotFoundError as e:
        print(f"Paper not found: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Extracting paper_state for {paper_name} (mode={mode}) ===\n")
    paper_md = paper_path.read_text(encoding="utf-8")
    sections = parse_paper_md(str(paper_path))
    entities = extract_from_paper(str(paper_path), sections)

    print(f"paper.md: {len(paper_md):,} chars")
    print(f"sections: {len(sections)}, entities: "
          f"{sum(len(v) for v in entities.values())}\n")

    out_dir = PROJECT_ROOT / "outputs" / paper_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "skeleton":
        state, usage = extract_skeleton(paper_name, paper_md, sections, entities)
        out_path = out_dir / "paper_state.json"
        usage_summary = {"call1_usage": usage}
    elif mode == "detail":
        existing_path = out_dir / "paper_state.json"
        if not existing_path.exists():
            print(f"No skeleton found at {existing_path}; run skeleton first")
            sys.exit(1)
        skeleton = json.loads(existing_path.read_text(encoding="utf-8"))
        state, usage = extract_detail(skeleton, paper_md, sections)
        out_path = out_dir / "paper_state.json"
        usage_summary = {"call2_usage": usage}
    else:
        state, usage_summary = extract_full_paper_state(
            paper_name, paper_md, sections, entities,
        )
        out_path = out_dir / "paper_state.json"

    out_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Wrote: {out_path}")

    # Summary
    print("\n--- Final summary ---")
    cm = state["core_method"]
    print(f"core_method.name:     {cm['name']}")
    print(f"core_method.one_line: {cm['one_line']}")
    print(f"atomic_steps:         {len(cm['atomic_steps'])}")
    if cm["atomic_steps"]:
        print(f"  first step: {cm['atomic_steps'][0].get('description', '')[:80]}")
        print(f"  last  step: {cm['atomic_steps'][-1].get('description', '')[:80]}")
    print(f"datasets:             {len(state['datasets'])}")
    print(f"models:               {len(state['models'])}")
    print(f"baselines:            {len(state['baselines'])}")
    print(f"metrics:              {len(state['metrics'])}")
    print(f"main_experiments:     {len(state['main_experiments'])}  "
          f"(axes filled: {sum(1 for e in state['main_experiments'] if e.get('axes'))})")
    print(f"expected_claims:      {len(state['expected_claims'])}")
    print(f"settings:             {len(cm['settings'])}")
    print(f"key_hyperparameters:  {len(cm['key_hyperparameters'])}")

    # Token total
    print("\n--- Token usage ---")
    for key, log in usage_summary.items():
        if isinstance(log, list):
            in_total = sum(u["input_tokens"] or 0 for u in log)
            out_total = sum(u["output_tokens"] or 0 for u in log)
            print(f"{key}: {in_total:,} in / {out_total:,} out, attempts={len(log)}")