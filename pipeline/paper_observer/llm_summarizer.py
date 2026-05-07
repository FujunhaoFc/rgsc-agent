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
        max_tokens=8000,
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
        if last_error and attempt > 1:
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
# Driver: run on AMUN
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline.paper_observer.section_parser import parse_paper_md
    from pipeline.paper_observer.entity_extractor import extract_from_paper

    paper_name = sys.argv[1] if len(sys.argv) > 1 else "AMUN"

    paper_path = PROJECT_ROOT / "data" / "train_valid" / paper_name / "paper.md"
    if not paper_path.exists():
        print(f"Paper not found: {paper_path}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Extracting skeleton paper_state for: {paper_name} ===\n")

    paper_md = paper_path.read_text(encoding="utf-8")
    sections = parse_paper_md(str(paper_path))
    entities = extract_from_paper(str(paper_path), sections)

    print(f"Loaded paper.md: {len(paper_md):,} chars")
    print(f"Sections: {len(sections)}")
    print(f"Entities: {sum(len(v) for v in entities.values())} "
          f"(tables={len(entities['tables'])}, "
          f"figures={len(entities['figures'])}, "
          f"algorithms={len(entities['algorithms'])}, "
          f"equations={len(entities['equations'])})")
    print()

    state, usage = extract_skeleton(paper_name, paper_md, sections, entities)

    out_dir = PROJECT_ROOT / "outputs" / paper_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paper_state.json"
    out_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n✓ Wrote: {out_path}")
    print(f"  Total attempts: {len(usage)}")
    total_in = sum(u["input_tokens"] or 0 for u in usage)
    total_out = sum(u["output_tokens"] or 0 for u in usage)
    print(f"  Total tokens: {total_in:,} in / {total_out:,} out")

    # Quick summary of what was extracted
    print("\n--- Extracted summary ---")
    print(f"core_method.name:     {state['core_method']['name']}")
    print(f"core_method.one_line: {state['core_method']['one_line']}")
    print(f"datasets:             {len(state['datasets'])}  ({[d['name'] for d in state['datasets']]})")
    print(f"models:               {len(state['models'])}  ({[m['name'] for m in state['models']]})")
    print(f"baselines:            {len(state['baselines'])}  ({[b['name'] for b in state['baselines']]})")
    print(f"metrics:              {len(state['metrics'])}  ({[m['name'] for m in state['metrics']]})")
    print(f"main_experiments:     {len(state['main_experiments'])}  "
          f"primary={sum(1 for e in state['main_experiments'] if e.get('primary'))}")
    print(f"expected_claims:      {len(state['expected_claims'])}")
    print(f"settings:             {len(state['core_method']['settings'])}")
    print(f"key_hyperparameters:  {len(state['core_method']['key_hyperparameters'])}")
