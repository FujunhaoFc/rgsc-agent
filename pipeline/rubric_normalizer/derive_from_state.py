"""
Derive a self-checklist from paper_state.

This module simulates what an agent would do at test time when no official
rubrics are available: extract a checklist of items the agent should
complete, in rubric-style natural language, from paper_state alone.

Each derived item is mapped to one of 5 rubric types (same as official rubric):
  - Paper Observation
  - Plan Writing
  - Code Implementation
  - Command Execution
  - Result Matching

The generators are pure rule-based: deterministic, no LLM, no randomness.
LLM judgment only enters later in coverage_diagnostic.py when we score
semantic match between derived items and official rubric criteria.
"""

from __future__ import annotations

from typing import Dict, List


# Section ids that are non-substantive and should NOT generate
# Paper Observation items.
_TRIVIAL_SECTION_IDS = {
    "sec-abstract", "sec-references", "sec-acknowledgments",
    "sec-acknowledgements", "sec-acknowledgement", "sec-impact-statement",
    "sec-ethics-statement", "sec-limitations", "sec-contribution-statement",
    "sec-reproducibility-statement",
}


def _format_section_label(sec: Dict) -> str:
    """Return a short label for a section, e.g. 'Section 3.1 (Notation)'."""
    sec_id = sec.get("id", "")
    title = sec.get("title", "").strip()
    # Strip leading numeric prefix from title for cleaner output
    parts = title.split(maxsplit=1)
    if parts and (parts[0].rstrip(".").replace(".", "").isalnum()):
        clean_title = parts[1] if len(parts) > 1 else parts[0]
    else:
        clean_title = title
    if sec_id.startswith("sec-"):
        sec_num = sec_id[len("sec-"):]
        return f"Section {sec_num} ({clean_title})"
    return clean_title


def _is_substantive_section(sec: Dict) -> bool:
    """A section is substantive if it's not trivial AND has a non-trivial body."""
    if sec.get("id") in _TRIVIAL_SECTION_IDS:
        return False
    # Need actual content
    if (sec.get("line_end", 0) - sec.get("line_start", 0)) < 2:
        return False
    return True


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def _gen_paper_observation(state: Dict) -> List[Dict]:
    items: List[Dict] = []

    # One item per substantive section
    for sec in state.get("sections", []):
        if not _is_substantive_section(sec):
            continue
        label = _format_section_label(sec)
        items.append({
            "type": "Paper Observation",
            "description": f"The agent has read {label} to understand its content.",
            "source": {"field": "sections", "ref": sec["id"]},
            "anchor": None,
        })

    # One item per entity (tables, figures, algorithms)
    entities = state.get("entities", {}) or {}
    for kind in ("tables", "figures", "algorithms"):
        for ent in entities.get(kind, []):
            ent_id = ent.get("id")
            label_kind = kind[:-1].capitalize()  # "table" -> "Table"
            label_num = ent.get("label", "")
            items.append({
                "type": "Paper Observation",
                "description": f"The agent has read {label_kind} {label_num} "
                               f"and understood the information it presents.",
                "source": {"field": f"entities.{kind}", "ref": ent_id},
                "anchor": ent_id,
            })

    return items


def _gen_plan_writing(state: Dict) -> List[Dict]:
    items: List[Dict] = []
    cm = state.get("core_method", {})

    # One item per atomic_step
    for step in cm.get("atomic_steps", []):
        desc = step.get("description", "")
        items.append({
            "type": "Plan Writing",
            "description": f"The plan includes {_lowercase_first(desc)}",
            "source": {"field": "core_method.atomic_steps", "ref": step.get("id")},
            "anchor": None,
        })

    # One item per setting
    for setting in cm.get("settings", []):
        name = setting.get("name", "")
        desc = setting.get("description", "")
        items.append({
            "type": "Plan Writing",
            "description": f"The plan includes the setting '{name}': {desc}",
            "source": {"field": "core_method.settings", "ref": setting.get("id")},
            "anchor": None,
        })

    # One item per dataset
    for ds in state.get("datasets", []):
        name = ds.get("name", "")
        items.append({
            "type": "Plan Writing",
            "description": f"The plan includes the use of the {name} dataset.",
            "source": {"field": "datasets", "ref": name},
            "anchor": None,
        })

    # One item per model
    for m in state.get("models", []):
        name = m.get("name", "")
        items.append({
            "type": "Plan Writing",
            "description": f"The plan includes the use of the {name} model.",
            "source": {"field": "models", "ref": name},
            "anchor": None,
        })

    # One item per baseline
    for b in state.get("baselines", []):
        name = b.get("name", "")
        items.append({
            "type": "Plan Writing",
            "description": f"The plan includes the {name} baseline for comparison.",
            "source": {"field": "baselines", "ref": name},
            "anchor": None,
        })

    # One item per metric
    for m in state.get("metrics", []):
        name = m.get("name", "")
        items.append({
            "type": "Plan Writing",
            "description": f"The plan includes the {name} evaluation metric.",
            "source": {"field": "metrics", "ref": name},
            "anchor": None,
        })

    # One item per key hyperparameter
    for hp in cm.get("key_hyperparameters", []):
        name = hp.get("name", "")
        value = hp.get("value", "")
        items.append({
            "type": "Plan Writing",
            "description": f"The plan specifies the hyperparameter '{name}' with value {value}.",
            "source": {"field": "core_method.key_hyperparameters", "ref": name},
            "anchor": None,
        })

    return items


def _gen_code_implementation(state: Dict) -> List[Dict]:
    items: List[Dict] = []
    cm = state.get("core_method", {})

    # One item per atomic_step (the "implement this step" view)
    for step in cm.get("atomic_steps", []):
        desc = step.get("description", "")
        items.append({
            "type": "Code Implementation",
            "description": f"The code implementing the following step is present: {desc}",
            "source": {"field": "core_method.atomic_steps", "ref": step.get("id")},
            "anchor": None,
        })

    # One item per baseline (implementing each baseline is its own code task)
    for b in state.get("baselines", []):
        name = b.get("name", "")
        items.append({
            "type": "Code Implementation",
            "description": f"The code for the {name} baseline is implemented.",
            "source": {"field": "baselines", "ref": name},
            "anchor": None,
        })

    # One item per metric (computing each metric is also a code task)
    for m in state.get("metrics", []):
        name = m.get("name", "")
        defn = m.get("definition", "")
        items.append({
            "type": "Code Implementation",
            "description": f"The code computing the {name} metric is implemented.",
            "source": {"field": "metrics", "ref": name},
            "anchor": None,
        })

    return items


def _gen_command_execution(state: Dict) -> List[Dict]:
    items: List[Dict] = []

    # Per-experiment command execution
    for exp in state.get("main_experiments", []):
        exp_id = exp.get("id", "")
        evidence = exp.get("evidence_in_paper", "")
        claim = exp.get("claim", "")
        items.append({
            "type": "Command Execution",
            "description": f"The agent has executed the command(s) that produce the "
                           f"results shown in {_format_entity_label(evidence)} "
                           f"({claim[:80]}{'...' if len(claim) > 80 else ''}).",
            "source": {"field": "main_experiments", "ref": exp_id},
            "anchor": evidence if evidence else None,
        })

    # Process-level command execution (training, evaluation pipelines run successfully)
    cm = state.get("core_method", {})
    method_name = cm.get("name", "the method")

    if state.get("reproducibility_meta", {}).get("needs_training", False):
        items.append({
            "type": "Command Execution",
            "description": f"The training process for {method_name} has been successfully "
                           f"executed and finished without errors.",
            "source": {"field": "reproducibility_meta", "ref": "training"},
            "anchor": None,
        })

    items.append({
        "type": "Command Execution",
        "description": f"The evaluation pipeline for {method_name} has been successfully "
                       f"executed and finished without errors.",
        "source": {"field": "reproducibility_meta", "ref": "evaluation"},
        "anchor": None,
    })

    # Per-dataset evaluation command
    for ds in state.get("datasets", []):
        name = ds.get("name", "")
        items.append({
            "type": "Command Execution",
            "description": f"The evaluation on the {name} dataset has been "
                           f"successfully executed without errors.",
            "source": {"field": "datasets", "ref": name},
            "anchor": None,
        })

    # Per-baseline execution command
    for b in state.get("baselines", []):
        name = b.get("name", "")
        items.append({
            "type": "Command Execution",
            "description": f"The {name} baseline has been successfully executed "
                           f"and produced its expected output.",
            "source": {"field": "baselines", "ref": name},
            "anchor": None,
        })

    # Per-setting execution command (if multiple settings exist)
    settings = cm.get("settings", [])
    if len(settings) >= 2:
        for setting in settings:
            name = setting.get("name", "")
            items.append({
                "type": "Command Execution",
                "description": f"The experiment under the '{name}' setting has been "
                               f"successfully executed and produced its expected output.",
                "source": {"field": "core_method.settings", "ref": setting.get("id")},
                "anchor": None,
            })

    return items


def _gen_result_matching(state: Dict) -> List[Dict]:
    items: List[Dict] = []

    # One item per main_experiment (Result Matching is anchored to evidence)
    for exp in state.get("main_experiments", []):
        exp_id = exp.get("id", "")
        evidence = exp.get("evidence_in_paper", "")
        claim = exp.get("claim", "")
        items.append({
            "type": "Result Matching",
            "description": f"The execution outcomes match the results in "
                           f"{_format_entity_label(evidence)}: {claim}",
            "source": {"field": "main_experiments", "ref": exp_id},
            "anchor": evidence if evidence else None,
        })

    # One additional item per expected_claim (more granular than main_exp)
    for ec in state.get("expected_claims", []):
        ec_id = ec.get("id", "")
        evidence = ec.get("evidence", "")
        claim_text = ec.get("claim_text", "")
        items.append({
            "type": "Result Matching",
            "description": f"The execution outcomes match the specific claim "
                           f"(based on {_format_entity_label(evidence)}): {claim_text}",
            "source": {"field": "expected_claims", "ref": ec_id},
            "anchor": evidence if evidence else None,
        })

    return items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lowercase_first(s: str) -> str:
    """Lowercase the first character. Used for 'The plan includes <lowercased>'."""
    return s[:1].lower() + s[1:] if s else s


def _format_entity_label(entity_id: str) -> str:
    """Convert 'table-1' to 'Table 1', 'figure-2' to 'Figure 2'."""
    if not entity_id:
        return "the relevant experiment"
    parts = entity_id.split("-", 1)
    if len(parts) == 2:
        kind, num = parts
        return f"{kind.capitalize()} {num}"
    return entity_id


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def derive_checklist(paper_state: Dict) -> List[Dict]:
    """Derive a self-checklist from a paper_state.

    Returns a list of derived items with id assigned in iteration order.
    """
    generators = [
        ("po", _gen_paper_observation),
        ("pw", _gen_plan_writing),
        ("ci", _gen_code_implementation),
        ("ce", _gen_command_execution),
        ("rm", _gen_result_matching),
    ]

    all_items: List[Dict] = []
    for prefix, gen in generators:
        items = gen(paper_state)
        for i, item in enumerate(items):
            item["id"] = f"derived-{prefix}-{i + 1:03d}"
        all_items.extend(items)

    return all_items


def summarize_checklist(items: List[Dict]) -> Dict:
    """Group items by type and return counts."""
    summary = {}
    for item in items:
        t = item["type"]
        summary[t] = summary.get(t, 0) + 1
    summary["TOTAL"] = sum(summary.values())
    return summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

    paper = sys.argv[1] if len(sys.argv) > 1 else "AMUN"

    state_path = PROJECT_ROOT / "outputs" / paper / "paper_state.json"
    if not state_path.exists():
        print(f"paper_state not found: {state_path}", file=sys.stderr)
        sys.exit(1)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    items = derive_checklist(state)

    out_path = PROJECT_ROOT / "outputs" / paper / "derived_checklist.json"
    out_path.write_text(
        json.dumps(items, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = summarize_checklist(items)
    print(f"=== Derived checklist for {paper} ===\n")
    for t in ["Paper Observation", "Plan Writing", "Code Implementation",
              "Command Execution", "Result Matching"]:
        print(f"  {t:<22}: {summary.get(t, 0)}")
    print(f"  {'-' * 30}")
    print(f"  {'TOTAL':<22}: {summary['TOTAL']}")
    print(f"\n✓ Wrote: {out_path}")

    # Print first 2 items of each type for inspection
    print(f"\n=== Sample items (first 2 of each type) ===\n")
    seen_types = {}
    for item in items:
        t = item["type"]
        if seen_types.get(t, 0) >= 2:
            continue
        seen_types[t] = seen_types.get(t, 0) + 1
        print(f"[{item['id']}] {t}")
        print(f"  {item['description'][:120]}")
        print(f"  source: {item['source']}")
        if item.get("anchor"):
            print(f"  anchor: {item['anchor']}")
        print()
