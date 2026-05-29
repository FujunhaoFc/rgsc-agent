"""
Stage Planner: split derived_checklist into ordered stages (Phase 2, module 1).

Steps 1-4 (rule-based): filter Paper Obs, coarse map by source, detect training,
form stages, split large stages, compute coverage.
Step 5 (LLM refinement): rewrite actions, reassign atomic_steps, review depends_on.

LLM call config (empirically tuned on 5 papers, 2026-05-29):
- max_tokens=8000: 4000 was insufficient for Beyond-Ngram (7 stages, 10 atomic_steps,
  17k char prompt). At 8000, 5/5 papers pass LLM refinement (4/5 on first attempt).
  Beyond-Ngram still needs 2-3 retries (LLM non-determinism with large JSON output).
- atomic_step_assignment: the output schema asks for it, but DeepSeek V4-Pro
  consistently omits this field. Assignment is implicit in which actions the LLM
  writes for each stage. Accepted as a known compromise.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from jsonschema import Draft7Validator
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROMPT_DIR = PROJECT_ROOT / "pipeline" / "stage_planner" / "prompts"
SCHEMA_PATH = PROJECT_ROOT / "pipeline" / "schemas" / "task_plan.schema.json"

# ---------------------------------------------------------------------------
# Step 1 & 2: mapping config
# ---------------------------------------------------------------------------

PAPER_OBS_TYPE = "Paper Observation"

SOURCE_TO_STAGE_TYPE: Dict[str, str] = {
    "datasets": "setup",
    "models": "setup",
    "reproducibility_meta": "setup",
    "core_method.atomic_steps": "training",
    "baselines": "training",
    "core_method.settings": "training",
    "core_method.key_hyperparameters": "training",
    "metrics": "evaluation",
    "main_experiments": "evaluation",
    "expected_claims": "verification",
}

# ---------------------------------------------------------------------------
# LLM client helpers
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
    temperature: float = 0.0,
    max_tokens: int = 16000,
) -> Tuple[str, dict]:
    """Make a single LLM call. Returns (text_response, usage_info)."""
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
# Step 3: training detection
# ---------------------------------------------------------------------------

TRAINING_KEYWORDS = [
    "train", "fine-tune", "finetune", "optimize",
    "loss", "gradient", "epoch", "backprop",
]


def _text_hits_keywords(text: str, keywords: List[str]) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def detect_has_training(paper_state: dict) -> bool:
    """Check whether the paper involves training (vs pure inference/decoding)."""
    atomic_steps = paper_state.get("core_method", {}).get("atomic_steps", [])
    for step in atomic_steps:
        if _text_hits_keywords(step.get("description", ""), TRAINING_KEYWORDS):
            return True

    baselines = paper_state.get("baselines", [])
    if baselines:
        for b in baselines:
            combined = f"{b.get('name','')} {b.get('type','')}"
            if _text_hits_keywords(combined, TRAINING_KEYWORDS):
                return True

    return False


# ---------------------------------------------------------------------------
# Step 1: filter Paper Observation items
# ---------------------------------------------------------------------------

def filter_paper_observations(checklist: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Return (paper_obs_items, plannable_items)."""
    po_items = []
    plannable = []
    for item in checklist:
        if item.get("type") == PAPER_OBS_TYPE:
            po_items.append(item)
        else:
            plannable.append(item)
    return po_items, plannable


# ---------------------------------------------------------------------------
# Step 2: coarse mapping by source.field
# ---------------------------------------------------------------------------

def coarse_map(plannable_items: List[dict]) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """
    Map each plannable item to a stage_type via SOURCE_TO_STAGE_TYPE.
    Returns (stage_type -> [items], fallback_items).
    """
    buckets: Dict[str, List[dict]] = defaultdict(list)
    fallback: List[dict] = []

    for item in plannable_items:
        source_field = item.get("source", {}).get("field", "")
        stage_type = SOURCE_TO_STAGE_TYPE.get(source_field)
        if stage_type:
            buckets[stage_type].append(item)
        else:
            fallback.append(item)

    return dict(buckets), fallback


# ---------------------------------------------------------------------------
# Step 4: form stages
# ---------------------------------------------------------------------------

def _make_stage_id(name: str) -> str:
    """Convert a human name to kebab-case stage_id."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _derive_actions_from_items(items: List[dict], max_actions: int = 8) -> List[str]:
    """Generate simple imperative actions from item descriptions."""
    actions: List[str] = []
    for item in items:
        desc = item.get("description", "")
        # Extract key phrase: remove "The plan includes", "The code implementing...", etc.
        cleaned = re.sub(r"^The (plan includes|code implementing the following step is present|agent has executed the command\(s\) that produce the results shown in|execution outcomes match the results in) ", "", desc)
        cleaned = cleaned.rstrip(".")
        if cleaned not in actions and len(cleaned) > 10:
            actions.append(cleaned)
        if len(actions) >= max_actions:
            break
    return actions


def _generate_actions_for_stage(
    stage_type: str, items: List[dict], paper_state: dict, stage_id: str = ""
) -> List[str]:
    """Generate 3-8 actions for a stage from its items and paper context."""
    actions: List[str] = []

    if stage_type == "setup":
        # Extract dataset/model names
        datasets = paper_state.get("datasets", [])
        models = paper_state.get("models", [])
        actions.append("Create project repository skeleton with src/, data/, configs/ directories")
        actions.append("Install required Python packages (pytorch, torchvision, numpy, etc.)")
        for ds in datasets[:3]:
            actions.append(f"Download and prepare {ds.get('name', 'dataset')} dataset")
        for m in models[:2]:
            actions.append(f"Set up {m.get('name', 'model')} model ({m.get('role', 'backbone')})")
        meta = paper_state.get("reproducibility_meta", {})
        if meta.get("code_partial_available"):
            actions.append("Clone and review official code repository")
        actions = actions[:8]

    elif stage_type == "training":
        if stage_id == "baseline-training":
            # Generate actions from baseline items
            baselines = paper_state.get("baselines", [])
            for b in baselines[:7]:
                actions.append(f"Implement {b.get('name','')} baseline: {b.get('type','')}")
            actions.append("Set up training infrastructure shared across baselines")
        else:
            # main-training: actions from atomic_steps
            atomic_steps = paper_state.get("core_method", {}).get("atomic_steps", [])
            for step in atomic_steps:
                desc = step.get("description", "")
                if desc:
                    actions.append(f"Implement: {desc}")
            hparams = paper_state.get("core_method", {}).get("key_hyperparameters", [])
            if hparams:
                actions.append(f"Configure hyperparameters: {', '.join(h.get('name','') for h in hparams[:5])}")

    elif stage_type == "evaluation":
        metrics = paper_state.get("metrics", [])
        for m in metrics[:4]:
            actions.append(f"Compute {m.get('name', 'metric')}: {m.get('definition', '')[:80]}")
        experiments = paper_state.get("main_experiments", [])
        for exp in experiments[:4]:
            claim_short = exp.get("claim", "")[:80]
            if claim_short:
                actions.append(f"Run experiment to verify: {claim_short}")

    elif stage_type == "preprocessing":
        datasets = paper_state.get("datasets", [])
        for ds in datasets[:5]:
            actions.append(f"Download {ds.get('name', 'dataset')} dataset")
        for ds in datasets[:3]:
            actions.append(f"Preprocess and split {ds.get('name', 'dataset')} into train/val/test")
        if not datasets:
            actions = _derive_actions_from_items(items, max_actions=8)

    elif stage_type == "verification":
        claims = paper_state.get("expected_claims", [])
        for claim in claims[:8]:
            actions.append(f"Verify: {claim.get('claim_text', '')[:100]}")

    elif stage_type == "inference":
        atomic_steps = paper_state.get("core_method", {}).get("atomic_steps", [])
        for step in atomic_steps:
            desc = step.get("description", "")
            if desc:
                actions.append(f"Implement inference: {desc}")

    elif stage_type == "analysis":
        actions = _derive_actions_from_items(items, max_actions=8)

    # Ensure 3-8 actions; pad or trim
    if len(actions) < 3:
        # Pad with generic actions based on type
        generic: Dict[str, List[str]] = {
            "setup": ["Set up logging and configuration", "Validate environment setup"],
            "training": ["Implement training loop boilerplate", "Add checkpoint saving"],
            "evaluation": ["Aggregate results into summary table", "Save evaluation outputs"],
            "verification": ["Compile verification report", "Cross-check with paper claims"],
            "inference": ["Set up inference pipeline", "Validate inference outputs"],
            "analysis": ["Generate summary statistics", "Create comparison tables"],
            "preprocessing": ["Validate preprocessed data", "Save processed datasets"],
        }
        for g in generic.get(stage_type, []):
            if g not in actions:
                actions.append(g)
            if len(actions) >= 3:
                break

    return actions[:8]


def _split_setup_stages(items: List[dict], paper_state: dict) -> List[dict]:
    """Split setup items into 1-3 stages."""
    # Group by source sub-type
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for item in items:
        src = item.get("source", {}).get("field", "other")
        by_source[src].append(item)

    stages: List[dict] = []
    if len(items) <= 15:
        stages.append({
            "stage_id": "env-setup",
            "name": "Environment Setup",
            "stage_type": "setup",
            "description": "Create repository, install dependencies, download datasets and models",
            "actions": _generate_actions_for_stage("setup", items, paper_state),
            "checklist_items_covered": [it["id"] for it in items],
            "depends_on": [],
            "expected_artifacts": ["repo_skeleton", "requirements.txt", "downloaded_datasets"],
        })
    else:
        # Split into 2-3 substages
        env_items = by_source.get("reproducibility_meta", []) + by_source.get("models", [])
        data_items = by_source.get("datasets", [])
        if env_items:
            stages.append({
                "stage_id": "env-setup",
                "name": "Environment Setup",
                "stage_type": "setup",
                "description": "Create repository, install dependencies, clone code",
                "actions": _generate_actions_for_stage("setup", env_items, paper_state),
                "checklist_items_covered": [it["id"] for it in env_items],
                "depends_on": [],
                "expected_artifacts": ["repo_skeleton", "requirements.txt"],
            })
        if data_items:
            prev = [stages[-1]["stage_id"]] if stages else []
            stages.append({
                "stage_id": "data-prep",
                "name": "Data Preparation",
                "stage_type": "preprocessing",
                "description": "Download and preprocess all required datasets",
                "actions": _generate_actions_for_stage("preprocessing", data_items, paper_state),
                "checklist_items_covered": [it["id"] for it in data_items],
                "depends_on": prev,
                "expected_artifacts": ["downloaded_datasets"],
            })

    return stages


def _split_training_stages(
    items: List[dict], paper_state: dict, has_training: bool
) -> List[dict]:
    """Split training items based on has_training and baselines."""
    has_baselines = bool(paper_state.get("baselines", []))

    if not has_training:
        # Pure inference paper — no training stage; items go to inference-setup
        return [{
            "stage_id": "inference-setup",
            "name": "Inference Setup",
            "stage_type": "inference",
            "description": "Set up inference pipeline (no training required)",
            "actions": _generate_actions_for_stage("inference", items, paper_state),
            "checklist_items_covered": [it["id"] for it in items],
            "depends_on": [],
            "expected_artifacts": ["inference_pipeline", "model_weights"],
        }]

    # Split items by source
    baseline_items = [it for it in items if it.get("source", {}).get("field") == "baselines"]
    core_items = [it for it in items if it not in baseline_items]

    stages: List[dict] = []

    if has_baselines and baseline_items:
        stages.append({
            "stage_id": "baseline-training",
            "name": "Train Baselines",
            "stage_type": "training",
            "description": "Implement and train all baseline methods for comparison",
            "actions": _generate_actions_for_stage("training", baseline_items, paper_state, stage_id="baseline-training"),
            "checklist_items_covered": [it["id"] for it in baseline_items],
            "depends_on": [],
            "expected_artifacts": ["baseline_models", "baseline_checkpoints"],
        })

    if core_items:
        prev = [stages[0]["stage_id"]] if stages else []
        stages.append({
            "stage_id": "main-training",
            "name": "Train Proposed Method",
            "stage_type": "training",
            "description": "Implement and train the proposed method",
            "actions": _generate_actions_for_stage("training", core_items, paper_state, stage_id="main-training"),
            "checklist_items_covered": [it["id"] for it in core_items],
            "depends_on": prev,
            "expected_artifacts": ["trained_model", "training_logs"],
        })

    return stages


def _split_large_stages(stages: List[dict], item_lookup: Dict[str, dict], paper_state: dict) -> List[dict]:
    """Split any stage with >=40 checklist items by derived item type.

    Splits into up to three sub-stages:
      - Plan Writing items       → {stage_id}-plan
      - Code Implementation items → {stage_id}-impl
      - Command Execution items   → {stage_id}-exec

    Each sub-stage inherits the parent's stage_type. Sub-stages with 0 items
    are skipped. Items with types other than the three above stay in the -plan
    stage as a catch-all.

    Stages with stage_type="inference" are skipped — in pure-inference papers
    the bundled inference-setup stage intentionally combines items from multiple
    sources (baselines + atomic_steps + settings) that would otherwise be split
    across training sub-stages.
    """
    TYPE_TO_SUFFIX = {
        "Plan Writing": "plan",
        "Code Implementation": "impl",
        "Command Execution": "exec",
    }
    SUFFIX_ORDER = ["plan", "impl", "exec"]

    result: List[dict] = []
    # Track parent → last sub-stage for depends_on fixup
    parent_to_last: Dict[str, str] = {}

    for stage in stages:
        items = stage.get("checklist_items_covered", [])
        if len(items) < 40 or stage.get("stage_type") == "inference":
            result.append(stage)
            continue

        # Group item IDs by derived type
        by_type: Dict[str, List[str]] = defaultdict(list)
        other_ids: List[str] = []
        for cid in items:
            item = item_lookup.get(cid, {})
            item_type = item.get("type", "")
            suffix = TYPE_TO_SUFFIX.get(item_type)
            if suffix:
                by_type[suffix].append(cid)
            else:
                other_ids.append(cid)

        parent_id = stage["stage_id"]
        parent_type = stage["stage_type"]
        prev_ids: List[str] = []

        for suffix in SUFFIX_ORDER:
            ids = by_type.get(suffix, [])
            # Attach "other" items to the first sub-stage (plan)
            if suffix == "plan":
                ids = ids + other_ids
            if not ids:
                continue

            sub_stage = {
                "stage_id": f"{parent_id}-{suffix}",
                "name": f"{stage['name']} ({suffix.title()})",
                "stage_type": parent_type,
                "description": f"{stage['description']} — {suffix} phase",
                "actions": _generate_actions_for_stage(parent_type, ids, paper_state),
                "checklist_items_covered": ids,
                "depends_on": [prev_ids[-1]] if prev_ids else stage.get("depends_on", []),
                "expected_artifacts": [f"{parent_id}_{suffix}_outputs"],
            }
            result.append(sub_stage)
            prev_ids.append(sub_stage["stage_id"])

        if prev_ids:
            parent_to_last[parent_id] = prev_ids[-1]

    # Fixup: replace stale depends_on references to split parents
    if parent_to_last:
        for s in result:
            s["depends_on"] = [
                parent_to_last.get(d, d) for d in s.get("depends_on", [])
            ]

    return result


def _form_stages(
    buckets: Dict[str, List[dict]],
    fallback_items: List[dict],
    has_training: bool,
    paper_state: dict,
) -> List[dict]:
    """Build the stage list from coarse buckets."""
    stages: List[dict] = []

    # --- setup ---
    setup_items = buckets.get("setup", [])
    if setup_items:
        stages.extend(_split_setup_stages(setup_items, paper_state))

    # --- training ---
    training_items = buckets.get("training", [])
    if training_items:
        stages.extend(_split_training_stages(training_items, paper_state, has_training))

    # --- evaluation ---
    eval_items = buckets.get("evaluation", [])
    if eval_items:
        stages.append({
            "stage_id": "evaluation",
            "name": "Evaluation",
            "stage_type": "evaluation",
            "description": "Run evaluation on all experiments and compute metrics",
            "actions": _generate_actions_for_stage("evaluation", eval_items, paper_state),
            "checklist_items_covered": [it["id"] for it in eval_items],
            "depends_on": [],
            "expected_artifacts": ["evaluation_results", "metric_tables"],
        })

    # --- fallback items → stage-misc (analysis) ---
    if fallback_items:
        stages.append({
            "stage_id": "stage-misc",
            "name": "Additional Analysis",
            "stage_type": "analysis",
            "description": "Handle remaining checklist items not assigned to other stages",
            "actions": _generate_actions_for_stage("analysis", fallback_items, paper_state),
            "checklist_items_covered": [it["id"] for it in fallback_items],
            "depends_on": [],
            "expected_artifacts": ["misc_outputs"],
        })

    # --- verification (always last in stage_type order) ---
    verif_items = buckets.get("verification", [])
    if verif_items:
        stages.append({
            "stage_id": "final-verification",
            "name": "Final Verification",
            "stage_type": "verification",
            "description": "Compare reproduced results against paper claims",
            "actions": _generate_actions_for_stage("verification", verif_items, paper_state),
            "checklist_items_covered": [it["id"] for it in verif_items],
            "depends_on": [],
            "expected_artifacts": ["verification_report"],
        })

    return stages


# ---------------------------------------------------------------------------
# Execution order
# ---------------------------------------------------------------------------

STAGE_TYPE_ORDER: Dict[str, int] = {
    "setup": 0,
    "preprocessing": 1,
    "training": 2,
    "inference": 3,
    "evaluation": 4,
    "analysis": 5,
    "verification": 6,
}


def _compute_execution_order(stages: List[dict]) -> List[str]:
    """Topological sort on depends_on DAG, fallback to stage_type order."""
    # Build adjacency
    stage_ids = {s["stage_id"] for s in stages}
    in_degree: Dict[str, int] = {s["stage_id"]: 0 for s in stages}
    adj: Dict[str, List[str]] = defaultdict(list)

    for s in stages:
        for dep in s.get("depends_on", []):
            if dep in stage_ids:
                adj[dep].append(s["stage_id"])
                in_degree[s["stage_id"]] += 1

    # Kahn's algorithm
    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    # Stable sort: within same in_degree, use stage_type order
    id_to_type = {s["stage_id"]: s["stage_type"] for s in stages}
    queue.sort(key=lambda sid: STAGE_TYPE_ORDER.get(id_to_type.get(sid, ""), 99))

    order: List[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                queue.sort(key=lambda sid: STAGE_TYPE_ORDER.get(id_to_type.get(sid, ""), 99))

    if len(order) != len(stages):
        # Cycle detected — fallback to stage_type order
        sorted_stages = sorted(stages, key=lambda s: STAGE_TYPE_ORDER.get(s["stage_type"], 99))
        order = [s["stage_id"] for s in sorted_stages]

    return order


def _wire_depends_on(stages: List[dict], execution_order: List[str]) -> None:
    """Set linear depends_on chain based on execution_order for rule-based version."""
    id_to_stage = {s["stage_id"]: s for s in stages}
    for i, sid in enumerate(execution_order):
        if i == 0:
            continue
        prev_sid = execution_order[i - 1]
        stage = id_to_stage.get(sid)
        if stage and not stage.get("depends_on"):
            stage["depends_on"] = [prev_sid]


# ---------------------------------------------------------------------------
# Coverage stats
# ---------------------------------------------------------------------------

def _compute_coverage_stats(
    total_items: int,
    po_excluded: int,
    plannable_items: List[dict],
    stages: List[dict],
    fallback_ids: List[str],
) -> dict:
    covered_ids: set[str] = set()
    for s in stages:
        for cid in s.get("checklist_items_covered", []):
            covered_ids.add(cid)

    plannable_ids = {it["id"] for it in plannable_items}
    uncovered = sorted(plannable_ids - covered_ids)
    covered_count = len(covered_ids)
    plannable_count = len(plannable_ids)

    return {
        "total_derived_items": total_items,
        "paper_observation_items_excluded": po_excluded,
        "plannable_items": plannable_count,
        "covered_items": covered_count,
        "coverage_ratio": round(covered_count / plannable_count, 4) if plannable_count > 0 else 0.0,
        "uncovered_item_ids": uncovered,
    }


# ---------------------------------------------------------------------------
# Step 5: LLM refinement
# ---------------------------------------------------------------------------

def _build_current_atomic_step_assignment(
    atomic_steps: List[dict], stages: List[dict], has_training: bool
) -> Dict[str, str]:
    """Build default step_idx -> stage_id mapping.

    Rule-based coarse mapping puts all atomic_step-derived items into
    main-training (or inference-setup for pure inference papers).
    """
    default_stage = ""
    if has_training:
        for s in stages:
            if s["stage_id"] == "main-training":
                default_stage = s["stage_id"]
                break
    else:
        for s in stages:
            if s["stage_id"] == "inference-setup":
                default_stage = s["stage_id"]
                break

    if not default_stage:
        for s in stages:
            if s["stage_type"] in ("training", "inference"):
                default_stage = s["stage_id"]
                break

    return {str(i): default_stage for i in range(len(atomic_steps))}


def _detect_cycle(adj: Dict[str, List[str]]) -> Tuple[bool, List[str]]:
    """Detect if there's a cycle in the DAG via DFS. Returns (has_cycle, involved_stage_ids)."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {node: WHITE for node in adj}
    cycle_nodes: set = set()

    def dfs(node: str, path: List[str]) -> bool:
        color[node] = GRAY
        for neighbor in adj.get(node, []):
            if neighbor not in color:
                continue
            if color[neighbor] == GRAY:
                try:
                    idx = path.index(neighbor)
                    cycle_nodes.update(path[idx:])
                except ValueError:
                    pass
                cycle_nodes.add(neighbor)
                return True
            elif color[neighbor] == WHITE:
                if dfs(neighbor, path + [neighbor]):
                    return True
        color[node] = BLACK
        return False

    for node in adj:
        if color.get(node) == WHITE:
            dfs(node, [node])

    return len(cycle_nodes) > 0, list(cycle_nodes)


def _validate_and_apply_llm_refinement(
    result: dict,
    stages: List[dict],
    original_depends_on: Dict[str, List[str]],
    warnings: List[str],
    *,
    verbose: bool = True,
) -> Tuple[bool, str]:
    """Validate LLM output and apply valid refinements to stages in-place.

    Returns (ok, error_message). On success, error_message is "".
    """
    valid_stage_ids = {s["stage_id"] for s in stages}
    id_to_stage = {s["stage_id"]: s for s in stages}

    refinements = result.get("stage_refinements", [])
    if not isinstance(refinements, list) or len(refinements) == 0:
        return False, "stage_refinements must be a non-empty list"

    # Check coverage: all input stages must appear in LLM output
    llm_stage_ids = set()
    for ref in refinements:
        if not isinstance(ref, dict):
            continue
        sid = ref.get("stage_id", "")
        if sid in valid_stage_ids:
            llm_stage_ids.add(sid)

    missing = valid_stage_ids - llm_stage_ids
    if missing:
        return False, f"Missing stage_refinements for: {sorted(missing)}"

    # Check atomic_step_assignment references valid stage_ids
    atomic_assignment = result.get("atomic_step_assignment", {})
    if isinstance(atomic_assignment, dict):
        for step_idx, sid in atomic_assignment.items():
            if sid and sid not in valid_stage_ids:
                warnings.append(
                    f"atomic_step_assignment[{step_idx}]: unknown stage_id '{sid}' (ignored)"
                )

    # Apply refinements with per-field fallbacks
    actions_reverted = 0
    deps_reverted = 0

    for ref in refinements:
        if not isinstance(ref, dict):
            continue
        sid = ref.get("stage_id", "")
        if sid not in valid_stage_ids:
            warnings.append(f"Ignoring refinement for unknown stage_id '{sid}'")
            continue

        stage = id_to_stage[sid]

        # Actions: validate count 3-8
        actions = ref.get("actions", [])
        if isinstance(actions, list) and 3 <= len(actions) <= 8:
            stage["actions"] = actions
        else:
            actions_reverted += 1
            warnings.append(
                f"actions for '{sid}': expected 3-8, got "
                f"{len(actions) if isinstance(actions, list) else type(actions).__name__}; "
                f"keeping rule-based"
            )

        # depends_on: validate stage_ids exist
        deps = ref.get("depends_on", [])
        if isinstance(deps, list):
            valid_deps = [d for d in deps if d in valid_stage_ids]
            bad_deps = [d for d in deps if d not in valid_stage_ids]
            if bad_deps:
                warnings.append(
                    f"depends_on for '{sid}' references unknown ids (discarded): {bad_deps}"
                )
            stage["depends_on"] = valid_deps
        else:
            deps_reverted += 1
            warnings.append(
                f"depends_on for '{sid}' is not a list; keeping rule-based"
            )

    # Cycle detection on the full DAG
    stage_adj = {s["stage_id"]: s.get("depends_on", []) for s in stages}
    has_cycle, cycle_sids = _detect_cycle(stage_adj)
    if has_cycle:
        for sid in cycle_sids:
            if sid in id_to_stage:
                id_to_stage[sid]["depends_on"] = original_depends_on.get(sid, [])
        warnings.append(
            f"Cycle detected involving {sorted(cycle_sids)}; "
            f"reverted depends_on to rule-based for affected stages"
        )
        if verbose:
            print(f"    ⚠ Cycle detected: {sorted(cycle_sids)}")

    if verbose:
        if actions_reverted:
            print(f"    ⚠ {actions_reverted} stage(s) actions reverted to rule-based")
        if deps_reverted:
            print(f"    ⚠ {deps_reverted} stage(s) depends_on reverted to rule-based")

    return True, ""


def _save_debug_response(
    paper_id: str,
    raw_response: dict,
    usage: dict,
    warnings: List[str],
) -> None:
    """Save raw LLM response and metadata to _debug/ before any validation discards."""
    debug_dir = OUTPUTS_DIR / paper_id / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / "llm_response_step5.json"
    debug_data = {
        "paper_id": paper_id,
        "model": _model_name(),
        "temperature": 0.3,
        "max_tokens": 8000,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "raw_response": raw_response,
        "warnings": warnings,
    }
    debug_path.write_text(
        json.dumps(debug_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _llm_refine_stages(
    stages: List[dict],
    paper_state: dict,
    paper_id: str,
    has_training: bool,
    *,
    verbose: bool = True,
) -> Tuple[List[dict], List[str], List[dict]]:
    """Call LLM to refine actions, atomic_step_assignment, and depends_on.

    Returns (refined_stages, warnings, usage_log).
    Modifies stages in-place.
    """
    atomic_steps = paper_state.get("core_method", {}).get("atomic_steps", [])
    if not atomic_steps:
        return stages, [], []

    # Save originals for fallback
    original_depends_on = {s["stage_id"]: list(s.get("depends_on", [])) for s in stages}

    current_assignment = _build_current_atomic_step_assignment(
        atomic_steps, stages, has_training
    )

    # Build compact LLM input (summary only, no full checklist)
    llm_input = {
        "paper_id": paper_id,
        "stages": [
            {
                "stage_id": s["stage_id"],
                "name": s["name"],
                "stage_type": s["stage_type"],
                "description": s["description"],
                "items_count": len(s["checklist_items_covered"]),
                "current_actions": s["actions"],
                "current_depends_on": s["depends_on"],
            }
            for s in stages
        ],
        "atomic_steps": [
            {"index": i, "description": step.get("description", "")}
            for i, step in enumerate(atomic_steps)
        ],
        "current_atomic_step_assignment": current_assignment,
    }

    # Render prompt
    prompt_template = (PROMPT_DIR / "stage_refinement.txt").read_text(encoding="utf-8")
    prompt = prompt_template.replace(
        "{LLM_INPUT_JSON}", json.dumps(llm_input, indent=2, ensure_ascii=False)
    )

    if verbose:
        print(f"  [LLM refine] prompt_chars={len(prompt):,}")

    # Call LLM with retries
    client = _make_client()
    warnings: List[str] = []
    usage_log: List[dict] = []

    MAX_RETRIES = 3
    last_raw = ""
    last_error: Optional[str] = None
    last_parsed_result: Optional[dict] = None
    last_usage: dict = {}

    for attempt in range(1, MAX_RETRIES + 1):
        if verbose:
            print(f"  [LLM refine] attempt {attempt}/{MAX_RETRIES}...")

        if last_error and attempt > 1:
            retry_prompt = (
                "Your previous response had errors. "
                "Below is your previous JSON output and the errors. "
                "Output a CORRECTED full JSON object — keep all parts that "
                "were correct, fix only the errors. Output ONLY the JSON.\n\n"
                "=== Your previous output ===\n"
                f"{last_raw}\n\n"
                "=== Errors ===\n"
                f"{last_error}\n\n"
                "Now output the corrected full JSON:"
            )
            text, usage = _call_llm(client, retry_prompt, temperature=0.3, max_tokens=8000)
        else:
            text, usage = _call_llm(client, prompt, temperature=0.3, max_tokens=8000)

        usage_log.append({**usage, "attempt": attempt})
        last_raw = text
        last_usage = usage

        if verbose:
            print(f"    → {usage['input_tokens']} in / {usage['output_tokens']} out tokens")

        # Parse JSON
        try:
            result = _parse_json_loose(text)
        except json.JSONDecodeError as e:
            last_error = f"Failed to parse JSON: {e}\n(first 300 chars): {text[:300]}"
            if verbose:
                print(f"    ✗ JSON parse failed")
            continue

        last_parsed_result = result

        # Save raw response to _debug/ before validation (preserves LLM output uncensored)
        _save_debug_response(paper_id, result, usage, list(warnings))

        # Validate and apply
        ok, err_msg = _validate_and_apply_llm_refinement(
            result, stages, original_depends_on, warnings, verbose=verbose
        )
        if ok:
            if verbose:
                print(f"    ✓ refinement valid on attempt {attempt}")
            break
        else:
            last_error = err_msg
            if verbose:
                print(f"    ✗ {err_msg[:120]}")
            continue
    else:
        # All retries exhausted — save debug with last parsed result if available
        if last_parsed_result is not None:
            _save_debug_response(paper_id, last_parsed_result, last_usage, list(warnings))
        warnings.append(
            f"LLM refinement failed after {MAX_RETRIES} attempts; "
            f"keeping rule-based stages. Last error: {last_error}"
        )
        if verbose:
            print(f"  ⚠ All retries exhausted; keeping rule-based stages")

    # Update debug file with final warnings after validation
    if last_parsed_result is not None:
        _save_debug_response(paper_id, last_parsed_result, last_usage, list(warnings))

    return stages, warnings, usage_log


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def _validate_task_plan(task_plan: dict) -> Optional[str]:
    """Validate task_plan against schema. Return None if OK, else error message."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(task_plan), key=lambda e: list(e.path))
    if not errors:
        return None
    msgs = []
    for e in errors[:5]:
        path = "/".join(str(p) for p in e.path) or "<root>"
        msgs.append(f"  - at '{path}': {e.message}")
    if len(errors) > 5:
        msgs.append(f"  ... and {len(errors) - 5} more")
    return "Schema validation failed:\n" + "\n".join(msgs)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plan(paper_id: str) -> dict:
    """Run Steps 1-5 (rule-based + LLM refinement) and return a task_plan dict."""
    paper_dir = OUTPUTS_DIR / paper_id
    paper_state_path = paper_dir / "paper_state.json"
    checklist_path = paper_dir / "derived_checklist.json"

    if not paper_state_path.exists():
        raise FileNotFoundError(f"paper_state not found: {paper_state_path}")
    if not checklist_path.exists():
        raise FileNotFoundError(f"derived_checklist not found: {checklist_path}")

    with open(paper_state_path) as f:
        paper_state = json.load(f)
    with open(checklist_path) as f:
        checklist = json.load(f)

    # Step 1: filter Paper Observation items
    po_items, plannable_items = filter_paper_observations(checklist)

    # Step 2: coarse mapping by source.field
    buckets, fallback_items = coarse_map(plannable_items)

    # Step 3: detect training
    has_training = detect_has_training(paper_state)

    # Step 4: form stages
    stages = _form_stages(buckets, fallback_items, has_training, paper_state)

    # Step 4b: split large stages (>=40 items) by derived item type
    item_lookup: Dict[str, dict] = {it["id"]: it for it in checklist}
    stages = _split_large_stages(stages, item_lookup, paper_state)

    # Coverage stats (rule-based, computed before LLM; never recomputed)
    fallback_ids = [it["id"] for it in fallback_items]
    coverage_stats = _compute_coverage_stats(
        len(checklist), len(po_items), plannable_items, stages, fallback_ids
    )

    # Step 5: LLM refinement (actions, atomic_step_assignment, depends_on)
    stages, warnings, usage_log = _llm_refine_stages(
        stages, paper_state, paper_id, has_training, verbose=True
    )

    # Execution order (recomputed after LLM may have changed depends_on)
    execution_order = _compute_execution_order(stages)
    _wire_depends_on(stages, execution_order)

    result = {
        "paper_id": paper_id,
        "stages": stages,
        "execution_order": execution_order,
        "coverage_stats": coverage_stats,
    }

    # Print warnings collected during refinement
    for w in warnings:
        print(f"  ⚠ {w}")

    # Print token usage summary
    if usage_log:
        total_in = sum(u["input_tokens"] for u in usage_log)
        total_out = sum(u["output_tokens"] for u in usage_log)
        attempts = len(usage_log)
        print(f"  [LLM refine] total tokens: {total_in:,} in / {total_out:,} out ({attempts} attempt(s))")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python planner.py <paper_id> [paper_id ...]", file=sys.stderr)
        sys.exit(1)

    for paper_id in sys.argv[1:]:
        try:
            task_plan = plan(paper_id)
        except FileNotFoundError as e:
            print(f"[SKIP] {paper_id}: {e}", file=sys.stderr)
            continue

        output_path = OUTPUTS_DIR / paper_id / "task_plan.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(task_plan, f, indent=2, ensure_ascii=False)

        cs = task_plan["coverage_stats"]
        print(
            f"[{paper_id}] {len(task_plan['stages'])} stages, "
            f"coverage={cs['coverage_ratio']:.1%} "
            f"({cs['covered_items']}/{cs['plannable_items']} plannable, "
            f"{cs['paper_observation_items_excluded']} PO excluded)"
        )

        # Schema validation
        schema_err = _validate_task_plan(task_plan)
        if schema_err:
            print(f"[{paper_id}] ✗ schema: {schema_err}", file=sys.stderr)
        else:
            print(f"[{paper_id}] ✓ schema valid")


if __name__ == "__main__":
    main()
