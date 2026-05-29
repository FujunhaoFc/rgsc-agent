"""
Tests for stage_planner (Steps 1-5): schema validation, DAG invariants,
coverage arithmetic, stage-type invariants.

Fixtures are the 5 paper task_plan.json files already produced by the planner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Load schemas and paper list
# ---------------------------------------------------------------------------

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "schemas" / "task_plan.schema.json"
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"

ALL_PAPERS = ["AMUN", "Beyond-Ngram", "I0T", "INCLINE", "min-p"]

STAGE_TYPE_ENUM = {
    "setup", "preprocessing", "training", "inference",
    "evaluation", "analysis", "verification",
}


def _load_task_plan(paper_id: str) -> dict:
    path = OUTPUTS_DIR / paper_id / "task_plan.json"
    if not path.exists():
        pytest.skip(f"task_plan.json not found for {paper_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paper_id", ALL_PAPERS)
def test_schema_validation(paper_id: str) -> None:
    """Every paper's task_plan.json must pass task_plan.schema.json validation."""
    task_plan = _load_task_plan(paper_id)
    schema = _load_schema()
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(task_plan), key=lambda e: list(e.path))
    assert not errors, (
        f"{paper_id}: schema validation failed — "
        + "; ".join(f"{'/'.join(str(p) for p in e.path)}: {e.message}" for e in errors[:5])
    )


# ---------------------------------------------------------------------------
# 2. stage_type enum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paper_id", ALL_PAPERS)
def test_stage_type_enum(paper_id: str) -> None:
    """Every stage_type must be one of the 7 allowed values."""
    task_plan = _load_task_plan(paper_id)
    for s in task_plan["stages"]:
        st = s["stage_type"]
        assert st in STAGE_TYPE_ENUM, (
            f"{paper_id}: stage '{s['stage_id']}' has invalid stage_type '{st}'"
        )


# ---------------------------------------------------------------------------
# 3. depends_on acyclic
# ---------------------------------------------------------------------------


def _topological_sort(stages: list[dict]) -> list[str] | None:
    """Return topological order, or None if cycle detected."""
    stage_ids = {s["stage_id"] for s in stages}
    in_degree: dict[str, int] = {s["stage_id"]: 0 for s in stages}
    adj: dict[str, list[str]] = {s["stage_id"]: [] for s in stages}

    for s in stages:
        for dep in s.get("depends_on", []):
            if dep in stage_ids:
                adj[dep].append(s["stage_id"])
                in_degree[s["stage_id"]] += 1

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(stages):
        return None
    return order


@pytest.mark.parametrize("paper_id", ALL_PAPERS)
def test_depends_on_acyclic(paper_id: str) -> None:
    """depends_on graph must be a DAG (no cycles)."""
    task_plan = _load_task_plan(paper_id)
    order = _topological_sort(task_plan["stages"])
    assert order is not None, (
        f"{paper_id}: depends_on contains a cycle — topological sort failed"
    )


# ---------------------------------------------------------------------------
# 4. execution_order consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paper_id", ALL_PAPERS)
def test_execution_order_consistency(paper_id: str) -> None:
    """execution_order must be a 1:1 bijection with stages, and topologically valid."""
    task_plan = _load_task_plan(paper_id)
    stage_ids = {s["stage_id"] for s in task_plan["stages"]}
    exec_order = task_plan["execution_order"]

    # 1:1 correspondence
    assert len(exec_order) == len(stage_ids), (
        f"{paper_id}: execution_order length {len(exec_order)} != stages {len(stage_ids)}"
    )
    assert len(set(exec_order)) == len(exec_order), (
        f"{paper_id}: execution_order has duplicates"
    )
    assert set(exec_order) == stage_ids, (
        f"{paper_id}: execution_order set does not match stage_ids set"
    )

    # Topological compatibility: every dep must appear before its dependent
    pos = {sid: i for i, sid in enumerate(exec_order)}
    for s in task_plan["stages"]:
        for dep in s.get("depends_on", []):
            if dep in stage_ids:
                assert pos[dep] < pos[s["stage_id"]], (
                    f"{paper_id}: dependency violation — '{dep}' appears after "
                    f"'{s['stage_id']}' in execution_order"
                )


# ---------------------------------------------------------------------------
# 5. actions count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paper_id", ALL_PAPERS)
def test_actions_count(paper_id: str) -> None:
    """Every stage must have 3–8 actions."""
    task_plan = _load_task_plan(paper_id)
    for s in task_plan["stages"]:
        n = len(s["actions"])
        assert 3 <= n <= 8, (
            f"{paper_id}: stage '{s['stage_id']}' has {n} actions, expected 3–8"
        )


# ---------------------------------------------------------------------------
# 6. Paper Observation items excluded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paper_id", ALL_PAPERS)
def test_paper_obs_excluded(paper_id: str) -> None:
    """No checklist item with type 'Paper Observation' may appear in any stage's checklist."""
    task_plan = _load_task_plan(paper_id)
    # Paper Obs items start with "derived-po-"
    for s in task_plan["stages"]:
        for cid in s["checklist_items_covered"]:
            assert not cid.startswith("derived-po-"), (
                f"{paper_id}: stage '{s['stage_id']}' contains Paper Obs item '{cid}'"
            )


# ---------------------------------------------------------------------------
# 7. min-p has no training stage
# ---------------------------------------------------------------------------


def test_min_p_no_training_stage() -> None:
    """min-p is a pure inference paper — must not have any training stage."""
    task_plan = _load_task_plan("min-p")
    training_stages = [s for s in task_plan["stages"] if s["stage_type"] == "training"]
    assert len(training_stages) == 0, (
        f"min-p should have 0 training stages, found {len(training_stages)}: "
        f"{[s['stage_id'] for s in training_stages]}"
    )


# ---------------------------------------------------------------------------
# 8. coverage arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paper_id", ALL_PAPERS)
def test_coverage_arithmetic(paper_id: str) -> None:
    """Three arithmetic invariants must hold in coverage_stats."""
    task_plan = _load_task_plan(paper_id)
    cs = task_plan["coverage_stats"]

    # covered + uncovered == plannable
    assert cs["covered_items"] + len(cs["uncovered_item_ids"]) == cs["plannable_items"], (
        f"{paper_id}: {cs['covered_items']} + {len(cs['uncovered_item_ids'])} "
        f"!= {cs['plannable_items']}"
    )

    # po_excluded + plannable == total
    assert cs["paper_observation_items_excluded"] + cs["plannable_items"] == cs["total_derived_items"], (
        f"{paper_id}: {cs['paper_observation_items_excluded']} + {cs['plannable_items']} "
        f"!= {cs['total_derived_items']}"
    )

    # ratio == covered / plannable
    if cs["plannable_items"] > 0:
        expected_ratio = cs["covered_items"] / cs["plannable_items"]
        assert abs(cs["coverage_ratio"] - expected_ratio) < 1e-9, (
            f"{paper_id}: coverage_ratio {cs['coverage_ratio']} != "
            f"{cs['covered_items']} / {cs['plannable_items']} = {expected_ratio}"
        )


# ---------------------------------------------------------------------------
# 9. depends_on references exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paper_id", ALL_PAPERS)
def test_stage_id_in_depends_on_exists(paper_id: str) -> None:
    """Every stage_id referenced in depends_on must exist in the stages list."""
    task_plan = _load_task_plan(paper_id)
    stage_ids = {s["stage_id"] for s in task_plan["stages"]}
    for s in task_plan["stages"]:
        for dep in s.get("depends_on", []):
            assert dep in stage_ids, (
                f"{paper_id}: stage '{s['stage_id']}' depends_on unknown '{dep}'"
            )


# ---------------------------------------------------------------------------
# 10. verification stages are last in execution_order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paper_id", ALL_PAPERS)
def test_verification_last(paper_id: str) -> None:
    """All verification-type stages must appear at the end of execution_order."""
    task_plan = _load_task_plan(paper_id)
    exec_order = task_plan["execution_order"]
    stage_by_id = {s["stage_id"]: s for s in task_plan["stages"]}

    verif_ids = [
        sid for sid in exec_order
        if stage_by_id.get(sid, {}).get("stage_type") == "verification"
    ]
    non_verif_ids = [
        sid for sid in exec_order
        if stage_by_id.get(sid, {}).get("stage_type") != "verification"
    ]

    if verif_ids and non_verif_ids:
        last_non_verif_idx = max(exec_order.index(sid) for sid in non_verif_ids)
        first_verif_idx = min(exec_order.index(sid) for sid in verif_ids)
        assert last_non_verif_idx < first_verif_idx, (
            f"{paper_id}: verification stage(s) {verif_ids} are not last in execution_order"
        )
