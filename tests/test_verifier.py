"""
Tests for Verifier module (Phase 3.1).

Tests 1-2 validate schema conformance of generated artifacts.
Tests 3-5 are unit tests using hand-crafted fixtures — no real LLM calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from jsonschema import Draft7Validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.verifier.verifier import (
    compute_overall_score,
    find_exp_by_evidence,
    is_placeholder,
    load_paper_state,
    load_results,
    verify,
)

# ---------------------------------------------------------------------------
# Shared paths
# ---------------------------------------------------------------------------

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
RESULTS_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "pipeline" / "schemas" / "results.schema.json"
)
REPORT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "pipeline"
    / "schemas"
    / "verification_report.schema.json"
)

ALL_PAPERS = ["AMUN", "Beyond-Ngram", "I0T", "INCLINE", "min-p"]


def _load_results_schema() -> dict:
    return json.loads(RESULTS_SCHEMA_PATH.read_text(encoding="utf-8"))


def _load_report_schema() -> dict:
    return json.loads(REPORT_SCHEMA_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Schema validation: mock_results.json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paper_id", ALL_PAPERS)
def test_schema_validation_results(paper_id: str):
    """Every placeholder mock_results.json must pass results.schema.json."""
    path = OUTPUTS_DIR / paper_id / "mock_results.json"
    if not path.exists():
        pytest.skip(f"mock_results.json not found for {paper_id}")

    results = json.loads(path.read_text(encoding="utf-8"))
    schema = _load_results_schema()
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(results), key=lambda e: list(e.path))

    for e in errors:
        path_str = "/".join(str(p) for p in e.path) or "<root>"
        print(f"  ✗ at '{path_str}': {e.message}")

    assert len(errors) == 0, (
        f"{paper_id} mock_results.json has {len(errors)} schema errors"
    )


# ---------------------------------------------------------------------------
# 2. Schema validation: verification_report.json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paper_id", ALL_PAPERS)
def test_schema_validation_report(paper_id: str):
    """Every verification_report.json must pass verification_report.schema.json."""
    path = OUTPUTS_DIR / paper_id / "verification_report.json"
    if not path.exists():
        pytest.skip(f"verification_report.json not found for {paper_id}")

    report = json.loads(path.read_text(encoding="utf-8"))
    schema = _load_report_schema()
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(report), key=lambda e: list(e.path))

    for e in errors:
        path_str = "/".join(str(p) for p in e.path) or "<root>"
        print(f"  ✗ at '{path_str}': {e.message}")

    assert len(errors) == 0, (
        f"{paper_id} verification_report.json has {len(errors)} schema errors"
    )


# ---------------------------------------------------------------------------
# 3. Placeholder detection → skipped
# ---------------------------------------------------------------------------


# Hand-crafted paper_state fixture for test 3
HANDCRAFTED_PAPER_STATE = {
    "paper_id": "TEST",
    "title": "Test Paper for Verifier Unit Tests",
    "main_experiments": [
        {
            "id": "exp-table1",
            "evidence_in_paper": "table-1",
            "primary": True,
            "claim": "Method X achieves best accuracy in Table 1",
            "axes": {"rows": ["MethodX", "Baseline"], "cols": ["Accuracy"]},
        },
        {
            "id": "exp-figure1",
            "evidence_in_paper": "figure-1",
            "primary": False,
            "claim": "Adv curves maintain test accuracy above 80%",
            "axes": {"rows": ["Curve1"], "cols": ["Test Acc"]},
        },
    ],
    "expected_claims": [
        {
            "id": "claim-t1",
            "claim_text": "Method X achieves the best accuracy in Table 1",
            "evidence": "table-1",
            "verification_hint": "Compare MethodX vs Baseline rows",
        },
        {
            "id": "claim-f1",
            "claim_text": "Adv curves maintain test accuracy above 80%",
            "evidence": "figure-1",
            "verification_hint": "Check test accuracy throughout training",
        },
    ],
}


def test_placeholder_skipped():
    """Placeholder mock input → all claim verdicts should be 'skipped'."""
    # Build placeholder mock results matching the handcrafted state
    mock_results = {
        "paper_id": "TEST",
        "experiments": {
            "exp-table1": {
                "evidence_in_paper": "table-1",
                "axes": {"rows": ["MethodX", "Baseline"], "cols": ["Accuracy"]},
                "values": [[None], [None]],
            },
            "exp-figure1": {
                "evidence_in_paper": "figure-1",
                "summary": "[PLACEHOLDER - 待手工填充]",
            },
        },
    }

    with (
        patch(
            "pipeline.verifier.verifier.load_paper_state",
            return_value=HANDCRAFTED_PAPER_STATE,
        ),
        patch(
            "pipeline.verifier.verifier.load_results",
            return_value=mock_results,
        ),
    ):
        # Pass a MagicMock client — it should NEVER be called because
        # all results are placeholders and get skipped before LLM path.
        mock_client = MagicMock()
        report = verify("TEST", client=mock_client, verbose=False)

        # Assert no LLM calls were made
        mock_client.chat.completions.create.assert_not_called()

        # All claims should be skipped
        assert len(report["claim_results"]) == 2
        for cr in report["claim_results"]:
            assert cr["verdict"] == "skipped", (
                f"Expected skipped, got {cr['verdict']} for {cr['claim_id']}"
            )
            assert cr["confidence"] == 0.0
            assert "placeholder" in cr["reasoning"].lower()

        # overall_score should be null when all skipped
        assert report["overall_score"] is None


# ---------------------------------------------------------------------------
# 4. overall_score arithmetic
# ---------------------------------------------------------------------------


def test_overall_score_arithmetic():
    """overall_score should be mean of pass=1.0, partial=0.5, fail=0.0, skipping excludes."""
    # All pass
    all_pass = [
        {"claim_id": "c1", "verdict": "pass"},
        {"claim_id": "c2", "verdict": "pass"},
        {"claim_id": "c3", "verdict": "pass"},
    ]
    assert compute_overall_score(all_pass) == 1.0

    # All fail
    all_fail = [
        {"claim_id": "c1", "verdict": "fail"},
        {"claim_id": "c2", "verdict": "fail"},
    ]
    assert compute_overall_score(all_fail) == 0.0

    # Mixed
    mixed = [
        {"claim_id": "c1", "verdict": "pass"},     # 1.0
        {"claim_id": "c2", "verdict": "partial"},  # 0.5
        {"claim_id": "c3", "verdict": "fail"},     # 0.0
        {"claim_id": "c4", "verdict": "pass"},     # 1.0
    ]
    # Mean = (1.0 + 0.5 + 0.0 + 1.0) / 4 = 2.5/4 = 0.625
    assert compute_overall_score(mixed) == pytest.approx(0.625)

    # With skipped (should be excluded)
    with_skipped = [
        {"claim_id": "c1", "verdict": "pass"},     # 1.0
        {"claim_id": "c2", "verdict": "skipped"},  # excluded
        {"claim_id": "c3", "verdict": "fail"},     # 0.0
    ]
    # Mean = (1.0 + 0.0) / 2 = 0.5
    assert compute_overall_score(with_skipped) == pytest.approx(0.5)

    # All skipped → null
    all_skipped = [
        {"claim_id": "c1", "verdict": "skipped"},
        {"claim_id": "c2", "verdict": "skipped"},
    ]
    assert compute_overall_score(all_skipped) is None

    # Empty list → null
    assert compute_overall_score([]) is None


# ---------------------------------------------------------------------------
# 5. main_experiment lookup
# ---------------------------------------------------------------------------


def test_main_experiment_lookup():
    """claim.evidence → main_experiment.id lookup logic."""
    main_experiments = [
        {
            "id": "exp-table1",
            "evidence_in_paper": "table-1",
            "claim": "Method X achieves the best accuracy in Table 1",
        },
        {
            "id": "exp-figure2",
            "evidence_in_paper": "figure-2",
            "claim": "Continuous unlearning works",
        },
        {
            "id": "exp-table3",
            "evidence_in_paper": "table-3",
            "claim": "Extended settings comparison",
        },
    ]

    # Exact match
    exp = find_exp_by_evidence(main_experiments, "table-1")
    assert exp is not None
    assert exp["id"] == "exp-table1"

    exp = find_exp_by_evidence(main_experiments, "figure-2")
    assert exp is not None
    assert exp["id"] == "exp-figure2"

    # No match
    exp = find_exp_by_evidence(main_experiments, "table-99")
    assert exp is None

    # Empty list
    exp = find_exp_by_evidence([], "table-1")
    assert exp is None


# ---------------------------------------------------------------------------
# Bonus: is_placeholder function
# ---------------------------------------------------------------------------


def test_is_placeholder():
    """is_placeholder should correctly detect null values and placeholder strings."""
    # None → placeholder
    assert is_placeholder(None) is True

    # Table with null values → placeholder
    table_null = {
        "evidence_in_paper": "table-1",
        "axes": {"rows": ["A", "B"], "cols": ["X"]},
        "values": [[None], [None]],
    }
    assert is_placeholder(table_null) is True

    # Table with all real values → not placeholder
    table_real = {
        "evidence_in_paper": "table-1",
        "axes": {"rows": ["A", "B"], "cols": ["X"]},
        "values": [[0.95], [0.87]],
    }
    assert is_placeholder(table_real) is False

    # Table with one null among real values → placeholder
    table_mixed = {
        "evidence_in_paper": "table-1",
        "axes": {"rows": ["A", "B"], "cols": ["X"]},
        "values": [[None], [0.87]],
    }
    assert is_placeholder(table_mixed) is True

    # Figure with placeholder summary → placeholder
    fig_placeholder = {
        "evidence_in_paper": "figure-1",
        "summary": "[PLACEHOLDER - 待手工填充]",
    }
    assert is_placeholder(fig_placeholder) is True

    # Figure with real summary → not placeholder
    fig_real = {
        "evidence_in_paper": "figure-1",
        "summary": "Test accuracy remains above 80% throughout training",
    }
    assert is_placeholder(fig_real) is False

    # Empty values array → placeholder
    empty_values = {
        "evidence_in_paper": "table-1",
        "axes": {"rows": [], "cols": []},
        "values": [],
    }
    assert is_placeholder(empty_values) is True
