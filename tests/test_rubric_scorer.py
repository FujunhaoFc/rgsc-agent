"""Tests for pipeline.rubric_scorer — Card 2.

All tests use mock LLM (judge_rubric returns fixed "hit" verdict).
No real API calls. Uses monkeypatch to override judge_rubric where needed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.rubric_scorer.scorer import (
    score,
    judge_rubric,
    compute_earned_score,
    aggregate,
    _validate_report,
)
from pipeline.rubric_scorer.retrieval import (
    filter_actions,
    _filter_by_type,
    _rank_by_keyword_overlap,
    _truncate,
    _extract_keywords,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_rubric(
    idx: int,
    criteria: str = "test criteria",
    rtype: str = "Code Implementation",
    score: float = 4,
) -> dict:
    return {
        "criteria": criteria,
        "score": score,
        "type": rtype,
        "comment": None,
        "rubric_idx": idx,
    }


def _make_judgement(
    verdict: str = "hit",
    confidence: float = 0.95,
    reasoning: str = "mock judgement",
    evidence: str = "mock evidence",
) -> dict:
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Test 1: compute_earned_score
# ---------------------------------------------------------------------------


class TestComputeEarnedScore:
    def test_hit_returns_full_score(self):
        assert compute_earned_score("hit", 4) == 4.0
        assert compute_earned_score("hit", 2) == 2.0
        assert compute_earned_score("hit", 1.5) == 1.5

    def test_partial_returns_half_score(self):
        assert compute_earned_score("partial", 4) == 2.0
        assert compute_earned_score("partial", 3) == 1.5
        assert compute_earned_score("partial", 1) == 0.5

    def test_miss_returns_zero(self):
        assert compute_earned_score("miss", 4) == 0.0
        assert compute_earned_score("miss", 100) == 0.0

    def test_unknown_verdict_raises(self):
        with pytest.raises(ValueError, match="Unknown verdict"):
            compute_earned_score("invalid", 4)


# ---------------------------------------------------------------------------
# Test 2: aggregate by_type breakdown
# ---------------------------------------------------------------------------


class TestAggregateByType:
    def test_by_type_breakdown(self):
        """Verify by_type totals and estimated scores aggregate correctly."""
        rubrics = [
            _make_rubric(0, "read sec 2", "Paper Observation", 2),
            _make_rubric(1, "read sec 3", "Paper Observation", 1),
            _make_rubric(2, "plan pgd", "Plan Writing", 3),
            _make_rubric(3, "code pgd", "Code Implementation", 4),
            _make_rubric(4, "run train", "Command Execution", 3),
            _make_rubric(5, "check result", "Result Matching", 2),
        ]
        # All hit → earned_score = rubric_score for each
        judgements = [
            _make_judgement("hit"),
            _make_judgement("hit"),
            _make_judgement("hit"),
            _make_judgement("hit"),
            _make_judgement("hit"),
            _make_judgement("hit"),
        ]

        report = aggregate("test", rubrics, judgements)

        assert report["paper_id"] == "test"
        assert report["total_score"] == 15.0  # 2+1+3+4+3+2
        assert report["estimated_score"] == 15.0
        assert report["estimated_recall"] == 1.0

        by_type = report["by_type"]
        assert by_type["Paper Observation"]["total"] == 3.0
        assert by_type["Paper Observation"]["estimated"] == 3.0
        assert by_type["Paper Observation"]["rate"] == 1.0

        assert by_type["Plan Writing"]["total"] == 3.0
        assert by_type["Code Implementation"]["total"] == 4.0
        assert by_type["Command Execution"]["total"] == 3.0
        assert by_type["Result Matching"]["total"] == 2.0

    def test_by_type_mixed_verdicts(self):
        """Verify partial/miss verdicts affect by_type correctly."""
        rubrics = [
            _make_rubric(0, "a", "Paper Observation", 2),
            _make_rubric(1, "b", "Paper Observation", 2),
            _make_rubric(2, "c", "Code Implementation", 4),
            _make_rubric(3, "d", "Code Implementation", 4),
        ]
        judgements = [
            _make_judgement("hit"),       # earned=2
            _make_judgement("partial"),   # earned=1
            _make_judgement("hit"),       # earned=4
            _make_judgement("miss"),      # earned=0
        ]

        report = aggregate("test", rubrics, judgements)

        assert report["total_score"] == 12.0
        assert report["estimated_score"] == 7.0  # 2+1+4+0
        assert report["estimated_recall"] == 7.0 / 12.0

        po = report["by_type"]["Paper Observation"]
        assert po["total"] == 4.0
        assert po["estimated"] == 3.0  # 2 + 1
        assert po["rate"] == 0.75

        ci = report["by_type"]["Code Implementation"]
        assert ci["total"] == 8.0
        assert ci["estimated"] == 4.0
        assert ci["rate"] == 0.5


# ---------------------------------------------------------------------------
# Test 3: estimated_recall formula
# ---------------------------------------------------------------------------


class TestEstimatedRecall:
    def test_all_hit_gives_recall_1(self):
        rubrics = [_make_rubric(i, score=3) for i in range(5)]
        judgements = [_make_judgement("hit") for _ in range(5)]
        report = aggregate("test", rubrics, judgements)
        assert report["estimated_recall"] == 1.0

    def test_all_miss_gives_recall_0(self):
        rubrics = [_make_rubric(i, score=3) for i in range(5)]
        judgements = [_make_judgement("miss") for _ in range(5)]
        report = aggregate("test", rubrics, judgements)
        assert report["estimated_recall"] == 0.0
        assert report["estimated_score"] == 0.0

    def test_half_hit_gives_recall_0_5(self):
        rubrics = [_make_rubric(0, score=2), _make_rubric(1, score=2)]
        judgements = [_make_judgement("hit"), _make_judgement("miss")]
        report = aggregate("test", rubrics, judgements)
        assert report["estimated_recall"] == 0.5

    def test_zero_rubrics_recall_is_zero(self):
        report = aggregate("test", [], [])
        assert report["estimated_recall"] == 0.0
        assert report["total_score"] == 0.0


# ---------------------------------------------------------------------------
# Test 4: Load rubrics for 5 papers, verify total_score
# ---------------------------------------------------------------------------


PAPER_EXPECTED_TOTALS = {
    "AMUN": 228,
    "Beyond-Ngram": 169,
    "I0T": 257,
    "INCLINE": 202,
    "min-p": 272,
}


class TestLoadRubrics:
    @pytest.mark.parametrize("paper_id", list(PAPER_EXPECTED_TOTALS))
    def test_rubrics_load_and_total(self, paper_id):
        """Each paper's rubrics.json loads and has correct total_score."""
        rubrics_path = PROJECT_ROOT / "data" / "train_valid" / paper_id / "rubrics.json"
        assert rubrics_path.exists(), f"Missing rubrics: {rubrics_path}"

        rubrics = json.loads(rubrics_path.read_text(encoding="utf-8"))
        total = sum(r["score"] for r in rubrics)
        expected = PAPER_EXPECTED_TOTALS[paper_id]

        assert total == expected, (
            f"{paper_id}: total_score={total}, expected={expected}"
        )

    @pytest.mark.parametrize("paper_id", list(PAPER_EXPECTED_TOTALS))
    def test_rubrics_have_valid_types(self, paper_id):
        """All rubric items have one of the 5 valid types."""
        valid_types = {
            "Paper Observation",
            "Plan Writing",
            "Code Implementation",
            "Command Execution",
            "Result Matching",
        }
        rubrics_path = PROJECT_ROOT / "data" / "train_valid" / paper_id / "rubrics.json"
        rubrics = json.loads(rubrics_path.read_text(encoding="utf-8"))

        for i, r in enumerate(rubrics):
            assert r.get("type") in valid_types, (
                f"{paper_id}[{i}]: invalid type '{r.get('type')}'"
            )
            assert isinstance(r.get("criteria"), str), f"{paper_id}[{i}]: no criteria"
            assert isinstance(r.get("score"), (int, float)), f"{paper_id}[{i}]: no score"
            assert r["score"] > 0, f"{paper_id}[{i}]: score <= 0"


# ---------------------------------------------------------------------------
# Test 5: End-to-end with mock LLM (min-p, all hit)
# ---------------------------------------------------------------------------


class TestScoreEndToEndMock:
    def test_min_p_all_hit(self):
        """Run min-p with mock LLM (all hit). estimated_score == total_score."""
        paper_id = "min-p"

        # Use monkeypatch to ensure mock behaviour (Card 2 default is mock)
        report = score(
            paper_id,
            actions_path=str(PROJECT_ROOT / "tests" / "fixtures" / "empty_actions.json"),
            repo_dir=tempfile.mkdtemp(),
            verbose=False,
            dry_run=True,
        )

        assert report["paper_id"] == paper_id
        assert report["total_score"] == 272
        assert report["estimated_score"] == 272.0
        assert report["estimated_recall"] == 1.0
        assert len(report["rubric_results"]) == 82

        # All verdicts should be "hit" (mock)
        for rr in report["rubric_results"]:
            assert rr["verdict"] == "hit"
            assert rr["earned_score"] == rr["rubric_score"]

    def test_empty_actions_still_runs(self):
        """With mock LLM, empty actions.json still produces results."""
        rubrics = [
            {"criteria": "test", "score": 2, "type": "Paper Observation", "comment": None}
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tf:
            json.dump(rubrics, tf)
            rubrics_path = tf.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as af:
            json.dump([], af)
            actions_path = af.name

        try:
            report = score(
                "test-paper",
                rubrics_path=rubrics_path,
                actions_path=actions_path,
                repo_dir=tempfile.mkdtemp(),
                verbose=False,
                dry_run=True,
            )
            assert report["total_score"] == 2.0
            assert len(report["rubric_results"]) == 1
        finally:
            Path(rubrics_path).unlink(missing_ok=True)
            Path(actions_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 6: Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    @pytest.mark.parametrize("paper_id", list(PAPER_EXPECTED_TOTALS))
    def test_all_5_papers_produce_valid_report(self, paper_id):
        """Each paper produces a report that passes schema validation."""
        report = score(
            paper_id,
            actions_path=str(PROJECT_ROOT / "tests" / "fixtures" / "empty_actions.json"),
            repo_dir=tempfile.mkdtemp(),
            verbose=False,
            dry_run=True,
        )

        err = _validate_report(report)
        assert err is None, f"{paper_id} schema validation failed:\n{err}"

        # Basic integrity checks
        assert report["paper_id"] == paper_id
        assert report["total_score"] == PAPER_EXPECTED_TOTALS[paper_id]
        assert report["estimated_score"] == report["total_score"]  # all hit
        assert 0.0 <= report["estimated_recall"] <= 1.0
        assert len(report["rubric_results"]) > 0

        # by_type should have all 5 keys
        for t in ["Paper Observation", "Plan Writing", "Code Implementation",
                   "Command Execution", "Result Matching"]:
            assert t in report["by_type"], f"{paper_id}: missing by_type key '{t}'"

    def test_report_with_miss_validates(self):
        """A report with a 'miss' verdict should still pass schema."""
        rubrics = [_make_rubric(0, "test", "Plan Writing", 3)]
        judgements = [_make_judgement("miss")]
        report = aggregate("test", rubrics, judgements)

        err = _validate_report(report)
        assert err is None, f"Schema validation failed: {err}"
        assert report["estimated_score"] == 0.0

    def test_report_with_partial_validates(self):
        """A report with a 'partial' verdict should still pass schema."""
        rubrics = [_make_rubric(0, "test", "Code Implementation", 4)]
        judgements = [_make_judgement("partial")]
        report = aggregate("test", rubrics, judgements)

        err = _validate_report(report)
        assert err is None, f"Schema validation failed: {err}"
        assert report["estimated_score"] == 2.0


# ---------------------------------------------------------------------------
# Card 3: retrieval per-type filtering tests
# ---------------------------------------------------------------------------


# --- Action fixtures ---------------------------------------------------------


def _make_read_paper(idx: int = 1, path: str = "paper.md", content: str = "abc") -> dict:
    return {
        "id": idx,
        "tool": "Read",
        "arguments": {"path": path},
        "result": {"content": content, "success": True},
        "timestamp": f"2026-06-08T00:00:{idx:02d}Z",
        "duration_ms": 100,
    }


def _make_read_other(idx: int = 2, path: str = "README.md") -> dict:
    return {
        "id": idx,
        "tool": "Read",
        "arguments": {"path": path},
        "result": {"content": "some content", "success": True},
    }


def _make_write_py(idx: int = 3, path: str = "train.py", content: str = "def train(): pass") -> dict:
    return {
        "id": idx,
        "tool": "Write",
        "arguments": {"path": path, "content": content},
        "result": {"success": True},
    }


def _make_write_plan(idx: int = 4, path: str = "plan.md", content: str = "plan content") -> dict:
    return {
        "id": idx,
        "tool": "Write",
        "arguments": {"path": path, "content": content},
        "result": {"success": True},
    }


def _make_execute(idx: int = 5, command: str = "python train.py",
                  stdout: str = "accuracy: 0.95", exit_code: int = 0) -> dict:
    return {
        "id": idx,
        "tool": "Execute",
        "arguments": {"command": command},
        "result": {"stdout": stdout, "stderr": "", "exit_code": exit_code, "success": True},
    }


def _make_edit_py(idx: int = 6, path: str = "model.py", content: str = "class Model: pass") -> dict:
    return {
        "id": idx,
        "tool": "Edit",
        "arguments": {"path": path, "content": content},
        "result": {"success": True},
    }


def _make_write_paper_md(idx: int = 7) -> dict:
    """Write to paper.md — should NOT be counted as Plan Writing."""
    return {
        "id": idx,
        "tool": "Write",
        "arguments": {"path": "paper.md", "content": "paper notes"},
        "result": {"success": True},
    }


def _make_read_metrics(idx: int = 8) -> dict:
    return {
        "id": idx,
        "tool": "Read",
        "arguments": {"path": "results/metrics.json"},
        "result": {"content": '{"accuracy": 0.95}', "success": True},
    }


class TestFilterByType:
    """Test _filter_by_type for each rubric type."""

    def test_paper_observation_only_reads_paper(self):
        actions = [
            _make_read_paper(1, "paper.md"),
            _make_read_other(2, "README.md"),
            _make_write_py(3),
            _make_execute(4),
        ]
        result = _filter_by_type("Paper Observation", actions)
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_paper_observation_excludes_other_reads(self):
        actions = [
            _make_read_other(1, "README.md"),
            _make_read_other(2, "data.md"),
        ]
        result = _filter_by_type("Paper Observation", actions)
        assert len(result) == 0

    def test_plan_writing_includes_plan_design(self):
        actions = [
            _make_write_plan(1, "plan.md"),
            _make_write_plan(2, "design.md"),
        ]
        result = _filter_by_type("Plan Writing", actions)
        ids = {a["id"] for a in result}
        assert ids == {1, 2}

    def test_plan_writing_excludes_paper_md(self):
        actions = [
            _make_write_plan(1, "plan.md"),
            _make_write_paper_md(2),
            _make_write_plan(3, "notes.md"),
        ]
        result = _filter_by_type("Plan Writing", actions)
        ids = {a["id"] for a in result}
        # plan.md and notes.md (both .md and not paper), but paper.md excluded
        assert 1 in ids
        assert 2 not in ids  # paper.md excluded
        assert 3 in ids      # notes.md ok

    def test_code_implementation_only_py(self):
        actions = [
            _make_write_py(1, "train.py"),
            _make_write_py(2, "eval.py"),
            _make_write_plan(3, "plan.md"),
            _make_read_paper(4),
        ]
        result = _filter_by_type("Code Implementation", actions)
        ids = {a["id"] for a in result}
        assert ids == {1, 2}

    def test_code_implementation_includes_edit_py(self):
        actions = [
            _make_edit_py(1, "model.py"),
            _make_write_py(2, "train.py"),
        ]
        result = _filter_by_type("Code Implementation", actions)
        assert len(result) == 2

    def test_command_execution_all_execute(self):
        actions = [
            _make_execute(1, "python train.py"),
            _make_execute(2, "ls -la"),
            _make_read_paper(3),
            _make_write_py(4),
        ]
        result = _filter_by_type("Command Execution", actions)
        ids = {a["id"] for a in result}
        assert ids == {1, 2}

    def test_result_matching_read_metrics(self):
        actions = [
            _make_read_metrics(1),
            _make_read_paper(2, "paper.md"),
            _make_read_other(3, "README.md"),
        ]
        result = _filter_by_type("Result Matching", actions)
        # Read on metrics.json → matches; paper.md and README.md → no
        ids = {a["id"] for a in result}
        assert 1 in ids
        assert 2 not in ids
        assert 3 not in ids

    def test_result_matching_execute_with_metrics(self):
        actions = [
            _make_execute(1, "python train.py", stdout="accuracy: 0.95 loss: 0.02"),
            _make_execute(2, "ls -la", stdout="total 0"),  # no metric keywords
        ]
        result = _filter_by_type("Result Matching", actions)
        # Execute with metric keywords passes; Execute without metric keywords fails
        ids = {a["id"] for a in result}
        assert 1 in ids
        # Action 2 with no metrics in stdout/command → excluded


class TestKeywordExtraction:
    def test_extract_keywords_basic(self):
        kw = _extract_keywords("load the BLOOMZ model and fine-tune")
        assert "load" in kw
        assert "bloomz" in kw
        assert "model" in kw
        assert "fine-tune" in kw
        # Stop words excluded
        assert "the" not in kw
        assert "and" not in kw

    def test_extract_keywords_removes_short_tokens(self):
        kw = _extract_keywords("in the we go to")
        assert kw == []  # all ≤ 2 chars or stop words

    def test_extract_keywords_preserves_hyphens(self):
        kw = _extract_keywords("fine-tune BLOOMZ-560m model")
        assert "fine-tune" in kw
        assert "bloomz-560m" in kw


class TestKeywordRanking:
    def test_rank_by_keyword_overlap(self):
        rubric = {"criteria": "load BLOOMZ model"}
        actions = [
            {
                "id": 1,
                "tool": "Write",
                "arguments": {
                    "path": "model.py",
                    "content": "standard model definition",
                },
            },
            {
                "id": 2,
                "tool": "Write",
                "arguments": {
                    "path": "load_bloomz.py",
                    "content": "def load_bloomz(): download BLOOMZ model",
                },
            },
        ]
        ranked = _rank_by_keyword_overlap(rubric, actions)
        # Action 2 has "bloomz" and "model" keywords → higher score
        assert ranked[0]["id"] == 2
        assert ranked[1]["id"] == 1

    def test_rank_by_keyword_preserves_order_on_tie(self):
        rubric = {"criteria": "implement training loop"}
        actions = [
            {"id": 1, "tool": "Read", "arguments": {"path": "paper.md"},
             "result": {"content": "pytorch training code"}},
            {"id": 2, "tool": "Read", "arguments": {"path": "paper.md"},
             "result": {"content": "pytorch training code"}},
        ]
        ranked = _rank_by_keyword_overlap(rubric, actions)
        assert [a["id"] for a in ranked] == [1, 2]  # tie → original order

    def test_rank_empty_keywords_returns_original_order(self):
        rubric = {"criteria": ""}
        actions = [
            _make_read_paper(1),
            _make_read_paper(2),
        ]
        ranked = _rank_by_keyword_overlap(rubric, actions)
        assert [a["id"] for a in ranked] == [1, 2]


class TestTruncate:
    def test_truncate_content_1500_chars(self):
        """Write content 1800 chars → truncated to ≤ 1500 + marker length."""
        long_content = "x" * 1800
        actions = [
            {
                "id": 1,
                "tool": "Write",
                "arguments": {"path": "model.py", "content": long_content},
            },
        ]
        result = _truncate(actions, max_actions=10, content_max_chars=1500)
        truncated = result[0]["arguments"]["content"]
        assert len(truncated) <= 1500 + len("...[truncated]")
        assert truncated.endswith("...[truncated]")

    def test_truncate_short_content_unchanged(self):
        """Content under 1500 chars should not be truncated."""
        short = "def train(): pass"
        actions = [
            {
                "id": 1,
                "tool": "Write",
                "arguments": {"path": "model.py", "content": short},
            },
        ]
        result = _truncate(actions, max_actions=10, content_max_chars=1500)
        assert result[0]["arguments"]["content"] == short

    def test_truncate_preserves_metadata(self):
        """id, timestamp, duration_ms, tool should survive truncation."""
        actions = [
            {
                "id": 42,
                "tool": "Execute",
                "arguments": {"command": "python train.py"},
                "result": {"stdout": "x" * 2000, "stderr": "", "exit_code": 0, "success": True},
                "timestamp": "2026-06-08T12:00:00Z",
                "duration_ms": 5000,
            },
        ]
        result = _truncate(actions, max_actions=10, content_max_chars=1500)
        a = result[0]
        assert a["id"] == 42
        assert a["tool"] == "Execute"
        assert a["timestamp"] == "2026-06-08T12:00:00Z"
        assert a["duration_ms"] == 5000
        assert a["result"]["exit_code"] == 0
        assert a["result"]["success"] is True

    def test_truncate_read_content(self):
        actions = [
            {
                "id": 1,
                "tool": "Read",
                "arguments": {"path": "paper.md"},
                "result": {"content": "y" * 2000, "success": True},
            },
        ]
        result = _truncate(actions, max_actions=10, content_max_chars=1500)
        assert len(result[0]["result"]["content"]) <= 1500 + len("...[truncated]")

    def test_truncate_deep_copies(self):
        """Original actions should not be mutated by truncation."""
        actions = [
            {
                "id": 1,
                "tool": "Write",
                "arguments": {"path": "model.py", "content": "x" * 2000},
            },
        ]
        original_content = actions[0]["arguments"]["content"]
        _truncate(actions, max_actions=10, content_max_chars=1500)
        # Original unchanged
        assert actions[0]["arguments"]["content"] == original_content


class TestFilterActionsIntegration:
    """End-to-end filter_actions tests (combining all 3 steps)."""

    def test_filter_empty_actions_returns_empty(self):
        result = filter_actions(
            {"type": "Paper Observation", "criteria": "read section 3"}, []
        )
        assert result == []

    def test_filter_no_match_returns_empty(self):
        actions = [_make_write_py(1), _make_execute(2)]
        # Paper Observation doesn't match Write or Execute
        result = filter_actions(
            {"type": "Paper Observation", "criteria": "read paper"}, actions
        )
        assert result == []

    def test_max_actions_cap(self):
        """30 matching actions, max_actions=15 → returns ≤ 15."""
        actions = [
            _make_read_paper(i) for i in range(30)
        ]
        result = filter_actions(
            {"type": "Paper Observation", "criteria": "read paper"},
            actions,
            max_actions=15,
        )
        assert len(result) == 15

    def test_max_actions_respects_input(self):
        """When fewer than max_actions match, return all matches."""
        actions = [
            _make_read_paper(1), _make_read_paper(2), _make_read_paper(3)
        ]
        result = filter_actions(
            {"type": "Paper Observation", "criteria": "read paper"},
            actions,
            max_actions=10,
        )
        assert len(result) == 3

    def test_keyword_ranking_determines_top_n(self):
        """Actions with matching keywords should be in top N."""
        actions = []
        for i in range(20):
            if i == 7:
                actions.append({
                    "id": 7,
                    "tool": "Write",
                    "arguments": {
                        "path": "attention.py",
                        "content": "multi-head attention implementation with BLOOMZ",
                    },
                })
            else:
                actions.append({
                    "id": i,
                    "tool": "Write",
                    "arguments": {
                        "path": f"file_{i}.py",
                        "content": f"generic code {i}",
                    },
                })

        result = filter_actions(
            {"type": "Code Implementation", "criteria": "implement BLOOMZ attention"},
            actions,
            max_actions=5,
        )
        # Action 7 with "bloomz" keyword should be in top 5
        ids = {a["id"] for a in result}
        assert 7 in ids

    def test_truncation_applied_in_integration(self):
        """Content > 1500 chars should be truncated in filter_actions output."""
        long_content = "z" * 2000
        actions = [
            {
                "id": 1,
                "tool": "Write",
                "arguments": {"path": "model.py", "content": long_content},
            },
        ]
        result = filter_actions(
            {"type": "Code Implementation", "criteria": "implement model"},
            actions,
            max_actions=5,
        )
        assert len(result) == 1
        truncated = result[0]["arguments"]["content"]
        assert len(truncated) <= 1500 + len("...[truncated]")
        assert truncated.endswith("...[truncated]")

    def test_case_insensitive_tool_matching(self):
        """Tool name matching should be case-insensitive."""
        actions = [
            {"id": 1, "tool": "read", "arguments": {"path": "paper.md"},
             "result": {"content": "abc"}},
            {"id": 2, "tool": "READ", "arguments": {"path": "paper.md"},
             "result": {"content": "def"}},
        ]
        result = filter_actions(
            {"type": "Paper Observation", "criteria": "read paper"}, actions
        )
        assert len(result) == 2

    def test_rubric_without_criteria_still_filters(self):
        """Missing criteria shouldn't crash — just no keyword ranking."""
        actions = [_make_read_paper(1), _make_read_other(2)]
        result = filter_actions(
            {"type": "Paper Observation"}, actions
        )
        assert len(result) == 1
        assert result[0]["id"] == 1


# ---------------------------------------------------------------------------
# Card 4: LLM judge tests (mock LLM via monkeypatch)
# ---------------------------------------------------------------------------

import pytest as _pytest
import os as _os


def _mock_llm_response(verdict="hit", confidence=0.95, reasoning="test judgement", evidence="test evidence"):
    """Return a valid JSON string tuple matching the expected output schema."""
    resp = {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "evidence": evidence,
    }
    return json.dumps(resp), {"input_tokens": 100, "output_tokens": 50, "model": "mock"}


class TestJudgeWithMockLLM:
    """Test judge_rubric using monkeypatched _call_llm."""

    def test_judge_hit(self, monkeypatch):
        """Monkeypatch _call_llm to return 'hit' verdict."""
        import pipeline.rubric_scorer.scorer as S
        monkeypatch.setattr(S, "_call_llm", lambda c, p, **kw: _mock_llm_response("hit"))
        monkeypatch.setattr(S, "_make_client", lambda: None)

        rubric = {"type": "Paper Observation", "criteria": "read section 2", "score": 2, "rubric_idx": 0}
        result = S.judge_rubric(
            rubric, [], [],
            client=None, paper_id="test",
            verbose=False, dry_run=False,
        )
        assert result["verdict"] == "hit"
        assert result["confidence"] == 0.95

    def test_judge_partial(self, monkeypatch):
        """Monkeypatch _call_llm to return 'partial' verdict."""
        import pipeline.rubric_scorer.scorer as S
        monkeypatch.setattr(S, "_call_llm", lambda c, p, **kw: _mock_llm_response("partial"))

        rubric = {"type": "Code Implementation", "criteria": "train function", "score": 4, "rubric_idx": 1}
        result = S.judge_rubric(
            rubric, [], [],
            client=None, paper_id="test",
            verbose=False, dry_run=False,
        )
        assert result["verdict"] == "partial"

    def test_judge_malformed_json_retries(self, monkeypatch):
        """First call returns broken JSON, second call returns valid → recover."""
        import pipeline.rubric_scorer.scorer as S

        call_count = [0]

        def broken_then_ok(client, prompt, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return "this is not json at all", {"input_tokens": 100, "output_tokens": 20, "model": "mock"}
            else:
                return _mock_llm_response("hit")

        monkeypatch.setattr(S, "_call_llm", broken_then_ok)
        monkeypatch.setattr(S, "_make_client", lambda: None)

        rubric = {"type": "Plan Writing", "criteria": "PGD attack plan", "score": 3, "rubric_idx": 2}
        result = S.judge_rubric(
            rubric, [], [],
            client=None, paper_id="test",
            verbose=False, dry_run=False,
        )
        assert result["verdict"] == "hit"
        assert call_count[0] == 2  # First attempt + 1 retry

    def test_judge_all_retries_fail(self, monkeypatch):
        """All calls return broken JSON → fallback to miss."""
        import pipeline.rubric_scorer.scorer as S

        monkeypatch.setattr(S, "_call_llm",
            lambda c, p, **kw: ("{{{broken", {"input_tokens": 100, "output_tokens": 10, "model": "mock"}))
        monkeypatch.setattr(S, "_make_client", lambda: None)

        rubric = {"type": "Command Execution", "criteria": "run training", "score": 3, "rubric_idx": 3}
        result = S.judge_rubric(
            rubric, [], [],
            client=None, paper_id="test",
            verbose=False, dry_run=False,
        )
        assert result["verdict"] == "miss"
        assert result["confidence"] == 0.3
        assert "malformed JSON" in result["reasoning"]
        assert "3 retries" in result["reasoning"]

    def test_judge_invalid_verdict_retries(self, monkeypatch):
        """Valid JSON but wrong verdict enum → retry."""
        import pipeline.rubric_scorer.scorer as S

        call_count = [0]

        def bad_verdict_then_ok(client, prompt, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({"verdict": "pass", "confidence": 0.8, "reasoning": "x", "evidence": "y"}), \
                    {"input_tokens": 100, "output_tokens": 30, "model": "mock"}
            else:
                return _mock_llm_response("miss")

        monkeypatch.setattr(S, "_call_llm", bad_verdict_then_ok)
        monkeypatch.setattr(S, "_make_client", lambda: None)

        rubric = {"type": "Result Matching", "criteria": "check accuracy", "score": 2, "rubric_idx": 4}
        result = S.judge_rubric(
            rubric, [], [],
            client=None, paper_id="test",
            verbose=False, dry_run=False,
        )
        assert result["verdict"] == "miss"
        assert call_count[0] == 2


class TestCLIDryRun:
    """Test --dry-run CLI behaviour."""

    def test_dry_run_produces_all_hit(self, tmp_path, monkeypatch):
        """With --dry-run, score() produces all-hit report without LLM."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        import pipeline.rubric_scorer.scorer as S

        rubrics = [
            {"criteria": "test", "score": 2, "type": "Paper Observation", "comment": None},
            {"criteria": "test 2", "score": 3, "type": "Code Implementation", "comment": None},
        ]

        rubrics_path = tmp_path / "rubrics.json"
        rubrics_path.write_text(json.dumps(rubrics))

        actions_path = tmp_path / "actions.json"
        actions_path.write_text("[]")

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        report = S.score(
            "test",
            rubrics_path=str(rubrics_path),
            actions_path=str(actions_path),
            repo_dir=str(repo_dir),
            verbose=False,
            dry_run=True,
        )
        assert report["estimated_score"] == 5.0
        assert report["estimated_recall"] == 1.0
        assert all(rr["verdict"] == "hit" for rr in report["rubric_results"])

    def test_missing_api_key_raises_without_dry_run(self, tmp_path, monkeypatch):
        """Without --dry-run and without DEEPSEEK_API_KEY, score() raises ValueError."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        import pipeline.rubric_scorer.scorer as S

        rubrics_path = tmp_path / "rubrics.json"
        rubrics_path.write_text(json.dumps([
            {"criteria": "test", "score": 2, "type": "Paper Observation", "comment": None}
        ]))
        actions_path = tmp_path / "actions.json"
        actions_path.write_text("[]")
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        with _pytest.raises(ValueError, match="DEEPSEEK_API_KEY not set"):
            S.score(
                "test",
                rubrics_path=str(rubrics_path),
                actions_path=str(actions_path),
                repo_dir=str(repo_dir),
                verbose=False,
                dry_run=False,
            )

    def test_client_injection_bypasses_api_key_check(self, tmp_path, monkeypatch):
        """When client is injected, API key check is skipped."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        import pipeline.rubric_scorer.scorer as S

        monkeypatch.setattr(S, "_call_llm", lambda c, p, **kw: _mock_llm_response("partial"))
        mock_client = object()  # arbitrary non-None client

        rubrics_path = tmp_path / "rubrics.json"
        rubrics_path.write_text(json.dumps([
            {"criteria": "test", "score": 3, "type": "Plan Writing", "comment": None}
        ]))
        actions_path = tmp_path / "actions.json"
        actions_path.write_text("[]")
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        # Should NOT raise ValueError even without API key
        report = S.score(
            "test",
            rubrics_path=str(rubrics_path),
            actions_path=str(actions_path),
            repo_dir=str(repo_dir),
            client=mock_client,
            verbose=False,
            dry_run=False,
            resume=False,
        )
        assert report["estimated_score"] == 1.5  # partial = 0.5 * 3


@pytest.mark.skipif(
    not _os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set — skip real API smoke test",
)
class TestRealAPISmoke:
    """Smoke tests that call the real DeepSeek API. Skipped when API key is absent."""

    def test_real_api_returns_valid_json(self):
        """Call real API with a simple rubric, verify valid JSON response."""
        import pipeline.rubric_scorer.scorer as S

        rubric = {
            "type": "Paper Observation",
            "criteria": "The agent has read section 2.1 to understand notations",
            "score": 1,
            "comment": "Section 2",
            "rubric_idx": 0,
        }
        actions = [
            {
                "id": 1,
                "tool": "Read",
                "arguments": {"path": "paper.md"},
                "result": {"content": "section 2.1 preliminaries and notation", "success": True},
            }
        ]
        repo_files = []

        result = S.judge_rubric(
            rubric, actions, repo_files,
            paper_id="smoke-test",
            verbose=True,
            dry_run=False,
        )

        assert result["verdict"] in ("hit", "partial", "miss")
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["reasoning"], str)
        assert isinstance(result["evidence"], str)


# ---------------------------------------------------------------------------
# Card 4.5: API-level retry + resume tests
# ---------------------------------------------------------------------------


class TestAPIRetry:
    """Tests for API-level retry in _call_llm."""

    def test_api_retry_recovers_from_timeout(self, monkeypatch):
        """First call raises APITimeoutError, second succeeds → recovers."""
        import openai as _oa
        call_count = [0]

        def fail_then_ok(client, prompt, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise _oa.APITimeoutError("Request timed out")
            else:
                return '{"verdict":"hit","confidence":0.8,"reasoning":"ok","evidence":"e"}', {
                    "input_tokens": 100, "output_tokens": 30, "model": "mock",
                }

        import pipeline.rubric_scorer.scorer as S
        monkeypatch.setattr(S, "_call_llm", fail_then_ok)
        monkeypatch.setattr(S, "_model_name", lambda: "mock")
        monkeypatch.setattr(S, "_make_client", lambda: type("DummyClient", (), {})())

        rubric = {"type": "Paper Observation", "criteria": "read paper", "score": 2, "rubric_idx": 0}
        result = S.judge_rubric(
            rubric, [], [],
            client=type("DummyClient", (), {})(), paper_id="test-api-retry",
            verbose=False, dry_run=False,
        )
        assert result["verdict"] == "hit"
        assert call_count[0] == 2  # First failed, second succeeded

    def test_api_retry_exhausts_raises(self, monkeypatch):
        """All calls raise APITimeoutError → the error propagates past API retry."""
        import openai as _oa
        call_count = [0]

        def always_fail(client, prompt, **kw):
            call_count[0] += 1
            raise _oa.APITimeoutError("Request timed out")

        import pipeline.rubric_scorer.scorer as S
        monkeypatch.setattr(S, "_call_llm", always_fail)
        monkeypatch.setattr(S, "_model_name", lambda: "mock")
        monkeypatch.setattr(S, "_make_client", lambda: type("DummyClient", (), {})())

        rubric = {"type": "Paper Observation", "criteria": "read paper", "score": 2, "rubric_idx": 0}
        result = S.judge_rubric(
            rubric, [], [],
            client=type("DummyClient", (), {})(), paper_id="test-api-exhaust",
            verbose=False, dry_run=False,
        )
        # API retry exhausted → judge_rubric's JSON-retry also fails → fallback to miss
        assert call_count[0] == 3  # 3 API-level retries
        assert result["verdict"] == "miss"  # fallback after all retries fail


class TestResume:
    """Tests for resume-from-_debug functionality."""

    def test_resume_skips_existing(self, tmp_path, monkeypatch):
        """Pre-write a cached verdict in _debug; resume should skip it."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        import pipeline.rubric_scorer.scorer as S

        # Create a mock _debug entry
        paper_id = "test-resume"
        debug_dir = tmp_path / paper_id / "_debug" / "rubric_scorer"
        debug_dir.mkdir(parents=True)
        cached = {
            "paper_id": paper_id,
            "rubric_idx": 0,
            "rubric_type": "Paper Observation",
            "rubric_criteria": "read section 2",
            "model": "mock",
            "temperature": 0.2,
            "max_tokens": 4000,
            "input_tokens": 100,
            "output_tokens": 30,
            "timestamp": "2026-06-08 12:00:00",
            "sent_prompt": "mock prompt",
            "raw_response": '{"verdict":"hit","confidence":0.9,"reasoning":"cached","evidence":"cache"}',
            "parsed_verdict": {
                "verdict": "hit",
                "confidence": 0.9,
                "reasoning": "cached verdict",
                "evidence": "cached evidence",
            },
            "warnings": [],
        }
        debug_path = debug_dir / "llm_response_rubric_000.json"
        debug_path.write_text(json.dumps(cached))

        # Create test inputs
        rubrics = [{"criteria": "read section 2", "score": 2, "type": "Paper Observation", "comment": None}]
        rubrics_path = tmp_path / "rubrics.json"
        rubrics_path.write_text(json.dumps(rubrics))

        actions_path = tmp_path / "actions.json"
        actions_path.write_text("[]")

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        # Patch OUTPUTS_DIR to use tmp_path
        monkeypatch.setattr(S, "OUTPUTS_DIR", tmp_path)

        # Mock _call_llm so we can detect if it's called
        llm_called = [False]
        def detect_call(*args, **kw):
            llm_called[0] = True
            return '{"verdict":"miss","confidence":0.1,"reasoning":"SHOULD NOT BE SEEN","evidence":"err"}', {
                "input_tokens": 0, "output_tokens": 0, "model": "mock",
            }
        monkeypatch.setattr(S, "_call_llm", detect_call)
        mock_client = type("DummyClient", (), {})()

        report = S.score(
            paper_id,
            rubrics_path=str(rubrics_path),
            actions_path=str(actions_path),
            repo_dir=str(repo_dir),
            client=mock_client,
            verbose=False,
            dry_run=False,
            resume=True,
        )
        # Should use cached verdict, NOT call LLM
        assert llm_called[0] is False
        assert report["estimated_score"] == 2.0  # hit = full score
        assert report["rubric_results"][0]["verdict"] == "hit"
        assert report["rubric_results"][0]["reasoning"] == "cached verdict"

    def test_no_resume_overrides_cache(self, tmp_path, monkeypatch):
        """With resume=False, cached verdicts are ignored and LLM is called."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        import pipeline.rubric_scorer.scorer as S

        # Create a pre-existing _debug entry
        paper_id = "test-no-resume"
        debug_dir = tmp_path / paper_id / "_debug" / "rubric_scorer"
        debug_dir.mkdir(parents=True)
        cached = {
            "paper_id": paper_id, "rubric_idx": 0,
            "rubric_type": "Plan Writing", "rubric_criteria": "plan PGD",
            "model": "mock", "temperature": 0.2, "max_tokens": 4000,
            "input_tokens": 100, "output_tokens": 30,
            "timestamp": "2026-06-08 12:00:00",
            "sent_prompt": "mock", "raw_response": "...",
            "parsed_verdict": {
                "verdict": "hit", "confidence": 0.9,
                "reasoning": "cached", "evidence": "cached",
            },
            "warnings": [],
        }
        (debug_dir / "llm_response_rubric_000.json").write_text(json.dumps(cached))

        rubrics_path = tmp_path / "rubrics.json"
        rubrics_path.write_text(json.dumps([
            {"criteria": "plan PGD", "score": 3, "type": "Plan Writing", "comment": None}
        ]))
        actions_path = tmp_path / "actions.json"
        actions_path.write_text("[]")
        (tmp_path / "repo").mkdir()

        monkeypatch.setattr(S, "OUTPUTS_DIR", tmp_path)

        # Mock LLM returns miss
        def mock_llm(*args, **kw):
            return '{"verdict":"miss","confidence":0.5,"reasoning":"freshly judged","evidence":"fresh"}', {
                "input_tokens": 100, "output_tokens": 30, "model": "mock",
            }
        monkeypatch.setattr(S, "_call_llm", mock_llm)
        mock_client = type("DummyClient", (), {})()

        report = S.score(
            paper_id,
            rubrics_path=str(rubrics_path),
            actions_path=str(actions_path),
            repo_dir=str(tmp_path / "repo"),
            client=mock_client,
            verbose=False,
            dry_run=False,
            resume=False,  # Explicitly override resume
        )
        # Should IGNORE cache and call LLM (which returns miss)
        assert report["rubric_results"][0]["verdict"] == "miss"
        assert report["rubric_results"][0]["reasoning"] == "freshly judged"
