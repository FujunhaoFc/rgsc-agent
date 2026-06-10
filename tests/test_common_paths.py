"""Unit tests for pipeline.common.paths — paper path resolution across splits."""

from pathlib import Path

import pytest

# We import the module-level constants for monkeypatching
from pipeline.common import paths as paths_mod
from pipeline.common.paths import (
    DATA_DIR,
    PROJECT_ROOT,
    SPLITS,
    find_paper_dir,
    find_paper_md,
    find_rubrics_json,
)


class TestFindPaperMd:
    """Tests for find_paper_md — the main convenience function."""

    def test_find_paper_md_train_valid_min_p(self):
        """min-p should resolve to data/train_valid/min-p/paper.md."""
        result = find_paper_md("min-p")
        expected = DATA_DIR / "train_valid" / "min-p" / "paper.md"
        assert result == expected
        assert result.exists()

    def test_find_paper_dir_with_required_files(self):
        """Requiring paper.md + rubrics.json should still find min-p."""
        result = find_paper_dir("min-p", ["paper.md", "rubrics.json"])
        expected = DATA_DIR / "train_valid" / "min-p"
        assert result == expected
        assert (result / "paper.md").exists()
        assert (result / "rubrics.json").exists()

    def test_find_paper_dir_not_found_raises(self):
        """Non-existent paper raises FileNotFoundError with searched paths."""
        with pytest.raises(FileNotFoundError) as exc_info:
            find_paper_dir("nonexistent-xyz")
        msg = str(exc_info.value)
        assert "nonexistent-xyz" in msg
        for split in SPLITS:
            assert split in msg, f"error message should mention split '{split}'"

    def test_find_paper_dir_test_split_priority(self, tmp_path, monkeypatch):
        """train_valid is checked before test — if both have the paper,
        train_valid wins."""
        # Set up a temp data directory with both splits
        fake_data = tmp_path / "data"
        tv_dir = fake_data / "train_valid" / "priority-test"
        test_dir = fake_data / "test" / "priority-test"
        tv_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        # Both have paper.md, but with distinguishing content
        (tv_dir / "paper.md").write_text("train_valid paper")
        (test_dir / "paper.md").write_text("test paper")

        # Monkeypatch DATA_DIR and SPLITS to use our temp tree
        monkeypatch.setattr(paths_mod, "DATA_DIR", fake_data)
        monkeypatch.setattr(paths_mod, "SPLITS", ("train_valid", "test"))

        result = find_paper_dir("priority-test")
        # Should return train_valid, not test
        assert result == tv_dir
        assert "train_valid" in str(result)


class TestIntegrationPaperMd:
    """Smoke test: verify that find_paper_md works for all 5 train papers."""

    def test_all_train_papers_have_paper_md(self):
        for paper_id in ["AMUN", "Beyond-Ngram", "I0T", "INCLINE", "min-p"]:
            path = find_paper_md(paper_id)
            assert path.name == "paper.md"
            assert path.exists(), f"Missing paper.md for {paper_id}"
