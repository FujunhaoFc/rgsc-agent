"""Unit tests for pipeline.paper_observer.section_parser."""

from pathlib import Path

import pytest

from pipeline.paper_observer.section_parser import (
    parse_paper_md,
    parse_sections,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AMUN_PAPER = PROJECT_ROOT / "data" / "train_valid" / "AMUN" / "paper.md"


# --------------------------------------------------------------------------
# Rule A: numbered sections
# --------------------------------------------------------------------------


def test_numbered_section_basic():
    md = "## 3.1 Notation\nsome content here\n"
    secs = parse_sections(md)
    assert len(secs) == 1
    assert secs[0]["id"] == "sec-3.1"
    assert secs[0]["level"] == 2
    assert secs[0]["title"].startswith("3.1")
    assert "Notation" in secs[0]["title"]


def test_numbered_section_with_dot():
    md = "## 3.1. Notation\ncontent\n"
    secs = parse_sections(md)
    assert len(secs) == 1
    assert secs[0]["id"] == "sec-3.1"


def test_numbered_section_top_level():
    md = "# 1 Introduction\ncontent\n"
    secs = parse_sections(md)
    assert secs[0]["id"] == "sec-1"
    assert secs[0]["level"] == 1


def test_numbered_section_three_levels():
    md = "### 3.1.1 Detail\ncontent\n"
    secs = parse_sections(md)
    assert secs[0]["level"] == 3
    assert secs[0]["id"] == "sec-3.1.1"


# --------------------------------------------------------------------------
# Rule B: appendix
# --------------------------------------------------------------------------


def test_appendix_top_level():
    md = "# A Appendix\ncontent\n"
    secs = parse_sections(md)
    assert secs[0]["id"] == "sec-A"
    assert secs[0]["level"] == 1


def test_appendix_nested():
    md = "## A.1 Proof\ncontent\n"
    secs = parse_sections(md)
    assert secs[0]["id"] == "sec-A.1"
    assert secs[0]["level"] == 2


# --------------------------------------------------------------------------
# Rule C: unnumbered top
# --------------------------------------------------------------------------


def test_unnumbered_top_section():
    md = "# Abstract\ncontent\n"
    secs = parse_sections(md)
    assert len(secs) == 1
    assert secs[0]["id"] == "sec-abstract"
    assert secs[0]["level"] == 1


def test_unnumbered_with_spaces():
    md = "# Related Work\ncontent\n"
    secs = parse_sections(md)
    assert secs[0]["id"] == "sec-related-work"


def test_unnumbered_with_punctuation():
    md = "# Acknowledgments!\ncontent\n"
    secs = parse_sections(md)
    assert secs[0]["id"] == "sec-acknowledgments"


# --------------------------------------------------------------------------
# Rule D: pollution filtering (min-p case)
# --------------------------------------------------------------------------


def test_min_p_filter_dash():
    md = "# - List item\ncontent\n# 1 Real Section\nbody\n"
    secs = parse_sections(md)
    assert len(secs) == 1
    assert secs[0]["id"] == "sec-1"


def test_min_p_filter_too_short():
    md = "# x\nbody\n# 1 Real\nbody\n"
    secs = parse_sections(md)
    assert len(secs) == 1
    assert secs[0]["id"] == "sec-1"


def test_min_p_filter_starred_bullet():
    md = "# * Bullet line\nbody\n# 2 Real Section\nbody\n"
    secs = parse_sections(md)
    assert len(secs) == 1
    assert secs[0]["id"] == "sec-2"


# --------------------------------------------------------------------------
# parent chain
# --------------------------------------------------------------------------


def test_parent_chain():
    md = (
        "# 1 Intro\n"
        "intro body\n"
        "## 1.1 Background\n"
        "bg body\n"
        "### 1.1.1 Detail\n"
        "detail body\n"
        "## 1.2 Motivation\n"
        "mot body\n"
    )
    secs = parse_sections(md)
    by_id = {s["id"]: s for s in secs}
    assert by_id["sec-1"]["parent"] is None
    assert by_id["sec-1.1"]["parent"] == "sec-1"
    assert by_id["sec-1.1.1"]["parent"] == "sec-1.1"
    assert by_id["sec-1.2"]["parent"] == "sec-1"


def test_parent_chain_resets_on_top():
    """After dropping back to level 1, parent stack should reset."""
    md = (
        "# 1 First\n"
        "## 1.1 Sub\n"
        "# 2 Second\n"
        "## 2.1 Sub\n"
    )
    secs = parse_sections(md)
    by_id = {s["id"]: s for s in secs}
    assert by_id["sec-2"]["parent"] is None
    assert by_id["sec-2.1"]["parent"] == "sec-2"


# --------------------------------------------------------------------------
# line range correctness
# --------------------------------------------------------------------------


def test_line_range_correctness():
    md = (
        "# 1 First\n"        # line 1
        "first body\n"       # line 2
        "more body\n"        # line 3
        "# 2 Second\n"       # line 4
        "second body\n"      # line 5
        "# 3 Third\n"        # line 6
        "third body\n"       # line 7
    )
    secs = parse_sections(md)
    assert len(secs) == 3
    assert secs[0]["line_start"] == 1 and secs[0]["line_end"] == 3
    assert secs[1]["line_start"] == 4 and secs[1]["line_end"] == 5
    assert secs[2]["line_start"] == 6 and secs[2]["line_end"] == 7


def test_line_end_to_eof():
    """The last section's line_end should equal total file lines."""
    md = "# 1 Only\nline 2\nline 3\n"
    secs = parse_sections(md)
    assert secs[-1]["line_end"] == 3


def test_char_offsets_monotonic():
    md = "# 1 A\nbody1\n# 2 B\nbody2\n"
    secs = parse_sections(md)
    for i in range(len(secs) - 1):
        assert secs[i]["char_end"] == secs[i + 1]["char_start"]


# --------------------------------------------------------------------------
# real paper sanity checks
# --------------------------------------------------------------------------


@pytest.mark.skipif(not AMUN_PAPER.exists(), reason="AMUN paper.md not present")
def test_real_paper_amun():
    secs = parse_paper_md(str(AMUN_PAPER))
    assert len(secs) >= 10, f"AMUN should have at least 10 sections, got {len(secs)}"

    # At least one level=1 section
    assert any(s["level"] == 1 for s in secs)

    # last line_end should equal AMUN's total line count (855)
    with open(AMUN_PAPER, "r", encoding="utf-8") as f:
        total_lines = len(f.read().splitlines())
    assert secs[-1]["line_end"] == total_lines, (
        f"Last line_end {secs[-1]['line_end']} != total {total_lines}"
    )


@pytest.mark.skipif(not AMUN_PAPER.exists(), reason="AMUN paper.md not present")
def test_real_paper_amun_has_intro_or_abstract():
    secs = parse_paper_md(str(AMUN_PAPER))
    early_titles = [s["title"].lower() for s in secs[:8]]
    assert any(
        "abstract" in t or "introduction" in t for t in early_titles
    ), f"Expected Abstract or Introduction in first 8 sections, got: {early_titles}"
