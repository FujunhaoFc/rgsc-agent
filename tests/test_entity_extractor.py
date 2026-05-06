"""Unit tests for pipeline.paper_observer.entity_extractor."""

import pytest

from pipeline.paper_observer.entity_extractor import (
    extract_entities,
    extract_from_paper,
)
from pipeline.paper_observer.section_parser import parse_sections


# --------- table extraction ------------------------------------------------


def test_table_basic():
    md = "Some text mentions Table 1 here.\nAnd more text.\n"
    out = extract_entities(md)
    assert len(out["tables"]) == 1
    assert out["tables"][0]["id"] == "table-1"
    assert out["tables"][0]["first_mention_line"] == 1


def test_table_caption_recognized():
    md = (
        "We compare methods.\n"
        "Table 1: Main results comparing AMUN with baselines.\n"
        "Below are the numbers.\n"
    )
    out = extract_entities(md)
    assert len(out["tables"]) == 1
    cap = out["tables"][0]["caption"]
    assert "Main results" in cap or "AMUN" in cap


def test_table_multiple_first_mention_only():
    """Even if Table 1 mentioned multiple times, first_mention_line is the first."""
    md = (
        "Line 1 talks about Table 1.\n"
        "Line 2 also mentions Table 1.\n"
        "Line 3 mentions Table 2.\n"
    )
    out = extract_entities(md)
    labels_to_line = {t["label"]: t["first_mention_line"] for t in out["tables"]}
    assert labels_to_line["1"] == 1
    assert labels_to_line["2"] == 3


def test_table_with_letter_suffix():
    md = "See Table 1a for the ablation.\n"
    out = extract_entities(md)
    assert out["tables"][0]["label"] == "1a"


# --------- figure / algorithm / equation -----------------------------------


def test_figure_basic():
    md = "Figure 3 illustrates the architecture.\n"
    out = extract_entities(md)
    assert out["figures"][0]["id"] == "figure-3"


def test_algorithm_basic():
    md = "Algorithm 1 describes the procedure.\n"
    out = extract_entities(md)
    assert out["algorithms"][0]["id"] == "algorithm-1"


def test_algorithm_typo():
    """Match the 'Algorihm' typo seen in AMUN."""
    md = "We follow Algorihm 2 in our implementation.\n"
    out = extract_entities(md)
    assert out["algorithms"][0]["label"] == "2"


def test_equation_paren():
    md = "From equation (5), we derive the bound.\n"
    out = extract_entities(md)
    assert out["equations"][0]["label"] == "5"


def test_equation_bare():
    md = "Substituting equation 12 yields ...\n"
    out = extract_entities(md)
    assert out["equations"][0]["label"] == "12"


# --------- in_section attribution -----------------------------------------


def test_in_section_attribution():
    md = (
        "# 1 Introduction\n"        # line 1
        "intro text\n"               # line 2
        "# 2 Method\n"               # line 3
        "We use Table 1 here.\n"     # line 4 ← Table 1 first mention
        "# 3 Results\n"              # line 5
        "Table 2 shows the data.\n"  # line 6 ← Table 2 first mention
    )
    sections = parse_sections(md)
    out = extract_entities(md, sections)
    by_label = {t["label"]: t["in_section"] for t in out["tables"]}
    assert by_label["1"] == "sec-2"
    assert by_label["2"] == "sec-3"


def test_in_section_none_when_no_sections():
    md = "Random Table 1 mention.\n"
    out = extract_entities(md, sections=[])
    assert out["tables"][0]["in_section"] is None


# --------- context_lines ---------------------------------------------------


def test_context_lines_within_bounds():
    md = "\n".join([f"line {i}" for i in range(1, 51)]) + "\n"
    md = md.replace("line 25", "We see Table 1 in line 25")
    out = extract_entities(md)
    ctx = out["tables"][0]["context_lines"]
    # first_mention should be 25, context should span [20, 45]
    assert out["tables"][0]["first_mention_line"] == 25
    assert ctx[0] == 20  # 25 - 5
    assert ctx[1] == 45  # 25 + 20


def test_context_lines_clamped_at_start():
    md = "We see Table 1 in line 1.\nmore text\n"
    out = extract_entities(md)
    ctx = out["tables"][0]["context_lines"]
    # ctx[0] cannot go below 1
    assert ctx[0] == 1


# --------- real paper sanity ----------------------------------------------


def test_amun_real_paper():
    """AMUN should yield non-trivial counts for tables and at least one algorithm."""
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[1]
    amun = project_root / "data" / "train_valid" / "AMUN" / "paper.md"
    if not amun.exists():
        pytest.skip("AMUN paper.md not present")

    sections = None
    from pipeline.paper_observer.section_parser import parse_paper_md
    sections = parse_paper_md(str(amun))
    out = extract_from_paper(str(amun), sections)

    assert len(out["tables"]) >= 1
    # AMUN has at least one Algorithm referenced (Algorihm 1)
    assert len(out["algorithms"]) >= 1
    # All entities should have line numbers
    for kind_list in out.values():
        for e in kind_list:
            assert e["first_mention_line"] >= 1
