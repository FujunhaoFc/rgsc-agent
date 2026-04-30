"""Unit tests for pipeline.rubric_normalizer.anchor_parser."""

import pytest

from pipeline.rubric_normalizer.anchor_parser import (
    parse_anchors,
    has_any_anchor,
)


# ---------- section ---------------------------------------------------------


def test_section_basic():
    out = parse_anchors("see section 3.1 for details")
    assert out["sections"] == ["3.1"]


def test_section_capitalized():
    out = parse_anchors("Section 3.1 explains")
    assert out["sections"] == ["3.1"]


def test_section_abbrev():
    out = parse_anchors("sec. 4 covers this")
    assert out["sections"] == ["4"]


def test_section_section_sign():
    out = parse_anchors("see §3.2.1 for proof")
    assert out["sections"] == ["3.2.1"]


def test_section_letter():
    out = parse_anchors("appendix Section A.2 has the proof")
    assert out["sections"] == ["A.2"]


def test_section_multiple_dedup():
    out = parse_anchors("Section 3.1 and Section 3.1 again")
    assert out["sections"] == ["3.1"]


# ---------- table -----------------------------------------------------------


def test_table_basic():
    out = parse_anchors("see Table 1")
    assert out["tables"] == ["1"]


def test_table_abbrev():
    out = parse_anchors("Tab. 3 shows results")
    assert out["tables"] == ["3"]


def test_table_with_letter():
    out = parse_anchors("Results in Table 1a")
    assert out["tables"] == ["1a"]


def test_table_multiple():
    out = parse_anchors("Table 1, Table 2, and Table 3")
    assert out["tables"] == ["1", "2", "3"]


# ---------- figure ----------------------------------------------------------


def test_figure_basic():
    out = parse_anchors("Figure 2 illustrates the architecture")
    assert out["figures"] == ["2"]


def test_figure_abbrev():
    out = parse_anchors("Fig. 3 shows the trend")
    assert out["figures"] == ["3"]


def test_figure_with_letter():
    out = parse_anchors("Figure 3a is the loss curve")
    assert out["figures"] == ["3a"]


# ---------- algorithm ------------------------------------------------------


def test_algorithm_basic():
    out = parse_anchors("Algorithm 1 describes the procedure")
    assert out["algorithms"] == ["1"]


def test_algorithm_abbrev():
    out = parse_anchors("Algo. 2 is implemented in code")
    assert out["algorithms"] == ["2"]


def test_algorithm_typo_amun():
    """AMUN's rubrics.json contains 'Algorihm' (missing t). Must match."""
    out = parse_anchors("the agent has read Algorihm 1")
    assert out["algorithms"] == ["1"]


# ---------- equation -------------------------------------------------------


def test_equation_paren():
    out = parse_anchors("see equation (5)")
    assert out["equations"] == ["5"]


def test_equation_bare():
    out = parse_anchors("from equation 12")
    assert out["equations"] == ["12"]


def test_equation_abbrev():
    out = parse_anchors("Eq. (3) gives the loss")
    assert out["equations"] == ["3"]


# ---------- combined / no anchors ------------------------------------------


def test_combined_anchors():
    text = "The method in Section 3.1 uses Algorithm 1 with equation (5), see Table 2 and Figure 3."
    out = parse_anchors(text)
    assert out["sections"] == ["3.1"]
    assert out["tables"] == ["2"]
    assert out["figures"] == ["3"]
    assert out["algorithms"] == ["1"]
    assert out["equations"] == ["5"]
    assert has_any_anchor(out) is True


def test_no_anchors():
    out = parse_anchors("This is plain text with no references at all.")
    assert out["sections"] == []
    assert out["tables"] == []
    assert out["figures"] == []
    assert out["algorithms"] == []
    assert out["equations"] == []
    assert has_any_anchor(out) is False


def test_raw_matches_present():
    out = parse_anchors("see Table 1 and Algorithm 2")
    assert len(out["raw_matches"]) == 2
    kinds = {m["kind"] for m in out["raw_matches"]}
    assert kinds == {"table", "algorithm"}


def test_raw_matches_have_span():
    out = parse_anchors("Table 1 here")
    m = out["raw_matches"][0]
    assert "span" in m
    assert m["span"][0] == 0   # "Table" starts at offset 0
