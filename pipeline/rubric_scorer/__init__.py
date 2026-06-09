"""Rubric Scorer — Phase 3.2 external evaluation tool.

Simulates NLPCC grading by comparing Agent actions.json + reproduced_repo against
ground truth rubrics.json, producing a rubric_score_report.json with per-rubric
verdicts and score breakdowns.
"""

from pipeline.rubric_scorer.scorer import score, judge_rubric, compute_earned_score, aggregate
from pipeline.rubric_scorer.retrieval import filter_actions

__all__ = ["score", "judge_rubric", "compute_earned_score", "aggregate", "filter_actions"]
