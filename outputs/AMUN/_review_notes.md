# AMUN paper_state Call 1 — Review Notes

**Status**: accepted, with known issues.
**Generated**: paper_state.json (v2)
**Cost**: 2 attempts × $0.03 = $0.06 total

## What's correct (verified against rubric)

- Title, venue, code_url all present and correct.
- problem_definition: 4 fields, accurate.
- core_method: name + one_line capture mechanism precisely.
- 2 datasets (CIFAR-10, Tiny ImageNet). matches rubric.
- 4 models incl. TRADES and Clipped ResNet-18. additional ablation models recognized.
- 7 baselines incl. Retrain (added in v2 prompt fix).
- 6 metrics incl. MIS.
- 9 main_experiments. covers all 5 Result Matching rubric witnesses:
  - table-1 (with-remain main)
  - table-2 (without-remain main)
  - table-3 (without-remain extended)
  - figure-1 (D_A choice ablation)
  - figure-2 (continuous unlearning)
- 3 settings (with_remain / without_remain / +SalUn variant).
- 5 expected_claims, with concrete numerical values and verification hints.

## Known issues (to address downstream)

### Issue 1: primary=true is too narrow
- LLM marked only table-1, table-2, figure-2 as primary.
- table-3, figure-1 are also Result Matching evidence per rubric, but
  marked primary=false.
- **Mitigation**: downstream `derive_from_state.py` should NOT rely solely
  on `primary` field. It should derive Result Matching items from
  `expected_claims.evidence` and any `main_experiments[].evidence_in_paper`
  regardless of primary flag.

### Issue 2: deferred_experiments includes figure-2/table-3/figure-1
- These are Result Matching evidence and should not be deferred.
- **Mitigation**: same as Issue 1 — downstream Stage Planner should compute
  its own priority list from rubric witnesses, not from
  reproducibility_meta.deferred_experiments.

### Issue 3: settings has 3 entries, last one ("AMUN+SalUn") is method variant not setting
- LLM included AMUN+SalUn as a 3rd "setting", but it's actually a method
  variant (proposed method combined with SalUn baseline).
- **Mitigation**: not blocking. main_experiments correctly attributes
  "AMUN+SalUn" claims to existing settings.

## Decision

**Accept this paper_state for Call 2.** Do not re-prompt Call 1.
Reasoning: prompt iteration on primary judgment yields diminishing returns
(LLM's per-paper judgment isn't fully steerable via universal rules).
Downstream derive logic must be designed to be primary-agnostic for
Result Matching rubrics.


---

## Cross-paper validation (2026-05) — Phase 1 closeout

After all 5 papers extracted via Call 1 + Call 2 with middle-50% method context:

| Paper | atomic_steps | main_exp | claims | Result Matching coverage |
|-------|-------------|----------|--------|-------------------------|
| AMUN  | 10 | 9  | 5 | 5/5 ✓ |
| Beyond-Ngram | 13 | 5* | 5 | 7/7 ✓ |
| I0T   | 11 | 7  | 6 | 7/7 ✓ |
| INCLINE | 11 | 10 | 6 | 10/10 ✓ |
| min-p | 10 | 9* | 7 | 6/6 ✓ |

*Manual patches applied:
- Beyond-Ngram: figure-1 (Elo score trends) added as exp-figure1
- min-p: table-8 (Llama 3 results) added as exp-table8

Both entities were correctly extracted by entity_extractor, visible to LLM
in Call 1 context (within 60% cutoff), but LLM failed to include them in
main_experiments. Confirms known issue: LLM judgment on main_experiment
completeness is paper-specific and not fully steerable via prompt rules.

Total RM rubric witnesses: 35/35 (100%) covered after patches.

## Known limitations (carry to v2)

1. main_experiments completeness depends on LLM judgment — 2/5 papers
   had a missing witness despite the entity being in the LLM context.
   Mitigation: add verification step in v2 — cross-check rubric anchors
   (training time) or every entity in result-bearing sections (test time)
   against main_experiments evidence.

2. primary/deferred labels in some papers (notably AMUN) are uneven.
   Mitigation: derive_from_state.py is designed primary-agnostic.

3. core_method.settings field may include method-variant entries
   (e.g. "AMUN+SalUn"), conflating settings with method variants.

4. baselines field semantics differ across papers (e.g. Beyond-Ngram lists
   metrics-as-baselines for a metric-eval paper). Downstream derive logic
   should not assume uniform semantics.

## Total cost (Phase 1)

5-paper paper_state extraction (final run): ~$0.55
Including all prompt iterations and bug fixes: ~$0.80 total.

---

## Phase 1 closeout — final v6 baseline (2026-05-12)

### Final recall by type (cross-paper)

| Type                | Target  | v6 Actual          |
|---------------------|---------|--------------------| 
| Paper Observation   | ≥ 90%   | **98.1%** ✓        |
| Result Matching     | ≥ 80%   | **100.0%** ✓       |
| Plan Writing        | ≥ 50%   | 48.1%              |
| Code Implementation | ≥ 50%   | 42.1%              |
| Command Execution   | ≥ 40%   | 30.4%              |
| **Overall**         | **≥60%**| **55.0%**          |

### By-paper recall

| Paper        | Total recall  |
|--------------|---------------|
| AMUN         | 63.6% (77)    |
| Beyond-Ngram | 66.1% (62)    |
| I0T          | 48.5% (102)   |
| INCLINE      | 61.8% (76)    |
| min-p        | 40.2% (82)    |
| Cross-agg    | 55.0% (399)   |

### Evaluator engineering — 6 iterations from 22% to 55%

| Version | Change | Total | Δ |
|---------|--------|-------|---|
| v1 | Initial LLM judge (buggy) | 22% | — |
| v2 | Fix JSON truncation (max_tokens 400→1200) | 40% | +18% |
| v3 | Prompt encourages partial credit | 43% | +3% |
| v4 | Expand Cmd Exec generator (pipeline/dataset/baseline) | 44% | +1% |
| v5 | Enforce reason ≤ 15 words | 49% | +5% |
| v6 | Retry on empty LLM response (max 3 attempts) | 55% | +6% |

### Key insight

The 22% → 55% trajectory is split roughly evenly between two causes:
1. **Real measurement improvements** (~12pp): prompt engineering for LLM
   judge, generator expansion, calibrated reason length.
2. **Evaluator infrastructure bugs** (~21pp): JSON truncation, empty LLM
   responses, parser failures. None of these reflected actual extraction
   quality — they were artifacts of the evaluation tooling.

This suggests that **evaluator engineering deserves the same rigor as
extraction engineering** in agent reproduction research. Many "low recall"
results in the literature may carry similar evaluator bugs.

### What we accept as Phase 1 baseline

55.0% overall recall, with:
- Anchor types (Paper Obs, Result Match) at design target
- Action types (Plan, Code, Cmd Exec) below target by 2-10pp
- Cmd Exec is the largest gap (30% vs 40%) — a v1 schema limit requiring
  runtime grounding in Phase 2/3 to close
- Total cost (Phase 1): ~$7 across paper_state extraction and 6 coverage
  diagnostic runs

### Important methodological note

**55% recall does NOT equal predicted agent score on the leaderboard.**
- 55% is the alignment between derived self-checklist text and official rubric text
- Actual agent score depends on how the organizers' evaluator scores actions.json
  + repo + results against the official rubric (which we don't have direct access to)
- Phase 2's Executor will produce actions.json. Many rubric items not in
  self-checklist may still be satisfied by the agent's actual execution
  (e.g. "training executed without errors" auto-satisfied by running
  `python train.py` successfully even if not on the checklist)
- The true agent score will only be measurable after Phase 2 produces
  actions.json on the training set

The 55% baseline is best interpreted as a **lower-bound signal** on
self-checklist coverage, not as a direct prediction of leaderboard score.

