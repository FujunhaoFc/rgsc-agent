# Paper State Schema & Extraction Pipeline Design

**Status**: v1 frozen 2026-04-30. To be validated on AMUN first, then 4 other papers.

## 1. Why This Document Exists

`paper_state.json` is the central data structure of the RGSC-Agent pipeline.
It is the single artifact that:

- Stage Planner reads to produce `plan.md`
- Repo Builder reads to scaffold the codebase
- Code Generator reads (per-step) to produce module implementations
- Self-Checklist Deriver reads to produce a derived rubric checklist when no
  official rubrics are provided (test-time scenario)

The fourth consumer is the most demanding: at test time, no official
`rubrics.json` exists. The agent must derive a checklist from `paper_state.json`
alone, and that derived checklist must have high *recall* against what the
hidden official rubrics would have asked. This drives every schema decision
below.

## 2. Empirical Anchor: What 5 Training Rubrics Tell Us

Total: 399 rubric items across 5 papers, by type:

| Type                | Count | anchor_parser coverage | Source in paper_state |
|---------------------|-------|------------------------|---------------------- |
| Paper Observation   |   52  | 100%                   | sections + entities (already covered) |
| Result Matching     |   35  | 100%                   | main_experiments + expected_claims |
| Plan Writing        |  134  | 0.7%                   | core_method.atomic_steps + settings + datasets/models/baselines/metrics |
| Code Implementation |  127  | 5.5%                   | core_method.atomic_steps + models/baselines |
| Command Execution   |   51  | 0.0%                   | main_experiments[].id (one cmd per experiment) |

Two findings shape the schema:

**Finding A**: 78% of rubrics (Plan Writing + Code Impl + Cmd Exec) describe
agent *actions*, not paper *locations*. They cannot be served by anchor
extraction; they require semantic fields in paper_state.

**Finding B**: rubric granularity is **atomic** — "calculate maximum
probability", "construct sampling pool", "use CIFAR-10 dataset" — not coarse.
If `core_method.key_steps` has only 3 entries, derived checklist max recall
on Plan Writing is 3/N where N is the rubric count. Schema must support
fine-grained atomic steps.

## 3. Schema Design Decisions (Frozen)

### Decision 1: Three-way split of `core_method`

Old: single `key_steps` list. Replaced with three fields:

- `atomic_steps[]` — algorithmic / computational steps. Each maps to one or
  more "calculation of X" / "implementation of Y" rubrics.
- `settings[]` — configuration variants ("with remain set", "without remain
  set"). Each maps to "setting with X" rubrics.
- Resource fields (`datasets`, `models`, `baselines`, `metrics`) stay at
  paper_state top level since they are also referenced by Result Matching.

### Decision 2: One main_experiment per cited Table/Figure

Each Table or Figure that appears in a Result Matching rubric criterion
corresponds to one entry in `main_experiments[]`. This keeps the mapping
mechanical and verifiable.

Edge case: qualitative figures (loss curves, t-SNE plots) that are *not*
cited by any Result Matching rubric → still recorded but flagged
`primary: false`.

### Decision 3: LLM-judge verification, not DSL predicates

`expected_claims[].claim_text` is natural language. At verification time, an
LLM judge compares (claim_text, agent_results) and returns yes/no/partial.

Rationale: DSL predicates require unified metric naming across papers
(impossible at test time when paper_state is auto-extracted), and a working
DSL parser is a non-trivial sub-project. v1 punts to LLM judge; future v2
may revisit if cost or reliability becomes a problem.

### Decision 4: Two-call LLM extraction

- **Call 1 — Skeleton**: paper.md outline (sections summary) + first 60% of
  paper full text → all paper_state fields *except* `atomic_steps` details
  and detailed `main_experiments.axes`.
- **Call 2 — Detail**: Call-1 output + method section full text (line range
  from sections[]) → `atomic_steps` and detailed `main_experiments.axes`.
- (Training-time only) **Call 3 — Calibration**: optional. Use official
  rubric criteria to fine-tune `atomic_steps` granularity.

Estimated cost per paper using DeepSeek-V4-Pro: ~$0.30. 5 papers ~$1.50.

### Decision 5: Recall targets (training-time evaluation)

| Rubric Type         | Target Recall |
|---------------------|---------------|
| Paper Observation   | ≥ 90%         |
| Result Matching     | ≥ 80%         |
| Plan Writing        | ≥ 50%         |
| Code Implementation | ≥ 50%         |
| Command Execution   | ≥ 40%         |
| **Overall (5-paper avg)** | **≥ 60%** |

These are the bar for proceeding to test-time submission. If overall recall
is below 50% after 5-paper validation, schema or prompts need redesign.

## 4. Schema (Frozen v1)

```json
{
  "paper_id": "AMUN",
  "title": "...",
  "venue": "ICML 2025",
  "code_url": "https://github.com/...",

  "sections": [...],          // from section_parser (already implemented)
  "entities": {...},          // from entity_extractor (next module)

  "problem_definition": {
    "task": "...",
    "input": "...",
    "output": "...",
    "evaluation_target": "..."
  },

  "core_method": {
    "name": "AMUN",
    "one_line": "...",
    "atomic_steps": [
      {
        "id": "step-1",
        "description": "...",
        "depends_on_method": null,
        "depends_on_equations": [],
        "source_section": "sec-4"
      }
    ],
    "settings": [
      {
        "id": "set-1",
        "name": "with_remain_set",
        "description": "..."
      }
    ],
    "key_hyperparameters": [
      {"name": "epsilon", "value": "8/255", "source_section": "sec-5"}
    ]
  },

  "datasets": [
    {"name": "CIFAR-10", "splits": ["train", "test"], "used_for": "main", "source_section": "sec-5"}
  ],

  "models": [
    {"name": "ResNet-18", "role": "classifier", "source_section": "sec-5"}
  ],

  "baselines": [
    {"name": "Retrain", "type": "exact unlearning", "source_section": "sec-5.1"}
  ],

  "metrics": [
    {"name": "Forget Acc", "definition": "...", "source_section": "sec-5.2"}
  ],

  "main_experiments": [
    {
      "id": "exp-table1",
      "claim": "...",
      "evidence_in_paper": "table-1",
      "primary": true,
      "axes": {
        "rows": ["AMUN", "Retrain", "SalUn"],
        "cols": ["Forget Acc", "Test Acc", "MIA"]
      },
      "settings_used": ["set-1"],
      "datasets_used": ["CIFAR-10"],
      "models_used": ["ResNet-18"]
    }
  ],

  "expected_claims": [
    {
      "id": "claim-table1-amun-best",
      "claim_text": "AMUN achieves the best forget accuracy among all baselines in Table 1",
      "evidence": "table-1",
      "verification_hint": "compare AMUN row vs Retrain/SalUn rows in Forget Acc column"
    }
  ],

  "reproducibility_meta": {
    "needs_training": true,
    "estimated_compute": "single GPU, hours",
    "data_availability": "public",
    "code_partial_available": true,
    "minimum_viable_experiments": ["exp-table1"],
    "deferred_experiments": []
  }
}
```

## 5. Implementation Plan

### Phase 1 — AMUN end-to-end (next session)

1. Write `pipeline/schemas/paper_state.schema.json` (jsonschema)
2. Write extraction prompts (Call 1 + Call 2) in `pipeline/paper_observer/prompts/`
3. Implement `llm_summarizer.py` (DeepSeek V4 client + retry + JSON validation)
4. Run on AMUN, manual review
5. Implement `derive_from_state.py` (paper_state → derived checklist)
6. Implement `coverage_diagnostic.py` (derived vs official recall)
7. Report AMUN recall, decide if schema needs revision

**Expected**: 60–70% chance schema needs adjustment after AMUN pass.
This is by design — AMUN is the canary.

### Phase 2 — Roll out to 5 papers

Run the same pipeline on Beyond-Ngram, I0T, INCLINE, min-p. Aggregate
recall by paper × type. Identify lowest-recall cell, root-cause.

### Phase 3 — Calibration (if needed)

If overall recall < 60%, decide whether to:
(a) Add Call 3 (rubric-driven calibration, training only)
(b) Refine schema fields
(c) Refine prompts

## 6. Out of Scope (for v1)

- DSL-based verifiable predicates (Decision 3)
- Multi-modal extraction from images (figure parsing beyond caption)
- Cross-paper dependency tracking (e.g., this paper builds on that paper)
- Embeddings-based fuzzy section retrieval

## 7. Open Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| atomic_steps granularity too coarse for some papers | High | Phase 1 catches it on AMUN |
| LLM hallucinates non-existent equations | Medium | Cross-check against entities[] from entity_extractor |
| Some papers have no Table/Figure-cited Result Matching | Low | Fall back to claim_text only |
| Two-call extraction unstable | Medium | Add JSON schema validation + 1 retry |