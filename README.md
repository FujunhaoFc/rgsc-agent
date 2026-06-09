# RGSC-Agent

**Rubric-Grounded, Stage-Controlled Experiment Reproduction Agent**

For NLPCC 2026 Shared Task 11 — Agent-Based Experiment Reproduction from Scientific Papers.

---

**Phase 1 ✓ · Phase 2 ✓ · Phase 3.1 ✓ · Phase 3.2 Step 1 ✓ · Phase 3.2 Step 2 in progress**

---

## 1. What is RGSC-Agent?

RGSC-Agent is an LLM-driven multi-agent system that automatically reproduces machine-learning experiments from scientific papers. It operates inside an MCP (Model Context Protocol) monitored environment, producing `actions.json` and a reproduced code repository that are scored against NLPCC rubric items — covering Paper Observation, Plan Writing, Code Implementation, Command Execution, and Result Matching.

### Why this is interesting

Existing general-purpose coding agents (Claude Code, OpenHands, SWE-Agent) can write code, but they are not guided by the rubric structure that NLPCC evaluators actually grade against. They lack a prior over what "good coverage" means for a scientific reproduction task — they may write a complete training loop but skip reading the paper's limitations section (losing rubric points in Paper Observation), or implement the method without verifying specific numeric claims (losing Result Matching points).

RGSC-Agent gives the agent a head start: a structured `paper_state` extracted from the paper via LLM, and a derived rubric checklist that maps every paper entity to concrete rubric-style items. The agent loop (Phase 3.2 Step 2) uses this checklist as its target, optimizing toward rubric coverage from the start rather than discovering the rubric structure at runtime.

### Scientific questions

This project investigates three questions relevant to PhD research on process-grounded agents:

1. **Process standard derivation.** Can an LLM extract a stable, rubric-aligned process specification (`paper_state`) from a scientific paper, good enough to guide downstream reproduction? (Phase 1)
2. **Process-level verification.** Can an independent LLM judge verify whether partial execution results match paper claims, without access to official rubrics? (Phase 3.1)
3. **Process-level preference alignment.** Does grounding agent behavior in rubric-derived checklists improve rubric score relative to unguided agent execution? (Phase 3.2 → confirmed via Rubric Scorer comparison against Phase 1 baseline of 38–62% derived-checklist recall)

---

## 2. Architecture

```
                  [data/train_valid/{paper}/]
                          │
                          ▼
              ┌──────────────────────┐
              │  Phase 1: Observer   │ ← LLM (DeepSeek V4-Pro)
              │  paper.md → state    │
              └──────────────────────┘
                          │
                          ├─ paper_state.json
                          ├─ derived_checklist.json
                          └─ coverage_diagnostic.json
                          │
                          ▼
              ┌──────────────────────┐
              │  Phase 2: Planner    │ ← LLM
              │  state → task_plan   │
              └──────────────────────┘
                          │
                          └─ task_plan.json
                          │
                          ▼
              ┌─────────────────────────────┐
              │  Phase 3.2 Step 2:           │
              │  Agent Loop (in progress)    │ ← Claude Code + MCP
              │  + paper.md                  │   Action Recorder
              │  + state + checklist         │
              └─────────────────────────────┘
                          │
                          ├─ actions.json
                          └─ reproduced_repo/
                          │
            ┌─────────────┴────────────┐
            ▼                          ▼
  ┌───────────────────┐    ┌─────────────────────┐
  │ Phase 3.1:         │   │ Phase 3.2 Step 1:    │
  │ Verifier (self-    │   │ Rubric Scorer        │
  │ check on results)  │   │ (NLPCC sim)          │
  └───────────────────┘    └─────────────────────┘
            │                          │
            ▼                          ▼
  verification_report.json   rubric_score_report.json
  (Result Matching only)     (all 5 rubric types)
```

### Module table

| Module | Input | Output | Status |
|---|---|---|---|
| Paper Observer | paper.md | paper_state.json | Phase 1 ✓ |
| Rubric Normalizer | paper_state.json | derived_checklist.json | Phase 1 ✓ |
| Stage Planner | paper_state.json + checklist | task_plan.json | Phase 2 ✓ |
| Verifier | paper_state + results.json | verification_report.json | Phase 3.1 ✓ |
| Rubric Scorer | actions.json + repo + rubrics.json | rubric_score_report.json | Phase 3.2 Step 1 ✓ |
| Agent Loop | paper.md + state + checklist | actions.json + repo | Phase 3.2 Step 2 in progress |

### NLPCC rubric type alignment

| Rubric Type | Phase 1 derived from | Scored by |
|---|---|---|
| Paper Observation | paper_state.sections | Rubric Scorer |
| Plan Writing | task_plan.stages | Rubric Scorer |
| Code Implementation | paper_state.atomic_steps (Phase 1 weak point) | Rubric Scorer |
| Command Execution | task_plan.stages | Rubric Scorer |
| Result Matching | paper_state.expected_claims | Verifier (self-check) + Rubric Scorer |

---

## 3. Quick Start

```bash
# 1. Clone
git clone https://github.com/FujunhaoFc/rgsc-agent.git
cd rgsc-agent

# 2. Python 3.11+
python3 --version   # should be >= 3.11

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env to set DEEPSEEK_API_KEY=sk-xxx

# 5. Prepare data
# NLPCC Task 11 dataset (5 train/val papers) is NOT in this repo.
# Obtain from: https://github.com/KOU-199024/NLPCC-2026-Shared-Task-11
# Place under: data/train_valid/{paper_name}/{paper.md, paper.pdf, rubrics.json, images/}

# 6. Verify installation
pytest tests/ -v
# Expected: 135+ test functions, all passing or with documented skips
```

---

## 4. Usage

Each module has an independent CLI entry point. All commands run from the project root.

### Paper Observer (Phase 1)

Extract `paper_state.json` from a paper's markdown:

```bash
# Full extraction (skeleton + detail in one pass)
python pipeline/paper_observer/llm_summarizer.py min-p

# Two-pass extraction (lower risk of truncation for long papers)
python pipeline/paper_observer/llm_summarizer.py min-p skeleton
python pipeline/paper_observer/llm_summarizer.py min-p detail

# Outputs:
#   outputs/min-p/paper_state.json
```

### Rubric Normalizer (Phase 1)

Derive a self-checklist from paper_state (rule-based, no LLM):

```bash
python pipeline/rubric_normalizer/derive_from_state.py min-p

# Outputs:
#   outputs/min-p/derived_checklist.json
```

### Coverage Diagnostic (Phase 1)

Score derived_checklist against ground truth rubrics.json to measure recall:

```bash
python eval/coverage_diagnostic.py min-p
# Or all 5 papers:
python eval/coverage_diagnostic.py AMUN Beyond-Ngram I0T INCLINE min-p

# Outputs:
#   outputs/{paper}/coverage_diagnostic.json
#   outputs/coverage_summary.json
```

### Stage Planner (Phase 2)

Split the derived checklist into an ordered, dependency-chained task plan:

```bash
python pipeline/stage_planner/planner.py min-p

# Outputs:
#   outputs/min-p/task_plan.json
#   outputs/min-p/_debug/llm_response_step5.json  (LLM refinement trace)
```

### Verifier (Phase 3.1)

Judge whether experiment results match paper claims (Result Matching type only):

```bash
python pipeline/verifier/verifier.py min-p

# Outputs:
#   outputs/min-p/verification_report.json
#   outputs/min-p/_debug/llm_response_verifier_{claim_id}.json
```

Requires `results.json` (or `mock_results.json` during development) in `outputs/min-p/`.

### Rubric Scorer (Phase 3.2 Step 1)

Simulate NLPCC grading of an agent submission (all 5 rubric types):

```bash
# Mock mode (no API calls, all verdicts = "hit")
python pipeline/rubric_scorer/scorer.py --dry-run min-p

# Real LLM scoring with resume support (cached rubric verdicts reused)
python pipeline/rubric_scorer/scorer.py min-p

# Force re-run all rubrics, bypassing cache
python pipeline/rubric_scorer/scorer.py --no-resume min-p

# Outputs:
#   outputs/min-p/rubric_score_report.json
#   outputs/min-p/_debug/rubric_scorer/llm_response_rubric_{idx:03d}.json
```

---

## 5. Project Layout

```
rgsc-agent/
├── pipeline/                       # Core modules
│   ├── paper_observer/             # Phase 1: paper.md → paper_state
│   │   ├── section_parser.py       #   Parse paper.md into section tree
│   │   ├── entity_extractor.py     #   Extract tables, figures, equations
│   │   ├── llm_summarizer.py       #   LLM extraction (skeleton + detail)
│   │   └── build_paper_state.py    #   Schema assembly utilities
│   ├── rubric_normalizer/          # Phase 1: paper_state → checklist
│   │   ├── derive_from_state.py    #   Rule-based checklist generation
│   │   ├── anchor_parser.py        #   Section/anchor extraction
│   │   └── normalize.py            #   Checklist normalization
│   ├── stage_planner/              # Phase 2: state → task_plan
│   │   ├── planner.py              #   Rule-based + LLM-refined stage planning
│   │   └── prompts/                #   LLM refinement prompts
│   ├── verifier/                   # Phase 3.1: Result Matching self-check
│   │   ├── verifier.py             #   LLM claim-judge engine
│   │   └── prompts/                #   claim_judge.txt
│   ├── rubric_scorer/              # Phase 3.2 Step 1: NLPCC rubric sim
│   │   ├── scorer.py               #   Main scoring engine
│   │   ├── retrieval.py            #   actions.json per-type filter
│   │   └── prompts/                #   Type-specific judge prompts
│   └── schemas/                    # JSON schemas for all pipeline outputs
├── eval/                           # Evaluation scripts
│   └── coverage_diagnostic.py      # derived_checklist vs official rubrics
├── data/
│   └── train_valid/                # NLPCC training set (not in git)
├── outputs/                        # Per-paper artifacts
├── tests/                          # 135+ pytest test functions
├── docs/                           # Design documents + worklog
│   ├── paper_state_design.md       # Schema decisions, recall targets
│   ├── rubric_scorer_design.md     # Scorer architecture, scoring rules
│   └── worklog.md                  # Engineering journal
├── requirements.txt
├── .env.example
└── README.md
```

---

## 6. Current Results

5 papers (AMUN, Beyond-Ngram, I0T, INCLINE, min-p), 399 rubric items total.

### Phase 1: derived_checklist coverage

| Type                | Target | Actual | Status |
|---------------------|--------|--------|--------|
| Paper Observation   | ≥ 90%  | 98.1%  | ✓      |
| Result Matching     | ≥ 80%  | 100.0% | ✓      |
| Plan Writing        | ≥ 50%  | 48.1%  | below  |
| Code Implementation | ≥ 50%  | 42.1%  | below  |
| Command Execution   | ≥ 40%  | 30.4%  | below  |
| **Overall**         | **≥ 60%** | **55.0%** | below |

Plan Writing and Code Implementation are the two weakest categories — these items require semantic understanding of atomic method steps, which the current prompt-based extraction struggles with at the granularity NLPCC rubrics demand. Closing this gap is the main objective of Phase 3.2 Step 2 (Agent Loop), which can cover ground the derived checklist misses through autonomous code generation and execution.

### Phase 3.1: Verifier mock validation

- All 5 papers passed mock end-to-end validation with overall score ≥ 0.929.
- AMUN, Beyond-Ngram, I0T, INCLINE: overall = 1.0.
- min-p: overall = 0.929 (6 pass / 1 partial).
- Verifier serves a dual role: Result Matching self-check at agent runtime, and post-hoc Paper Observer quality audit (see `outputs/AMUN/_review_notes.md`).

### Phase 3.2 Step 1: Rubric Scorer sanity

- All 5 papers scored on empty `actions.json` + empty `reproduced_repo/` → `estimated_recall = 0.0`.
- 399 LLM judge calls, token cost ~$0.14, zero hallucinations (Scorer correctly assigns "miss" when no evidence exists).

Full per-paper breakdowns in `outputs/{paper}/coverage_diagnostic.json`.

---

## 7. Tech Stack

- **Language**: Python 3.11+
- **LLM**: DeepSeek V4-Pro via OpenAI-compatible API (`base_url=https://api.deepseek.com/v1`)
- **Schemas**: jsonschema (Draft7)
- **Validation**: pydantic ≥ 2.5
- **Tests**: pytest (135 test functions, 191 collected; 70+ pass, some gated behind live API key)
- **Agent framework** (Phase 3.2 Step 2): Claude Code + MCP Action Recorder
- **LLM config**: max_tokens 4000 (Verifier/Scorer), 8000 (Stage Planner Step 5); temperature 0.2–0.3; retry strategy: 3 attempts with combined prompt + complete schema template on malformed JSON

---

## 8. Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Paper Observer + derived checklist + coverage diagnostic | ✓ complete |
| Phase 2 | Stage Planner (rule-based + LLM-refined task_plan) | ✓ complete |
| Phase 3.1 | Verifier (Result Matching LLM judge) | ✓ complete |
| Phase 3.2 Step 1 | Rubric Scorer (NLPCC simulator, empty-input sanity verified) | ✓ complete |
| Phase 3.2 Step 2 | Agent Loop — Claude Code + MCP Action Recorder on min-p | **next** |
| Phase 3.3 | Debugger module (auto-retry / code repair on execution failure) | planned |
| Phase 3.4–3.6 | Scale to all 5 train papers + benchmark vs PaperBench / ScienceAgentBench | planned |

The next milestone (Phase 3.2 Step 2) is to configure Claude Code with the MCP Action Recorder, run the agent loop on a single paper (min-p), and produce a real `actions.json`. The key measurement will be rubric_score_report.estimated_recall vs the Phase 1 derived_checklist baseline of 38–62% — the delta represents the agent's autonomous contribution beyond what the checklist captures.

---

## 9. References

- **NLPCC 2026 Shared Task 11**: [https://github.com/KOU-199024/NLPCC-2026-Shared-Task-11](https://github.com/KOU-199024/NLPCC-2026-Shared-Task-11) — official task definition, dataset, and evaluation protocol.
- **MCP Action Recorder**: documented in the Task 11 repository README — the MCP server that records all agent tool calls as structured `actions.json`.
- **PaperBench** (OpenAI, 2025): a benchmark for AI agents that reproduce ML experiments end-to-end. RGSC-Agent targets a compatible but rubric-oriented variant of the reproduction task.
- **ScienceAgentBench** (CMU, 2025): evaluates language agents on scientific discovery tasks including code generation and data analysis.
- **AutoReproduce**: a line of work on automatic reproduction of computational experiments; RGSC-Agent adds rubric grounding as the distinguishing contribution.
