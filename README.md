# RGSC-Agent

**A rubric-grounded scientific code agent for reproducing computational
experiments from papers. Ranked first on NLPCC 2026 Shared Task 11.**

[![leaderboard](https://img.shields.io/badge/NLPCC%202026%20Task%2011-Rank%201-brightgreen)]()
[![replication score](https://img.shields.io/badge/Replication%20Score-49.64%25-blue)]()
[![baseline gap](https://img.shields.io/badge/vs%20GPT--5.5%20baseline-%2B25.45pp-orange)]()

## NLPCC 2026 Shared Task 11 results

Our system, submitted as **YNU-HPCC-Task11-AgentRep**, ranked first on the
official leaderboard for [NLPCC 2026 Shared Task 11 (Agent-Based Experiment
Reproduction from Scientific
Papers)](https://github.com/KOU-199024/NLPCC-2026-Shared-Task-11):

| Rank | Team                                | Replication Score |
| ---- | ----------------------------------- | ----------------- |
| 1    | **YNU-HPCC-Task11-AgentRep (ours)** | **49.64%**        |
| 2    | zzunlp_wu                           | 24.70%            |
| 3    | QueenAgent                          | 7.03%             |
| —    | Codex-GPT5.5-Medium (baseline)      | 24.19%            |

Our system exceeds the organizer-reported Codex + GPT-5.5 baseline by 25.45
absolute percentage points and more than doubles the second-ranked team's
score. On a training-set self-evaluation using an LLM judge over 12 papers,
the same pipeline attains 42.0% aggregate score recall, with per-rubric-type
recall of **Paper Observation 71.8%**, **Plan Writing 86.4%**,
**Code Implementation 49.9%**.

Details, including per-paper scores, rubric-type breakdown, and Phase B
toolkit ablation, are reported in our NLPCC 2026 workshop paper (Section 5).

## The problem

Reproducing experimental pipelines from scientific papers is hard: papers
vary widely in length, formalism, and infrastructure assumptions, and a
reliable reproduction requires both broad paper understanding and disciplined
execution of long tool-use chains. Recent benchmarks — PaperBench, SciReplicate-Bench —
score LLM agents on end-to-end reproduction and reveal a large capability
gap between free-form agents and human researchers.

NLPCC 2026 Shared Task 11 formalizes this problem: given a paper, an agent
must produce a reproduction plan, code, and results, graded against a hidden
rubric across five types (Paper Observation, Plan Writing, Code
Implementation, Command Execution, Result Matching). Every action must be
logged through an official MCP action-recorder for evaluation.

## RGSC-Agent in one screen

RGSC-Agent (**R**ubric-**G**rounded **S**cientific **C**ode Agent) is a
structured six-phase pipeline built on Claude Code with DeepSeek V4-Pro as
the backbone LLM.

For each paper, the agent runs:

| Phase | Action | Output |
| ----- | ------ | ------ |
| A     | Read `paper.md` in ~5 targeted chunks (method, setup, datasets, baselines) | Internal paper understanding |
| B     | Invoke three Python toolkit modules via `execute_cmd` | `paper_state.json`, `derived_checklist.json`, `task_plan.json` |
| C     | Write `plan.md` addressing the derived checklist | Structured reproduction plan |
| D     | Write a Python code skeleton under `src/` | Method implementation stubs |
| E     | Execute experiments (skipped in skim mode; marked NOT REPRODUCED) | `results.md` |
| G     | Call `export_log` to persist the action trace | `log/actions.json` |

Under strict hardware constraints (single RTX 5090, 32 GB VRAM) and a ten-day
window, we adopt a breadth-first **skim mode**: cover all 138 test papers
with Phases A–D + G, honestly marking Phase E as NOT REPRODUCED rather than
fabricating results. Two design choices push results well past the free-form
agent baseline:

- **Parallel triage** dispatches seven Claude Code sub-agents to score all
  138 papers along five axes (reproducibility, scope, hardware, infra,
  length) before commitment — see `triage_report.md`.
- **Toolkit-mediated Phase B** produces structured JSON artifacts before
  free-form planning. On the training-set self-evaluation, Plan Writing
  reaches 86.4% score recall, and ablating `derived_checklist.json` drops
  aggregate recall by 4.6 pp.

## Repository layout

```
rgsc-agent/
├── agent-workspace/          # Per-paper workspace template and launcher scripts
│   ├── _template/            #   CLAUDE.md (v5, 13 KB), .mcp.json, .claude/
│   ├── setup_paper.sh        #   Build a workspace from the template
│   ├── batch_skim.sh         #   Launch a per-paper skim-mode session in tmux
│   └── batch_ablation.sh     #   Launch Phase-B toolkit ablation runs
├── pipeline/                 # Phase B toolkit
│   ├── paper_observer/       #   Extract structured paper_state.json
│   ├── rubric_normalizer/    #   Derive self-rubric checklist
│   ├── stage_planner/        #   Build stage-wise task plan
│   └── common/paths.py       #   Split-aware paper-id resolution
├── evaluation/               # LLM-judge rubric evaluator + ablation harness
│   ├── rubric_evaluator.py   #   454 lines, parallel LLM judge, --train_dir / --workspace_root / --output_dir CLI
│   └── evaluate_ablation.py  #   148 lines, evaluates 3 papers × 4 configs
├── figures/                  # matplotlib scripts to regenerate paper figures
│   ├── make_figures.py       #   fig3, fig4 (initial), fig6
│   ├── make_figure4_v2.py    #   fig4 with updated data
│   ├── make_figure5.py       #   fig5 ablation
│   └── make_figure2.py       #   fig2 workspace layout
├── results/                  # Aggregate JSON, safe to publish
│   ├── test_138_action_stats.json
│   ├── train_12_rubric_structure.json
│   ├── train_self_eval_aggregate.json
│   ├── paper_final_stats.json
│   └── ablation_summary.json
├── paper/figures/            # Rendered PNG figures
├── data/
│   ├── train_valid/          # 12 official training-set papers (flat + nested)
│   └── official_tools/       # MCP action-recorder from the organizer
├── triage_report.md          # 138-paper 5-axis triage
└── README.md
```

The 138-paper test set is not distributed (organizer-restricted). Agent
outputs (`outputs/`, `agent-workspace/<paper>/`) are per-run artifacts and
are also excluded from tracking.

## Setup

- Python 3.13 or later.
- DeepSeek API key (`DEEPSEEK_API_KEY`) for the toolkit LLM calls and the
  rubric evaluator.
- Claude Code (or a compatible MCP client) installed and reachable on `PATH`
  as `claude` for running the agent loop.
- The MCP action-recorder tool provided by the organizer under
  `data/official_tools/record_tools.py`.

```bash
git clone https://github.com/FujunhaoFc/rgsc-agent.git
cd rgsc-agent
python -m venv .venv && source .venv/bin/activate
pip install -e .
export DEEPSEEK_API_KEY=sk-...
```

Paths in `agent-workspace/_template/.mcp.json` assume the repo lives at
`/root/rgsc-agent` (our development environment). If you clone elsewhere,
adjust the `--work-dir` argument in that file and in
`agent-workspace/setup_paper.sh` accordingly.

## Running the pipeline

**Skim mode on a single paper** (~15 minutes):

```bash
./agent-workspace/batch_skim.sh AMUN
tmux attach -t skim-AMUN            # optional: watch the session
```

Outputs land in `~/agent-workspace/AMUN/`:
- `log/actions.json` (action trace via MCP action-recorder)
- `plan.md` (reproduction plan)
- `results.md` (honest reporting; NOT REPRODUCED with reason)
- `src/*.py` (method implementation skeleton)

**Batch mode** (loop over a list of papers):

```bash
for p in AMUN INCLINE min-p; do
    ./agent-workspace/batch_skim.sh $p
    # wait for tmux session to finish before next
done
```

**Phase B toolkit ablation** (produces the data in Section 5.4 of the paper):

```bash
for cfg in Full NoStagePlanner NoRubricNormalizer NoPhaseB; do
    ./agent-workspace/batch_ablation.sh AMUN $cfg
done
```

## Reproducing the reported numbers

Once workspaces exist, the rubric evaluator produces the per-type recall
tables in Sections 5.2 and 5.4 of the paper. It uses DeepSeek V4-Pro as an
impartial LLM judge:

```bash
# Training-set self-evaluation (Section 5.2)
python evaluation/rubric_evaluator.py \
    --train_dir data/train_valid \
    --workspace_root ~/agent-workspace \
    --output_dir ./eval_results \
    --papers AMUN CIForm gated-attention GSL-MPP I0T INCLINE \
             min-p n-grams REPAIR SCoRe ActorAttack representation-political-llm

# Phase B toolkit ablation (Section 5.4)
python evaluation/evaluate_ablation.py \
    --papers gated-attention AMUN min-p
```

The evaluator produces `<paper>_eval.json` per paper and an aggregate
`aggregate.json`. Aggregate JSONs are also included pre-computed under
`results/`.

**Regenerating the figures** (from the aggregate JSONs already in
`results/`):

```bash
python figures/make_figures.py       # fig3, fig4, fig6
python figures/make_figure4_v2.py    # fig4 with updated data
python figures/make_figure5.py       # fig5 ablation
python figures/make_figure2.py       # fig2 workspace layout
```

PNGs write to `paper/figures/` by default; set `RGSC_FIGURES_OUT` to
override.

## Design notes: honest NOT REPRODUCED

Under Constraints 4 and 5 of the task rubric, participants must not
fabricate results. When we cannot actually run an experiment — because the
paper needs a proprietary dataset, an 8×A100 training budget, or a gated
model — we mark it as NOT REPRODUCED with a concrete reason and leave the
implementation skeleton in place. All 138 test-set papers and all 12
training-set papers use this policy. This design deliberately caps our
theoretical maximum score at the sum of Paper Observation, Plan Writing, and
Code Implementation weights (64.6% under the training-set rubric
distribution). Our test-set result of 49.64% corresponds to 76.8% of that
upper bound.

## Citation

The NLPCC 2026 workshop paper describing this system is under preparation.
A BibTeX entry and PDF link will be added here on publication.

If you use RGSC-Agent in your research, please cite the NLPCC 2026 shared
task overview paper and this repository.

## License

MIT. See `LICENSE`.

## Team

- **System Name (ID)**: YNU-HPCC-Task11-AgentRep
- **Team Leader**: Junhao Fu (`fujunhaofc@outlook.com`)
- **Affiliations**: HPCC Lab, School of Information Science and Engineering,
  Yunnan University; School of Information Engineering, Yunnan Jiaotong
  Polytechnic University.
