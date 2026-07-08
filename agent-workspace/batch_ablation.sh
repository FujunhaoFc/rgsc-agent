#!/bin/bash
# batch_ablation.sh - Run a paper in a Phase-B ablation config.
#
# Usage: ./batch_ablation.sh <paper_id> <config>
#   config: Full | NoStagePlanner | NoRubricNormalizer | NoPhaseB
#
# Builds workspace at ~/agent-workspace-ablation/<paper_id>_<config>/
# and starts a tmux session 'abl-<paper_id>-<config>'.
#
# CRITICAL: prompt is byte-identical to batch_skim.sh's .skim_prompt.txt except
# for the Phase B block, which varies by ablation config. This ensures Full
# baseline reproduces the same behavior as the training-set self-eval runs.

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <paper_id> <config>"
    echo "Configs: Full, NoStagePlanner, NoRubricNormalizer, NoPhaseB"
    exit 1
fi

PAPER_ID=$1
CONFIG=$2
REPO=/root/rgsc-agent
TEMPLATE=$HOME/agent-workspace/_template
WORKSPACE=$HOME/agent-workspace-ablation/${PAPER_ID}_${CONFIG}
SESSION_NAME="abl-${PAPER_ID}-${CONFIG}"

# Validate config
case "$CONFIG" in
    Full|NoStagePlanner|NoRubricNormalizer|NoPhaseB) ;;
    *) echo "ERROR: unknown config '$CONFIG'"; exit 1 ;;
esac

# Find paper.md
PAPER_MD=""
for split in test train_valid; do
    for depth in 2 1; do
        candidate=$(find -L "$REPO/data/$split" -mindepth $depth -maxdepth $depth -type d -name "${PAPER_ID}_${CONFIG}" 2>/dev/null | head -1)
        if [ -n "$candidate" ] && [ -f "$candidate/paper.md" ]; then
            PAPER_MD="$candidate/paper.md"
            break 2
        fi
    done
done
if [ -z "$PAPER_MD" ]; then
    echo "ERROR: paper.md not found for '${PAPER_ID}_${CONFIG}' (need symlink from data/train_valid/)"
    exit 1
fi

# Step 1: Build workspace (fresh)
if [ -d "$WORKSPACE" ]; then
    echo "Removing existing workspace: $WORKSPACE"
    rm -rf "$WORKSPACE"
fi

echo "Setting up workspace $WORKSPACE..."
cp -r "$TEMPLATE" "$WORKSPACE"

# Substitute paper_id (setup_paper.sh convention)
sed -i "s|__PAPER_ID__|${PAPER_ID}_${CONFIG}|g" "$WORKSPACE/CLAUDE.md"
sed -i "s|__PAPER_ID__|${PAPER_ID}_${CONFIG}|g" "$WORKSPACE/.mcp.json"

# Force work-dir to ablation workspace
python3 -c "
import json
with open('$WORKSPACE/.mcp.json') as f:
    d = json.load(f)
d['mcpServers']['action-recorder']['args'] = [
    '/root/rgsc-agent/data/official_tools/record_tools.py',
    '--work-dir', '$WORKSPACE',
]
with open('$WORKSPACE/.mcp.json', 'w') as f:
    json.dump(d, f, indent=2)
print('Updated .mcp.json --work-dir to:', '$WORKSPACE')
"

cp "$PAPER_MD" "$WORKSPACE/paper.md"
mkdir -p "$WORKSPACE/log"

# Verify no leftover placeholders
if grep -l "__PAPER_ID__" "$WORKSPACE/CLAUDE.md" "$WORKSPACE/.mcp.json" 2>/dev/null; then
    echo "ERROR: __PAPER_ID__ placeholder still present"
    exit 1
fi

# Step 2: Check tmux session not running
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION_NAME' already exists"
    exit 1
fi

# Step 3: Build the prompt — BYTE-IDENTICAL to batch_skim.sh except for Phase B
PROMPT_FILE=$WORKSPACE/.ablation_prompt.txt

# Phase B block varies by config
case "$CONFIG" in
    Full)
        PHASE_B_BLOCK='Phase B — Invoke toolkit (record paper_state + checklist + plan)
- execute_cmd: `cd /root/rgsc-agent && PYTHONPATH=. python pipeline/paper_observer/llm_summarizer.py {paper_id}`
- execute_cmd: `cd /root/rgsc-agent && PYTHONPATH=. python pipeline/rubric_normalizer/derive_from_state.py {paper_id}`
- execute_cmd: `cd /root/rgsc-agent && PYTHONPATH=. python pipeline/stage_planner/planner.py {paper_id}`
- Time: 3-5 minutes (DeepSeek API calls)'
        ;;
    NoStagePlanner)
        PHASE_B_BLOCK='Phase B — Invoke toolkit (record paper_state + checklist ONLY, SKIP stage_planner)
- execute_cmd: `cd /root/rgsc-agent && PYTHONPATH=. python pipeline/paper_observer/llm_summarizer.py {paper_id}`
- execute_cmd: `cd /root/rgsc-agent && PYTHONPATH=. python pipeline/rubric_normalizer/derive_from_state.py {paper_id}`
- DO NOT run stage_planner in this ablation run.
- Time: 2-3 minutes (DeepSeek API calls)'
        ;;
    NoRubricNormalizer)
        PHASE_B_BLOCK='Phase B — Invoke toolkit (paper_observer ONLY, SKIP rubric_normalizer AND stage_planner)
- execute_cmd: `cd /root/rgsc-agent && PYTHONPATH=. python pipeline/paper_observer/llm_summarizer.py {paper_id}`
- DO NOT run rubric_normalizer in this ablation run.
- DO NOT run stage_planner in this ablation run.
- Time: 1-2 minutes (DeepSeek API call)'
        ;;
    NoPhaseB)
        PHASE_B_BLOCK='Phase B — SKIPPED in this ablation.
- Do not invoke any toolkit module (paper_observer, rubric_normalizer, stage_planner).
- Proceed directly to Phase C after Phase A.
- Time: 0 minutes'
        ;;
esac

# Build prompt — copied verbatim from batch_skim.sh, with $PHASE_B_BLOCK spliced in
cat > "$PROMPT_FILE" << PROMPT
This is SKIM MODE for NLPCC Task 11 test set submission.

Time budget: 15 MINUTES MAX. We are submitting 138 papers in 8 days; full reproduction of every paper is impossible.

CRITICAL CONSTRAINT — DO NOT MODIFY data/:
  - Do NOT write, copy, symlink, or modify ANYTHING under /root/rgsc-agent/data/.
  - This directory holds the canonical test set. Modifying it contaminates the experiment.
  - The toolkit (paper_observer/rubric_normalizer/stage_planner) finds the paper automatically by paper_id — you do not need to "place" the paper anywhere.
  - If a tool errors citing missing paper file, that is a TOOL bug, not your concern. Move to next phase or mark NOT REPRODUCED.
  - All your outputs must go to the workspace (current dir) or /root/rgsc-agent/outputs/, never /root/rgsc-agent/data/.

YOUR GOAL: Maximize Scorer credit on Paper Observation + Plan Writing + Code Implementation rubrics. SKIP Result Matching and Command Execution experiments entirely.

EXECUTE THESE PHASES IN ORDER:

Phase A — Read paper.md (5 chunks max via read_file with start_offset/end_offset)
- Focus: paper's METHOD, EXPERIMENTAL SETUP, DATASETS, BASELINES
- Skip: related work, appendix, deep math derivations
- Time: 3 minutes

$PHASE_B_BLOCK

Phase C — Write plan.md (covers ALL Plan Writing checklist items)
- Read /root/rgsc-agent/outputs/{paper_id}/derived_checklist.json to see what's required
- Write plan.md addressing each Plan Writing rubric item
- Time: 2 minutes

Phase D — Write src/ skeleton (paper-aligned code structure)
- Files matching paper's method: e.g., model.py, train.py, eval.py
- Each file: minimal functional code stub showing the paper's algorithm/method
- DO NOT load datasets, DO NOT load models, DO NOT run experiments
- Time: 3 minutes

Phase E — SKIP experiments. Write results.md noting:
"NOT REPRODUCED — Skim mode submission under NLPCC 8-day deadline. Full experimental reproduction would require [list specific datasets/models/compute needed]. Per Constraint 4+5 of the task: experimental results not fabricated, marked NOT REPRODUCED."
- For each Table/Figure of experimental results in the paper, write a separate NOT REPRODUCED entry with: status, reason, partial evidence (which src/ files implement the relevant method).
- Time: 1 minute

Phase F — Skip verification step (no experiments to verify)

Phase G — IMMEDIATELY call action-recorder export_log
- This writes log/actions.json to disk
- DO NOT skip this — without export_log all Paper Observation evidence is lost
- Time: 30 seconds

TOTAL BUDGET: 15 minutes. If you exceed 20 minutes, stop wherever you are and call export_log.

Begin Phase A now.
PROMPT

# Substitute {paper_id} (the toolkit param) — use ablation paper_id with _CONFIG suffix
sed -i "s|{paper_id}|${PAPER_ID}_${CONFIG}|g" "$PROMPT_FILE"

# Step 4: Start tmux
echo ""
echo "Starting tmux session '$SESSION_NAME'..."
echo "  attach: tmux attach -t $SESSION_NAME"
echo "  log:    tail -f /tmp/abl_${PAPER_ID}_${CONFIG}.log"
echo ""

tmux new-session -d -s "$SESSION_NAME" -c "$WORKSPACE" "
    source /root/setup_env.sh
    echo ''
    echo '======================================'
    echo 'Ablation: $PAPER_ID / $CONFIG'
    echo 'Workspace: $WORKSPACE'
    echo '======================================'
    echo ''
    ( echo '1'; sleep 2; cat $PROMPT_FILE ) | claude 2>&1 | tee /tmp/abl_${PAPER_ID}_${CONFIG}.log
    echo ''
    echo '======================================'
    echo 'Ablation session finished'
    echo '======================================'
    sleep 3600
"

sleep 2
echo "tmux session '$SESSION_NAME' started."
