#!/bin/bash
# Usage: ./batch_skim.sh <paper_id>
# Runs a paper in "skim mode" — fast (~15 min) Phase A-D + NOT REPRODUCED for Phase E.
# Uses tmux for SSH-disconnect resilience.

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <paper_id>"
    exit 1
fi

PAPER_ID=$1
WORKSPACE=$HOME/agent-workspace/$PAPER_ID
SESSION_NAME="skim-$PAPER_ID"

# Step 1: Setup workspace if not exists
if [ ! -d "$WORKSPACE" ]; then
    echo "Setting up workspace for $PAPER_ID..."
    $HOME/agent-workspace/setup_paper.sh "$PAPER_ID"
else
    echo "Workspace already exists at $WORKSPACE (using existing)"
fi

# Step 2: Check tmux session not running
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION_NAME' already exists"
    echo "  - to attach:  tmux attach -t $SESSION_NAME"
    echo "  - to kill:    tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Step 3: Build the Agent prompt for skim mode
PROMPT_FILE=$WORKSPACE/.skim_prompt.txt
cat > "$PROMPT_FILE" << 'PROMPT'
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

Phase B — Invoke toolkit (record paper_state + checklist + plan)
- execute_cmd: `cd /root/rgsc-agent && PYTHONPATH=. python pipeline/paper_observer/llm_summarizer.py {paper_id}`
- execute_cmd: `cd /root/rgsc-agent && PYTHONPATH=. python pipeline/rubric_normalizer/derive_from_state.py {paper_id}`
- execute_cmd: `cd /root/rgsc-agent && PYTHONPATH=. python pipeline/stage_planner/planner.py {paper_id}`
- Time: 3-5 minutes (DeepSeek API calls)

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

# Substitute paper_id in the prompt
sed -i "s|{paper_id}|$PAPER_ID|g" "$PROMPT_FILE"

# Step 4: Start tmux session, source env, cd workspace, start claude
echo ""
echo "Starting tmux session '$SESSION_NAME'..."
echo "  - to attach:  tmux attach -t $SESSION_NAME"
echo "  - to detach:  Ctrl+B then D"
echo ""

tmux new-session -d -s "$SESSION_NAME" -c "$WORKSPACE" "
    source /root/setup_env.sh
    echo ''
    echo '======================================'
    echo 'Starting skim mode for paper: $PAPER_ID'
    echo 'Workspace: $WORKSPACE'
    echo '======================================'
    echo ''
    ( echo "1"; sleep 2; cat $PROMPT_FILE ) | claude
    echo ''
    echo '======================================'
    echo 'Agent finished for $PAPER_ID'
    echo '======================================'
    sleep 60
"

sleep 2
echo "tmux session started. Monitor with: tmux attach -t $SESSION_NAME"
