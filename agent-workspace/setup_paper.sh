#!/bin/bash
# Usage: ./setup_paper.sh <paper_id>
# Recursively searches data/test/ then data/train_valid/ for <paper_id>/paper.md

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <paper_id>"
    echo "Example: $0 SAFM"
    exit 1
fi

PAPER_ID=$1
WORKSPACE_ROOT=$HOME/agent-workspace
TEMPLATE_DIR=$WORKSPACE_ROOT/_template
TARGET_DIR=$WORKSPACE_ROOT/$PAPER_ID
REPO_DIR=/root/rgsc-agent

# Refuse to overwrite existing workspace
if [ -e "$TARGET_DIR" ]; then
    echo "ERROR: workspace already exists at $TARGET_DIR"
    echo "Delete it first if you want to recreate: rm -rf $TARGET_DIR"
    exit 1
fi

# Find paper.md by recursive search (test first since we're in submission mode, then train_valid)
PAPER_MD=""
PAPER_CATEGORY=""
PAPER_SPLIT=""

for split in test train_valid; do
    # find at depth exactly 2 (<category>/<paper>/paper.md) and depth 1 (<paper>/paper.md)
    for depth in 2 1; do
        candidate=$(find "$REPO_DIR/data/$split" -mindepth $depth -maxdepth $depth -type d -name "$PAPER_ID" 2>/dev/null | head -1)
        if [ -n "$candidate" ] && [ -f "$candidate/paper.md" ]; then
            PAPER_MD="$candidate/paper.md"
            PAPER_SPLIT=$split
            # Extract category (parent dir of paper dir, or '' if at split root)
            parent=$(basename "$(dirname "$candidate")")
            if [ "$parent" != "$split" ]; then
                PAPER_CATEGORY=$parent
            fi
            break 2
        fi
    done
done

if [ -z "$PAPER_MD" ]; then
    echo "ERROR: paper.md not found for '$PAPER_ID'"
    echo "Searched (depth 1 and 2):"
    echo "  $REPO_DIR/data/test/"
    echo "  $REPO_DIR/data/train_valid/"
    exit 1
fi

echo "Found paper: $PAPER_MD"
echo "  split: $PAPER_SPLIT"
echo "  category: ${PAPER_CATEGORY:-<root>}"

# Copy _template to TARGET_DIR
cp -r $TEMPLATE_DIR $TARGET_DIR

# Substitute __PAPER_ID__ in CLAUDE.md and .mcp.json
sed -i "s|__PAPER_ID__|$PAPER_ID|g" $TARGET_DIR/CLAUDE.md
sed -i "s|__PAPER_ID__|$PAPER_ID|g" $TARGET_DIR/.mcp.json

# Copy paper.md
cp $PAPER_MD $TARGET_DIR/paper.md

# Create log dir
mkdir -p $TARGET_DIR/log

# Verify no leftover placeholders
LEFTOVER=$(grep -l "__PAPER_ID__" $TARGET_DIR/CLAUDE.md $TARGET_DIR/.mcp.json 2>/dev/null || true)
if [ -n "$LEFTOVER" ]; then
    echo "WARNING: __PAPER_ID__ placeholder still found in:"
    echo "$LEFTOVER"
    exit 1
fi

echo ""
echo "Workspace ready at $TARGET_DIR"
echo "  paper.md: $(wc -c < $TARGET_DIR/paper.md) bytes"
echo "  CLAUDE.md: $(wc -l < $TARGET_DIR/CLAUDE.md) lines"
echo "  log/: empty"
echo ""
echo "To start the Agent:"
echo "  cd $TARGET_DIR"
echo "  claude"
