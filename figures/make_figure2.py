"""
Generate Figure 2: Per-paper workspace layout as a monospace text figure.
Outputs paper/figures/fig2_workspace_layout.png (or $RGSC_FIGURES_OUT).

Usage:
    python make_figure2.py

Note: if you'd rather insert this as a monospace text box directly in Word,
you don't need this figure — just copy the ASCII tree into a text frame
with Consolas / Courier New font. This script is only for uniform figure style.
"""

import os
import matplotlib.pyplot as plt
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

OUT_DIR = Path(os.environ.get("RGSC_FIGURES_OUT", str(_REPO_ROOT / "paper" / "figures")))
OUT_DIR.mkdir(exist_ok=True)

TREE = r"""~/agent-workspace/<paper_id>/
├── CLAUDE.md                    # system prompt (13 KB, identical across papers)
├── .mcp.json                    # MCP server config (points at action-recorder)
├── .claude/
│   └── settings.local.json      # tool allow-list (mcp__action-recorder only)
├── paper.md                     # input paper (copied from data/train_valid/)
├── log/
│   └── actions.json             # avg 305 KB, 24.8 actions per paper
├── plan.md                      # avg 9.9 KB
├── results.md                   # avg 7.6 KB, "NOT REPRODUCED" reports
└── src/
    └── *.py                     # avg 8.6 files, 1,531 lines of Python
"""

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(6.8, 3.6))
ax.text(0.02, 0.98, TREE,
        family="monospace",
        fontsize=8.5,
        verticalalignment="top",
        horizontalalignment="left",
        transform=ax.transAxes)
ax.axis("off")
# Add a subtle border box around the whole figure
for spine in ax.spines.values():
    spine.set_visible(False)
fig.patch.set_edgecolor("gray")
fig.patch.set_linewidth(0.5)

plt.tight_layout()
out = OUT_DIR / "fig2_workspace_layout.png"
fig.savefig(out, edgecolor=fig.get_edgecolor())
plt.close(fig)
print(f"Saved {out}")
