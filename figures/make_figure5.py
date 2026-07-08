"""
Generate Figure 5: Phase B toolkit ablation grouped bar chart.
Outputs paper/figures/fig5_ablation.png (or $RGSC_FIGURES_OUT).

Usage:
    python make_figure5.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

OUT_DIR = Path(os.environ.get("RGSC_FIGURES_OUT", str(_REPO_ROOT / "paper" / "figures")))
OUT_DIR.mkdir(exist_ok=True)

# Data from evaluate_ablation.py output
# Columns match: Full, NoStagePlanner, NoRubricNormalizer, NoPhaseB
DATA = {
    "AMUN":              [61.4, 54.4, 57.9, 60.1],
    "min-p":             [41.2, 44.1, 37.5, 23.2],
    "gated-attention":   [27.0, 27.9, 17.2, 18.0],
    "Aggregate (avg)":   [43.2, 42.1, 37.5, 33.8],
}

CONFIGS = ["Full", "NoStagePlanner", "NoRubricNormalizer", "NoPhaseB"]
# Consistent colors: Full = green (best), degrading to red
COLORS = ["#2E7D32", "#66BB6A", "#EF9A9A", "#C62828"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(7.0, 4.0))

papers = list(DATA.keys())
n_papers = len(papers)
n_configs = len(CONFIGS)
x = np.arange(n_papers)
bar_width = 0.20

for i, (config, color) in enumerate(zip(CONFIGS, COLORS)):
    values = [DATA[p][i] for p in papers]
    offset = (i - (n_configs - 1) / 2) * bar_width
    bars = ax.bar(x + offset, values, bar_width, label=config,
                   color=color, edgecolor="black", linewidth=0.5)
    # Value labels above bars
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, rotation=0)

# Vertical separator between per-paper and aggregate
ax.axvline(x=n_papers - 1.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(papers, rotation=0)
ax.set_ylabel("Score recall (%)")
ax.set_ylim(0, 75)
ax.set_title("Phase B toolkit ablation across three training-set papers")
ax.legend(loc="upper right", ncol=2, framealpha=0.95, columnspacing=0.8)
ax.grid(axis="y", alpha=0.3)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# Annotation: aggregate columns are averages
ax.annotate("(mean over 3 papers)",
            xy=(3, 5), xytext=(3, 5),
            ha="center", va="bottom", fontsize=8, style="italic", alpha=0.7)

plt.tight_layout()
out = OUT_DIR / "fig5_ablation.png"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
