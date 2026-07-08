"""
Regenerate Figure 4 with updated 12-paper data (INCLINE rerun + v5 template).

Reads results/paper_final_stats.json which was updated after 12-paper
evaluator rerun.

Usage:
    python make_figure4_v2.py

Output:
    paper/figures/fig4_per_paper_recall.png  (or $RGSC_FIGURES_OUT)
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent

OUT_DIR = Path(os.environ.get("RGSC_FIGURES_OUT", str(_REPO_ROOT / "paper" / "figures")))
OUT_DIR.mkdir(exist_ok=True)

STATS = json.load(open(_REPO_ROOT / "results" / "paper_final_stats.json"))

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

per_paper = STATS["per_paper_recall"]
# Sort ascending so highest bar at top
sorted_items = sorted(per_paper.items(), key=lambda x: x[1])
names = [n for n, _ in sorted_items]
recalls = [r * 100 for _, r in sorted_items]

fig, ax = plt.subplots(figsize=(6.5, 4.2))

train_agg = STATS["train_self_eval"]["aggregate_score_recall"] * 100

# Color: green if above aggregate, red if below
bar_colors = [
    "#55A868" if r >= train_agg else "#C44E52"
    for r in recalls
]

bars = ax.barh(names, recalls, color=bar_colors, edgecolor="black", linewidth=0.5)

for bar, r in zip(bars, recalls):
    ax.text(
        bar.get_width() + 0.7, bar.get_y() + bar.get_height() / 2,
        f"{r:.1f}%",
        va="center", fontsize=8,
    )

# Reference lines
ax.axvline(train_agg, color="#4C72B0", linestyle="--", linewidth=1.3,
           label=f"Train aggregate = {train_agg:.1f}%")
test_score = STATS["test_official"]["replication_score"] * 100
ax.axvline(test_score, color="black", linestyle="-", linewidth=1.3,
           label=f"Test leaderboard = {test_score:.2f}%")

skim_ub = 64.6
ax.axvline(skim_ub, color="gray", linestyle=":", linewidth=1.0,
           label=f"Skim-mode upper bound = {skim_ub:.1f}%")

ax.set_xlabel("Score recall (%)")
ax.set_xlim(0, 70)
ax.set_title("Per-paper score recall on the 12-paper training set")
ax.legend(loc="lower right", framealpha=0.95)
ax.grid(axis="x", alpha=0.3)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

out = OUT_DIR / "fig4_per_paper_recall.png"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
print(f"\nUpdated data snapshot:")
print(f"  Train aggregate: {train_agg:.2f}%")
print(f"  Test leaderboard: {test_score:.2f}%")
print(f"  Papers (n={len(recalls)}):")
for name, r in sorted_items:
    print(f"    {name:<35} {r*100:>5.1f}%")
