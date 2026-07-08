"""
Generate paper figures from real data on autodl.

Outputs high-resolution PNGs to paper/figures/ (or RGSC_FIGURES_OUT env).
Uses matplotlib only. Font sizes tuned for insertion into a 2-column-ish
Springer LNAI docx layout at ~4-6 inch wide.

Usage:
    pip install matplotlib
    python make_figures.py
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Output dir
OUT_DIR = Path(os.environ.get("RGSC_FIGURES_OUT", str(_REPO_ROOT / "paper" / "figures")))
OUT_DIR.mkdir(exist_ok=True)

# Load data
STATS = json.load(open(_REPO_ROOT / "results" / "paper_final_stats.json"))
TRAIN_RUBRIC = json.load(open(_REPO_ROOT / "results" / "train_12_rubric_structure.json"))

# Common style — LNCS papers use serif fonts, keep consistent
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

# Consistent color palette — muted, printer-friendly
COLORS = {
    "Paper Observation": "#4C72B0",     # blue
    "Plan Writing": "#55A868",          # green
    "Code Implementation": "#C44E52",   # red
    "Command Execution": "#8172B2",     # purple
    "Result Matching": "#CCB974",       # yellow-brown
}

RUBRIC_TYPES = [
    "Paper Observation",
    "Plan Writing",
    "Code Implementation",
    "Command Execution",
    "Result Matching",
]

# ============================================================================
# Figure 3: Rubric Type Score Distribution
# ============================================================================
# Stacked horizontal bar showing % of total score per rubric type.
# Highlight the skim-mode achievable region (PO + PW + CI = 64.6%).

def fig3_rubric_distribution():
    scores = TRAIN_RUBRIC["total_scores"]
    total = sum(scores.values())
    pcts = {t: scores[t] / total * 100 for t in RUBRIC_TYPES}

    fig, ax = plt.subplots(figsize=(6.5, 2.0))

    left = 0
    for t in RUBRIC_TYPES:
        w = pcts[t]
        # skim mode achievable (PO, PW, CI) gets solid; skipped (CE, RM) gets hatched
        skim_achievable = t in ("Paper Observation", "Plan Writing", "Code Implementation")
        hatch = "" if skim_achievable else "///"
        alpha = 1.0 if skim_achievable else 0.55
        bar = ax.barh(
            [0], [w], left=left, height=0.55,
            color=COLORS[t], edgecolor="black", linewidth=0.6,
            hatch=hatch, alpha=alpha,
        )
        # inside-bar label
        label_txt = f"{t}\n{w:.1f}%"
        ax.text(
            left + w / 2, 0, label_txt,
            ha="center", va="center", fontsize=8,
            color="white" if t in ("Code Implementation", "Command Execution") else "black",
        )
        left += w

    # Skim-mode upper bound line
    skim_ub = pcts["Paper Observation"] + pcts["Plan Writing"] + pcts["Code Implementation"]
    ax.axvline(x=skim_ub, color="black", linestyle="--", linewidth=1.2, zorder=10)
    ax.annotate(
        f"Skim-mode upper bound = {skim_ub:.1f}%",
        xy=(skim_ub, 0.32), xytext=(skim_ub - 20, 0.55),
        fontsize=9, ha="right",
        arrowprops=dict(arrowstyle="->", lw=0.7),
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.9)
    ax.set_yticks([])
    ax.set_xlabel("Share of total rubric score (%)")
    ax.set_title(
        f"Rubric type distribution across 12 training-set papers "
        f"(total {total:,} points)"
    )
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)

    out = OUT_DIR / "fig3_rubric_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ============================================================================
# Figure 4: Per-Paper Recall Bar Chart
# ============================================================================
# Horizontal bar chart of 12 training papers, sorted by recall.
# Show aggregate line (41.4%) and test official line (49.64%).

def fig4_per_paper_recall():
    per_paper = STATS["per_paper_recall"]
    # sort ascending so highest bar at top
    sorted_items = sorted(per_paper.items(), key=lambda x: x[1])
    names = [n for n, _ in sorted_items]
    recalls = [r * 100 for _, r in sorted_items]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # Color bar by whether above/below aggregate
    train_agg = STATS["train_self_eval"]["aggregate_score_recall"] * 100
    bar_colors = [
        "#55A868" if r >= train_agg else "#C44E52"
        for r in recalls
    ]

    bars = ax.barh(names, recalls, color=bar_colors, edgecolor="black", linewidth=0.5)

    # Value labels
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


# ============================================================================
# Figure 6: Actions.json tool type breakdown on 138 test papers
# ============================================================================
# Aggregate Read/Write/Execute counts across all 138 test papers.

def fig6_action_types():
    # Load paper_stats.json (138 test papers)
    paper_stats = json.load(open(_REPO_ROOT / "results" / "test_138_action_stats.json"))
    total_reads = sum(s["reads"] for s in paper_stats)
    total_writes = sum(s["writes"] for s in paper_stats)
    total_executes = sum(s["executes"] for s in paper_stats)
    total_all = total_reads + total_writes + total_executes

    labels = ["Read", "Write", "Execute"]
    counts = [total_reads, total_writes, total_executes]
    pcts = [c / total_all * 100 for c in counts]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    # Left: pie chart
    wedges, texts, autotexts = ax1.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax1.set_title(f"Tool call distribution\n(n = {total_all:,} calls, 138 papers)")

    # Right: per-domain average bar chart
    from collections import defaultdict
    by_domain = defaultdict(lambda: {"reads": [], "writes": [], "executes": []})
    for s in paper_stats:
        d = s["domain"]
        by_domain[d]["reads"].append(s["reads"])
        by_domain[d]["writes"].append(s["writes"])
        by_domain[d]["executes"].append(s["executes"])

    domain_order = ["Astronomy", "Biology", "Chemistry", "Environment", "ML", "Materials", "Medical"]
    x = range(len(domain_order))
    reads_avg = [sum(by_domain[d]["reads"]) / len(by_domain[d]["reads"]) if by_domain[d]["reads"] else 0 for d in domain_order]
    writes_avg = [sum(by_domain[d]["writes"]) / len(by_domain[d]["writes"]) if by_domain[d]["writes"] else 0 for d in domain_order]
    executes_avg = [sum(by_domain[d]["executes"]) / len(by_domain[d]["executes"]) if by_domain[d]["executes"] else 0 for d in domain_order]

    ax2.bar(x, reads_avg, label="Read", color=colors[0], edgecolor="black", linewidth=0.4)
    ax2.bar(x, writes_avg, bottom=reads_avg, label="Write", color=colors[1], edgecolor="black", linewidth=0.4)
    bottoms_ex = [r + w for r, w in zip(reads_avg, writes_avg)]
    ax2.bar(x, executes_avg, bottom=bottoms_ex, label="Execute", color=colors[2], edgecolor="black", linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d[:4] for d in domain_order], rotation=0, fontsize=8)
    ax2.set_ylabel("Avg. tool calls per paper")
    ax2.set_title("Per-domain average\n(n varies: 4–110 per domain)")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "fig6_action_types.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ============================================================================
# Figure 5 placeholder — ablation, to be run after ablation experiment
# ============================================================================
# Left commented; will be generated later when ablation data is available.
# See make_figure5_ablation.py (to be created after ablation runs complete).


if __name__ == "__main__":
    print("Generating figures...")
    fig3_rubric_distribution()
    fig4_per_paper_recall()
    fig6_action_types()
    print("\nAll figures saved to", OUT_DIR)
    print("\nFigure descriptions:")
    print("  fig3: Rubric type score distribution with skim-mode upper bound")
    print("  fig4: Per-paper recall bar chart with train/test reference lines")
    print("  fig6: Tool call distribution (pie) + per-domain averages (stacked bars)")
