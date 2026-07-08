"""
Ablation Evaluator — runs rubric_evaluator.py on each ablation workspace.

For each paper (AMUN, min-p, gated-attention) × config (Full, NoStagePlanner,
NoRubricNormalizer, NoPhaseB), evaluate the ablation workspace against the
same ground-truth rubrics.json and produce a comparative report.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python evaluate_ablation.py

Outputs:
    /root/eval_results_ablation/<paper>_<config>_eval.json  (per run)
    /root/eval_results_ablation/ablation_summary.json       (aggregate)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PAPERS = ["AMUN", "min-p", "gated-attention"]
CONFIGS = ["Full", "NoStagePlanner", "NoRubricNormalizer", "NoPhaseB"]

# Paths — support env-var override for portability across clones.
# Defaults match the layout used on our autodl development instance.
_REPO_ROOT = Path(__file__).resolve().parent.parent  # rgsc-agent/
TRAIN_DIR = os.environ.get("RGSC_TRAIN_DIR", str(_REPO_ROOT / "data" / "train_valid"))
ABL_ROOT = os.environ.get("RGSC_ABL_ROOT", "/root/agent-workspace-ablation")
OUT_DIR = Path(os.environ.get("RGSC_OUT_DIR", "/root/eval_results_ablation"))
EVALUATOR = str(_REPO_ROOT / "evaluation" / "rubric_evaluator.py")


def evaluate_one(paper: str, config: str, dry_run: bool = False) -> dict:
    """Evaluate a single (paper, config) run.

    Symlinks the ablation workspace to a stable name so that the existing
    evaluator (which looks for workspace_root/<paper>/...) picks up the right
    per-config workspace. Uses the same rubrics.json for all configs (they
    all point to the same paper via symlinks).
    """
    ws_name = f"{paper}_{config}"
    ws_path = Path(f"{ABL_ROOT}/{ws_name}")
    if not (ws_path / "log" / "actions.json").exists():
        return {"paper": paper, "config": config, "status": "MISSING_ACTIONS"}

    # The evaluator looks up rubrics via train_dir/**/paper_name/rubrics.json.
    # We symlinked <paper>_<config> in data/train_valid to point at the real paper,
    # so rubrics.json exists at data/train_valid/<paper>_<config>/rubrics.json
    # through the symlink. The evaluator will therefore find it automatically
    # when passed --papers <paper>_<config>.

    OUT_DIR.mkdir(exist_ok=True, parents=True)

    cmd = [
        sys.executable, EVALUATOR,
        "--train_dir", TRAIN_DIR,
        "--workspace_root", ABL_ROOT,
        "--output_dir", str(OUT_DIR),
        "--papers", ws_name,
        "--model", "deepseek-chat",
        "--max_workers", "4",
    ]
    if dry_run:
        print("DRY:", " ".join(cmd))
        return {"paper": paper, "config": config, "status": "DRY_RUN"}

    print(f"\n>>> Evaluating {ws_name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    STDERR: {result.stderr[-500:]}")
        return {"paper": paper, "config": config, "status": "EVAL_FAILED",
                "stderr": result.stderr[-500:]}

    # Parse output JSON
    eval_path = OUT_DIR / f"{ws_name}_eval.json"
    if not eval_path.exists():
        return {"paper": paper, "config": config, "status": "NO_OUTPUT"}
    data = json.load(open(eval_path))
    return {
        "paper": paper,
        "config": config,
        "status": "OK",
        "overall": data["overall_score_recall"],
        "score_earned": data["total_score_earned"],
        "score_possible": data["total_score_possible"],
        "by_type": {t: bt["score_recall"] for t, bt in data["by_type"].items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", nargs="+", default=PAPERS,
                    help="which papers to evaluate")
    ap.add_argument("--configs", nargs="+", default=CONFIGS,
                    help="which configs to evaluate")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    results = []
    for paper in args.papers:
        for config in args.configs:
            r = evaluate_one(paper, config, dry_run=args.dry_run)
            results.append(r)

    if args.dry_run:
        return

    # Aggregate table
    print("\n" + "=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)
    print(f"{'Paper':<20} {'Full':>8} {'NoStagePl':>10} {'NoRubricN':>10} {'NoPhaseB':>10}")
    print("-" * 60)
    for paper in args.papers:
        row = {r["config"]: r for r in results if r["paper"] == paper}
        vals = []
        for cfg in CONFIGS:
            if cfg in row and row[cfg].get("status") == "OK":
                vals.append(f"{row[cfg]['overall']*100:>7.1f}%")
            else:
                vals.append(f"{'—':>8}")
        print(f"{paper:<20} " + " ".join(f"{v:>9}" for v in vals))

    # Save summary
    summary_path = OUT_DIR / "ablation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary saved to {summary_path}")

    # Cross-config comparison (avg over papers)
    print("\n" + "=" * 80)
    print("Averaged Score Recall by Config (across successful papers)")
    print("=" * 80)
    for cfg in args.configs:
        ok = [r for r in results if r["config"] == cfg and r.get("status") == "OK"]
        if not ok:
            print(f"  {cfg:<25}  (no data)")
            continue
        avg = sum(r["overall"] for r in ok) / len(ok)
        print(f"  {cfg:<25}  {avg*100:.2f}%  (n={len(ok)})")


if __name__ == "__main__":
    main()
