"""
Generate placeholder mock_results.json from paper_state.main_experiments.

For table experiments: fills a 2D values array with null, keeping the shape
from the axes row/col counts.
For figure experiments: writes a placeholder summary string.

All values are null/placeholder — no real numbers are fabricated.
The user will hand-edit AMUN's mock with real numbers for testing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PLACEHOLDER_SUMMARY = "[PLACEHOLDER - 待手工填充]"


def _is_figure_exp(exp: dict) -> bool:
    """Determine if experiment is figure-type based on axes content.

    axes=null or rows/cols both empty → figure type (qualitative evidence).
    axes with non-empty rows and cols → table type (structured values grid).
    """
    axes = exp.get("axes")
    if axes is None:
        return True
    rows = axes.get("rows", []) if isinstance(axes, dict) else []
    cols = axes.get("cols", []) if isinstance(axes, dict) else []
    return len(rows) == 0 or len(cols) == 0


def _build_placeholder_exp(exp: dict) -> dict:
    """Build a placeholder result entry for one main_experiment."""
    evidence = exp.get("evidence_in_paper", "")

    if _is_figure_exp(exp):
        return {
            "evidence_in_paper": evidence,
            "summary": PLACEHOLDER_SUMMARY,
        }
    else:
        # Table type: axes has non-null, non-empty rows and cols.
        # Fill a values grid with null preserving the paper_state shape.
        axes = exp["axes"]
        rows = axes["rows"]
        cols = axes["cols"]
        n_rows = len(rows)
        n_cols = len(cols)

        values = [[None for _ in range(n_cols)] for _ in range(n_rows)]

        return {
            "evidence_in_paper": evidence,
            "axes": {
                "rows": rows,
                "cols": cols,
            },
            "values": values,
        }


def generate(paper_id: str) -> dict:
    """Generate placeholder mock_results for a paper.

    Returns the mock_results dict (not yet written to disk).
    """
    paper_state_path = OUTPUTS_DIR / paper_id / "paper_state.json"
    if not paper_state_path.exists():
        raise FileNotFoundError(f"paper_state not found: {paper_state_path}")

    with open(paper_state_path) as f:
        paper_state = json.load(f)

    experiments = {}
    for exp in paper_state.get("main_experiments", []):
        exp_id = exp["id"]  # e.g. "exp-table1"
        experiments[exp_id] = _build_placeholder_exp(exp)

    return {
        "paper_id": paper_id,
        "experiments": experiments,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python generate_placeholder_mock.py <paper_id> [paper_id ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    for paper_id in sys.argv[1:]:
        try:
            mock = generate(paper_id)
        except FileNotFoundError as e:
            print(f"[SKIP] {paper_id}: {e}", file=sys.stderr)
            continue

        output_path = OUTPUTS_DIR / paper_id / "mock_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(mock, f, indent=2, ensure_ascii=False)

        n_table = sum(
            1 for v in mock["experiments"].values() if "values" in v
        )
        n_figure = sum(
            1 for v in mock["experiments"].values() if "summary" in v
        )
        print(
            f"[{paper_id}] {len(mock['experiments'])} experiments "
            f"({n_table} table + {n_figure} figure) → {output_path}"
        )


if __name__ == "__main__":
    main()
