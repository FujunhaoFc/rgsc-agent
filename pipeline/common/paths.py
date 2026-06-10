"""Paper path resolution for paper-agnostic CLI tools.

Searches data/train_valid/{paper}/ first, then data/test/{paper}/.
Allows Phase 1/3 modules to work on both training fixtures and test inputs
without changing the CLI signature.
"""

from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SPLITS = ("train_valid", "test")


def find_paper_dir(paper_id: str, required_files: Optional[list[str]] = None) -> Path:
    """Find paper directory across train_valid/ and test/ splits.

    Args:
        paper_id: Paper name (e.g. "min-p", "AMUN")
        required_files: Optional list of filenames that must exist
            (default: ["paper.md"])

    Returns:
        Absolute Path to the paper directory

    Raises:
        FileNotFoundError: if no split contains the paper with required files
    """
    required = required_files or ["paper.md"]

    tried = []
    for split in SPLITS:
        candidate = DATA_DIR / split / paper_id
        tried.append(str(candidate))
        if not candidate.is_dir():
            continue
        # Check all required files exist
        if all((candidate / f).exists() for f in required):
            return candidate

    raise FileNotFoundError(
        f"Paper {paper_id!r} not found with files {required}. "
        f"Searched: {tried}"
    )


def find_paper_md(paper_id: str) -> Path:
    """Convenience: return path to paper.md."""
    return find_paper_dir(paper_id, ["paper.md"]) / "paper.md"


def find_rubrics_json(paper_id: str) -> Path:
    """Find rubrics.json. May not exist for test set (test rubrics not released).

    Raises FileNotFoundError if not found in any split.
    """
    return find_paper_dir(paper_id, ["rubrics.json"]) / "rubrics.json"
