"""
Result output path: date-time stamped folders for inference/analysis.
Structure: base_dir / pin_results / YYYYMMDD_HHMMSS /
"""
from datetime import datetime
from pathlib import Path


def get_results_root(project_root: Path | None = None) -> Path:
    """Default results root: project_root / pin_results."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    return project_root / "pin_results"


def get_timestamped_dir(base_dir: Path | None = None) -> Path:
    """
    Return a new timestamped directory for this analysis run.
    base_dir: if None, uses project_root / pin_results.
    Returns: base_dir / YYYYMMDD_HHMMSS (created).
    """
    if base_dir is None:
        base_dir = get_results_root()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base_dir / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out
