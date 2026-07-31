"""Print a compact status summary for one example output directory."""

from pathlib import Path
import sys

import numpy as np
import yaml


def summarize(output_root: Path) -> None:
    """Print episode statuses, lift outcome, and statistics location.

    Args:
        output_root: Example output directory.

    Returns:
        None.
    """
    control_paths = sorted((output_root / "control").rglob("*.npy"))
    for path in control_paths:
        record = np.load(path, allow_pickle=True).item()
        status = record.get("episode_status", "legacy")
        poses = np.asarray(record.get("obj_pose", []))
        lifted = None
        if len(poses) >= 2:
            lifted = bool(poses[-1, 2] - poses[0, 2] > 0.1)
        print(f"episode={path.relative_to(output_root)} status={status} lifted={lifted}")
    statistics_paths = sorted((output_root / "control_stat_res").glob("*.yaml"))
    for path in statistics_paths:
        statistics = yaml.safe_load(path.read_text(encoding="utf-8"))
        print(
            f"statistics={path} success_rate={statistics.get('success_rate')} valid={statistics.get('num_valid_cases')}"
        )


def main() -> None:
    """Parse the output directory and print its summary.

    Returns:
        None.
    """
    if len(sys.argv) != 2:
        raise SystemExit("Usage: report_example.py <output-root>")
    summarize(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
