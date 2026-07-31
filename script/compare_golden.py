"""Compare control trajectories against a baseline with release tolerances."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


IGNORED_KEYS = {"schema_version", "episode_status", "solver_diagnostics"}


def _ignored(key: object) -> bool:
    """Identify timing and newly additive metadata fields.

    Args:
        key: Mapping key.

    Returns:
        Whether the field is excluded from numeric golden comparison.
    """
    return isinstance(key, str) and (key in IGNORED_KEYS or key.startswith("t_"))


def compare_values(expected: Any, actual: Any, location: str, failures: list[str]) -> None:
    """Recursively compare nested NumPy records.

    Args:
        expected: Baseline value.
        actual: Refactored value.
        location: Human-readable nested location.
        failures: Mutable list receiving mismatch descriptions.

    Returns:
        None.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            failures.append(f"{location}: expected mapping, got {type(actual).__name__}")
            return
        expected_keys = {key for key in expected if not _ignored(key)}
        missing = expected_keys.difference(actual)
        if missing:
            failures.append(f"{location}: missing keys {sorted(missing)}")
        for key in sorted(expected_keys.intersection(actual), key=str):
            compare_values(expected[key], actual[key], f"{location}.{key}", failures)
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            actual_length = len(actual) if hasattr(actual, "__len__") else "n/a"
            failures.append(f"{location}: sequence length {len(expected)} != {actual_length}")
            return
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            compare_values(expected_item, actual_item, f"{location}[{index}]", failures)
        return
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        expected_array = np.asarray(expected)
        actual_array = np.asarray(actual)
        if expected_array.shape != actual_array.shape:
            failures.append(f"{location}: shape {expected_array.shape} != {actual_array.shape}")
            return
        if np.issubdtype(expected_array.dtype, np.number) and np.issubdtype(actual_array.dtype, np.number):
            if not np.allclose(expected_array, actual_array, rtol=1e-5, atol=1e-6, equal_nan=False):
                difference = float(np.max(np.abs(expected_array - actual_array)))
                failures.append(f"{location}: numeric mismatch (max_abs={difference})")
        elif not np.array_equal(expected_array, actual_array):
            failures.append(f"{location}: array values differ")
        return
    if isinstance(expected, (float, np.floating, int, np.integer)) and isinstance(
        actual, (float, np.floating, int, np.integer)
    ):
        if not np.isclose(expected, actual, rtol=1e-5, atol=1e-6):
            failures.append(f"{location}: {expected!r} != {actual!r}")
        return
    if expected != actual:
        failures.append(f"{location}: {expected!r} != {actual!r}")


def classify(record: dict[str, Any], lift_height: float = 0.2) -> str:
    """Classify a historical control record using the published lift rule.

    Args:
        record: Control trajectory mapping.
        lift_height: Target vertical lift in metres.

    Returns:
        ``success``, ``failure``, ``invalid``, or ``solver_degraded``.
    """
    if record.get("episode_status") == "solver_degraded":
        return "solver_degraded"
    poses = np.asarray(record.get("obj_pose", []))
    if poses.size == 0:
        return "invalid"
    target_z = poses[0, 2] + lift_height
    return "success" if abs(target_z - poses[-1, 2]) < lift_height / 2 else "failure"


def compare_directories(baseline_root: Path, current_root: Path) -> list[str]:
    """Compare all baseline NPY paths against an equivalent current tree.

    Args:
        baseline_root: Root containing expected control files.
        current_root: Root containing refactored control files.

    Returns:
        Mismatch descriptions; an empty list means acceptance.
    """
    failures = []
    baseline_paths = sorted(baseline_root.rglob("*.npy"))
    for baseline_path in baseline_paths:
        relative = baseline_path.relative_to(baseline_root)
        current_path = current_root / relative
        if not current_path.is_file():
            failures.append(f"{relative}: current file missing")
            continue
        expected = np.load(baseline_path, allow_pickle=True).item()
        actual = np.load(current_path, allow_pickle=True).item()
        if classify(expected) != classify(actual):
            failures.append(f"{relative}: classification {classify(expected)} != {classify(actual)}")
        compare_values(expected, actual, str(relative), failures)
    extra = {path.relative_to(current_root) for path in current_root.rglob("*.npy")}.difference(
        path.relative_to(baseline_root) for path in baseline_paths
    )
    if extra:
        failures.append(f"Unexpected current files: {sorted(str(path) for path in extra)}")
    return failures


def parse_args() -> argparse.Namespace:
    """Parse baseline/current tree arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    return parser.parse_args()


def main() -> None:
    """Compare two trees and exit nonzero on any regression.

    Returns:
        None.
    """
    args = parse_args()
    failures = compare_directories(args.baseline.resolve(), args.current.resolve())
    if failures:
        print("Golden comparison failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    count = len(list(args.baseline.rglob("*.npy")))
    print(f"Golden comparison passed for {count} file(s).")


if __name__ == "__main__":
    main()
