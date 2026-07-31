"""Compare control trajectories against a baseline with release tolerances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


IGNORED_KEYS = {"schema_version", "episode_status", "solver_diagnostics"}
RTOL = 1e-5
ATOL = 1e-6


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
        actual_keys = {key for key in actual if not _ignored(key)}
        missing = expected_keys.difference(actual_keys)
        if missing:
            failures.append(f"{location}: missing keys {sorted(missing, key=str)}")
        unexpected = actual_keys.difference(expected_keys)
        if unexpected:
            failures.append(f"{location}: unexpected keys {sorted(unexpected, key=str)}")
        for key in sorted(expected_keys.intersection(actual_keys), key=str):
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
            if not np.allclose(expected_array, actual_array, rtol=RTOL, atol=ATOL, equal_nan=False):
                difference = float(np.max(np.abs(expected_array - actual_array)))
                failures.append(f"{location}: numeric mismatch (max_abs={difference})")
        elif not np.array_equal(expected_array, actual_array):
            failures.append(f"{location}: array values differ")
        return
    if isinstance(expected, (float, np.floating, int, np.integer)) and isinstance(
        actual, (float, np.floating, int, np.integer)
    ):
        if not np.isclose(expected, actual, rtol=RTOL, atol=ATOL):
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
        ``success``, ``failure``, ``invalid_initialization``,
        ``solver_degraded``, or ``execution_error``.
    """
    declared_status = record.get("episode_status")
    if declared_status in {"execution_error", "invalid_initialization", "solver_degraded"}:
        return declared_status
    poses = np.asarray(record.get("obj_pose", []))
    if poses.size == 0:
        return "invalid_initialization"
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
    if not baseline_root.is_dir():
        failures.append(f"Baseline root is not a directory: {baseline_root}")
        return failures
    if not current_root.is_dir():
        failures.append(f"Current root is not a directory: {current_root}")
        return failures
    baseline_paths = sorted(baseline_root.rglob("*.npy"))
    if not baseline_paths:
        failures.append(f"Baseline contains no .npy files: {baseline_root}")
        return failures
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
    parser.add_argument(
        "--json-report",
        type=Path,
        help="Write the machine-readable comparison result to this path.",
    )
    return parser.parse_args()


def build_report(baseline_root: Path, current_root: Path, failures: list[str]) -> dict[str, Any]:
    """Build a machine-readable summary for one directory comparison.

    Args:
        baseline_root: Root containing expected control files.
        current_root: Root containing current control files.
        failures: Mismatch descriptions produced by ``compare_directories``.

    Returns:
        JSON-serializable comparison report.
    """
    return {
        "schema_version": 1,
        "baseline_root": str(baseline_root),
        "current_root": str(current_root),
        "rtol": RTOL,
        "atol": ATOL,
        "expected_file_count": len(list(baseline_root.rglob("*.npy"))),
        "actual_file_count": len(list(current_root.rglob("*.npy"))),
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def write_json_report(path: Path, report: dict[str, Any]) -> None:
    """Persist a comparison report using stable, human-readable JSON.

    Args:
        path: Destination JSON path.
        report: JSON-serializable comparison report.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """Compare two trees and exit nonzero on any regression.

    Returns:
        None.
    """
    args = parse_args()
    baseline_root = args.baseline.resolve()
    current_root = args.current.resolve()
    failures = compare_directories(baseline_root, current_root)
    report = build_report(baseline_root, current_root, failures)
    if args.json_report is not None:
        write_json_report(args.json_report.resolve(), report)
    if failures:
        print("Golden comparison failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    count = report["expected_file_count"]
    print(f"Golden comparison passed for {count} file(s).")


if __name__ == "__main__":
    main()
