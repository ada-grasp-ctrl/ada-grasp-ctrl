"""Print a compact status summary for one example output directory."""

import json
from pathlib import Path
import sys

import numpy as np
import yaml


class ReportError(RuntimeError):
    """Raised when one example run lacks a self-consistent result set."""


def _load_json_mapping(path: Path) -> dict[str, object]:
    """Load one required JSON mapping with contextual errors.

    Args:
        path: JSON file to read.

    Returns:
        Parsed mapping.

    Raises:
        ReportError: If the file is missing, malformed, or not a mapping.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"Cannot read required report {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReportError(f"Required report is not a JSON mapping: {path}")
    return value


def _reported_control_paths(output_root: Path) -> list[Path]:
    """Return only outputs declared by this run's control-eval report.

    Args:
        output_root: Unique example run directory.

    Returns:
        Sorted control output paths.

    Raises:
        ReportError: If the report or any declared output is invalid.
    """
    report_path = output_root / "log" / "control_eval" / "run_report.json"
    report = _load_json_mapping(report_path)
    if report.get("task") != "control_eval" or not isinstance(report.get("results"), list):
        raise ReportError(f"Invalid control-eval report contract: {report_path}")
    control_root = (output_root / "control").resolve(strict=False)
    paths: set[Path] = set()
    for index, result in enumerate(report["results"]):
        if not isinstance(result, dict) or not isinstance(result.get("output_paths"), list):
            raise ReportError(f"Invalid output_paths in control-eval result {index}: {report_path}")
        for raw_path in result["output_paths"]:
            if not isinstance(raw_path, str):
                raise ReportError(f"Control-eval result {index} contains a non-string output path.")
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = report_path.parent / candidate
            candidate = candidate.resolve(strict=False)
            if not candidate.is_relative_to(control_root):
                raise ReportError(f"Control report references output outside this run: {candidate}")
            if not candidate.is_file():
                raise ReportError(f"Control report output is missing: {candidate}")
            paths.add(candidate)
    if not paths:
        raise ReportError(f"Control-eval report declares no output files: {report_path}")
    return sorted(paths)


def _validate_stat_report(report: dict[str, object], report_path: Path, control_paths: list[Path]) -> None:
    """Validate that control-stat processed exactly the current control outputs.

    Args:
        report: Parsed control-stat batch report.
        report_path: Report path used in contextual errors.
        control_paths: Outputs declared by the current control-eval report.

    Returns:
        None.

    Raises:
        ReportError: If counts or input paths do not match the current run.
    """
    results = report.get("results")
    if report.get("task") != "control_stat" or not isinstance(results, list):
        raise ReportError(f"Invalid control-stat report contract: {report_path}")
    expected_count = len(control_paths)
    for field in ("num_discovered", "num_processed"):
        value = report.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value != expected_count:
            raise ReportError(f"Control-stat report {field}={value!r}, expected {expected_count}: {report_path}")
    reported_inputs: set[Path] = set()
    for index, result in enumerate(results):
        if not isinstance(result, dict) or not isinstance(result.get("input_path"), str):
            raise ReportError(f"Invalid control-stat result {index}: {report_path}")
        candidate = Path(result["input_path"]).expanduser()
        if not candidate.is_absolute():
            candidate = report_path.parent / candidate
        reported_inputs.add(candidate.resolve(strict=False))
    if reported_inputs != set(control_paths):
        raise ReportError(f"Control-stat report inputs do not match the current control-eval outputs: {report_path}")


def _validate_statistics(statistics: object, path: Path, expected_count: int) -> dict[str, object]:
    """Validate the minimum statistics contract used by the example report.

    Args:
        statistics: Parsed YAML value.
        path: Statistics path used in contextual errors.
        expected_count: Number of current control outputs.

    Returns:
        Validated statistics mapping.

    Raises:
        ReportError: If required counts are absent or inconsistent.
    """
    if not isinstance(statistics, dict):
        raise ReportError(f"Statistics file is not a mapping: {path}")
    count_fields = (
        "num_total",
        "success",
        "failure",
        "invalid_initialization",
        "solver_degraded",
        "execution_error",
        "num_valid_cases",
        "success_rate_denominator",
    )
    counts: dict[str, int] = {}
    for field in count_fields:
        value = statistics.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ReportError(f"Statistics field {field} must be a non-negative integer: {path}")
        counts[field] = value
    classified = sum(
        counts[field]
        for field in ("success", "failure", "invalid_initialization", "solver_degraded", "execution_error")
    )
    valid = counts["success"] + counts["failure"]
    if counts["num_total"] != expected_count or classified != expected_count:
        raise ReportError(f"Statistics counts do not match {expected_count} current control output(s): {path}")
    if counts["num_valid_cases"] != valid or counts["success_rate_denominator"] != valid:
        raise ReportError(f"Statistics success-rate denominator is inconsistent: {path}")
    success_rate = statistics.get("success_rate")
    if valid == 0:
        if success_rate is not None:
            raise ReportError(f"Statistics success_rate must be null when no valid cases exist: {path}")
    elif (
        not isinstance(success_rate, (int, float))
        or isinstance(success_rate, bool)
        or not np.isclose(float(success_rate), counts["success"] / valid)
    ):
        raise ReportError(f"Statistics success_rate is inconsistent with success counts: {path}")
    return statistics


def summarize(output_root: Path) -> None:
    """Print episode statuses, lift outcome, and statistics location.

    Args:
        output_root: Example output directory.

    Returns:
        None.

    Raises:
        ReportError: If required current-run reports or outputs are missing.
    """
    control_paths = _reported_control_paths(output_root)
    episode_lines: list[str] = []
    for path in control_paths:
        try:
            record = np.load(path, allow_pickle=True).item()
        except Exception as error:
            raise ReportError(f"Cannot read control output {path}: {error}") from error
        if not isinstance(record, dict):
            raise ReportError(f"Control output is not a mapping: {path}")
        status = record.get("episode_status", "legacy")
        poses = np.asarray(record.get("obj_pose", []))
        lifted = None
        if len(poses) >= 2:
            lifted = bool(poses[-1, 2] - poses[0, 2] > 0.1)
        episode_lines.append(f"episode={path.relative_to(output_root)} status={status} lifted={lifted}")
    stat_report_path = output_root / "log" / "control_stat" / "run_report.json"
    stat_report = _load_json_mapping(stat_report_path)
    _validate_stat_report(stat_report, stat_report_path, control_paths)
    statistics_paths = sorted((output_root / "control_stat_res").glob("*.yaml"))
    if len(statistics_paths) != 1:
        raise ReportError(
            f"Expected exactly one current-run statistics file below {output_root / 'control_stat_res'}, "
            f"found {len(statistics_paths)}."
        )
    statistics_path = statistics_paths[0]
    try:
        statistics = yaml.safe_load(statistics_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ReportError(f"Cannot read statistics {statistics_path}: {error}") from error
    statistics = _validate_statistics(statistics, statistics_path, len(control_paths))

    # Emit nothing until every current-run artifact has passed validation, so
    # a stale or partial report cannot leave a misleading success-like prefix.
    for line in episode_lines:
        print(line)
    print(
        f"statistics={statistics_path} success_rate={statistics.get('success_rate')} "
        f"valid={statistics.get('num_valid_cases')}"
    )


def main() -> None:
    """Parse the output directory and print its summary.

    Returns:
        None.
    """
    if len(sys.argv) != 2:
        raise SystemExit("Usage: report_example.py <output-root>")
    try:
        summarize(Path(sys.argv[1]).resolve())
    except ReportError as error:
        print(f"example report failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
