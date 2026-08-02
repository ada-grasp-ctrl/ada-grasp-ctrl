"""Create or verify per-sample classifications for bundled quick runs."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

if __package__:
    from .build_example_fixtures import HAND_NAMES, QUICK_DATA_RELATIVE, sha256_file
else:
    from build_example_fixtures import HAND_NAMES, QUICK_DATA_RELATIVE, sha256_file


EXPECTED_COUNT = 100
SCIENTIFIC_CLASSIFICATIONS = {
    "success",
    "failure",
    "invalid_initialization",
    "solver_degraded",
    "execution_error",
}


class QuickResultError(RuntimeError):
    """Raised when a quick run or expected inventory is incomplete or inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    """Load one required JSON mapping.

    Args:
        path: JSON path.

    Returns:
        Parsed mapping.

    Raises:
        QuickResultError: If the file is missing or malformed.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QuickResultError(f"Cannot read required JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise QuickResultError(f"Required JSON file is not a mapping: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one required YAML mapping.

    Args:
        path: YAML path.

    Returns:
        Parsed mapping.

    Raises:
        QuickResultError: If the file is missing or malformed.
    """
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise QuickResultError(f"Cannot read required YAML file {path}: {error}") from error
    if not isinstance(value, dict):
        raise QuickResultError(f"Required YAML file is not a mapping: {path}")
    return value


def classification_digest(records: Sequence[Mapping[str, object]]) -> str:
    """Return a deterministic digest of sample paths and classifications.

    Args:
        records: Ordered per-sample classification entries.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["path"])):
        canonical = {"path": record["path"], "classification": record["classification"]}
        digest.update(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve_report_path(raw_path: object, report_path: Path, expected_root: Path) -> Path:
    """Resolve one report path and require it to stay inside a current-run root.

    Args:
        raw_path: JSON path value.
        report_path: Report containing the value.
        expected_root: Root that must contain the resolved path.

    Returns:
        Normalized absolute path.

    Raises:
        QuickResultError: If the value is invalid or escapes the run.
    """
    if not isinstance(raw_path, str):
        raise QuickResultError(f"Report contains a non-string path: {report_path}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = report_path.parent / path
    path = path.resolve(strict=False)
    if not path.is_relative_to(expected_root):
        raise QuickResultError(f"Report path escapes the current run: {path}")
    return path


def _manifest_input_paths(manifest: Mapping[str, Any], hand: str) -> list[str]:
    """Return the expected source-relative input identities for one hand.

    Args:
        manifest: Fixture manifest.
        hand: Maintained short hand name.

    Returns:
        Sorted source-relative paths.

    Raises:
        QuickResultError: If the manifest section is malformed.
    """
    try:
        records = manifest["hands"][hand]["records"]
        paths = [entry["source_relative_path"] for entry in records]
    except (KeyError, TypeError) as error:
        raise QuickResultError(f"Fixture manifest lacks the {hand} input inventory.") from error
    if len(paths) != EXPECTED_COUNT or any(not isinstance(path, str) for path in paths):
        raise QuickResultError(f"Fixture manifest does not contain exactly 100 {hand} input paths.")
    return sorted(paths)


def _validate_manifest_inputs(output_root: Path, input_root: Path, expected_inputs: Sequence[str]) -> None:
    """Verify the control-eval run manifest records exactly the bundled inputs.

    Args:
        output_root: Current quick run root.
        input_root: Selected hand fixture root.
        expected_inputs: Expected relative sample identities.

    Returns:
        None.

    Raises:
        QuickResultError: If the manifest is missing or lists different inputs.
    """
    path = output_root / "log/control_eval/run_manifest.yaml"
    manifest = _load_yaml(path)
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or any(not isinstance(value, str) for value in inputs):
        raise QuickResultError(f"Control-eval run manifest has an invalid input list: {path}")
    relative = []
    for value in inputs:
        candidate = Path(value).resolve(strict=False)
        if not candidate.is_relative_to(input_root):
            raise QuickResultError(f"Control-eval run manifest input is outside the bundled fixture root: {candidate}")
        relative.append(candidate.relative_to(input_root).as_posix())
    if sorted(relative) != sorted(expected_inputs):
        raise QuickResultError(f"Control-eval run manifest inputs differ from the bundled 100-record inventory: {path}")


def collect_run_classifications(
    hand: str,
    output_root: Path,
    fixture_manifest_path: Path,
) -> dict[str, object]:
    """Validate one fresh quick run and collect its per-sample classifications.

    Args:
        hand: Maintained short hand name.
        output_root: Unique quick run directory.
        fixture_manifest_path: Checked-in fixture manifest.

    Returns:
        Hand inventory containing records, counts, and aggregate digest.

    Raises:
        QuickResultError: If reports, outputs, or classifications are inconsistent.
    """
    if hand not in HAND_NAMES:
        raise QuickResultError(f"Unsupported quick hand: {hand}")
    output_root = Path(output_root).expanduser().resolve(strict=False)
    fixture_manifest_path = Path(fixture_manifest_path).expanduser().resolve(strict=False)
    project_root = fixture_manifest_path.parent.parent.resolve(strict=False)
    input_root = (project_root / QUICK_DATA_RELATIVE / hand / "dummy_arm").resolve(strict=False)
    fixture_manifest = _load_json(fixture_manifest_path)
    expected_inputs = _manifest_input_paths(fixture_manifest, hand)
    _validate_manifest_inputs(output_root, input_root, expected_inputs)

    eval_report_path = output_root / "log/control_eval/run_report.json"
    eval_report = _load_json(eval_report_path)
    eval_results = eval_report.get("results")
    for field in ("num_discovered", "num_processed"):
        if eval_report.get(field) != EXPECTED_COUNT:
            raise QuickResultError(f"Control-eval report {field} must equal 100: {eval_report_path}")
    if eval_report.get("num_skipped") != 0 or not isinstance(eval_results, list) or len(eval_results) != EXPECTED_COUNT:
        raise QuickResultError(f"Control-eval report must describe exactly 100 current results: {eval_report_path}")

    control_root = (output_root / "control").resolve(strict=False)
    output_to_input: dict[Path, str] = {}
    observed_inputs = []
    eval_status_by_input = {}
    for index, result in enumerate(eval_results):
        if not isinstance(result, Mapping):
            raise QuickResultError(f"Control-eval result {index} is invalid: {eval_report_path}")
        input_path = _resolve_report_path(result.get("input_path"), eval_report_path, input_root)
        relative = input_path.relative_to(input_root).as_posix()
        outputs = result.get("output_paths")
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise QuickResultError(f"Control-eval result for {relative} must declare exactly one output.")
        output_path = _resolve_report_path(outputs[0], eval_report_path, control_root)
        if not output_path.is_file():
            raise QuickResultError(f"Control-eval output is missing: {output_path}")
        if output_path in output_to_input:
            raise QuickResultError(f"Control-eval report repeats an output path: {output_path}")
        status = result.get("status")
        if status not in {"completed", "invalid_initialization", "solver_degraded", "execution_error"}:
            raise QuickResultError(f"Control-eval result has unsupported status for {relative}: {status!r}")
        output_to_input[output_path] = relative
        observed_inputs.append(relative)
        eval_status_by_input[relative] = status
    if sorted(observed_inputs) != expected_inputs:
        raise QuickResultError(f"Control-eval report inputs differ from the bundled inventory: {eval_report_path}")

    stat_report_path = output_root / "log/control_stat/run_report.json"
    stat_report = _load_json(stat_report_path)
    stat_results = stat_report.get("results")
    for field in ("num_discovered", "num_processed"):
        if stat_report.get(field) != EXPECTED_COUNT:
            raise QuickResultError(f"Control-stat report {field} must equal 100: {stat_report_path}")
    if not isinstance(stat_results, list) or len(stat_results) != EXPECTED_COUNT:
        raise QuickResultError(f"Control-stat report must describe exactly 100 current results: {stat_report_path}")
    stat_inputs = {
        _resolve_report_path(result.get("input_path"), stat_report_path, control_root)
        for result in stat_results
        if isinstance(result, Mapping)
    }
    if stat_inputs != set(output_to_input):
        raise QuickResultError(f"Control-stat report inputs differ from control-eval outputs: {stat_report_path}")

    statistics_paths = sorted((output_root / "control_stat_res").glob("*.yaml"))
    if len(statistics_paths) != 1:
        raise QuickResultError(f"Expected exactly one current-run statistics file below {output_root}.")
    statistics = _load_yaml(statistics_paths[0])
    sample_status = statistics.get("sample_status")
    if (
        statistics.get("num_total") != EXPECTED_COUNT
        or not isinstance(sample_status, list)
        or len(sample_status) != EXPECTED_COUNT
    ):
        raise QuickResultError(f"Statistics must classify exactly 100 current results: {statistics_paths[0]}")

    records = []
    classifications = Counter()
    seen_inputs = set()
    for index, status_record in enumerate(sample_status):
        if not isinstance(status_record, Mapping):
            raise QuickResultError(f"Statistics sample_status entry {index} is invalid: {statistics_paths[0]}")
        output_path = _resolve_report_path(status_record.get("path"), statistics_paths[0], control_root)
        relative = output_to_input.get(output_path)
        if relative is None or relative in seen_inputs:
            raise QuickResultError(f"Statistics sample_status contains an unknown or repeated path: {output_path}")
        status = status_record.get("status")
        if status == "completed":
            classification = status_record.get("scientific_outcome")
        else:
            classification = status
        if classification not in SCIENTIFIC_CLASSIFICATIONS:
            raise QuickResultError(f"Statistics classification is invalid for {relative}: {classification!r}")
        eval_status = eval_status_by_input[relative]
        if eval_status == "completed" and classification not in {"success", "failure"}:
            raise QuickResultError(f"Completed control-eval sample has non-scientific classification: {relative}")
        if eval_status != "completed" and classification != eval_status:
            raise QuickResultError(f"Control-eval/statistics status mismatch for {relative}.")
        records.append({"path": relative, "classification": classification})
        classifications[classification] += 1
        seen_inputs.add(relative)
    if sorted(seen_inputs) != expected_inputs:
        raise QuickResultError(f"Statistics sample identities differ from the bundled inputs: {statistics_paths[0]}")

    recorded_counts = {
        classification: int(statistics.get(classification, -1)) for classification in SCIENTIFIC_CLASSIFICATIONS
    }
    expected_counts = {classification: classifications[classification] for classification in SCIENTIFIC_CLASSIFICATIONS}
    if recorded_counts != expected_counts:
        raise QuickResultError(
            f"Statistics aggregate classifications disagree with sample_status: {statistics_paths[0]}"
        )
    records = sorted(records, key=lambda record: record["path"])
    return {
        "records": records,
        "counts": dict(sorted(expected_counts.items())),
        "aggregate_sha256": classification_digest(records),
    }


def create_expected_inventory(
    fixture_manifest_path: Path,
    run_roots: Mapping[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    """Create the deterministic expected-classification inventory from three runs.

    Args:
        fixture_manifest_path: Checked-in fixture manifest.
        run_roots: Hand-to-fresh-run mapping.
        output_path: Destination JSON file.

    Returns:
        Written inventory mapping.

    Raises:
        QuickResultError: If hand coverage or any run is invalid.
    """
    if set(run_roots) != set(HAND_NAMES):
        raise QuickResultError(f"Expected run roots for exactly {', '.join(HAND_NAMES)}.")
    fixture_manifest_path = Path(fixture_manifest_path).expanduser().resolve(strict=True)
    hands = {hand: collect_run_classifications(hand, run_roots[hand], fixture_manifest_path) for hand in HAND_NAMES}
    inventory = {
        "schema_version": 1,
        "fixture_manifest_sha256": sha256_file(fixture_manifest_path),
        "hands": hands,
    }
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return inventory


def verify_expected_inventory(
    hand: str,
    output_root: Path,
    fixture_manifest_path: Path,
    expected_path: Path,
) -> dict[str, object]:
    """Verify one fresh quick run against the checked-in expected inventory.

    Args:
        hand: Maintained short hand name.
        output_root: Unique quick run directory.
        fixture_manifest_path: Checked-in fixture manifest.
        expected_path: Checked-in expected-classification inventory.

    Returns:
        Verified current hand inventory.

    Raises:
        QuickResultError: If the run or expected inventory differs.
    """
    fixture_manifest_path = Path(fixture_manifest_path).expanduser().resolve(strict=True)
    expected = _load_json(Path(expected_path).expanduser().resolve(strict=False))
    if expected.get("schema_version") != 1:
        raise QuickResultError(f"Expected-status inventory is not schema v1: {expected_path}")
    if expected.get("fixture_manifest_sha256") != sha256_file(fixture_manifest_path):
        raise QuickResultError("Expected-status inventory is tied to a different fixture manifest.")
    hands = expected.get("hands")
    if not isinstance(hands, Mapping) or set(hands) != set(HAND_NAMES):
        raise QuickResultError("Expected-status inventory must describe exactly the three maintained hands.")
    current = collect_run_classifications(hand, output_root, fixture_manifest_path)
    if current != hands[hand]:
        raise QuickResultError(f"Quick classifications for {hand} differ from the expected inventory.")
    return current


def _parse_runs(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeatable ``HAND=/path/to/run`` assignments.

    Args:
        values: Raw assignments.

    Returns:
        Hand-to-run-root mapping.

    Raises:
        QuickResultError: If an assignment is malformed or duplicated.
    """
    runs = {}
    for value in values:
        hand, separator, raw_path = value.partition("=")
        if not separator or hand not in HAND_NAMES or not raw_path:
            raise QuickResultError(f"Quick run assignments use HAND=/path/to/run, got: {value}")
        if hand in runs:
            raise QuickResultError(f"Duplicate quick run assignment for {hand}.")
        runs[hand] = Path(raw_path)
    return runs


def _argument_parser() -> argparse.ArgumentParser:
    """Create the expected-status command-line parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="Create the inventory from three fresh quick runs.")
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--run", action="append", required=True, metavar="HAND=/PATH/TO/RUN")
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="Verify one fresh quick run against the inventory.")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected", type=Path, required=True)
    verify.add_argument("--hand", choices=HAND_NAMES, required=True)
    verify.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Create or verify expected quick classifications.

    Args:
        argv: Optional argument sequence for tests.

    Returns:
        Process exit code.
    """
    arguments = _argument_parser().parse_args(argv)
    try:
        if arguments.command == "create":
            inventory = create_expected_inventory(
                arguments.manifest,
                _parse_runs(arguments.run),
                arguments.output,
            )
            print(f"quick expected-status inventory written: {arguments.output} hands={len(inventory['hands'])}")
        else:
            current = verify_expected_inventory(
                arguments.hand,
                arguments.output_root,
                arguments.manifest,
                arguments.expected,
            )
            print(
                f"quick result verification passed: hand={arguments.hand} "
                f"records={len(current['records'])} counts={current['counts']}"
            )
    except QuickResultError as error:
        print(f"quick result verification failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
