"""Create and verify compact, machine-auditable golden release evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Iterable

import numpy as np
import yaml

if __package__:
    from .compare_golden import ATOL, RTOL, _ignored, classify, compare_directories, compare_values
else:
    from compare_golden import ATOL, RTOL, _ignored, classify, compare_directories, compare_values


HAND_NAMES = ("shadow", "allegro", "leap_tac3d")
METHOD_DIRECTORIES = ("ours_default", "op", "bs1", "bs2", "bs3")
STATISTIC_KEYS = (
    "success_rate",
    "ave_obj_pos_err",
    "ave_obj_angle_err",
    "ave_sum_cf_all",
    "ave_wrench_all",
    "ave_normalized_wrench_all",
    "failure_cases",
    "num_invalid_cases",
    "num_valid_cases",
    "num_total",
    "success",
    "failure",
    "invalid_initialization",
    "solver_degraded",
    "execution_error",
    "success_rate_denominator",
)


def sha256_file(path: Path) -> str:
    """Hash one file without loading the entire payload into memory.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_scientific_digest(digest: Any, value: Any) -> None:
    """Feed a nested NumPy record into a stable scientific-content digest.

    Args:
        digest: Hash object receiving canonical bytes.
        value: Nested mapping, sequence, array, scalar, or null value.

    Returns:
        None.
    """
    if isinstance(value, dict):
        digest.update(b"mapping{")
        keys = sorted((key for key in value if not _ignored(key)), key=str)
        for key in keys:
            _update_scientific_digest(digest, key)
            _update_scientific_digest(digest, value[key])
        digest.update(b"}")
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence[")
        for item in value:
            _update_scientific_digest(digest, item)
        digest.update(b"]")
        return
    if isinstance(value, np.ndarray):
        digest.update(b"array:")
        digest.update(value.dtype.str.encode("utf-8"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        if value.dtype.hasobject:
            for item in value.flat:
                _update_scientific_digest(digest, item)
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
        return
    if isinstance(value, np.generic):
        _update_scientific_digest(digest, value.item())
        return
    if isinstance(value, bool):
        digest.update(b"bool:1" if value else b"bool:0")
        return
    if isinstance(value, int):
        digest.update(f"int:{value}".encode("ascii"))
        return
    if isinstance(value, float):
        digest.update(b"float:")
        digest.update(struct.pack(">d", value))
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"str:{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
        return
    if value is None:
        digest.update(b"none")
        return
    raise TypeError(f"Unsupported golden value type: {type(value).__name__}")


def scientific_sha256(record: dict[str, Any]) -> str:
    """Hash scientific record content while excluding approved metadata.

    Args:
        record: Loaded control trajectory mapping.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    _update_scientific_digest(digest, record)
    return digest.hexdigest()


def inventory_files(paths: Iterable[Path], root: Path, include_scientific: bool = False) -> list[dict[str, Any]]:
    """Build sorted checksum entries for explicit files.

    Args:
        paths: Files to include.
        root: Root used for portable relative paths.
        include_scientific: Whether NPY records also receive scientific digests and classifications.

    Returns:
        JSON-serializable checksum entries sorted by relative path.
    """
    entries = []
    for path in sorted(paths):
        entry: dict[str, Any] = {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if include_scientific:
            record = np.load(path, allow_pickle=True).item()
            entry["scientific_sha256"] = scientific_sha256(record)
            entry["classification"] = classify(record)
        entries.append(entry)
    return entries


def inventory_tree(root: Path, include_scientific: bool = False) -> list[dict[str, Any]]:
    """Inventory every NPY file under a tree.

    Args:
        root: Tree containing NPY files.
        include_scientific: Whether to hash and classify loaded records.

    Returns:
        Sorted checksum entries relative to ``root``.
    """
    return inventory_files(root.rglob("*.npy"), root, include_scientific=include_scientific)


def inventory_digest(entries: list[dict[str, Any]], field: str) -> str:
    """Aggregate a portable tree digest from per-file entries.

    Args:
        entries: File inventory entries.
        field: Digest field to aggregate, such as ``sha256`` or ``scientific_sha256``.

    Returns:
        Lowercase hexadecimal SHA-256 digest for paths and selected digests.
    """
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry[field].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def comparison_evidence(baseline_root: Path, current_root: Path) -> dict[str, Any]:
    """Run a strict comparison and return compact evidence.

    Args:
        baseline_root: Expected trajectory tree.
        current_root: Repeated trajectory tree.

    Returns:
        Machine-readable comparison result without environment-specific failure output.
    """
    failures = compare_directories(baseline_root, current_root)
    return {
        "rtol": RTOL,
        "atol": ATOL,
        "expected_file_count": len(list(baseline_root.rglob("*.npy"))),
        "actual_file_count": len(list(current_root.rglob("*.npy"))),
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def _mismatch_category(message: str) -> str:
    """Classify one human-readable mismatch into a stable audit category.

    Args:
        message: Mismatch description from ``compare_values``.

    Returns:
        Stable mismatch category name.
    """
    if "missing keys " in message:
        return "missing_keys"
    if "unexpected keys " in message:
        return "unexpected_keys"
    if ": shape " in message:
        return "shape"
    if "sequence length " in message:
        return "sequence_length"
    if "numeric mismatch " in message:
        return "numeric"
    if "array values differ" in message:
        return "array_values"
    if "expected mapping" in message:
        return "type"
    return "scalar_value"


def record_difference(label: str, historical_path: Path, current_path: Path) -> dict[str, Any] | None:
    """Summarize scientific differences for one historical/current pair.

    Args:
        label: Portable logical path for the pair.
        historical_path: Previous promoted control record.
        current_path: Corrected control record.

    Returns:
        Difference entry, or ``None`` when the scientific records match.
    """
    historical = np.load(historical_path, allow_pickle=True).item()
    current = np.load(current_path, allow_pickle=True).item()
    failures: list[str] = []
    compare_values(historical, current, label, failures)
    if not failures:
        return None
    categories = Counter(_mismatch_category(failure) for failure in failures)
    return {
        "path": label,
        "historical_classification": classify(historical),
        "current_classification": classify(current),
        "classification_changed": classify(historical) != classify(current),
        "historical_sha256": sha256_file(historical_path),
        "current_sha256": sha256_file(current_path),
        "historical_scientific_sha256": scientific_sha256(historical),
        "current_scientific_sha256": scientific_sha256(current),
        "mismatch_count": len(failures),
        "mismatch_categories": dict(sorted(categories.items())),
    }


def difference_report(pairs: Iterable[tuple[str, Path, Path]]) -> dict[str, Any]:
    """Aggregate pairwise old-to-new differences.

    Args:
        pairs: Logical label, historical path, and corrected path triples.

    Returns:
        Compact report with per-file and aggregate mismatch counts.
    """
    entries = []
    categories: Counter[str] = Counter()
    for label, historical_path, current_path in pairs:
        entry = record_difference(label, historical_path, current_path)
        if entry is None:
            continue
        entries.append(entry)
        categories.update(entry["mismatch_categories"])
    return {
        "changed_file_count": len(entries),
        "classification_change_count": sum(entry["classification_changed"] for entry in entries),
        "mismatch_count": sum(entry["mismatch_count"] for entry in entries),
        "mismatch_categories": dict(sorted(categories.items())),
        "files": entries,
    }


def fixed_pairs(historical_root: Path, current_root: Path) -> list[tuple[str, Path, Path]]:
    """Pair historical and corrected fixed-matrix files despite renamed fixtures.

    Args:
        historical_root: Historical three-hand matrix root.
        current_root: Corrected three-hand matrix root.

    Returns:
        Logical label and file path triples for all 15 cases.
    """
    pairs = []
    for hand_name in HAND_NAMES:
        for method_directory in METHOD_DIRECTORIES:
            historical_path = next((historical_root / hand_name / "control" / method_directory).glob("*.npy"))
            current_path = next((current_root / hand_name / "control" / method_directory).glob("*.npy"))
            label = f"{hand_name}/control/{method_directory}/{current_path.name}"
            pairs.append((label, historical_path, current_path))
    return pairs


def release_pairs(historical_root: Path, current_root: Path) -> list[tuple[str, Path, Path]]:
    """Pair historical and corrected 300-case release files.

    Args:
        historical_root: Previous release root organized by hand.
        current_root: Corrected release root organized by hand.

    Returns:
        Logical label and file path triples for all 300 cases.
    """
    pairs = []
    for hand_name in HAND_NAMES:
        historical_control = historical_root / hand_name / "control"
        current_control = current_root / hand_name / "control"
        for historical_path in sorted(historical_control.rglob("*.npy")):
            relative = historical_path.relative_to(historical_control)
            pairs.append((f"{hand_name}/{relative}", historical_path, current_control / relative))
    return pairs


def load_run_manifests(paths: Iterable[Path], root: Path) -> list[dict[str, Any]]:
    """Load full runtime manifests with portable evidence paths.

    Args:
        paths: Run-manifest YAML paths.
        root: Root used for relative evidence paths.

    Returns:
        Sorted manifest entries containing paths and parsed YAML data.
    """
    return [
        {
            "path": str(path.relative_to(root)),
            "manifest": yaml.safe_load(path.read_text(encoding="utf-8")),
        }
        for path in sorted(paths)
    ]


def release_statistics(root: Path) -> dict[str, dict[str, Any]]:
    """Load stable release summary fields while excluding absolute sample paths.

    Args:
        root: Release root organized by hand.

    Returns:
        Statistics keyed by public hand name.
    """
    statistics = {}
    for hand_name in HAND_NAMES:
        path = root / hand_name / "control_stat_res" / "dist_0_ours_default.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        statistics[hand_name] = {key: data[key] for key in STATISTIC_KEYS}
    return statistics


def fixed_input_inventory(project_root: Path) -> list[dict[str, Any]]:
    """Inventory checked-in inputs and collision assets for the fixed matrix.

    Args:
        project_root: Repository root.

    Returns:
        Portable raw checksum entries.
    """
    paths = []
    for hand_name in HAND_NAMES:
        paths.extend((project_root / "examples" / "data" / hand_name / "dummy_arm").glob("*.npy"))
    object_root = project_root / "examples" / "assets" / "object" / "core_bottle_15787789482f045d8add95bf56d3d2fa"
    paths.append(object_root / "info" / "simplified.json")
    paths.extend((object_root / "urdf" / "meshes").glob("convex_piece_*.obj"))
    return inventory_files(paths, project_root)


def release_input_inventory(project_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Inventory the external 100-case input set for each hand.

    Args:
        project_root: Repository root containing local release inputs.

    Returns:
        Per-hand raw checksum entries relative to each grasp-data root.
    """
    inventory = {}
    for hand_name in HAND_NAMES:
        root = project_root / "output" / f"learn_dummy_arm_{hand_name}" / "graspdata"
        inventory[hand_name] = inventory_tree(root)
    return inventory


def release_output_inventory(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Inventory scientific and raw output checksums for each release hand.

    Args:
        root: Release root organized by hand.

    Returns:
        Per-hand output checksum and classification entries.
    """
    return {
        hand_name: inventory_tree(root / hand_name / "control", include_scientific=True) for hand_name in HAND_NAMES
    }


def verify_inventory(
    expected: list[dict[str, Any]],
    root: Path,
    digest_field: str,
) -> list[str]:
    """Verify a current tree against a stored portable inventory.

    Args:
        expected: Stored inventory entries.
        root: Current tree root.
        digest_field: ``sha256`` or ``scientific_sha256``.

    Returns:
        Verification failures; an empty list means acceptance.
    """
    include_scientific = digest_field == "scientific_sha256"
    actual = inventory_tree(root, include_scientific=include_scientific)
    expected_by_path = {entry["path"]: entry for entry in expected}
    actual_by_path = {entry["path"]: entry for entry in actual}
    failures = []
    missing = sorted(expected_by_path.keys() - actual_by_path.keys())
    extra = sorted(actual_by_path.keys() - expected_by_path.keys())
    if missing:
        failures.append(f"Missing files: {missing}")
    if extra:
        failures.append(f"Unexpected files: {extra}")
    for relative in sorted(expected_by_path.keys() & actual_by_path.keys()):
        if expected_by_path[relative][digest_field] != actual_by_path[relative][digest_field]:
            failures.append(f"{relative}: {digest_field} differs")
        elif (
            include_scientific
            and expected_by_path[relative]["classification"] != actual_by_path[relative]["classification"]
        ):
            failures.append(f"{relative}: classification differs")
    return failures


def create_artifact(args: argparse.Namespace) -> dict[str, Any]:
    """Build a complete fixed-matrix and 300-case audit artifact.

    Args:
        args: Parsed create-command arguments.

    Returns:
        JSON-serializable release artifact.
    """
    project_root = args.project_root.resolve()
    fixed_root = args.fixed_root.resolve()
    fixed_repeat_root = args.fixed_repeat_root.resolve()
    release_root = args.release_root.resolve()
    release_repeat_root = args.release_repeat_root.resolve()

    fixed_comparison = comparison_evidence(fixed_root, fixed_repeat_root)
    release_comparison = comparison_evidence(release_root, release_repeat_root)
    if not fixed_comparison["passed"] or not release_comparison["passed"]:
        raise RuntimeError("Repeated golden outputs do not match the promoted run.")

    fixed_outputs = inventory_tree(fixed_root, include_scientific=True)
    release_outputs = release_output_inventory(release_root)
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "fixed_matrix": {
            "promoted_path": "release/golden/fixed_matrix",
            "inputs": fixed_input_inventory(project_root),
            "outputs": fixed_outputs,
            "raw_tree_sha256": inventory_digest(fixed_outputs, "sha256"),
            "scientific_tree_sha256": inventory_digest(fixed_outputs, "scientific_sha256"),
            "repeat_comparison": fixed_comparison,
            "run_manifests": load_run_manifests(fixed_root.rglob("run_manifest.yaml"), fixed_root),
        },
        "release_300": {
            "inputs": release_input_inventory(project_root),
            "outputs": release_outputs,
            "statistics": release_statistics(release_root),
            "repeat_statistics": release_statistics(release_repeat_root),
            "repeat_comparison": release_comparison,
            "run_manifests": load_run_manifests(
                (release_root / hand_name / "log" / "run_manifest.yaml" for hand_name in HAND_NAMES),
                release_root,
            ),
        },
    }
    if args.historical_fixed_root is not None:
        artifact["fixed_matrix"]["historical_difference"] = difference_report(
            fixed_pairs(args.historical_fixed_root.resolve(), fixed_root)
        )
    if args.historical_release_root is not None:
        historical_root = args.historical_release_root.resolve()
        artifact["release_300"]["historical_statistics"] = release_statistics(historical_root)
        artifact["release_300"]["historical_difference"] = difference_report(
            release_pairs(historical_root, release_root)
        )
    return artifact


def verify_artifact(args: argparse.Namespace) -> list[str]:
    """Verify checked-in and optionally regenerated evidence against an artifact.

    Args:
        args: Parsed verify-command arguments.

    Returns:
        Verification failures; an empty list means acceptance.
    """
    artifact_path = args.artifact.resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    project_root = args.project_root.resolve()
    fixed_root = args.fixed_root.resolve() if args.fixed_root else artifact_path.parent / "fixed_matrix"
    failures = []

    fixed_expected = artifact["fixed_matrix"]["outputs"]
    failures.extend(verify_inventory(fixed_expected, fixed_root, "sha256"))
    for entry in artifact["fixed_matrix"]["inputs"]:
        path = project_root / entry["path"]
        if not path.is_file():
            failures.append(f"Fixed input missing: {entry['path']}")
        elif sha256_file(path) != entry["sha256"]:
            failures.append(f"Fixed input checksum differs: {entry['path']}")

    if args.fixed_current is not None:
        failures.extend(compare_directories(fixed_root, args.fixed_current.resolve()))
    if args.release_root is not None:
        release_root = args.release_root.resolve()
        for hand_name in HAND_NAMES:
            failures.extend(
                f"{hand_name}: {failure}"
                for failure in verify_inventory(
                    artifact["release_300"]["outputs"][hand_name],
                    release_root / hand_name / "control",
                    "scientific_sha256",
                )
            )
    if args.release_input_root is not None:
        input_root = args.release_input_root.resolve()
        for hand_name in HAND_NAMES:
            root = input_root / f"learn_dummy_arm_{hand_name}" / "graspdata"
            failures.extend(
                f"{hand_name} input: {failure}"
                for failure in verify_inventory(
                    artifact["release_300"]["inputs"][hand_name],
                    root,
                    "sha256",
                )
            )
    return failures


def parse_args() -> argparse.Namespace:
    """Parse create and verify command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create an audit artifact from completed runs.")
    create.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    create.add_argument("--fixed-root", type=Path, required=True)
    create.add_argument("--fixed-repeat-root", type=Path, required=True)
    create.add_argument("--release-root", type=Path, required=True)
    create.add_argument("--release-repeat-root", type=Path, required=True)
    create.add_argument("--historical-fixed-root", type=Path)
    create.add_argument("--historical-release-root", type=Path)
    create.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="Verify checked-in or regenerated golden evidence.")
    verify.add_argument("artifact", type=Path)
    verify.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    verify.add_argument("--fixed-root", type=Path)
    verify.add_argument("--fixed-current", type=Path)
    verify.add_argument("--release-root", type=Path)
    verify.add_argument("--release-input-root", type=Path)
    return parser.parse_args()


def main() -> None:
    """Create or verify an audit artifact and expose a stable process result.

    Returns:
        None.
    """
    args = parse_args()
    if args.command == "create":
        artifact = create_artifact(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Golden audit artifact written to {args.output}")
        return

    failures = verify_artifact(args)
    if failures:
        print("Golden audit verification failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("Golden audit verification passed.")


if __name__ == "__main__":
    main()
