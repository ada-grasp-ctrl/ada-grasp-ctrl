"""Build the checked-in three-hand quick fixtures and their DGN asset subset."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from ada_grasp_ctrl.schema import load_npy_record, validate_grasp_record, with_current_schema
from ada_grasp_ctrl.utils.robots.base import RobotFactory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAND_NAMES = ("shadow", "allegro", "leap_tac3d")
EXPECTED_RECORDS_PER_HAND = 100
EXPECTED_OBJECT_COUNT = 89
LEGACY_OBJECT_ID = "core_bottle_15787789482f045d8add95bf56d3d2fa"
QUICK_DATA_RELATIVE = Path("examples/data")
QUICK_ASSET_RELATIVE = Path("examples/assets/object/DGN_2k")
QUICK_MANIFEST_RELATIVE = Path("examples/quick_manifest.json")
REQUIRED_SCENE_FIELDS = ("file_path", "xml_path", "urdf_path", "info_path")


class FixtureBuildError(ValueError):
    """Raised when accepted fixture or asset sources are incomplete or malformed."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file.

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


def inventory_digest(entries: Sequence[Mapping[str, object]]) -> str:
    """Return a deterministic digest for an ordered file inventory.

    Args:
        entries: Inventory entries containing paths, sizes, and SHA-256 values.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        canonical = {
            "path": entry["path"],
            "size": entry["size"],
            "sha256": entry["sha256"],
        }
        digest.update(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _inventory_entry(path: Path, project_root: Path, **metadata: object) -> dict[str, object]:
    """Describe one generated file relative to its checkout root.

    Args:
        path: Generated file.
        project_root: Root used for portable paths.
        **metadata: Additional manifest fields.

    Returns:
        JSON-compatible inventory mapping.
    """
    entry: dict[str, object] = {
        "path": path.relative_to(project_root).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    entry.update(metadata)
    return entry


def _validate_source_roots(roots: Mapping[str, Path]) -> dict[str, Path]:
    """Validate one explicit dummy-arm source root for every maintained hand.

    Args:
        roots: Hand-to-source-root mapping.

    Returns:
        Absolute roots in maintained hand order.

    Raises:
        FixtureBuildError: If coverage or a root is invalid.
    """
    missing = [hand for hand in HAND_NAMES if hand not in roots]
    extra = sorted(set(roots) - set(HAND_NAMES))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unsupported {', '.join(extra)}")
        raise FixtureBuildError(f"Dummy-arm roots must cover exactly {', '.join(HAND_NAMES)} ({'; '.join(details)}).")

    normalized = {}
    for hand in HAND_NAMES:
        root = Path(roots[hand]).expanduser()
        if not root.is_absolute():
            raise FixtureBuildError(f"Dummy-arm root for {hand} must be absolute: {root}")
        root = root.resolve(strict=False)
        if not root.is_dir():
            raise FixtureBuildError(f"Dummy-arm root for {hand} is not a directory: {root}")
        normalized[hand] = root
    return normalized


def _parse_hand_roots(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeatable ``HAND=/absolute/path`` command-line values.

    Args:
        values: Raw assignments.

    Returns:
        Validated hand-to-root mapping.

    Raises:
        FixtureBuildError: If an assignment is malformed or duplicated.
    """
    roots: dict[str, Path] = {}
    for value in values:
        hand, separator, raw_path = value.partition("=")
        if not separator or not hand or not raw_path:
            raise FixtureBuildError(f"Dummy-arm roots use HAND=/absolute/path, got: {value}")
        if hand in roots:
            raise FixtureBuildError(f"Duplicate dummy-arm root for hand '{hand}'.")
        roots[hand] = Path(raw_path)
    return _validate_source_roots(roots)


def _robot(hand: str) -> Any:
    """Create the configured dummy-arm robot for exact joint validation.

    Args:
        hand: Maintained short hand name.

    Returns:
        Configured robot instance.
    """
    robot_name = f"dummy_arm_{hand}"
    prefix = "rh_" if hand != "allegro" else ""
    return RobotFactory.create_robot(robot_type=robot_name, prefix=prefix)


def _load_dummy_record(path: Path, hand: str) -> dict[str, Any]:
    """Load and validate one accepted legacy or current dummy-arm record.

    Args:
        path: Source record.
        hand: Maintained short hand name.

    Returns:
        Validated mutable record.

    Raises:
        FixtureBuildError: If the record violates the accepted contract.
    """
    try:
        record = load_npy_record(path)
        robot = _robot(hand)
        validate_grasp_record(
            record,
            path,
            expected_joint_dim=robot.n_dof,
            expected_joint_names=robot.dof_names,
            require_joint_names=True,
        )
    except Exception as error:
        raise FixtureBuildError(f"Invalid dummy-arm fixture source {path}: {error}") from error

    if "approach_qpos" not in record:
        raise FixtureBuildError(f"Dummy-arm fixture source lacks approach_qpos: {path}")
    approach = np.asarray(record["approach_qpos"])
    qpos_dimension = np.asarray(record["grasp_qpos"]).shape[0]
    if (
        approach.ndim not in (1, 2)
        or approach.shape[-1] != qpos_dimension
        or not np.issubdtype(approach.dtype, np.number)
        or not np.all(np.isfinite(approach))
    ):
        raise FixtureBuildError(f"Dummy-arm fixture source has invalid approach_qpos: {path}")
    return record


def _discover_sources(roots: Mapping[str, Path]) -> tuple[list[Path], dict[str, dict[Path, dict[str, Any]]]]:
    """Load all three accepted 100-record inventories before writing anything.

    Args:
        roots: Validated hand-to-source-root mapping.

    Returns:
        Shared relative paths and loaded records by hand/path.

    Raises:
        FixtureBuildError: If counts or relative identities differ.
    """
    relative_paths: dict[str, list[Path]] = {}
    records: dict[str, dict[Path, dict[str, Any]]] = {}
    for hand in HAND_NAMES:
        discovered = sorted(path for path in roots[hand].rglob("*.npy") if path.is_file())
        if len(discovered) != EXPECTED_RECORDS_PER_HAND:
            raise FixtureBuildError(
                f"Dummy-arm source for {hand} contains {len(discovered)} .npy records; "
                f"expected {EXPECTED_RECORDS_PER_HAND}."
            )
        relative_paths[hand] = [path.relative_to(roots[hand]) for path in discovered]
        records[hand] = {
            relative: _load_dummy_record(roots[hand] / relative, hand) for relative in relative_paths[hand]
        }

    shared = relative_paths[HAND_NAMES[0]]
    for hand in HAND_NAMES[1:]:
        if relative_paths[hand] != shared:
            raise FixtureBuildError(f"Dummy-arm source paths for {hand} do not match the accepted shared inventory.")
    return shared, records


def _scene_path(dgn_root: Path, relative: Path) -> Path:
    """Map one grasp sample identity to its DGN scene configuration.

    Args:
        dgn_root: Real DGN 2k source root.
        relative: Four-component grasp path.

    Returns:
        Scene configuration path.

    Raises:
        FixtureBuildError: If the sample layout is unexpected.
    """
    if len(relative.parts) != 4 or relative.suffix != ".npy":
        raise FixtureBuildError(f"Unexpected accepted grasp path layout: {relative}")
    object_id, setting, scene_name, _sample_name = relative.parts
    return dgn_root / "scene_cfg" / object_id / setting / f"{scene_name}.npy"


def _portable_object_path(relative: Path) -> str:
    """Return the checkout-relative processed-object path used by one grasp.

    Args:
        relative: Accepted grasp identity.

    Returns:
        Portable path preserving the original DGN scene-relative traversal.
    """
    object_id, setting, _scene_name, _sample_name = relative.parts
    return (QUICK_ASSET_RELATIVE / "scene_cfg" / object_id / setting / "../../../processed_data" / object_id).as_posix()


def _load_scene(path: Path, relative: Path, reference_record: Mapping[str, Any]) -> tuple[str, set[Path]]:
    """Validate one selected DGN scene and return its required source files.

    Args:
        path: Scene configuration file.
        relative: Accepted grasp identity.
        reference_record: One hand's matching grasp record.

    Returns:
        Object ID and every directly referenced processed-data file.

    Raises:
        FixtureBuildError: If the scene is missing, malformed, or inconsistent.
    """
    if not path.is_file():
        raise FixtureBuildError(f"Referenced DGN scene configuration is missing: {path}")
    try:
        scene_record = np.load(path, allow_pickle=True).item()
    except (OSError, ValueError) as error:
        raise FixtureBuildError(f"Cannot load DGN scene configuration {path}: {error}") from error
    if not isinstance(scene_record, Mapping):
        raise FixtureBuildError(f"DGN scene configuration must contain a mapping: {path}")

    object_id = relative.parts[0]
    task = scene_record.get("task")
    scene = scene_record.get("scene")
    if not isinstance(task, Mapping) or task.get("obj_name") != object_id or not isinstance(scene, Mapping):
        raise FixtureBuildError(f"DGN scene configuration does not identify object {object_id}: {path}")
    object_record = scene.get(object_id)
    if not isinstance(object_record, Mapping):
        raise FixtureBuildError(f"DGN scene configuration lacks object entry {object_id}: {path}")

    pose = np.asarray(object_record.get("pose"))
    scale = np.asarray(object_record.get("scale"))
    if not np.array_equal(pose, np.asarray(reference_record["obj_pose"])):
        raise FixtureBuildError(f"DGN scene pose differs from accepted grasp record: {path}")
    if scale.shape != (3,) or not np.all(scale == float(reference_record["obj_scale"])):
        raise FixtureBuildError(f"DGN scene scale differs from accepted grasp record: {path}")

    dependencies = set()
    for field in REQUIRED_SCENE_FIELDS:
        value = object_record.get(field)
        if not isinstance(value, str):
            raise FixtureBuildError(f"DGN scene object field {field} is invalid: {path}")
        dependency = (path.parent / value).resolve(strict=False)
        if not dependency.is_file():
            raise FixtureBuildError(f"DGN scene dependency is missing: {dependency}")
        dependencies.add(dependency)
    return object_id, dependencies


def _select_assets(
    dgn_root: Path,
    relative_paths: Sequence[Path],
    records: Mapping[str, Mapping[Path, Mapping[str, Any]]],
) -> tuple[list[Path], list[Path], list[str]]:
    """Select the exact scene and processed-object file closure.

    Args:
        dgn_root: Real DGN 2k source root.
        relative_paths: Shared accepted sample identities.
        records: Loaded hand records.

    Returns:
        Scene files, processed-data files, and object IDs.

    Raises:
        FixtureBuildError: If paths escape the DGN root or required files are absent.
    """
    scene_files: list[Path] = []
    processed_files: set[Path] = set()
    object_ids: set[str] = set()
    resolved_dgn_root = dgn_root.resolve(strict=True)
    for relative in relative_paths:
        scene_path = _scene_path(resolved_dgn_root, relative)
        object_id, dependencies = _load_scene(scene_path, relative, records[HAND_NAMES[0]][relative])
        object_ids.add(object_id)
        scene_files.append(scene_path)
        processed_files.update(dependencies)

        processed_root = resolved_dgn_root / "processed_data" / object_id
        required = (
            processed_root / "info/simplified.json",
            processed_root / "mesh/simplified.obj",
            processed_root / "urdf/coacd.xml",
            processed_root / "urdf/coacd.urdf",
        )
        for path in required:
            if not path.is_file():
                raise FixtureBuildError(f"Runtime-required DGN object file is missing: {path}")
            processed_files.add(path.resolve())
        collision_meshes = sorted((processed_root / "urdf/meshes").glob("convex_piece_*.obj"))
        if not collision_meshes:
            raise FixtureBuildError(f"DGN object has no runtime collision meshes: {processed_root}")
        processed_files.update(path.resolve() for path in collision_meshes if path.is_file())

        for hand in HAND_NAMES:
            record = records[hand][relative]
            original = Path(str(record["obj_path"]))
            if original.is_absolute():
                resolved_original = original.resolve(strict=False)
            else:
                resolved_original = (PROJECT_ROOT / original).resolve(strict=False)
            if resolved_original != processed_root.resolve(strict=False):
                raise FixtureBuildError(
                    f"Accepted {hand} record {relative} resolves to {resolved_original}, expected {processed_root}."
                )

    if len(object_ids) != EXPECTED_OBJECT_COUNT:
        raise FixtureBuildError(f"Accepted quick inventory references {len(object_ids)} objects; expected 89.")
    for path in (*scene_files, *processed_files):
        try:
            path.resolve(strict=True).relative_to(resolved_dgn_root)
        except ValueError as error:
            raise FixtureBuildError(f"Selected DGN file escapes the accepted source root: {path}") from error
    return sorted(set(scene_files)), sorted(processed_files), sorted(object_ids)


def _save_record(path: Path, record: Mapping[str, Any]) -> None:
    """Write one version-one NumPy record.

    Args:
        path: Destination file.
        record: Record to serialize.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, with_current_schema(record))


def _copy_dgn_file(source: Path, dgn_root: Path, staging_root: Path) -> Path:
    """Copy one real DGN file into the staged checkout tree.

    Args:
        source: Source file, possibly reached through a workstation symlink.
        dgn_root: Real DGN source root.
        staging_root: Staged checkout root.

    Returns:
        Staged destination file.
    """
    relative = source.resolve(strict=True).relative_to(dgn_root.resolve(strict=True))
    destination = staging_root / QUICK_ASSET_RELATIVE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.resolve(strict=True), destination)
    return destination


def _stage_full_fixture_updates(project_root: Path, staging_root: Path) -> list[Path]:
    """Retarget the existing one-sample full fixtures to the deduplicated bottle.

    Args:
        project_root: Current checkout root.
        staging_root: Staged checkout root.

    Returns:
        Staged files to install.

    Raises:
        FixtureBuildError: If the maintained full fixtures are missing or malformed.
    """
    source_scene = project_root / QUICK_DATA_RELATIVE / "scene.npy"
    try:
        scene = np.load(source_scene, allow_pickle=True).item()
        target = scene["scene"]["target"]
    except Exception as error:
        raise FixtureBuildError(f"Cannot load maintained full-example scene {source_scene}: {error}") from error
    if not isinstance(scene, dict) or not isinstance(target, dict):
        raise FixtureBuildError(f"Maintained full-example scene is malformed: {source_scene}")
    scene = dict(scene)
    scene["scene"] = dict(scene["scene"])
    scene["scene"]["target"] = dict(target)
    scene["scene"]["target"]["file_path"] = (
        f"../assets/object/DGN_2k/processed_data/{LEGACY_OBJECT_ID}/mesh/simplified.obj"
    )
    staged_scene = staging_root / QUICK_DATA_RELATIVE / "scene.npy"
    staged_scene.parent.mkdir(parents=True, exist_ok=True)
    np.save(staged_scene, scene)
    staged = [staged_scene]

    object_path = (QUICK_ASSET_RELATIVE / "processed_data" / LEGACY_OBJECT_ID).as_posix()
    for hand in HAND_NAMES:
        source = project_root / QUICK_DATA_RELATIVE / hand / "formatted/grasp.npy"
        try:
            record = load_npy_record(source)
            validate_grasp_record(record, source)
        except Exception as error:
            raise FixtureBuildError(f"Cannot load maintained full-example fixture {source}: {error}") from error
        current = dict(record)
        current["obj_path"] = object_path
        destination = staging_root / QUICK_DATA_RELATIVE / hand / "formatted/grasp.npy"
        _save_record(destination, current)
        staged.append(destination)
    return staged


def _remove_path(path: Path) -> None:
    """Remove one exact file, symlink, or directory before installing staged data.

    Args:
        path: Exact scoped target.

    Returns:
        None.
    """
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def build_fixtures(
    dummy_arm_roots: Mapping[str, Path],
    dgn_root: Path,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Build all quick fixtures, the DGN subset, and their manifest.

    Every external record and asset is validated and staged before any tracked
    destination is replaced.

    Args:
        dummy_arm_roots: Accepted per-hand dummy-arm source roots.
        dgn_root: DGN 2k root containing ``scene_cfg`` and ``processed_data``.
        project_root: Checkout receiving the generated files.

    Returns:
        Generated manifest path.

    Raises:
        FixtureBuildError: If sources or maintained full fixtures are invalid.
    """
    roots = _validate_source_roots(dummy_arm_roots)
    dgn_root = Path(dgn_root).expanduser()
    if not dgn_root.is_absolute():
        raise FixtureBuildError(f"DGN root must be absolute: {dgn_root}")
    if not dgn_root.is_dir():
        raise FixtureBuildError(f"DGN root is not a directory: {dgn_root}")
    dgn_root = dgn_root.resolve(strict=True)
    project_root = Path(project_root).expanduser().resolve(strict=False)

    relative_paths, records = _discover_sources(roots)
    scene_files, processed_files, object_ids = _select_assets(dgn_root, relative_paths, records)

    with tempfile.TemporaryDirectory(prefix="ada-grasp-quick-fixtures-") as temporary:
        staging_root = Path(temporary) / "checkout"
        record_entries: dict[str, list[dict[str, object]]] = {hand: [] for hand in HAND_NAMES}
        for hand in HAND_NAMES:
            for relative in relative_paths:
                current = dict(records[hand][relative])
                current["obj_path"] = _portable_object_path(relative)
                destination = staging_root / QUICK_DATA_RELATIVE / hand / "dummy_arm" / relative
                _save_record(destination, current)
                record_entries[hand].append(
                    _inventory_entry(
                        destination,
                        staging_root,
                        source_relative_path=relative.as_posix(),
                        object_id=relative.parts[0],
                    )
                )

        scene_entries = []
        for source in scene_files:
            destination = _copy_dgn_file(source, dgn_root, staging_root)
            scene_entries.append(
                _inventory_entry(
                    destination,
                    staging_root,
                    source_relative_path=source.relative_to(dgn_root).as_posix(),
                    object_id=source.relative_to(dgn_root / "scene_cfg").parts[0],
                )
            )
        object_entries = []
        for source in processed_files:
            destination = _copy_dgn_file(source, dgn_root, staging_root)
            object_entries.append(
                _inventory_entry(
                    destination,
                    staging_root,
                    source_relative_path=source.relative_to(dgn_root).as_posix(),
                    object_id=source.relative_to(dgn_root / "processed_data").parts[0],
                )
            )

        staged_full_files = _stage_full_fixture_updates(project_root, staging_root)
        all_inventory_entries = [entry for hand in HAND_NAMES for entry in record_entries[hand]]
        all_inventory_entries.extend(scene_entries)
        all_inventory_entries.extend(object_entries)
        manifest = {
            "schema_version": 1,
            "source": {
                "collection": "DGN 2k",
                "grasp_roots": {hand: f"output/learn_dummy_arm_{hand}/graspdata" for hand in HAND_NAMES},
                "object_root": "assets/object/DGN_2k",
                "operation": (
                    "Three accepted 100-record dummy-arm inventories and their exact 89-object runtime subset."
                ),
            },
            "counts": {
                "records_per_hand": {hand: len(record_entries[hand]) for hand in HAND_NAMES},
                "grasp_records": sum(len(record_entries[hand]) for hand in HAND_NAMES),
                "scene_configs": len(scene_entries),
                "object_ids": len(object_ids),
                "object_files": len(object_entries),
                "asset_files": len(scene_entries) + len(object_entries),
            },
            "hands": {
                hand: {
                    "records": record_entries[hand],
                    "aggregate_sha256": inventory_digest(record_entries[hand]),
                }
                for hand in HAND_NAMES
            },
            "assets": {
                "object_ids": object_ids,
                "scene_configs": scene_entries,
                "scene_aggregate_sha256": inventory_digest(scene_entries),
                "object_files": object_entries,
                "object_aggregate_sha256": inventory_digest(object_entries),
            },
            "inventory_sha256": inventory_digest(all_inventory_entries),
        }
        staged_manifest = staging_root / QUICK_MANIFEST_RELATIVE
        staged_manifest.parent.mkdir(parents=True, exist_ok=True)
        staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        for hand in HAND_NAMES:
            target = project_root / QUICK_DATA_RELATIVE / hand / "dummy_arm"
            _remove_path(target)
            shutil.copytree(staging_root / QUICK_DATA_RELATIVE / hand / "dummy_arm", target)
        target_assets = project_root / QUICK_ASSET_RELATIVE
        _remove_path(target_assets)
        shutil.copytree(staging_root / QUICK_ASSET_RELATIVE, target_assets)
        for staged in staged_full_files:
            relative = staged.relative_to(staging_root)
            destination = project_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, destination)
        manifest_path = project_root / QUICK_MANIFEST_RELATIVE
        shutil.copy2(staged_manifest, manifest_path)

    legacy_object = project_root / "examples/assets/object" / LEGACY_OBJECT_ID
    _remove_path(legacy_object)
    return manifest_path


def _argument_parser() -> argparse.ArgumentParser:
    """Create the fixture-builder command-line parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dummy-arm-root",
        action="append",
        required=True,
        metavar="HAND=/ABSOLUTE/PATH",
        help="Repeat once per supported hand; each root contains the accepted 100-record tree.",
    )
    parser.add_argument("--dgn-root", type=Path, required=True, help="Absolute DGN_2k source root.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Destination checkout root (default: this source checkout).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build fixtures from explicit accepted sources.

    Args:
        argv: Optional argument sequence for tests.

    Returns:
        Process exit code.
    """
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        roots = _parse_hand_roots(arguments.dummy_arm_root)
        manifest = build_fixtures(roots, arguments.dgn_root, arguments.project_root)
    except FixtureBuildError as error:
        parser.error(str(error))
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
