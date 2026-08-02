"""Audit the bundled 3x100 quick fixtures and their exact DGN asset subset."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np

from ada_grasp_ctrl.schema import load_npy_record, validate_grasp_record
from ada_grasp_ctrl.utils.robots.base import RobotFactory

if __package__:
    from .build_example_fixtures import (
        EXPECTED_OBJECT_COUNT,
        EXPECTED_RECORDS_PER_HAND,
        HAND_NAMES,
        LEGACY_OBJECT_ID,
        QUICK_ASSET_RELATIVE,
        QUICK_DATA_RELATIVE,
        inventory_digest,
        sha256_file,
    )
else:
    from build_example_fixtures import (
        EXPECTED_OBJECT_COUNT,
        EXPECTED_RECORDS_PER_HAND,
        HAND_NAMES,
        LEGACY_OBJECT_ID,
        QUICK_ASSET_RELATIVE,
        QUICK_DATA_RELATIVE,
        inventory_digest,
        sha256_file,
    )


class FixtureAuditError(RuntimeError):
    """Raised when checked-in quick data differs from its manifest contract."""


def _load_manifest(path: Path) -> dict[str, Any]:
    """Load one fixture manifest mapping.

    Args:
        path: Manifest path.

    Returns:
        Parsed mapping.

    Raises:
        FixtureAuditError: If the manifest is missing or malformed.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixtureAuditError(f"Cannot read fixture manifest {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise FixtureAuditError(f"Fixture manifest must be a schema-version-one mapping: {path}")
    return value


def _robot(hand: str) -> Any:
    """Create the configured dummy-arm robot for one maintained hand.

    Args:
        hand: Maintained short hand name.

    Returns:
        Configured robot instance.
    """
    prefix = "rh_" if hand != "allegro" else ""
    return RobotFactory.create_robot(robot_type=f"dummy_arm_{hand}", prefix=prefix)


def _entry_path(entry: Mapping[str, object], project_root: Path) -> Path:
    """Resolve one manifest entry without allowing path traversal.

    Args:
        entry: File inventory entry.
        project_root: Checkout root.

    Returns:
        Absolute checked-in file path.

    Raises:
        FixtureAuditError: If the path is malformed or escapes the checkout.
    """
    raw_path = entry.get("path")
    if not isinstance(raw_path, str):
        raise FixtureAuditError(f"Manifest inventory entry has invalid path: {entry!r}")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise FixtureAuditError(f"Manifest inventory path must be relative: {raw_path}")
    resolved = (project_root / relative).resolve(strict=False)
    if not resolved.is_relative_to(project_root):
        raise FixtureAuditError(f"Manifest inventory path escapes the checkout: {raw_path}")
    return resolved


def _verify_entry(entry: Mapping[str, object], project_root: Path) -> Path:
    """Verify one file size and SHA-256 entry.

    Args:
        entry: File inventory entry.
        project_root: Checkout root.

    Returns:
        Verified file path.

    Raises:
        FixtureAuditError: If the file or metadata differs.
    """
    path = _entry_path(entry, project_root)
    if not path.is_file():
        raise FixtureAuditError(f"Manifest file is missing: {path}")
    expected_size = entry.get("size")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or path.stat().st_size != expected_size:
        raise FixtureAuditError(f"Manifest file size differs: {path}")
    expected_digest = entry.get("sha256")
    if not isinstance(expected_digest, str) or sha256_file(path) != expected_digest:
        raise FixtureAuditError(f"Manifest SHA-256 differs: {path}")
    return path


def _portable_object_path(relative: Path) -> str:
    """Return the exact checkout-relative object path required by a quick record.

    Args:
        relative: Source-relative grasp identity.

    Returns:
        Portable DGN processed-object path.
    """
    object_id, setting, _scene_name, _sample_name = relative.parts
    return (QUICK_ASSET_RELATIVE / "scene_cfg" / object_id / setting / "../../../processed_data" / object_id).as_posix()


def _audit_hand(
    hand: str,
    hand_manifest: Mapping[str, object],
    project_root: Path,
) -> tuple[list[dict[str, object]], list[Path]]:
    """Audit one exact 100-record hand inventory.

    Args:
        hand: Maintained short hand name.
        hand_manifest: Manifest section for the hand.
        project_root: Checkout root.

    Returns:
        Verified entries and their shared source-relative identities.

    Raises:
        FixtureAuditError: If files, schema, or object paths differ.
    """
    entries = hand_manifest.get("records")
    if not isinstance(entries, list) or len(entries) != EXPECTED_RECORDS_PER_HAND:
        raise FixtureAuditError(f"Manifest hand {hand} must contain exactly 100 record entries.")
    if hand_manifest.get("aggregate_sha256") != inventory_digest(entries):
        raise FixtureAuditError(f"Manifest aggregate digest differs for hand {hand}.")

    hand_root = project_root / QUICK_DATA_RELATIVE / hand / "dummy_arm"
    actual_files = sorted(path for path in hand_root.rglob("*.npy") if path.is_file())
    expected_files = sorted(_entry_path(entry, project_root) for entry in entries)
    if actual_files != expected_files:
        raise FixtureAuditError(f"Checked-in dummy-arm tree for {hand} differs from the manifest inventory.")

    robot = _robot(hand)
    relative_paths = []
    for entry in entries:
        path = _verify_entry(entry, project_root)
        raw_relative = entry.get("source_relative_path")
        object_id = entry.get("object_id")
        if not isinstance(raw_relative, str) or not isinstance(object_id, str):
            raise FixtureAuditError(f"Manifest record identity is invalid for {path}.")
        relative = Path(raw_relative)
        if len(relative.parts) != 4 or relative.parts[0] != object_id or relative.suffix != ".npy":
            raise FixtureAuditError(f"Manifest record identity is malformed for {path}.")
        if path.relative_to(hand_root) != relative:
            raise FixtureAuditError(f"Manifest source-relative path does not match checked-in location: {path}")
        try:
            record = load_npy_record(path)
            validate_grasp_record(
                record,
                path,
                expected_joint_dim=robot.n_dof,
                expected_joint_names=robot.dof_names,
                require_joint_names=True,
            )
        except Exception as error:
            raise FixtureAuditError(f"Checked-in quick record is invalid {path}: {error}") from error
        if record.get("schema_version") != 1:
            raise FixtureAuditError(f"Checked-in quick record is not schema v1: {path}")
        if record.get("obj_path") != _portable_object_path(relative):
            raise FixtureAuditError(f"Checked-in quick record has a nonportable object path: {path}")
        resolved_object = (project_root / str(record["obj_path"])).resolve(strict=False)
        expected_object = (project_root / QUICK_ASSET_RELATIVE / "processed_data" / object_id).resolve(strict=False)
        if resolved_object != expected_object or not resolved_object.is_dir():
            raise FixtureAuditError(f"Checked-in quick record resolves to the wrong object: {path}")
        relative_paths.append(relative)
    return entries, relative_paths


def _audit_assets(manifest: Mapping[str, object], project_root: Path, relative_paths: Sequence[Path]) -> list[dict]:
    """Audit the exact selected scene and processed-object file closure.

    Args:
        manifest: Assets section of the fixture manifest.
        project_root: Checkout root.
        relative_paths: Shared quick sample identities.

    Returns:
        All verified asset inventory entries.

    Raises:
        FixtureAuditError: If the asset tree or scene linkage differs.
    """
    object_ids = manifest.get("object_ids")
    scene_entries = manifest.get("scene_configs")
    object_entries = manifest.get("object_files")
    if (
        not isinstance(object_ids, list)
        or len(object_ids) != EXPECTED_OBJECT_COUNT
        or object_ids != sorted(set(object_ids))
        or not isinstance(scene_entries, list)
        or not isinstance(object_entries, list)
    ):
        raise FixtureAuditError("Manifest asset inventory structure is invalid.")
    if manifest.get("scene_aggregate_sha256") != inventory_digest(scene_entries):
        raise FixtureAuditError("Manifest scene aggregate digest differs.")
    if manifest.get("object_aggregate_sha256") != inventory_digest(object_entries):
        raise FixtureAuditError("Manifest object aggregate digest differs.")

    asset_root = project_root / QUICK_ASSET_RELATIVE
    if not (asset_root / "ATTRIBUTION.md").is_file():
        raise FixtureAuditError(f"Bundled DGN source attribution is missing: {asset_root / 'ATTRIBUTION.md'}")
    symlinks = sorted(path for path in asset_root.rglob("*") if path.is_symlink())
    if symlinks:
        raise FixtureAuditError(f"Bundled DGN asset tree contains symlinks: {symlinks}")

    actual_scenes = sorted(path for path in (asset_root / "scene_cfg").rglob("*") if path.is_file())
    actual_objects = sorted(path for path in (asset_root / "processed_data").rglob("*") if path.is_file())
    expected_scenes = sorted(_entry_path(entry, project_root) for entry in scene_entries)
    expected_objects = sorted(_entry_path(entry, project_root) for entry in object_entries)
    if actual_scenes != expected_scenes:
        raise FixtureAuditError("Checked-in DGN scene tree differs from the manifest inventory.")
    if actual_objects != expected_objects:
        raise FixtureAuditError("Checked-in DGN processed-object tree differs from the manifest inventory.")
    for entry in (*scene_entries, *object_entries):
        _verify_entry(entry, project_root)

    expected_scene_paths = []
    seen_objects = set()
    reference_root = project_root / QUICK_DATA_RELATIVE / HAND_NAMES[0] / "dummy_arm"
    for relative in relative_paths:
        object_id, setting, scene_name, _sample_name = relative.parts
        scene_path = asset_root / "scene_cfg" / object_id / setting / f"{scene_name}.npy"
        expected_scene_paths.append(scene_path)
        try:
            scene_record = np.load(scene_path, allow_pickle=True).item()
            scene_object = scene_record["scene"][object_id]
            task_object = scene_record["task"]["obj_name"]
            grasp_record = load_npy_record(reference_root / relative)
        except Exception as error:
            raise FixtureAuditError(f"Cannot validate bundled DGN scene {scene_path}: {error}") from error
        if task_object != object_id or not np.array_equal(scene_object.get("pose"), grasp_record["obj_pose"]):
            raise FixtureAuditError(f"Bundled DGN scene identity or pose differs: {scene_path}")
        scale = np.asarray(scene_object.get("scale"))
        if scale.shape != (3,) or not np.all(scale == float(grasp_record["obj_scale"])):
            raise FixtureAuditError(f"Bundled DGN scene scale differs: {scene_path}")
        for field in ("file_path", "xml_path", "urdf_path", "info_path"):
            value = scene_object.get(field)
            if not isinstance(value, str) or not (scene_path.parent / value).resolve(strict=False).is_file():
                raise FixtureAuditError(f"Bundled DGN scene dependency is missing for {field}: {scene_path}")
        processed_root = asset_root / "processed_data" / object_id
        required = (
            processed_root / "info/simplified.json",
            processed_root / "mesh/simplified.obj",
            processed_root / "urdf/coacd.xml",
            processed_root / "urdf/coacd.urdf",
        )
        if any(not path.is_file() for path in required):
            raise FixtureAuditError(f"Bundled DGN object lacks a runtime-required file: {processed_root}")
        if not sorted((processed_root / "urdf/meshes").glob("convex_piece_*.obj")):
            raise FixtureAuditError(f"Bundled DGN object lacks collision meshes: {processed_root}")
        seen_objects.add(object_id)
    if sorted(expected_scene_paths) != expected_scenes or sorted(seen_objects) != object_ids:
        raise FixtureAuditError("Manifest DGN scene/object identities differ from the quick inputs.")

    legacy_object = project_root / "examples/assets/object" / LEGACY_OBJECT_ID
    if legacy_object.exists() or legacy_object.is_symlink():
        raise FixtureAuditError(f"Legacy duplicate example object remains present: {legacy_object}")
    return [*scene_entries, *object_entries]


def _audit_full_fixture_paths(project_root: Path) -> None:
    """Verify the one-sample full examples use the deduplicated bottle path.

    Args:
        project_root: Checkout root.

    Returns:
        None.

    Raises:
        FixtureAuditError: If a full fixture still references the removed copy.
    """
    expected_object = (QUICK_ASSET_RELATIVE / "processed_data" / LEGACY_OBJECT_ID).as_posix()
    for hand in HAND_NAMES:
        path = project_root / QUICK_DATA_RELATIVE / hand / "formatted/grasp.npy"
        record = load_npy_record(path)
        if record.get("obj_path") != expected_object:
            raise FixtureAuditError(f"Full-example formatted fixture has a stale object path: {path}")
    scene_path = project_root / QUICK_DATA_RELATIVE / "scene.npy"
    try:
        scene = np.load(scene_path, allow_pickle=True).item()
        file_path = scene["scene"]["target"]["file_path"]
    except Exception as error:
        raise FixtureAuditError(f"Cannot validate the full-example scene path: {error}") from error
    expected_mesh = f"../assets/object/DGN_2k/processed_data/{LEGACY_OBJECT_ID}/mesh/simplified.obj"
    if file_path != expected_mesh or not (scene_path.parent / file_path).resolve(strict=False).is_file():
        raise FixtureAuditError(f"Full-example scene has a stale or missing object mesh path: {scene_path}")


def audit_manifest(manifest_path: Path, project_root: Path | None = None, hand: str | None = None) -> dict[str, int]:
    """Audit one checked-in fixture manifest and return verified counts.

    Args:
        manifest_path: Machine-readable quick fixture manifest.
        project_root: Optional explicit checkout root.
        hand: Optional hand restriction for record validation.

    Returns:
        Verified count summary.

    Raises:
        FixtureAuditError: If any checked-in file or contract differs.
    """
    manifest_path = Path(manifest_path).expanduser().resolve(strict=False)
    project_root = (
        Path(project_root).expanduser().resolve(strict=False)
        if project_root is not None
        else manifest_path.parent.parent.resolve(strict=False)
    )
    if hand is not None and hand not in HAND_NAMES:
        raise FixtureAuditError(f"Unsupported quick fixture hand: {hand}")
    manifest = _load_manifest(manifest_path)
    hands = manifest.get("hands")
    assets = manifest.get("assets")
    counts = manifest.get("counts")
    if not isinstance(hands, Mapping) or set(hands) != set(HAND_NAMES) or not isinstance(assets, Mapping):
        raise FixtureAuditError("Fixture manifest must describe exactly the three maintained hands and assets.")
    if not isinstance(counts, Mapping):
        raise FixtureAuditError("Fixture manifest count section is invalid.")

    selected_hands = (hand,) if hand is not None else HAND_NAMES
    all_record_entries = []
    shared_relative_paths: list[Path] | None = None
    for hand_name in HAND_NAMES:
        entries, relative_paths = _audit_hand(hand_name, hands[hand_name], project_root)
        all_record_entries.extend(entries)
        if shared_relative_paths is None:
            shared_relative_paths = relative_paths
        elif relative_paths != shared_relative_paths:
            raise FixtureAuditError(f"Quick fixture identities for {hand_name} differ from the other hands.")
    assert shared_relative_paths is not None
    asset_entries = _audit_assets(assets, project_root, shared_relative_paths)
    _audit_full_fixture_paths(project_root)

    expected_counts = {
        "grasp_records": len(all_record_entries),
        "scene_configs": len(assets["scene_configs"]),
        "object_ids": len(assets["object_ids"]),
        "object_files": len(assets["object_files"]),
        "asset_files": len(asset_entries),
    }
    for field, value in expected_counts.items():
        if counts.get(field) != value:
            raise FixtureAuditError(f"Fixture manifest count {field}={counts.get(field)!r}, expected {value}.")
    records_per_hand = counts.get("records_per_hand")
    if records_per_hand != {hand_name: EXPECTED_RECORDS_PER_HAND for hand_name in HAND_NAMES}:
        raise FixtureAuditError("Fixture manifest per-hand counts differ from 100/100/100.")
    if manifest.get("inventory_sha256") != inventory_digest([*all_record_entries, *asset_entries]):
        raise FixtureAuditError("Fixture manifest overall inventory digest differs.")

    return {
        "hands_audited": len(selected_hands),
        "grasp_records": len(all_record_entries),
        "scene_configs": len(assets["scene_configs"]),
        "object_ids": len(assets["object_ids"]),
        "asset_files": len(asset_entries),
    }


def _argument_parser() -> argparse.ArgumentParser:
    """Create the audit command-line parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--hand", choices=HAND_NAMES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Audit fixtures and print a compact verified count summary.

    Args:
        argv: Optional argument sequence for tests.

    Returns:
        Process exit code.
    """
    arguments = _argument_parser().parse_args(argv)
    try:
        counts = audit_manifest(arguments.manifest, arguments.project_root, arguments.hand)
    except FixtureAuditError as error:
        print(f"example fixture audit failed: {error}")
        return 1
    print(
        "example fixture audit passed: "
        f"records={counts['grasp_records']} scenes={counts['scene_configs']} "
        f"objects={counts['object_ids']} asset_files={counts['asset_files']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
