"""Build the checked-in three-hand fixtures from explicit archived sources."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAND_NAMES = ("shadow", "allegro", "leap_tac3d")
OBJECT_ID = "core_bottle_15787789482f045d8add95bf56d3d2fa"
SAMPLE_RELATIVE = Path(OBJECT_ID) / "tabletop_ur10e" / "scale006_pose004_0" / "partial_pc_00_6.npy"
OBJECT_RELATIVE = Path("examples/assets/object") / OBJECT_ID
REQUIRED_RECORD_FIELDS = (
    "obj_path",
    "obj_pose",
    "obj_scale",
    "pregrasp_qpos",
    "grasp_qpos",
    "squeeze_qpos",
)


class FixtureBuildError(ValueError):
    """Raised when archived fixture sources are incomplete or malformed."""


def _is_finite(array: np.ndarray) -> bool:
    """Return whether a possibly object-typed array is entirely finite.

    Args:
        array: Candidate numeric array.

    Returns:
        ``True`` only for values accepted by NumPy's finite check.
    """
    try:
        return bool(np.all(np.isfinite(array)))
    except TypeError:
        return False


def _load(path: Path) -> dict[str, object]:
    """Load and minimally validate one archived grasp record.

    Args:
        path: Absolute source record path.

    Returns:
        Mutable mapping copy.

    Raises:
        FixtureBuildError: If the file is missing, unreadable, or does not
            contain the fields required by the bundled examples.
    """
    if not path.is_file():
        raise FixtureBuildError(f"Fixture source is missing: {path}")
    try:
        value = np.load(path, allow_pickle=True).item()
    except (OSError, ValueError) as error:
        raise FixtureBuildError(f"Cannot load fixture source {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise FixtureBuildError(f"Fixture source must contain a mapping: {path}")
    record = dict(value)
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        raise FixtureBuildError(f"Fixture source {path} is missing fields: {', '.join(missing)}")

    for field in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos"):
        array = np.asarray(record[field])
        if array.ndim != 1 or not _is_finite(array):
            raise FixtureBuildError(f"Fixture source {path} has invalid finite one-dimensional {field}.")
    qpos_shapes = {np.asarray(record[field]).shape for field in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")}
    if len(qpos_shapes) != 1:
        raise FixtureBuildError(f"Fixture source {path} has inconsistent hand qpos dimensions.")

    object_pose = np.asarray(record["obj_pose"])
    object_scale = np.asarray(record["obj_scale"])
    if object_pose.shape != (7,) or not _is_finite(object_pose):
        raise FixtureBuildError(f"Fixture source {path} has invalid finite seven-value obj_pose.")
    if object_scale.shape != () or not _is_finite(object_scale) or float(object_scale) <= 0.0:
        raise FixtureBuildError(f"Fixture source {path} has invalid positive scalar obj_scale.")
    return record


def _save(path: Path, record: Mapping[str, object]) -> None:
    """Save one version-one fixture.

    Args:
        path: Destination record.
        record: Mapping to serialize.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    current = dict(record)
    current["schema_version"] = 1
    np.save(path, current)


def _validate_hand_roots(roots: Mapping[str, Path], label: str) -> dict[str, Path]:
    """Validate one exact source root for every supported hand.

    Args:
        roots: Hand-to-directory mapping.
        label: Human-readable source group name.

    Returns:
        Normalized absolute roots in maintained hand order.

    Raises:
        FixtureBuildError: If a hand is missing, unsupported, relative, or not
            backed by an existing directory.
    """
    missing = [hand for hand in HAND_NAMES if hand not in roots]
    extra = sorted(set(roots) - set(HAND_NAMES))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unsupported {', '.join(extra)}")
        raise FixtureBuildError(f"{label} roots must cover exactly {', '.join(HAND_NAMES)} ({'; '.join(details)}).")

    normalized = {}
    for hand in HAND_NAMES:
        root = Path(roots[hand]).expanduser()
        if not root.is_absolute():
            raise FixtureBuildError(f"{label} root for {hand} must be absolute: {root}")
        root = root.resolve(strict=False)
        if not root.is_dir():
            raise FixtureBuildError(f"{label} root for {hand} is not a directory: {root}")
        normalized[hand] = root
    return normalized


def _parse_hand_roots(values: Sequence[str], label: str) -> dict[str, Path]:
    """Parse repeatable ``HAND=/absolute/path`` command-line values.

    Args:
        values: Raw command-line assignments.
        label: Human-readable source group name.

    Returns:
        Validated hand-to-directory mapping.

    Raises:
        FixtureBuildError: If an assignment is malformed or duplicated.
    """
    roots: dict[str, Path] = {}
    for value in values:
        hand, separator, raw_path = value.partition("=")
        if not separator or not hand or not raw_path:
            raise FixtureBuildError(f"{label} roots use HAND=/absolute/path, got: {value}")
        if hand in roots:
            raise FixtureBuildError(f"Duplicate {label} root for hand '{hand}'.")
        roots[hand] = Path(raw_path)
    return _validate_hand_roots(roots, label)


def build_scene(destination_root: Path) -> Path:
    """Write the shared Learning-converter scene record.

    Args:
        destination_root: Root of the generated example data tree.

    Returns:
        Scene path.
    """
    scene_path = destination_root / "scene.npy"
    scene = {
        "task": {"obj_name": "target"},
        "scene": {
            "target": {
                "file_path": f"../assets/object/{OBJECT_ID}/mesh/simplified.obj",
                "pose": np.array([0.71100098, 0.04075842, 0.02034247, 0.58787514, -0.37770758, 0.61379698, 0.36741404]),
                "scale": [0.06],
            }
        },
    }
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(scene_path, scene)
    return scene_path


def build_hand(
    hand: str,
    formatted: Mapping[str, object],
    dummy_arm: Mapping[str, object],
    destination_root: Path,
) -> None:
    """Write raw, formatted, and dummy-arm fixtures for one hand.

    Args:
        hand: ``shadow``, ``allegro``, or ``leap_tac3d``.
        formatted: Archived formatted grasp record.
        dummy_arm: Archived dummy-arm grasp record.
        destination_root: Root of the generated example data tree.

    Returns:
        None.
    """
    formatted_record = dict(formatted)
    dummy_record = dict(dummy_arm)
    for record in (formatted_record, dummy_record):
        record["obj_path"] = str(OBJECT_RELATIVE)

    hand_root = destination_root / hand
    _save(hand_root / "formatted/grasp.npy", formatted_record)
    _save(hand_root / "dummy_arm/grasp.npy", dummy_record)
    raw = {
        "schema_version": 1,
        "scene_path": "../../scene.npy",
        "pregrasp_qpos": formatted_record["pregrasp_qpos"],
        "grasp_qpos": formatted_record["grasp_qpos"],
        "squeeze_qpos": formatted_record["squeeze_qpos"],
    }
    _save(hand_root / "raw/learning.npy", raw)


def build_fixtures(
    formatted_roots: Mapping[str, Path],
    dummy_arm_roots: Mapping[str, Path],
    destination_root: Path = PROJECT_ROOT / "examples/data",
) -> list[Path]:
    """Build all checked-in example records from explicit archived roots.

    Every source is loaded before any destination is written, preventing an
    incomplete archive from partially replacing the checked-in fixture set.

    Args:
        formatted_roots: Per-hand roots containing the fixed formatted sample.
        dummy_arm_roots: Per-hand roots containing the fixed dummy-arm sample.
        destination_root: Destination data tree, anchored explicitly rather
            than to the caller's current working directory.

    Returns:
        Generated fixture paths in deterministic order.

    Raises:
        FixtureBuildError: If roots or records are incomplete or malformed.
    """
    formatted_roots = _validate_hand_roots(formatted_roots, "formatted")
    dummy_arm_roots = _validate_hand_roots(dummy_arm_roots, "dummy-arm")
    destination_root = Path(destination_root).expanduser()
    if not destination_root.is_absolute():
        destination_root = PROJECT_ROOT / destination_root
    destination_root = destination_root.resolve(strict=False)

    sources: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    for hand in HAND_NAMES:
        sources[hand] = (
            _load(formatted_roots[hand] / SAMPLE_RELATIVE),
            _load(dummy_arm_roots[hand] / SAMPLE_RELATIVE),
        )

    generated = [build_scene(destination_root)]
    for hand in HAND_NAMES:
        formatted, dummy_arm = sources[hand]
        build_hand(hand, formatted, dummy_arm, destination_root)
        generated.extend(
            (
                destination_root / hand / "formatted/grasp.npy",
                destination_root / hand / "dummy_arm/grasp.npy",
                destination_root / hand / "raw/learning.npy",
            )
        )
    return generated


def _argument_parser() -> argparse.ArgumentParser:
    """Create the fixture-builder command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formatted-root",
        action="append",
        required=True,
        metavar="HAND=/ABSOLUTE/PATH",
        help="Repeat once per supported hand; each root directly contains the fixed sample path.",
    )
    parser.add_argument(
        "--dummy-arm-root",
        action="append",
        required=True,
        metavar="HAND=/ABSOLUTE/PATH",
        help="Repeat once per supported hand; each root directly contains the fixed sample path.",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=PROJECT_ROOT / "examples/data",
        help="Destination fixture tree (default: the checkout's examples/data directory).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build all fixtures from command-line source roots.

    Args:
        argv: Optional argument sequence for tests.

    Returns:
        Process exit code.
    """
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        formatted_roots = _parse_hand_roots(arguments.formatted_root, "formatted")
        dummy_arm_roots = _parse_hand_roots(arguments.dummy_arm_root, "dummy-arm")
        generated = build_fixtures(formatted_roots, dummy_arm_roots, arguments.destination_root)
    except FixtureBuildError as error:
        parser.error(str(error))
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
