"""Versioned validation for raw, grasp, and control NumPy records."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


class SchemaError(ValueError):
    """Describe one invalid field together with its source sample."""

    def __init__(self, sample: str | Path, field: str, expected: str, actual: object):
        """Initialize a contextual validation error.

        Args:
            sample: Path or logical name of the invalid sample.
            field: Dot-separated field name.
            expected: Human-readable field requirement.
            actual: Observed value or concise description.

        Returns:
            None.
        """
        self.sample = str(sample)
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(f"{self.sample}: field '{self.field}' expected {self.expected}; actual {self.actual!r}.")


def load_npy_record(path: str | Path) -> dict[str, Any]:
    """Load a pickled NumPy mapping with a stable error message.

    Args:
        path: Input ``.npy`` path.

    Returns:
        Mutable record dictionary.

    Raises:
        SchemaError: If the file does not contain a mapping.
    """
    input_path = Path(path)
    value = np.load(input_path, allow_pickle=True).item()
    if not isinstance(value, Mapping):
        raise SchemaError(input_path, "<root>", "a mapping", type(value).__name__)
    return dict(value)


def schema_version(record: Mapping[str, Any], sample: str | Path) -> int:
    """Read and validate a record schema version.

    Args:
        record: Record whose version is inspected.
        sample: Source used in validation errors.

    Returns:
        Version zero for legacy records or the explicit supported version.

    Raises:
        SchemaError: If the declared version is unsupported.
    """
    version = record.get("schema_version", 0)
    if isinstance(version, np.ndarray) and version.size == 1:
        version = version.item()
    if version not in (0, SCHEMA_VERSION):
        raise SchemaError(sample, "schema_version", "0 or 1", version)
    return int(version)


def with_current_schema(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a record and mark it with the current schema version.

    Args:
        record: Legacy or current record.

    Returns:
        Mutable version-one record preserving all existing fields.
    """
    result = dict(record)
    result["schema_version"] = SCHEMA_VERSION
    return result


def _require(record: Mapping[str, Any], field: str, sample: str | Path) -> Any:
    """Return a required field.

    Args:
        record: Mapping containing the field.
        field: Required top-level field name.
        sample: Source used in validation errors.

    Returns:
        Field value.

    Raises:
        SchemaError: If the field is absent.
    """
    if field not in record:
        raise SchemaError(sample, field, "a required field", "missing")
    return record[field]


def _finite_array(
    record: Mapping[str, Any],
    field: str,
    sample: str | Path,
    *,
    ndim: int | None = None,
    trailing: int | None = None,
) -> np.ndarray:
    """Validate and return a finite numeric array.

    Args:
        record: Mapping containing the array.
        field: Field name.
        sample: Source used in validation errors.
        ndim: Optional exact number of dimensions.
        trailing: Optional required final dimension.

    Returns:
        Numeric NumPy array view.

    Raises:
        SchemaError: If shape, dtype, or finiteness is invalid.
    """
    value = _require(record, field, sample)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise SchemaError(sample, field, "a numeric array", type(value).__name__) from error
    if not np.issubdtype(array.dtype, np.number):
        raise SchemaError(sample, field, "a numeric array", str(array.dtype))
    if ndim is not None and array.ndim != ndim:
        raise SchemaError(sample, field, f"an array with ndim={ndim}", array.shape)
    if trailing is not None and (array.ndim == 0 or array.shape[-1] != trailing):
        raise SchemaError(sample, field, f"an array ending in dimension {trailing}", array.shape)
    if not np.all(np.isfinite(array)):
        raise SchemaError(sample, field, "only finite values", "contains NaN or infinity")
    return array


def _validate_pose(record: Mapping[str, Any], field: str, sample: str | Path) -> np.ndarray:
    """Validate a position plus WXYZ quaternion pose.

    Args:
        record: Mapping containing the pose.
        field: Pose field name.
        sample: Source used in validation errors.

    Returns:
        Finite pose array.

    Raises:
        SchemaError: If pose shape or quaternion norm is invalid.
    """
    pose = _finite_array(record, field, sample, trailing=7)
    quaternion_norm = np.linalg.norm(pose[..., 3:7], axis=-1)
    if np.any(quaternion_norm <= 1e-12):
        raise SchemaError(sample, field, "a nonzero WXYZ quaternion", pose.shape)
    return pose


def validate_scene_record(record: Mapping[str, Any], sample: str | Path) -> dict[str, Any]:
    """Validate the scene fields consumed by all converters.

    Args:
        record: Loaded scene configuration.
        sample: Scene path used in validation errors.

    Returns:
        Mutable validated scene mapping.
    """
    scene = _require(record, "scene", sample)
    task = _require(record, "task", sample)
    if not isinstance(scene, Mapping):
        raise SchemaError(sample, "scene", "a mapping", type(scene).__name__)
    if not isinstance(task, Mapping):
        raise SchemaError(sample, "task", "a mapping", type(task).__name__)
    object_name = _require(task, "obj_name", sample)
    if object_name not in scene or not isinstance(scene[object_name], Mapping):
        raise SchemaError(sample, f"scene.{object_name}", "an object mapping", "missing")
    object_record = scene[object_name]
    for field in ("file_path", "pose", "scale"):
        _require(object_record, field, sample)
    _validate_pose(object_record, "pose", sample)
    scale = np.asarray(object_record["scale"])
    if scale.size == 0 or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise SchemaError(sample, f"scene.{object_name}.scale", "positive finite values", scale)
    return dict(record)


def validate_raw_record(record: Mapping[str, Any], converter: str, sample: str | Path) -> dict[str, Any]:
    """Validate fields specific to one supported raw converter.

    Args:
        record: Loaded raw record.
        converter: ``BODex``, ``Learning``, or ``Batched``.
        sample: Input path used in validation errors.

    Returns:
        Mutable validated raw record.

    Raises:
        SchemaError: If required fields or aligned batch dimensions are invalid.
    """
    schema_version(record, sample)
    _require(record, "scene_path", sample)
    if converter == "BODex":
        robot_pose = _finite_array(record, "robot_pose", sample)
        if robot_pose.ndim < 3 or robot_pose.shape[-1] < 7:
            raise SchemaError(sample, "robot_pose", "a pose trajectory ending in at least 7 values", robot_pose.shape)
        joint_names = _require(record, "joint_names", sample)
        if not isinstance(joint_names, (list, tuple, np.ndarray)):
            raise SchemaError(sample, "joint_names", "a sequence", type(joint_names).__name__)
    elif converter in {"Learning", "Batched"}:
        arrays = {
            field: _finite_array(record, field, sample) for field in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")
        }
        if any(array.shape != arrays["grasp_qpos"].shape for array in arrays.values()):
            raise SchemaError(sample, "*_qpos", "identical shapes", {k: v.shape for k, v in arrays.items()})
        if arrays["grasp_qpos"].shape[-1] < 7:
            raise SchemaError(sample, "grasp_qpos", "a final dimension of at least 7", arrays["grasp_qpos"].shape)
        for field, array in arrays.items():
            quaternion_norm = np.linalg.norm(array[..., 3:7], axis=-1)
            if np.any(quaternion_norm <= 1e-12):
                raise SchemaError(sample, field, "nonzero WXYZ quaternions", array.shape)
        if converter == "Batched":
            scene_scale = _finite_array(record, "scene_scale", sample)
            if arrays["grasp_qpos"].ndim != 2:
                raise SchemaError(sample, "grasp_qpos", "a two-dimensional batch", arrays["grasp_qpos"].shape)
            if scene_scale.reshape(-1).shape[0] != arrays["grasp_qpos"].shape[0]:
                raise SchemaError(
                    sample,
                    "scene_scale",
                    f"{arrays['grasp_qpos'].shape[0]} aligned values",
                    scene_scale.shape,
                )
    else:
        raise SchemaError(sample, "converter", "BODex, Learning, or Batched", converter)
    return dict(record)


def validate_grasp_record(
    record: Mapping[str, Any],
    sample: str | Path,
    *,
    expected_joint_dim: int | None = None,
    require_assets: bool = False,
) -> dict[str, Any]:
    """Validate a legacy-v0 or current-v1 grasp record.

    Args:
        record: Loaded grasp mapping.
        sample: Input path used in validation errors.
        expected_joint_dim: Optional exact qpos length.
        require_assets: Whether the object directory and metadata must exist.

    Returns:
        Mutable validated grasp record without changing its schema version.
    """
    schema_version(record, sample)
    object_path = _require(record, "obj_path", sample)
    if not isinstance(object_path, (str, Path)):
        raise SchemaError(sample, "obj_path", "a filesystem path", type(object_path).__name__)
    object_scale = np.asarray(_require(record, "obj_scale", sample))
    if object_scale.size != 1 or not np.all(np.isfinite(object_scale)) or float(object_scale.reshape(-1)[0]) <= 0:
        raise SchemaError(sample, "obj_scale", "one positive finite scalar", object_scale)
    _validate_pose(record, "obj_pose", sample)

    qpos = {
        field: _finite_array(record, field, sample, ndim=1) for field in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")
    }
    dimensions = {field: value.shape[0] for field, value in qpos.items()}
    if len(set(dimensions.values())) != 1:
        raise SchemaError(sample, "*_qpos", "equal one-dimensional lengths", dimensions)
    if expected_joint_dim is not None and dimensions["grasp_qpos"] != expected_joint_dim:
        raise SchemaError(sample, "grasp_qpos", f"length {expected_joint_dim}", dimensions["grasp_qpos"])
    if "joint_names" in record and len(record["joint_names"]) != dimensions["grasp_qpos"]:
        raise SchemaError(sample, "joint_names", f"length {dimensions['grasp_qpos']}", len(record["joint_names"]))

    if require_assets:
        asset_path = Path(object_path)
        if not asset_path.is_dir():
            raise SchemaError(sample, "obj_path", "an existing object directory", str(asset_path))
        metadata_path = asset_path / "info" / "simplified.json"
        if not metadata_path.is_file():
            raise SchemaError(sample, "obj_path", f"metadata at {metadata_path}", "missing")
    return dict(record)


def validate_control_record(record: Mapping[str, Any], sample: str | Path) -> dict[str, Any]:
    """Validate fields needed by statistics while accepting historical results.

    Args:
        record: Loaded control result.
        sample: Result path used in validation errors.

    Returns:
        Mutable validated control mapping.
    """
    schema_version(record, sample)
    object_poses = _require(record, "obj_pose", sample)
    if len(object_poses):
        poses = np.asarray(object_poses)
        if poses.ndim != 2 or poses.shape[1] != 7 or not np.all(np.isfinite(poses)):
            raise SchemaError(sample, "obj_pose", "a finite (T, 7) trajectory or an empty sequence", poses.shape)
    contacts = _require(record, "contacts", sample)
    if not isinstance(contacts, (list, tuple, np.ndarray)):
        raise SchemaError(sample, "contacts", "a sequence", type(contacts).__name__)
    status = record.get("episode_status")
    supported = {"completed", "invalid_initialization", "solver_degraded", "execution_error", None}
    if status not in supported:
        raise SchemaError(sample, "episode_status", f"one of {sorted(value for value in supported if value)}", status)
    return dict(record)
