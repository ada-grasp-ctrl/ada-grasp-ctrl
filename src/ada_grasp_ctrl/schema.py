"""Versioned validation for raw, grasp, and control NumPy records."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any, Sequence

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


def _validate_joint_names(
    value: object,
    field: str,
    sample: str | Path,
    *,
    expected_length: int,
    expected_names: Sequence[str] | None = None,
) -> list[str]:
    """Validate a unique one-dimensional joint-name sequence.

    Args:
        value: Candidate joint-name container.
        field: Contextual field path.
        sample: Source used in validation errors.
        expected_length: Exact number of joint names.
        expected_names: Optional exact ordered names.

    Returns:
        Joint names as a plain list.

    Raises:
        SchemaError: If type, length, uniqueness, or order is invalid.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, np.ndarray)):
        raise SchemaError(sample, field, "a one-dimensional string sequence", type(value).__name__)
    array = np.asarray(value, dtype=object)
    if array.ndim != 1:
        raise SchemaError(sample, field, "a one-dimensional string sequence", array.shape)
    names = array.tolist()
    if len(names) != expected_length:
        raise SchemaError(sample, field, f"length {expected_length}", len(names))
    if any(not isinstance(name, str) or not name for name in names):
        raise SchemaError(sample, field, "nonempty strings", names)
    if len(set(names)) != len(names):
        raise SchemaError(sample, field, "unique names", names)
    if expected_names is not None and names != list(expected_names):
        raise SchemaError(sample, field, "the configured joint order", names)
    return names


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
        if field not in object_record:
            raise SchemaError(sample, f"scene.{object_name}.{field}", "a required field", "missing")
    file_field = f"scene.{object_name}.file_path"
    file_path = object_record["file_path"]
    if not isinstance(file_path, (str, Path)):
        raise SchemaError(sample, file_field, "a filesystem path", type(file_path).__name__)
    if not Path(file_path).is_file():
        raise SchemaError(sample, file_field, "an existing object mesh file", str(file_path))
    pose_field = f"scene.{object_name}.pose"
    _validate_pose({pose_field: object_record["pose"]}, pose_field, sample)
    scale = np.asarray(object_record["scale"])
    if scale.size == 0 or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise SchemaError(sample, f"scene.{object_name}.scale", "positive finite values", scale)
    return dict(record)


def validate_raw_record(
    record: Mapping[str, Any],
    converter: str,
    sample: str | Path,
    *,
    expected_qpos_dim: int | None = None,
    minimum_trajectory_steps: int = 3,
) -> dict[str, Any]:
    """Validate fields specific to one supported raw converter.

    Args:
        record: Loaded raw record.
        converter: ``BODex``, ``Learning``, or ``Batched``.
        sample: Input path used in validation errors.
        expected_qpos_dim: Optional exact final qpos/robot-pose dimension.
        minimum_trajectory_steps: Minimum BODex approach/control trajectory length.

    Returns:
        Mutable validated raw record.

    Raises:
        SchemaError: If required fields or aligned batch dimensions are invalid.
    """
    schema_version(record, sample)
    _require(record, "scene_path", sample)
    if converter == "BODex":
        robot_pose = _finite_array(record, "robot_pose", sample, ndim=4)
        if robot_pose.shape[0] != 1:
            raise SchemaError(sample, "robot_pose", "shape (1, N, T, D)", robot_pose.shape)
        if robot_pose.shape[2] < minimum_trajectory_steps:
            raise SchemaError(
                sample,
                "robot_pose",
                f"at least {minimum_trajectory_steps} trajectory steps",
                robot_pose.shape,
            )
        if expected_qpos_dim is not None and robot_pose.shape[-1] != expected_qpos_dim:
            raise SchemaError(sample, "robot_pose", f"final dimension {expected_qpos_dim}", robot_pose.shape)
        if robot_pose.shape[-1] < 8:
            raise SchemaError(sample, "robot_pose", "a pose plus at least one joint", robot_pose.shape)
        _validate_joint_names(
            _require(record, "joint_names", sample),
            "joint_names",
            sample,
            expected_length=robot_pose.shape[-1] - 7,
        )
    elif converter in {"Learning", "Batched"}:
        expected_ndim = 1 if converter == "Learning" else 2
        arrays = {
            field: _finite_array(record, field, sample, ndim=expected_ndim)
            for field in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")
        }
        if any(array.shape != arrays["grasp_qpos"].shape for array in arrays.values()):
            raise SchemaError(sample, "*_qpos", "identical shapes", {k: v.shape for k, v in arrays.items()})
        if expected_qpos_dim is not None and arrays["grasp_qpos"].shape[-1] != expected_qpos_dim:
            raise SchemaError(sample, "grasp_qpos", f"final dimension {expected_qpos_dim}", arrays["grasp_qpos"].shape)
        if arrays["grasp_qpos"].shape[-1] < 8:
            raise SchemaError(sample, "grasp_qpos", "a pose plus at least one joint", arrays["grasp_qpos"].shape)
        for field, array in arrays.items():
            quaternion_norm = np.linalg.norm(array[..., 3:7], axis=-1)
            if np.any(quaternion_norm <= 1e-12):
                raise SchemaError(sample, field, "nonzero WXYZ quaternions", array.shape)
        if converter == "Batched":
            scene_scale = _finite_array(record, "scene_scale", sample, ndim=1)
            if scene_scale.shape[0] != arrays["grasp_qpos"].shape[0]:
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
    expected_joint_names: Sequence[str] | None = None,
    qpos_prefix_dim: int = 0,
    require_joint_names: bool = False,
    require_assets: bool = False,
) -> dict[str, Any]:
    """Validate a legacy-v0 or current-v1 grasp record.

    Args:
        record: Loaded grasp mapping.
        sample: Input path used in validation errors.
        expected_joint_dim: Optional exact qpos length.
        expected_joint_names: Optional exact ordered joint-name sequence.
        qpos_prefix_dim: Leading non-joint qpos values, such as a seven-value free pose.
        require_joint_names: Whether the record must declare joint names.
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
    if qpos_prefix_dim < 0 or qpos_prefix_dim > dimensions["grasp_qpos"]:
        raise SchemaError(sample, "qpos_prefix_dim", f"between 0 and {dimensions['grasp_qpos']}", qpos_prefix_dim)
    expected_name_count = dimensions["grasp_qpos"] - qpos_prefix_dim
    if expected_joint_names is not None and len(expected_joint_names) != expected_name_count:
        raise SchemaError(
            sample,
            "joint_names",
            f"{expected_name_count} configured names",
            len(expected_joint_names),
        )
    if "joint_names" in record:
        _validate_joint_names(
            record["joint_names"],
            "joint_names",
            sample,
            expected_length=expected_name_count,
            expected_names=expected_joint_names,
        )
    elif require_joint_names:
        raise SchemaError(sample, "joint_names", "a required field", "missing")

    if require_assets:
        asset_path = Path(object_path)
        if not asset_path.is_dir():
            raise SchemaError(sample, "obj_path", "an existing object directory", str(asset_path))
        metadata_path = asset_path / "info" / "simplified.json"
        if not metadata_path.is_file():
            raise SchemaError(sample, "obj_path", f"metadata at {metadata_path}", "missing")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SchemaError(sample, "obj_path.info.simplified.json", "valid JSON metadata", str(error)) from error
        if not isinstance(metadata, Mapping):
            raise SchemaError(sample, "obj_path.info.simplified.json", "a mapping", type(metadata).__name__)
        for field in ("mass", "density", "scale"):
            value = np.asarray(metadata.get(field))
            if value.size != 1 or not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
                raise SchemaError(
                    sample, f"obj_path.info.simplified.json.{field}", "one finite number", metadata.get(field)
                )
            if float(value.reshape(-1)[0]) <= 0:
                raise SchemaError(
                    sample,
                    f"obj_path.info.simplified.json.{field}",
                    "a positive finite number",
                    metadata.get(field),
                )
        collision_directory = asset_path / "urdf" / "meshes"
        if not collision_directory.is_dir():
            raise SchemaError(
                sample,
                "obj_path",
                f"a collision mesh directory at {collision_directory}",
                "missing",
            )
        collision_meshes = sorted(path for path in collision_directory.glob("convex_piece_*.obj") if path.is_file())
        if not collision_meshes:
            raise SchemaError(
                sample,
                "obj_path",
                f"at least one collision mesh below {collision_directory}",
                "none found",
            )
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
        poses = _validate_pose(record, "obj_pose", sample)
        if poses.ndim != 2:
            raise SchemaError(sample, "obj_pose", "a finite (T, 7) trajectory or an empty sequence", poses.shape)
    contacts = _require(record, "contacts", sample)
    if not isinstance(contacts, (list, tuple, np.ndarray)):
        raise SchemaError(sample, "contacts", "a sequence", type(contacts).__name__)
    contact_count = len(contacts)
    for field in ("dof", "doa", "planned_dof"):
        if field not in record:
            continue
        trajectory = np.asarray(record[field])
        if len(trajectory) != contact_count:
            raise SchemaError(sample, field, f"{contact_count} steps aligned with contacts", len(trajectory))
        if len(trajectory) and (trajectory.ndim != 2 or not np.issubdtype(trajectory.dtype, np.number)):
            raise SchemaError(sample, field, "a finite two-dimensional trajectory", trajectory.shape)
        if len(trajectory) and not np.all(np.isfinite(trajectory)):
            raise SchemaError(sample, field, "only finite values", "contains NaN or infinity")

    for step_index, step_contacts in enumerate(contacts):
        step_field = f"contacts[{step_index}]"
        if isinstance(step_contacts, (str, bytes)) or not isinstance(step_contacts, (list, tuple, np.ndarray)):
            raise SchemaError(sample, step_field, "a contact sequence", type(step_contacts).__name__)
        for contact_index, contact in enumerate(step_contacts):
            contact_field = f"{step_field}[{contact_index}]"
            if not isinstance(contact, Mapping):
                raise SchemaError(sample, contact_field, "a contact mapping", type(contact).__name__)
            for field, shape in (("contact_pos", (3,)), ("contact_force", (6,)), ("contact_frame", (3, 3))):
                field_path = f"{contact_field}.{field}"
                array = _finite_array({field_path: contact.get(field)}, field_path, sample)
                if array.shape != shape:
                    raise SchemaError(sample, field_path, f"shape {shape}", array.shape)
            frame = np.asarray(contact["contact_frame"])
            if not np.allclose(frame.T @ frame, np.eye(3), rtol=1e-5, atol=1e-6):
                raise SchemaError(sample, f"{contact_field}.contact_frame", "an orthonormal frame", frame)
            determinant = float(np.linalg.det(frame))
            if not np.isclose(determinant, 1.0, rtol=1e-5, atol=1e-6):
                raise SchemaError(
                    sample,
                    f"{contact_field}.contact_frame",
                    "a right-handed frame with determinant +1",
                    determinant,
                )
    status = record.get("episode_status")
    supported = {"completed", "invalid_initialization", "solver_degraded", "execution_error", None}
    if status not in supported:
        raise SchemaError(sample, "episode_status", f"one of {sorted(value for value in supported if value)}", status)
    return dict(record)
