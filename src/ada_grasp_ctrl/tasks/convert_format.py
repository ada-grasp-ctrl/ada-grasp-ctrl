"""Convert supported raw grasp formats into the versioned grasp schema."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from glob import glob
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ada_grasp_ctrl.batch import (
    SampleResult,
    SampleStatus,
    choose_inputs,
    execution_error,
    raise_for_batch_failures,
    write_batch_report,
)
from ada_grasp_ctrl.errors import PreflightError
from ada_grasp_ctrl.paths import map_path
from ada_grasp_ctrl.runtime import activate_runtime_roots, seed_sample, write_run_manifest
from ada_grasp_ctrl.schema import (
    SchemaError,
    load_npy_record,
    validate_grasp_record,
    validate_raw_record,
    validate_scene_record,
    with_current_schema,
)
from ada_grasp_ctrl.utils.rot_util import torch_matrix_to_quaternion, torch_quaternion_to_matrix
from ada_grasp_ctrl.utils.robots.base import RobotFactory


Converter = Callable[[str, Any], list[str]]


def _path_scalar(value: object) -> str:
    """Extract one path string from scalar or one-element legacy containers.

    Args:
        value: Raw scene-path value.

    Returns:
        Path encoded as a string.
    """
    array = np.asarray(value, dtype=object)
    if array.size == 0:
        return ""
    return str(array.reshape(-1)[0])


def _resolve_scene_path(raw_path: object, data_file: str, configs: Any) -> Path:
    """Resolve scene paths emitted by DexLearn and BODex layouts.

    Args:
        raw_path: Stored path or one-element container.
        data_file: Raw record path used as a relative-path anchor.
        configs: Runtime-normalized configuration containing the data root.

    Returns:
        Existing absolute scene path.

    Raises:
        SchemaError: If no supported relative or absolute candidate exists.
    """
    text = _path_scalar(raw_path)
    # Direct converter calls in downstream libraries may predate runtime root
    # normalization; the input directory remains a safe local fallback there.
    data_root = Path(configs.get("data_root", Path(data_file).parent))
    stored_path = Path(text).expanduser()
    candidates = (
        [stored_path] if stored_path.is_absolute() else [Path(data_file).parent / stored_path, data_root / stored_path]
    )
    marker = "src/curobo/content/"
    if marker in text:
        suffix = Path(text.split(marker, maxsplit=1)[1])
        candidates.extend([Path(data_file).parent / suffix, data_root / suffix])
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.is_file():
            return expanded.resolve()
    raise SchemaError(data_file, "scene_path", "an existing scene .npy file", text)


def _hand_qpos_contract(configs: Any) -> tuple[int, list[str]]:
    """Return the pose-plus-hand qpos dimension and exact hand joint order.

    Args:
        configs: Composed application configuration.

    Returns:
        Total pose-plus-hand dimension and ordered hand joint names.
    """
    robot_prefix = "" if configs.hand_name == "allegro" else "rh_"
    robot = RobotFactory.create_robot(robot_type=configs.hand_name, prefix=robot_prefix)
    return 7 + robot.n_dof, list(robot.dof_names)


def load_scene_cfg(scene_path: str | Path) -> dict[str, Any]:
    """Load a scene and make embedded relative file paths scene-relative.

    Args:
        scene_path: Scene ``.npy`` path.

    Returns:
        Validated mutable scene configuration.
    """
    path = Path(scene_path)
    scene_cfg = load_npy_record(path)

    def update_relative_paths(mapping: dict[str, Any]) -> None:
        """Resolve path-valued fields recursively in place.

        Args:
            mapping: Nested scene mapping.

        Returns:
            None.
        """
        for key, value in mapping.items():
            if isinstance(value, dict):
                update_relative_paths(value)
            elif key.endswith("_path") and isinstance(value, str) and not Path(value).is_absolute():
                mapping[key] = str((path.parent / value).resolve(strict=False))

    scene = scene_cfg.get("scene")
    if isinstance(scene, dict):
        update_relative_paths(scene)
    return validate_scene_record(scene_cfg, path)


def _object_fields(scene_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the common object fields written to grasp records.

    Args:
        scene_cfg: Validated scene configuration.

    Returns:
        Object path, pose, and base scale.
    """
    object_name = scene_cfg["task"]["obj_name"]
    object_cfg = scene_cfg["scene"][object_name]
    return {
        "obj_path": str(Path(object_cfg["file_path"]).parent.parent),
        "obj_pose": np.asarray(object_cfg["pose"]).copy(),
        "obj_scale": float(np.asarray(object_cfg["scale"]).reshape(-1)[0]),
    }


def _mapped_output(data_file: str, configs: Any) -> Path:
    """Map one raw input below the configured grasp output root.

    Args:
        data_file: Raw input path.
        configs: Composed application configuration.

    Returns:
        Output path retaining the input-relative hierarchy.
    """
    return map_path(data_file, configs.task.data_path, configs.grasp_dir)


def _bodex_output_paths(data_file: str, count: int, configs: Any) -> list[Path]:
    """Build BODex per-grasp paths compatible with the historical layout.

    Args:
        data_file: Raw BODex input path.
        count: Number of grasps in the record.
        configs: Composed application configuration.

    Returns:
        Deterministically ordered output paths.
    """
    mapped = _mapped_output(data_file, configs)
    suffix = "_grasp" if mapped.stem.endswith("_grasp") else "_mogen"
    directory = mapped.with_name(mapped.stem[: -len(suffix)])
    return [directory / f"{index}{suffix}.npy" for index in range(count)]


def _batched_output_paths(data_file: str, count: int, configs: Any) -> list[Path]:
    """Build sibling Batched outputs without recursively nesting indices.

    Args:
        data_file: Raw Batched input path.
        count: Number of grasps in the record.
        configs: Composed application configuration.

    Returns:
        Paths ``<mapped-stem>/<index>.npy`` for every batch element.
    """
    directory = _mapped_output(data_file, configs).with_suffix("")
    return [directory / f"{index}.npy" for index in range(count)]


def _save_grasp(
    path: Path,
    record: Mapping[str, Any],
    *,
    expected_qpos_dim: int,
    expected_joint_names: list[str],
) -> str:
    """Validate and save one version-one grasp record.

    Args:
        path: Destination ``.npy`` path.
        record: Grasp fields to serialize.
        expected_qpos_dim: Exact pose-plus-hand qpos dimension.
        expected_joint_names: Exact hand joint order when names are present.

    Returns:
        Absolute output path string.
    """
    current = with_current_schema(record)
    validate_grasp_record(
        current,
        path,
        expected_joint_dim=expected_qpos_dim,
        expected_joint_names=expected_joint_names,
        qpos_prefix_dim=7,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, current)
    return str(path.resolve(strict=False))


def BODex(params: tuple[str, Any]) -> list[str]:
    """Convert one BODex record while preserving its published numeric transforms.

    Args:
        params: Pair of raw path and composed configuration.

    Returns:
        Paths of all per-grasp outputs.
    """
    data_file, configs = params
    expected_qpos_dim, expected_joint_names = _hand_qpos_contract(configs)
    minimum_steps = 3 if configs.hand.mocap else 4
    raw_data = validate_raw_record(
        load_npy_record(data_file),
        "BODex",
        data_file,
        expected_qpos_dim=expected_qpos_dim,
        minimum_trajectory_steps=minimum_steps,
    )
    robot_pose = np.asarray(raw_data["robot_pose"])[0].copy()
    scene_path = _resolve_scene_path(raw_data["scene_path"], data_file, configs)
    scene_cfg = load_scene_cfg(scene_path)
    common = _object_fields(scene_cfg)
    common["scene_path"] = str(scene_path)

    if configs.hand_name == "shadow":
        common["joint_names"] = list(raw_data["joint_names"])[5:] + list(raw_data["joint_names"])[:5]
        robot_pose = np.concatenate([robot_pose[:, :, :7], robot_pose[:, :, 12:], robot_pose[:, :, 7:12]], axis=-1)
        rotation = torch_quaternion_to_matrix(torch.tensor(robot_pose[:, :, 3:7], dtype=torch.float32))
        robot_pose[:, :, :3] -= (rotation @ torch.tensor([0, 0, 0.034]).view(1, 1, 3, 1)).squeeze(-1).numpy()
    elif configs.hand_name == "allegro":
        common["joint_names"] = list(raw_data["joint_names"])
        rotation = torch_quaternion_to_matrix(torch.tensor(robot_pose[:, :, 3:7]))
        # Keep the fixed frame correction beside the source tensor so float32
        # and float64 BODex records follow the same supported code path.
        delta_quaternion = rotation.new_tensor([0, 1, 0, 1]).view(1, 1, 4)
        delta = torch_quaternion_to_matrix(delta_quaternion)
        robot_pose[:, :, 3:7] = torch_matrix_to_quaternion(rotation @ delta.transpose(-1, -2))
    elif configs.hand_name == "leap_tac3d":
        common["joint_names"] = list(raw_data["joint_names"])
    else:
        raise ValueError(
            f"{data_file}: BODex converter supports shadow, allegro, and leap_tac3d; got {configs.hand_name}."
        )

    output_paths = _bodex_output_paths(data_file, len(robot_pose), configs)
    saved = []
    for index, output_path in enumerate(output_paths):
        grasp = deepcopy(common)
        if configs.hand.mocap:
            grasp.update(
                pregrasp_qpos=robot_pose[index, 0],
                grasp_qpos=robot_pose[index, 1],
                squeeze_qpos=robot_pose[index, 2],
            )
        else:
            grasp.update(
                approach_qpos=robot_pose[index, :-4],
                pregrasp_qpos=robot_pose[index, -4],
                grasp_qpos=robot_pose[index, -3],
                squeeze_qpos=robot_pose[index, -2],
                lift_qpos=robot_pose[index, -1],
            )
        saved.append(
            _save_grasp(
                output_path,
                grasp,
                expected_qpos_dim=expected_qpos_dim,
                expected_joint_names=expected_joint_names,
            )
        )
    return saved


def Learning(params: tuple[str, Any]) -> list[str]:
    """Convert one single-grasp DexLearn record.

    Args:
        params: Pair of raw path and composed configuration.

    Returns:
        A one-element output path list.
    """
    data_file, configs = params
    expected_qpos_dim, expected_joint_names = _hand_qpos_contract(configs)
    raw_data = validate_raw_record(
        load_npy_record(data_file),
        "Learning",
        data_file,
        expected_qpos_dim=expected_qpos_dim,
    )
    scene_path = _resolve_scene_path(raw_data["scene_path"], data_file, configs)
    grasp = _object_fields(load_scene_cfg(scene_path))
    for field in ("grasp_qpos", "pregrasp_qpos", "squeeze_qpos"):
        grasp[field] = np.asarray(raw_data[field]).copy()
    return [
        _save_grasp(
            _mapped_output(data_file, configs),
            grasp,
            expected_qpos_dim=expected_qpos_dim,
            expected_joint_names=expected_joint_names,
        )
    ]


def Batched(params: tuple[str, Any]) -> list[str]:
    """Convert every element of one batched DexLearn record.

    Args:
        params: Pair of raw path and composed configuration.

    Returns:
        Sibling output paths for every batch element.
    """
    data_file, configs = params
    expected_qpos_dim, expected_joint_names = _hand_qpos_contract(configs)
    raw_data = validate_raw_record(
        load_npy_record(data_file),
        "Batched",
        data_file,
        expected_qpos_dim=expected_qpos_dim,
    )
    common = _object_fields(load_scene_cfg(_resolve_scene_path(raw_data["scene_path"], data_file, configs)))
    count = np.asarray(raw_data["grasp_qpos"]).shape[0]
    output_paths = _batched_output_paths(data_file, count, configs)
    saved = []
    for index, output_path in enumerate(output_paths):
        grasp = deepcopy(common)
        grasp["obj_scale"] = common["obj_scale"] * float(np.asarray(raw_data["scene_scale"]).reshape(-1)[index])
        for field in ("grasp_qpos", "pregrasp_qpos", "squeeze_qpos"):
            grasp[field] = np.asarray(raw_data[field])[index].copy()
        saved.append(
            _save_grasp(
                output_path,
                grasp,
                expected_qpos_dim=expected_qpos_dim,
                expected_joint_names=expected_joint_names,
            )
        )
    return saved


CONVERTER_REGISTRY: dict[str, Converter] = {
    "BODex": BODex,
    "Learning": Learning,
    "Batched": Batched,
}


def _expected_outputs(data_file: str, configs: Any) -> list[str]:
    """Predict outputs for skip filtering without writing files.

    Args:
        data_file: Raw record path.
        configs: Composed application configuration.

    Returns:
        Expected output paths, or an empty list if the damaged input must be processed.
    """
    try:
        raw = load_npy_record(data_file)
        converter = configs.task.data_name
        if converter == "BODex":
            count = np.asarray(raw["robot_pose"])[0].shape[0]
            return [str(path) for path in _bodex_output_paths(data_file, count, configs)]
        if converter == "Batched":
            count = np.asarray(raw["grasp_qpos"]).shape[0]
            return [str(path) for path in _batched_output_paths(data_file, count, configs)]
        return [str(_mapped_output(data_file, configs))]
    except Exception:
        # Damaged records are intentionally sent to the worker so the report
        # contains their contextual validation error instead of silently skipping them.
        return []


def _convert_one(params: tuple[str, Any, int]) -> SampleResult:
    """Run one converter and capture an exception as a structured result.

    Args:
        params: Raw path, composed configuration, and stable sample index.

    Returns:
        Completed or execution-error sample result.
    """
    data_file, configs, sample_index = params
    activate_runtime_roots(configs)
    derived_seed = seed_sample(int(configs.seed), sample_index)
    try:
        output_paths = CONVERTER_REGISTRY[configs.task.data_name]((data_file, configs))
        return SampleResult(
            data_file,
            SampleStatus.COMPLETED,
            output_paths=output_paths,
            details={"sample_index": sample_index, "sample_seed": derived_seed},
        )
    except Exception as error:
        result = execution_error(data_file, error)
        result.details.update({"sample_index": sample_index, "sample_seed": derived_seed})
        return result


def task_format(configs: Any) -> None:
    """Discover and convert raw records with complete batch reporting.

    Args:
        configs: Composed application configuration.

    Returns:
        None.

    Raises:
        PreflightError: If the converter is unknown or the raw input is empty.
        BatchExecutionError: After processing when at least one record failed.
    """
    converter = configs.task.data_name
    if converter not in CONVERTER_REGISTRY:
        supported = ", ".join(sorted(CONVERTER_REGISTRY))
        raise PreflightError(f"Unsupported converter '{converter}'. Supported converters: {supported}.")
    pattern = "*_grasp.npy" if converter == "BODex" and configs.hand.mocap else "*_mogen.npy"
    if converter != "BODex":
        pattern = "*.npy"
    discovered = sorted(glob(str(Path(configs.task.data_path) / "**" / pattern), recursive=True))
    if not discovered:
        raise PreflightError(f"No raw '{converter}' .npy inputs found below {configs.task.data_path}.")

    expected = [_expected_outputs(path, configs) for path in discovered]
    selected, num_skipped = choose_inputs(
        discovered,
        skip=bool(configs.skip),
        output_paths=expected,
        max_num=int(configs.task.max_num),
        seed=int(configs.seed),
    )
    write_run_manifest(configs, discovered)
    logging.info(
        "Found %d raw files below %s, skipped %d, processing %d.",
        len(discovered),
        configs.task.data_path,
        num_skipped,
        len(selected),
    )

    sample_indices = {path: index for index, path in enumerate(discovered)}
    params = [(path, configs, sample_indices[path]) for path in selected]
    if params:
        with multiprocessing.Pool(processes=configs.n_worker) as pool:
            results = list(pool.imap_unordered(_convert_one, params))
    else:
        results = []
    summary = write_batch_report(
        Path(configs.log_dir),
        "format",
        results,
        num_discovered=len(discovered),
        num_skipped=num_skipped,
    )
    logging.info("Format conversion wrote %d grasp file(s).", len(list(Path(configs.grasp_dir).rglob("*.npy"))))
    raise_for_batch_failures(summary)
