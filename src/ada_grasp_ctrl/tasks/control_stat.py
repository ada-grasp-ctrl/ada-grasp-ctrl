"""Compute robust statistics for legacy and versioned control trajectories."""

from __future__ import annotations

from glob import glob
import logging
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ada_grasp_ctrl.batch import (
    SampleResult,
    SampleStatus,
    execution_error,
    raise_for_batch_failures,
    write_batch_report,
)
from ada_grasp_ctrl.errors import PreflightError
from ada_grasp_ctrl.runtime import write_run_manifest
from ada_grasp_ctrl.schema import load_npy_record, validate_control_record
from ada_grasp_ctrl.utils.grasp_controller import GraspController
from ada_grasp_ctrl.utils.robots.base import RobotFactory


SUPPORTED_METHODS = {"ours", "op", "bs1", "bs2", "bs3"}


def read_data(npy_path: str) -> dict[str, Any]:
    """Load and validate one historical or versioned control result.

    Args:
        npy_path: Control ``.npy`` path.

    Returns:
        Validated control mapping.
    """
    return validate_control_record(load_npy_record(npy_path), npy_path)


def read_data_with_index(args: tuple[int, str]) -> tuple[int, str, dict[str, Any] | None, SampleResult | None]:
    """Read one result while retaining deterministic source order.

    Args:
        args: Original index and input path.

    Returns:
        Index, path, optional data, and optional structured read error.
    """
    index, npy_path = args
    try:
        return index, npy_path, read_data(npy_path), None
    except Exception as error:
        return index, npy_path, None, execution_error(npy_path, error)


def _nullable_summary(values: list[float]) -> dict[str, float | None]:
    """Return finite mean/std values or YAML-null placeholders.

    Args:
        values: Scalar observations.

    Returns:
        Mapping with ``mean`` and ``std``.
    """
    if not values:
        return {"mean": None, "std": None}
    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "std": float(np.std(array))}


def _build_grasp_controller(configs: Any) -> GraspController:
    """Build the grasp-matrix helper for the configured robot.

    Args:
        configs: Composed application configuration.

    Returns:
        Controller used only for contact wrench calculations.
    """
    from ada_grasp_ctrl.utils.pin_helper import PinocchioHelper
    from ada_grasp_ctrl.utils.robot_adaptor import RobotAdaptor

    robot_prefix = "rh_" if "allegro" not in configs.hand_name else ""
    robot = RobotFactory.create_robot(robot_type=configs.hand_name, prefix=robot_prefix)
    robot_model = PinocchioHelper(robot_file_path=robot.get_file_path("mjcf"), robot_file_type="mjcf")
    robot_adaptor = RobotAdaptor(
        robot_model=robot_model,
        dof_names=robot.dof_names,
        doa_names=robot.doa_names,
        doa2dof_matrix=robot.doa2dof_matrix,
    )
    return GraspController(configs=None, robot=robot, robot_adaptor=robot_adaptor)


def _quaternion_angle_degrees(target_wxyz: np.ndarray, actual_wxyz: np.ndarray) -> float:
    """Compute the relative rotation magnitude for WXYZ quaternions.

    Args:
        target_wxyz: Target quaternion.
        actual_wxyz: Actual quaternion.

    Returns:
        Angular error in degrees.
    """
    target_xyzw = np.roll(target_wxyz, -1)
    actual_xyzw = np.roll(actual_wxyz, -1)
    relative = Rotation.from_quat(target_xyzw).inv() * Rotation.from_quat(actual_xyzw)
    return float(np.rad2deg(relative.magnitude()))


def _method_directory(configs: Any) -> str:
    """Resolve the control output directory name for a method.

    Args:
        configs: Composed application configuration.

    Returns:
        Directory component such as ``ours_default`` or ``op``.
    """
    method = str(configs.task.method)
    if method == "ours":
        return f"{method}_{configs.task.ablation_name}"
    return method


def _statistics_path(configs: Any) -> Path:
    """Build the YAML statistics output path.

    Args:
        configs: Composed application configuration.

    Returns:
        Destination YAML path.
    """
    output_dir = Path(configs.control_dir).parent / "control_stat_res"
    return output_dir / f"{configs.task.setting_name}_{_method_directory(configs)}.yaml"


def get_control_results(
    indexed_data: list[tuple[int, str, dict[str, Any] | None, SampleResult | None]],
    configs: Any,
) -> tuple[dict[str, Any], list[SampleResult], Path]:
    """Classify trajectories and save statistics with explicit denominator semantics.

    Args:
        indexed_data: Ordered loaded records and per-file read failures.
        configs: Composed application configuration.

    Returns:
        Statistics mapping, per-sample results, and YAML output path.
    """
    non_scientific_statuses = {
        SampleStatus.EXECUTION_ERROR.value,
        SampleStatus.INVALID_INITIALIZATION.value,
        SampleStatus.SOLVER_DEGRADED.value,
    }
    usable_data = [
        data
        for _, _, data, error in indexed_data
        if data is not None
        and error is None
        and data.get("episode_status") not in non_scientific_statuses
        and len(data["obj_pose"]) > 0
    ]
    grasp_ctrl = _build_grasp_controller(configs) if usable_data else None
    lift_height = float(configs.task.lift_height)
    n_terminal_steps = int(configs.task.n_terminal_steps)

    position_errors: list[float] = []
    angle_errors: list[float] = []
    summed_forces: list[float] = []
    wrench_norms: list[float] = []
    normalized_wrench_norms: list[float] = []
    success_cases: list[int] = []
    failure_cases: list[int] = []
    invalid_cases: list[int] = []
    degraded_cases: list[int] = []
    sample_results: list[SampleResult] = []
    sample_status: list[dict[str, Any]] = []

    for index, path, record, read_error in indexed_data:
        if read_error is not None:
            sample_results.append(read_error)
            sample_status.append({"index": index, "path": path, "status": SampleStatus.EXECUTION_ERROR.value})
            continue
        assert record is not None
        declared_status = record.get("episode_status")
        if declared_status == SampleStatus.EXECUTION_ERROR.value:
            message = record.get("error_message") or record.get("message") or "Control record declares execution_error."
            recorded_traceback = record.get("traceback")
            result = SampleResult(
                path,
                SampleStatus.EXECUTION_ERROR,
                output_paths=[path],
                message=str(message),
                traceback=str(recorded_traceback) if recorded_traceback is not None else None,
            )
            sample_results.append(result)
            sample_status.append({"index": index, "path": path, "status": result.status.value})
            continue
        if declared_status == SampleStatus.SOLVER_DEGRADED.value:
            degraded_cases.append(index)
            result = SampleResult(path, SampleStatus.SOLVER_DEGRADED, output_paths=[path])
            sample_results.append(result)
            sample_status.append({"index": index, "path": path, "status": result.status.value})
            continue
        if declared_status == SampleStatus.INVALID_INITIALIZATION.value or len(record["obj_pose"]) == 0:
            invalid_cases.append(index)
            result = SampleResult(path, SampleStatus.INVALID_INITIALIZATION, output_paths=[path])
            sample_results.append(result)
            sample_status.append({"index": index, "path": path, "status": result.status.value})
            continue

        sequence = np.asarray(record["obj_pose"])
        initial_pose = sequence[0]
        target_pose = initial_pose.copy()
        target_pose[2] += lift_height
        final_pose = sequence[-1]
        position_error = float(np.linalg.norm(target_pose[:3] - final_pose[:3]))
        z_error = float(abs(target_pose[2] - final_pose[2]))
        angle_error = _quaternion_angle_degrees(target_pose[3:], final_pose[3:])

        contacts_sequence = record["contacts"]
        sequence_wrench = np.zeros((len(contacts_sequence), 6))
        sequence_normalized_wrench = np.zeros((len(contacts_sequence), 6))
        sequence_sum_force = np.zeros(len(contacts_sequence))
        for step, contacts in enumerate(contacts_sequence):
            if not contacts:
                continue
            assert grasp_ctrl is not None
            grasp_matrix = grasp_ctrl.compute_grasp_matrix(contacts)
            contact_forces = np.concatenate([contact["contact_force"][:3] for contact in contacts])
            sequence_wrench[step] = (grasp_matrix @ contact_forces.reshape(-1, 1)).reshape(-1)
            sequence_normalized_wrench[step] = grasp_ctrl.compute_normalized_wrench(grasp_matrix, contact_forces)
            sequence_sum_force[step] = np.sum(contact_forces.reshape(-1, 3)[:, 0])

        if z_error < lift_height / 2.0:
            outcome = "success"
            success_cases.append(index)
            position_errors.append(position_error)
            angle_errors.append(angle_error)
            terminal = slice(-n_terminal_steps, None) if n_terminal_steps > 0 else slice(None)
            summed_forces.append(float(np.mean(sequence_sum_force[terminal])))
            wrench_norms.append(float(np.mean(np.linalg.norm(sequence_wrench[terminal], axis=-1))))
            normalized_wrench_norms.append(
                float(np.mean(np.linalg.norm(sequence_normalized_wrench[terminal], axis=-1)))
            )
        else:
            outcome = "failure"
            failure_cases.append(index)
        result = SampleResult(
            path,
            SampleStatus.COMPLETED,
            output_paths=[path],
            details={"scientific_outcome": outcome},
        )
        sample_results.append(result)
        sample_status.append(
            {"index": index, "path": path, "status": result.status.value, "scientific_outcome": outcome}
        )

    valid_count = len(success_cases) + len(failure_cases)
    success_rate = len(success_cases) / valid_count if valid_count else None
    execution_error_count = sum(result.status == SampleStatus.EXECUTION_ERROR for result in sample_results)
    stat_results = {
        # Preserve all historical keys and replace undefined floating values with YAML null.
        "success_rate": success_rate,
        "ave_obj_pos_err": _nullable_summary(position_errors),
        "ave_obj_angle_err": _nullable_summary(angle_errors),
        "ave_sum_cf_all": _nullable_summary(summed_forces),
        "ave_wrench_all": _nullable_summary(wrench_norms),
        "ave_normalized_wrench_all": _nullable_summary(normalized_wrench_norms),
        "failure_cases": failure_cases,
        "num_invalid_cases": len(invalid_cases),
        "num_valid_cases": valid_count,
        # New counts make the primary success-rate denominator auditable.
        "num_total": len(indexed_data),
        "success": len(success_cases),
        "failure": len(failure_cases),
        "invalid_initialization": len(invalid_cases),
        "solver_degraded": len(degraded_cases),
        "execution_error": execution_error_count,
        "success_rate_denominator": valid_count,
        "sample_status": sample_status,
    }
    save_path = _statistics_path(configs)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(stat_results, file_obj, sort_keys=False)
    return stat_results, sample_results, save_path


def task_control_stat(configs: Any) -> None:
    """Load matching control files, save robust statistics, and write reports.

    Args:
        configs: Composed application configuration.

    Returns:
        None.

    Raises:
        PreflightError: If method configuration is unsupported.
        BatchExecutionError: After statistics are saved if any input is unreadable or degraded.
    """
    if configs.task.method not in SUPPORTED_METHODS:
        raise PreflightError(
            f"Unsupported statistics method '{configs.task.method}'. Supported: {sorted(SUPPORTED_METHODS)}."
        )
    method_directory = _method_directory(configs)
    setting_name = str(configs.task.setting_name)
    discovered = sorted(
        path
        for path in glob(str(Path(configs.control_dir) / "**" / "*.npy"), recursive=True)
        if Path(path).parent.name == method_directory and setting_name in path
    )
    write_run_manifest(configs, discovered)
    logging.info(
        "Found %d control result(s) for method '%s' and setting '%s' in %s.",
        len(discovered),
        method_directory,
        setting_name,
        configs.control_dir,
    )

    indexed_data: list[tuple[int, str, dict[str, Any] | None, SampleResult | None]] = []
    if discovered:
        with multiprocessing.Pool(processes=configs.n_worker) as pool:
            unordered = pool.imap_unordered(read_data_with_index, enumerate(discovered))
            indexed_data = sorted(unordered, key=lambda value: value[0])
    statistics, results, output_path = get_control_results(indexed_data, configs)
    summary = write_batch_report(
        Path(configs.log_dir),
        "control_stat",
        results,
        num_discovered=len(discovered),
        num_skipped=0,
    )
    logging.info(
        "Saved statistics to %s (success=%d, failure=%d, invalid=%d, degraded=%d, errors=%d).",
        output_path,
        statistics["success"],
        statistics["failure"],
        statistics["invalid_initialization"],
        statistics["solver_degraded"],
        statistics["execution_error"],
    )
    raise_for_batch_failures(summary)
