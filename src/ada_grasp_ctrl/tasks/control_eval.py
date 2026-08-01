"""Run supported control policies with explicit dispatch and batch reports."""

from __future__ import annotations

from glob import glob
import logging
import multiprocessing
from pathlib import Path
from typing import Any

from ada_grasp_ctrl.batch import (
    SampleResult,
    SampleStatus,
    choose_inputs,
    execution_error,
    raise_for_batch_failures,
    write_batch_report,
)
from ada_grasp_ctrl.errors import PreflightError
from ada_grasp_ctrl.optimization import DEFAULT_SOLVER_FAILURE_POLICY, SOLVER_FAILURE_POLICIES
from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.runtime import activate_runtime_roots, seed_sample, write_run_manifest
from ada_grasp_ctrl.schema import SchemaError, load_npy_record, validate_grasp_record
from ada_grasp_ctrl.tasks.control_eval_func.base import control_output_path
from ada_grasp_ctrl.tasks.control_eval_func.tabletop_dummy_arm_bs1 import tabletopDummyArmBS1Eval
from ada_grasp_ctrl.tasks.control_eval_func.tabletop_dummy_arm_bs2 import tabletopDummyArmBS2Eval
from ada_grasp_ctrl.tasks.control_eval_func.tabletop_dummy_arm_bs3 import tabletopDummyArmBS3Eval
from ada_grasp_ctrl.tasks.control_eval_func.tabletop_dummy_arm_op import tabletopDummyArmOpEval
from ada_grasp_ctrl.tasks.control_eval_func.tabletop_dummy_arm_ours import tabletopDummyArmOursEval
from ada_grasp_ctrl.utils.grasp_controller import GraspControllerParameters
from ada_grasp_ctrl.utils.viewer_session import DebugViewerError, create_debug_viewer_session


METHOD_REGISTRY = {
    ("tabletop", "op"): tabletopDummyArmOpEval,
    ("tabletop", "ours"): tabletopDummyArmOursEval,
    ("tabletop", "bs1"): tabletopDummyArmBS1Eval,
    ("tabletop", "bs2"): tabletopDummyArmBS2Eval,
    ("tabletop", "bs3"): tabletopDummyArmBS3Eval,
}

SUPPORTED_HANDS = {"dummy_arm_shadow", "dummy_arm_allegro", "dummy_arm_leap_tac3d"}


def _file_suffixes(offsets: list[float]) -> list[str]:
    """Expand configured offsets into deterministic result suffixes.

    Args:
        offsets: Planar object perturbation distances in metres.

    Returns:
        Suffixes matching :class:`BaseEval` output generation.
    """
    suffixes = []
    for offset in offsets:
        count = 1 if float(offset) == 0.0 else 8
        suffixes.extend(f"_dist_{int(100 * float(offset))}_pos_{index}" for index in range(count))
    return suffixes


def _expected_outputs(input_path: str, configs: Any) -> list[str]:
    """Return every output expected for one grasp input.

    Args:
        input_path: Source grasp record.
        configs: Composed application configuration.

    Returns:
        Output path strings for all configured perturbations.
    """
    return [
        str(control_output_path(input_path, configs, suffix)) for suffix in _file_suffixes(list(configs.task.offsets))
    ]


def safe_eval_one(params: tuple[str, Any, object, int]) -> SampleResult:
    """Evaluate one grasp and convert all outcomes into one structured result.

    Args:
        params: Input path, composed configuration, optional viewer session, and stable sample index.

    Returns:
        Completed, invalid-initialization, degraded, or execution-error result.
    """
    input_npy_path, configs, debug_viewer_session, sample_index = params
    activate_runtime_roots(configs)
    derived_seed = seed_sample(int(configs.seed), sample_index)
    try:
        evaluator_class = METHOD_REGISTRY[(configs.setting, configs.task.method)]
        output_paths, statuses = evaluator_class(
            input_npy_path,
            configs,
            debug_viewer_session=debug_viewer_session,
        ).run()
        if SampleStatus.SOLVER_DEGRADED in statuses:
            status = SampleStatus.SOLVER_DEGRADED
        elif statuses and all(value == SampleStatus.INVALID_INITIALIZATION for value in statuses):
            status = SampleStatus.INVALID_INITIALIZATION
        else:
            status = SampleStatus.COMPLETED
        return SampleResult(
            input_npy_path,
            status,
            output_paths=output_paths,
            details={
                "sample_index": sample_index,
                "sample_seed": derived_seed,
                "output_statuses": [value.value for value in statuses],
            },
        )
    except DebugViewerError:
        # Viewer startup is a task-level preflight failure, not a bad grasp.
        raise
    except Exception as error:
        result = execution_error(input_npy_path, error)
        result.details.update({"sample_index": sample_index, "sample_seed": derived_seed})
        return result


def _validate_control_preflight(configs: Any) -> None:
    """Validate public hand, setting, method, and input selector.

    Args:
        configs: Composed application configuration.

    Returns:
        None.

    Raises:
        PreflightError: If a requested registry key or configured field is unsupported.
    """
    if configs.hand_name not in SUPPORTED_HANDS:
        raise PreflightError(f"control_eval requires one of {sorted(SUPPORTED_HANDS)}; got '{configs.hand_name}'.")
    hand_xml = Path(configs.hand.xml_path)
    if not hand_xml.is_file():
        raise PreflightError(f"Hand MJCF asset does not exist: {hand_xml}.")
    key = (configs.setting, configs.task.method)
    if key not in METHOD_REGISTRY:
        methods = sorted(method for setting, method in METHOD_REGISTRY if setting == configs.setting)
        raise PreflightError(f"Unsupported control setting/method {key!r}. Methods for '{configs.setting}': {methods}.")
    if configs.task.input_data not in configs:
        raise PreflightError(f"task.input_data refers to unknown config field '{configs.task.input_data}'.")
    if not list(configs.task.offsets):
        raise PreflightError("task.offsets must contain at least one perturbation distance.")
    failure_policy = str(configs.task.control.get("solver_failure_policy", DEFAULT_SOLVER_FAILURE_POLICY))
    if failure_policy not in SOLVER_FAILURE_POLICIES:
        supported = ", ".join(SOLVER_FAILURE_POLICIES)
        raise PreflightError(
            f"Unsupported task.control.solver_failure_policy '{failure_policy}'. Supported values: {supported}."
        )
    try:
        GraspControllerParameters.from_config(configs.task.control)
    except ValueError as error:
        raise PreflightError(f"Invalid task.control configuration: {error}.") from error


def _validate_object_assets(input_paths: list[str], configs: Any) -> None:
    """Validate assets referenced by structurally valid selected grasp records.

    Malformed or unreadable records remain sample-level execution errors so one
    damaged file does not prevent the rest of a batch from running. Once a
    record satisfies the grasp schema, however, its shared object resources are
    task-level prerequisites and must fail before worker processes start.

    Args:
        input_paths: Deterministically selected grasp record paths.
        configs: Runtime-normalized configuration containing the data root.

    Returns:
        None.

    Raises:
        PreflightError: If a valid grasp references missing object resources.
    """
    activate_runtime_roots(configs)
    for input_path in input_paths:
        try:
            record = validate_grasp_record(load_npy_record(input_path), input_path)
        except Exception:
            continue
        resolved_record = dict(record)
        resolved_record["obj_path"] = str(resolve_from_root(resolved_record["obj_path"], root_kind="data"))
        try:
            validate_grasp_record(resolved_record, input_path, require_assets=True)
        except SchemaError as error:
            raise PreflightError(f"Object asset preflight failed: {error}") from error


def task_control_eval(configs: Any) -> None:
    """Run a control batch and report every input without swallowing errors.

    Args:
        configs: Composed application configuration.

    Returns:
        None.

    Raises:
        PreflightError: If configuration, input, or viewer startup is invalid.
        BatchExecutionError: After all samples finish if any execution or solver failed.
    """
    _validate_control_preflight(configs)
    input_dir = str(configs[configs.task.input_data])
    discovered = sorted(glob(str(Path(input_dir) / "**" / "*.npy"), recursive=True))
    if not discovered:
        raise PreflightError(f"No grasp .npy inputs found below {input_dir}.")
    expected = [_expected_outputs(path, configs) for path in discovered]
    selected, num_skipped = choose_inputs(
        discovered,
        skip=bool(configs.skip),
        output_paths=expected,
        max_num=int(configs.task.max_num),
        seed=int(configs.seed),
    )
    _validate_object_assets(selected, configs)
    write_run_manifest(configs, discovered)
    logging.info(
        "Found %d grasp files in %s, skipped %d, processing %d.",
        len(discovered),
        input_dir,
        num_skipped,
        len(selected),
    )

    debug_viewer_session = None
    results = []
    try:
        debug_viewer_session = create_debug_viewer_session(configs.task)
        sample_indices = {path: index for index, path in enumerate(discovered)}
        params = [(path, configs, debug_viewer_session, sample_indices[path]) for path in selected]
        if configs.task.debug_viewer or configs.task.debug_render:
            for index, item in enumerate(params):
                logging.info("Control sample index: %d", index)
                results.append(safe_eval_one(item))
        elif params:
            with multiprocessing.Pool(processes=configs.n_worker) as pool:
                results = list(pool.imap_unordered(safe_eval_one, params))
    except DebugViewerError as error:
        raise PreflightError(str(error)) from error
    finally:
        if debug_viewer_session is not None:
            debug_viewer_session.close()

    summary = write_batch_report(
        Path(configs.log_dir),
        "control_eval",
        results,
        num_discovered=len(discovered),
        num_skipped=num_skipped,
    )
    logging.info("Control evaluation completed %d input(s).", summary["num_processed"])
    raise_for_batch_failures(summary)
