"""Run the maintained control-method matrix in one fresh experiment tree."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_HANDS = ("shadow", "allegro", "leap_tac3d")
SUPPORTED_METHODS = ("ours", "op", "bs1", "bs2", "bs3")
SETTING_OFFSETS = {"dist_0": (0.0,), "dist_2": (0.02,)}
SAFE_ABLATION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class ExperimentRunnerError(RuntimeError):
    """Raised when the experiment cannot safely start or continue."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated experiment-runner configuration."""

    python_executable: str
    input_root: Path
    output_root: Path
    asset_root: Path
    data_root: Path
    hands: tuple[str, ...] = SUPPORTED_HANDS
    methods: tuple[str, ...] = SUPPORTED_METHODS
    settings: tuple[str, ...] = tuple(SETTING_OFFSETS)
    ours_ablations: tuple[str, ...] = ("default",)
    seed: int = 12
    workers: int = 12
    max_num: int = -1


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    """Return values once each while preserving command-line order.

    Args:
        values: User-selected identifiers.

    Returns:
        Stable tuple without duplicates.
    """
    return tuple(dict.fromkeys(values))


def _resolve_python(value: str) -> str:
    """Resolve a Python executable independently of the caller's directory.

    Args:
        value: Executable name or absolute path.

    Returns:
        Absolute executable path.

    Raises:
        ExperimentRunnerError: If the executable cannot be found.
    """
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        if not candidate.is_file():
            raise ExperimentRunnerError(f"Python executable is unavailable: {candidate}")
        return str(candidate)
    resolved = shutil.which(value)
    if resolved is None:
        raise ExperimentRunnerError(f"Python executable is unavailable: {value}")
    return str(Path(resolved).absolute())


def _validate_absolute_directory(path: Path, label: str) -> Path:
    """Validate an explicit existing directory.

    Args:
        path: User-provided directory.
        label: Human-readable option name.

    Returns:
        Normalized absolute directory.

    Raises:
        ExperimentRunnerError: If the path is relative or missing.
    """
    path = path.expanduser()
    if not path.is_absolute():
        raise ExperimentRunnerError(f"{label} must be absolute: {path}")
    path = path.resolve(strict=False)
    if not path.is_dir():
        raise ExperimentRunnerError(f"{label} is not a directory: {path}")
    return path


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    """Validate roots, selections, and the fresh-output invariant.

    Args:
        config: Candidate experiment configuration.

    Returns:
        Normalized immutable configuration.

    Raises:
        ExperimentRunnerError: If the run could consume stale output or lacks
            an input/configuration prerequisite.
    """
    python_executable = _resolve_python(config.python_executable)
    input_root = _validate_absolute_directory(config.input_root, "input_root")
    asset_root = _validate_absolute_directory(config.asset_root, "asset_root")
    data_root = _validate_absolute_directory(config.data_root, "data_root")
    output_root = config.output_root.expanduser()
    if not output_root.is_absolute():
        raise ExperimentRunnerError(f"output_root must be absolute: {output_root}")
    if output_root.exists() or output_root.is_symlink():
        raise ExperimentRunnerError(f"output_root already exists; choose a fresh path: {output_root}")
    output_root = output_root.resolve(strict=False)
    if config.workers <= 0:
        raise ExperimentRunnerError("workers must be greater than zero.")
    if not config.hands or not config.methods or not config.settings or not config.ours_ablations:
        raise ExperimentRunnerError("hands, methods, settings, and ours_ablations must be non-empty.")
    invalid_ablations = [
        value for value in config.ours_ablations if value in {".", ".."} or SAFE_ABLATION_NAME.fullmatch(value) is None
    ]
    if invalid_ablations:
        raise ExperimentRunnerError(
            "ours_ablations must be safe path/config identifiers; invalid values: " + ", ".join(invalid_ablations)
        )

    unsupported_hands = sorted(set(config.hands) - set(SUPPORTED_HANDS))
    unsupported_methods = sorted(set(config.methods) - set(SUPPORTED_METHODS))
    unsupported_settings = sorted(set(config.settings) - set(SETTING_OFFSETS))
    if unsupported_hands:
        raise ExperimentRunnerError(f"Unsupported hands: {', '.join(unsupported_hands)}")
    if unsupported_methods:
        raise ExperimentRunnerError(f"Unsupported methods: {', '.join(unsupported_methods)}")
    if unsupported_settings:
        raise ExperimentRunnerError(f"Unsupported settings: {', '.join(unsupported_settings)}")

    hands = _unique(config.hands)
    for hand in hands:
        input_directory = input_root / f"learn_dummy_arm_{hand}" / "graspdata"
        if not input_directory.is_dir():
            raise ExperimentRunnerError(f"Input directory for {hand} is missing: {input_directory}")

    return ExperimentConfig(
        python_executable=python_executable,
        input_root=input_root,
        output_root=output_root,
        asset_root=asset_root,
        data_root=data_root,
        hands=hands,
        methods=_unique(config.methods),
        settings=_unique(config.settings),
        ours_ablations=_unique(config.ours_ablations),
        seed=int(config.seed),
        workers=int(config.workers),
        max_num=int(config.max_num),
    )


def _method_directory(method: str, ablation: str) -> str:
    """Return the control-output directory name for one method.

    Args:
        method: Maintained public control method.
        ablation: Ours-controller ablation name.

    Returns:
        Method directory component.
    """
    return f"ours_{ablation}" if method == "ours" else method


def _case_roots(config: ExperimentConfig, setting: str, hand: str, method: str, ablation: str) -> dict[str, Path]:
    """Build the isolated paths used by one matrix case.

    Args:
        config: Validated experiment configuration.
        setting: ``dist_0`` or ``dist_2``.
        hand: Supported hand identifier without the dummy-arm prefix.
        method: Supported control method.
        ablation: Ours-controller ablation name.

    Returns:
        Input, case, control, and log paths.
    """
    case_root = config.output_root / setting / hand / _method_directory(method, ablation)
    return {
        "input": config.input_root / f"learn_dummy_arm_{hand}" / "graspdata",
        "case": case_root,
        "control": case_root / "control",
        "eval_log": case_root / "log/control_eval",
        "stat_log": case_root / "log/control_stat",
    }


def _common_command(config: ExperimentConfig, roots: dict[str, Path], hand: str, task: str) -> list[str]:
    """Build task-independent Hydra overrides for one application command.

    Args:
        config: Validated experiment configuration.
        roots: Isolated paths for the current case.
        hand: Supported hand identifier without the dummy-arm prefix.
        task: Application task name.

    Returns:
        Command prefix and common overrides.
    """
    return [
        config.python_executable,
        str(PROJECT_ROOT / "src/main.py"),
        "setting=tabletop",
        f"hand=dummy_arm_{hand}",
        f"task={task}",
        f"exp_name=ablation_{roots['case'].name}",
        f"seed={config.seed}",
        f"n_worker={config.workers}",
        f"asset_root={config.asset_root}",
        f"data_root={config.data_root}",
        f"output_root={roots['case']}",
        f"save_dir={roots['case']}",
        f"control_dir={roots['control']}",
    ]


def control_eval_command(
    config: ExperimentConfig,
    setting: str,
    hand: str,
    method: str,
    ablation: str,
) -> tuple[list[str], Path]:
    """Build one isolated ``control_eval`` command and report path.

    Args:
        config: Validated experiment configuration.
        setting: ``dist_0`` or ``dist_2``.
        hand: Supported hand identifier without the dummy-arm prefix.
        method: Supported control method.
        ablation: Ours-controller ablation name.

    Returns:
        Command arguments and expected current-run report path.
    """
    roots = _case_roots(config, setting, hand, method, ablation)
    offsets = json.dumps(SETTING_OFFSETS[setting], separators=(",", ":"))
    command = _common_command(config, roots, hand, "control_eval")
    command.extend(
        (
            f"grasp_dir={roots['input']}",
            f"log_dir={roots['eval_log']}",
            f"task.method={method}",
            f"task.control.ablation_name={ablation}",
            f"task.offsets={offsets}",
            "task.input_data=grasp_dir",
            "task.debug_viewer=false",
            "task.debug_render=false",
            f"task.max_num={config.max_num}",
        )
    )
    return command, roots["eval_log"] / "run_report.json"


def control_stat_command(
    config: ExperimentConfig,
    setting: str,
    hand: str,
    method: str,
    ablation: str,
    input_report: Path,
) -> tuple[list[str], Path]:
    """Build one isolated ``control_stat`` command and report path.

    Args:
        config: Validated experiment configuration.
        setting: ``dist_0`` or ``dist_2``.
        hand: Supported hand identifier without the dummy-arm prefix.
        method: Supported control method.
        ablation: Ours-controller ablation name.
        input_report: Exact current ``control_eval`` report.

    Returns:
        Command arguments and expected statistics report path.
    """
    roots = _case_roots(config, setting, hand, method, ablation)
    command = _common_command(config, roots, hand, "control_stat")
    command.extend(
        (
            f"log_dir={roots['stat_log']}",
            f"task.method={method}",
            f"task.ablation_name={ablation}",
            f"task.setting_name={setting}",
            f"task.input_report={input_report}",
        )
    )
    return command, roots["stat_log"] / "run_report.json"


def _write_experiment_report(config: ExperimentConfig, results: list[dict[str, Any]], status: int) -> Path:
    """Atomically write the runner-level structured report.

    Args:
        config: Validated experiment configuration.
        results: Completed and failed case records.
        status: Current runner exit status.

    Returns:
        Final report path.
    """
    report_path = config.output_root / "experiment_report.json"
    temporary_path = report_path.with_suffix(".json.tmp")
    report = {
        "schema_version": 1,
        "runner": "run_ablation_baselines",
        "input_root": str(config.input_root),
        "output_root": str(config.output_root),
        "asset_root": str(config.asset_root),
        "data_root": str(config.data_root),
        "seed": config.seed,
        "workers": config.workers,
        "max_num": config.max_num,
        "exit_code": status,
        "results": results,
    }
    try:
        temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_path.replace(report_path)
    except OSError as error:
        raise ExperimentRunnerError(f"Cannot write experiment report {report_path}: {error}") from error
    return report_path


def _record_stage(command: list[str], report_path: Path, exit_code: int) -> dict[str, Any]:
    """Build one structured stage result.

    Args:
        command: Executed application command.
        report_path: Expected task report.
        exit_code: Child process exit code.

    Returns:
        JSON-serializable stage record.
    """
    return {
        "command": command,
        "exit_code": exit_code,
        "run_report": str(report_path),
        "report_exists": report_path.is_file(),
    }


def run_experiment(
    config: ExperimentConfig,
    command_runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> int:
    """Run every selected case and preserve the documented exit semantics.

    Exit ``0`` means every child task returned zero. Exit ``1`` means all
    possible cases ran but at least one task reported execution errors or
    solver degradation. Exit ``2`` means runner preflight failed, a child task
    returned a preflight/error code, or a required current-run report was not
    produced.

    Args:
        config: Candidate experiment configuration.
        command_runner: Injectable subprocess runner for focused tests.

    Returns:
        Runner exit code ``0``, ``1``, or ``2``.

    Raises:
        ExperimentRunnerError: If runner preflight fails before output creation.
    """
    config = validate_config(config)
    try:
        config.output_root.mkdir(parents=True)
    except OSError as error:
        raise ExperimentRunnerError(f"Cannot create output_root {config.output_root}: {error}") from error
    results: list[dict[str, Any]] = []
    overall_status = 0
    _write_experiment_report(config, results, overall_status)

    for setting in config.settings:
        for hand in config.hands:
            for method in config.methods:
                ablations = config.ours_ablations if method == "ours" else ("default",)
                for ablation in ablations:
                    case_record: dict[str, Any] = {
                        "setting": setting,
                        "hand": hand,
                        "method": method,
                        "ablation": ablation,
                    }
                    eval_command, eval_report = control_eval_command(config, setting, hand, method, ablation)
                    print("Running:", " ".join(eval_command), flush=True)
                    try:
                        completed = command_runner(eval_command, cwd=PROJECT_ROOT, check=False)
                        eval_status = int(completed.returncode)
                    except OSError as error:
                        eval_status = 2
                        case_record["error"] = f"Cannot start control_eval: {error}"
                    case_record["control_eval"] = _record_stage(eval_command, eval_report, eval_status)

                    if eval_status > 1 or eval_status < 0 or not eval_report.is_file():
                        if eval_status <= 1 and eval_status >= 0:
                            case_record["error"] = f"control_eval did not produce its current report: {eval_report}"
                        results.append(case_record)
                        _write_experiment_report(config, results, 2)
                        return 2
                    overall_status = max(overall_status, eval_status)

                    stat_command, stat_report = control_stat_command(
                        config,
                        setting,
                        hand,
                        method,
                        ablation,
                        eval_report,
                    )
                    print("Running:", " ".join(stat_command), flush=True)
                    try:
                        completed = command_runner(stat_command, cwd=PROJECT_ROOT, check=False)
                        stat_status = int(completed.returncode)
                    except OSError as error:
                        stat_status = 2
                        case_record["error"] = f"Cannot start control_stat: {error}"
                    case_record["control_stat"] = _record_stage(stat_command, stat_report, stat_status)

                    if stat_status > 1 or stat_status < 0 or not stat_report.is_file():
                        if stat_status <= 1 and stat_status >= 0:
                            case_record["error"] = f"control_stat did not produce its current report: {stat_report}"
                        results.append(case_record)
                        _write_experiment_report(config, results, 2)
                        return 2
                    overall_status = max(overall_status, stat_status)
                    results.append(case_record)
                    _write_experiment_report(config, results, overall_status)

    report_path = _write_experiment_report(config, results, overall_status)
    print(f"[ada-grasp-ctrl] experiment report: {report_path}")
    return overall_status


def _argument_parser() -> argparse.ArgumentParser:
    """Create the experiment-runner command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Absolute archive root containing learn_dummy_arm_<hand>/graspdata directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Absolute fresh experiment path; it must not already exist.",
    )
    parser.add_argument("--asset-root", type=Path, default=PROJECT_ROOT / "assets")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--hands", nargs="+", choices=SUPPORTED_HANDS, default=SUPPORTED_HANDS)
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS, default=SUPPORTED_METHODS)
    parser.add_argument("--settings", nargs="+", choices=tuple(SETTING_OFFSETS), default=tuple(SETTING_OFFSETS))
    parser.add_argument("--ours-ablation", action="append", dest="ours_ablations", default=None)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-num", type=int, default=-1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected experiment matrix from command-line arguments.

    Args:
        argv: Optional argument sequence for tests.

    Returns:
        Process exit code ``0``, ``1``, or ``2``.
    """
    arguments = _argument_parser().parse_args(argv)
    config = ExperimentConfig(
        python_executable=arguments.python_executable,
        input_root=arguments.input_root,
        output_root=arguments.output_root,
        asset_root=arguments.asset_root,
        data_root=arguments.data_root,
        hands=tuple(arguments.hands),
        methods=tuple(arguments.methods),
        settings=tuple(arguments.settings),
        ours_ablations=tuple(arguments.ours_ablations or ("default",)),
        seed=arguments.seed,
        workers=arguments.workers,
        max_num=arguments.max_num,
    )
    try:
        return run_experiment(config)
    except ExperimentRunnerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
