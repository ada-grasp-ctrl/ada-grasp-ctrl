"""Structured sample results and task-level batch reports."""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
import traceback as traceback_module
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .errors import BatchExecutionError


class SampleStatus(str, Enum):
    """Supported outcomes for one batch input."""

    COMPLETED = "completed"
    INVALID_INITIALIZATION = "invalid_initialization"
    SOLVER_DEGRADED = "solver_degraded"
    EXECUTION_ERROR = "execution_error"


@dataclass
class SampleResult:
    """Serializable outcome for one input sample."""

    input_path: str
    status: SampleStatus
    output_paths: list[str] = field(default_factory=list)
    message: Optional[str] = None
    traceback: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to JSON-compatible data.

        Returns:
            Dictionary containing the serialized sample result.
        """
        result = asdict(self)
        result["status"] = self.status.value
        return result


def execution_error(input_path: str, error: BaseException) -> SampleResult:
    """Create a structured result from a caught worker exception.

    Args:
        input_path: Sample that caused the exception.
        error: Caught exception instance.

    Returns:
        Execution-error result including the current traceback.
    """
    return SampleResult(
        input_path=input_path,
        status=SampleStatus.EXECUTION_ERROR,
        message=f"{type(error).__name__}: {error}",
        traceback=traceback_module.format_exc(),
    )


def write_batch_report(
    report_dir: Path,
    task_name: str,
    results: Iterable[SampleResult],
    *,
    num_discovered: int,
    num_skipped: int,
) -> dict[str, Any]:
    """Write deterministic JSON and JSONL reports for a completed batch.

    Args:
        report_dir: Directory receiving report files.
        task_name: Public task identifier.
        results: Per-sample outcomes.
        num_discovered: Number of inputs found before skip filtering.
        num_skipped: Number of already processed inputs skipped.

    Returns:
        JSON-compatible summary dictionary.
    """
    ordered = sorted(results, key=lambda item: item.input_path)
    counts = {status.value: 0 for status in SampleStatus}
    for result in ordered:
        counts[result.status.value] += 1

    summary = {
        "task": task_name,
        "num_discovered": num_discovered,
        "num_skipped": num_skipped,
        "num_processed": len(ordered),
        **{f"num_{name}": count for name, count in counts.items()},
        "results": [result.to_dict() for result in ordered],
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "run_report.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")

    failures = [
        result for result in ordered if result.status in {SampleStatus.SOLVER_DEGRADED, SampleStatus.EXECUTION_ERROR}
    ]
    with (report_dir / "failures.jsonl").open("w", encoding="utf-8") as file_obj:
        for result in failures:
            file_obj.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    return summary


def raise_for_batch_failures(summary: dict[str, Any]) -> None:
    """Raise after all samples finish when the report contains failures.

    Args:
        summary: Summary returned by :func:`write_batch_report`.

    Returns:
        None.

    Raises:
        BatchExecutionError: If execution or solver failures were recorded.
    """
    failed = summary["num_execution_error"] + summary["num_solver_degraded"]
    if failed:
        raise BatchExecutionError(
            f"{summary['task']} finished with {failed} failed/degraded sample(s); "
            "see run_report.json and failures.jsonl."
        )


def choose_inputs(
    discovered: Sequence[str],
    *,
    skip: bool,
    output_paths: Sequence[Sequence[str]] | None,
    max_num: int,
    seed: int,
) -> tuple[list[str], int]:
    """Apply deterministic skip and maximum-count selection to inputs.

    Args:
        discovered: Complete sorted or unsorted input sequence.
        skip: Whether samples whose expected outputs all exist are omitted.
        output_paths: Expected output paths aligned with ``discovered``.
        max_num: Positive selection cap, or a non-positive value for all samples.
        seed: Stable random seed used when applying the cap.

    Returns:
        Sorted selected inputs and the number omitted because of ``skip``.

    Raises:
        ValueError: If expected outputs are not aligned with inputs.
    """
    ordered = sorted(str(path) for path in discovered)
    if output_paths is not None and len(output_paths) != len(discovered):
        raise ValueError("output_paths must be aligned with discovered inputs")
    outputs_by_input = None
    if output_paths is not None:
        outputs_by_input = {
            str(input_path): [Path(output) for output in outputs]
            for input_path, outputs in zip(discovered, output_paths)
        }

    selected = []
    num_skipped = 0
    for input_path in ordered:
        expected = outputs_by_input[input_path] if outputs_by_input is not None else []
        if skip and expected and all(path.is_file() for path in expected):
            num_skipped += 1
        else:
            selected.append(input_path)

    if max_num > 0 and len(selected) > max_num:
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(len(selected), size=max_num, replace=False).tolist())
        selected = [selected[index] for index in indices]
    return selected, num_skipped
