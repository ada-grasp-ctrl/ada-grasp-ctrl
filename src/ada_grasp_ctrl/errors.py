"""User-facing error types and process exit semantics."""


class AdaGraspError(RuntimeError):
    """Base error carrying the process exit code expected by the CLI."""

    exit_code = 1


class PreflightError(AdaGraspError):
    """Report a configuration, input, asset, or environment failure."""

    exit_code = 2


class BatchExecutionError(AdaGraspError):
    """Report a completed batch containing runtime or solver failures."""

    exit_code = 1
