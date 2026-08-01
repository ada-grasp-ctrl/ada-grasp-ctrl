"""Hydra command runner for supported public tasks."""

import logging

import hydra
from omegaconf import DictConfig

from .errors import AdaGraspError, PreflightError
from .runtime import configure_runtime, write_run_manifest
from .tasks import TASK_REGISTRY


class _ApplicationSystemExit(SystemExit):
    """Mark an exit code that was deliberately produced by an application error."""


@hydra.main(config_path="config", config_name="base", version_base=None)
def hydra_main(config: DictConfig) -> None:
    """Compose configuration and execute one registered task.

    Args:
        config: Hydra-composed application configuration.

    Returns:
        None.
    """
    try:
        configure_runtime(config)
        task = TASK_REGISTRY.get(config.task_name)
        if task is None:
            supported = ", ".join(sorted(TASK_REGISTRY))
            raise PreflightError(f"Unsupported task '{config.task_name}'. Supported tasks: {supported}.")
        write_run_manifest(config)
        task(config)
    except AdaGraspError as error:
        # Catch application errors before Hydra wraps the task exception so the
        # public CLI preserves its documented exit status without a traceback.
        logging.error("%s", error)
        raise _ApplicationSystemExit(error.exit_code) from error


def run() -> None:
    """Run the Hydra application with stable user-facing exit codes.

    Returns:
        None. The function exits the process when a known application error occurs.
    """
    try:
        hydra_main()
    except _ApplicationSystemExit as error:
        # Hydra composition failures and application failures can both use code
        # 1 internally. Preserve only codes explicitly assigned by our tasks.
        raise SystemExit(error.code) from None
    except SystemExit as error:
        if error.code in (None, 0):
            raise
        # Hydra emits a concise configuration error before raising SystemExit.
        # Public configuration failures consistently use the documented code 2.
        raise SystemExit(2) from None


def main() -> None:
    """Reject the removed setuptools console entry point.

    Raises:
        SystemExit: Always, with migration guidance for stale installations.
    """
    raise SystemExit("The 'ada-grasp-ctrl' console command has been removed; use 'python src/main.py' instead.")


if __name__ == "__main__":
    run()
