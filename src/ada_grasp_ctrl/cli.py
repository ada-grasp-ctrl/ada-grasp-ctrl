"""Hydra command-line entry point for supported public tasks."""

import logging

import hydra
from omegaconf import DictConfig

from .errors import AdaGraspError, PreflightError
from .runtime import configure_runtime, write_run_manifest
from .tasks import TASK_REGISTRY


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
        raise SystemExit(error.exit_code) from error


def main() -> None:
    """Run the Hydra application with stable user-facing exit codes.

    Returns:
        None. The function exits the process when a known application error occurs.
    """
    hydra_main()


if __name__ == "__main__":
    main()
