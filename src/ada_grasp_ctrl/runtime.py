"""Runtime configuration, reproducibility, and run-manifest helpers."""

import importlib.metadata
import importlib
import os
from pathlib import Path
import platform
import random
import subprocess
from typing import Iterable
import warnings

import numpy as np
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf, open_dict
import torch
import yaml

from .errors import PreflightError
from .paths import (
    configure_runtime_roots,
    resolve_external_root,
    resolve_from_root,
    source_checkout_root,
)

OUTPUT_PATH_FIELDS = (
    "save_root",
    "save_dir",
    "grasp_dir",
    "dummy_arm_grasp_dir",
    "control_dir",
    "log_dir",
)


def resolve_worker_count(value: object) -> int:
    """Resolve ``auto`` or validate an explicit worker count.

    Args:
        value: Hydra value, either ``auto`` or a positive integer.

    Returns:
        Positive number of worker processes.

    Raises:
        PreflightError: If the value is unsupported.
    """
    if isinstance(value, str) and value.lower() == "auto":
        return min(8, os.cpu_count() or 1)
    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise PreflightError("n_worker must be 'auto' or a positive integer.") from exc
    if workers <= 0:
        raise PreflightError("n_worker must be greater than zero.")
    return workers


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, Torch, and CUDA random generators.

    Args:
        seed: Global deterministic seed.

    Returns:
        None.
    """
    normalized_seed = int(seed) % (2**32)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized_seed)


def sample_seed(global_seed: int, sample_index: int) -> int:
    """Derive one stable worker seed from a global seed and sorted index.

    Args:
        global_seed: User-configured run seed.
        sample_index: Nonnegative index in the deterministically sorted input list.

    Returns:
        Unsigned 32-bit seed suitable for Python, NumPy, Torch, and CUDA.

    Raises:
        ValueError: If ``sample_index`` is negative.
    """
    if sample_index < 0:
        raise ValueError("sample_index must be nonnegative")
    sequence = np.random.SeedSequence([int(global_seed) % (2**32), int(sample_index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def seed_sample(global_seed: int, sample_index: int) -> int:
    """Reset every supported random generator for one worker sample.

    Args:
        global_seed: User-configured run seed.
        sample_index: Nonnegative index in the deterministically sorted input list.

    Returns:
        Derived sample seed applied to all random generators.
    """
    derived_seed = sample_seed(global_seed, sample_index)
    seed_everything(derived_seed)
    return derived_seed


def configure_runtime(config: DictConfig) -> DictConfig:
    """Normalize paths, worker count, and hand assets before a task runs.

    Args:
        config: Composed Hydra configuration.

    Returns:
        Mutated configuration with absolute paths and resolved runtime values.
    """
    configured_output = config.get("output_root")
    legacy_save_root = config.get("save_root")
    if configured_output in (None, "", "???") and legacy_save_root not in (None, "", "???"):
        configured_output = legacy_save_root
    try:
        roots = {
            "asset": resolve_external_root("asset", config.get("asset_root")),
            "data": resolve_external_root("data", config.get("data_root")),
            "output": resolve_external_root("output", configured_output),
        }
    except ValueError as error:
        raise PreflightError(str(error)) from error
    configure_runtime_roots(
        asset_root=roots["asset"],
        data_root=roots["data"],
        output_root=roots["output"],
    )

    checkout = source_checkout_root()
    with open_dict(config):
        config.project_root = str(checkout) if checkout is not None else None
        config.asset_root = str(roots["asset"])
        config.data_root = str(roots["data"])
        config.output_root = str(roots["output"])
        # Keep the historical field readable while making output_root canonical.
        config.save_root = str(roots["output"])
        config.n_worker = resolve_worker_count(config.n_worker)
        for field in OUTPUT_PATH_FIELDS:
            if field in config and config[field] not in (None, "???"):
                config[field] = str(resolve_from_root(config[field], root_kind="output"))
        if "hand" in config and "xml_path" in config.hand:
            config.hand.xml_path = str(resolve_from_root(config.hand.xml_path, root_kind="asset"))
        if "task" in config:
            if "data_path" in config.task:
                try:
                    data_path = config.task.data_path
                except MissingMandatoryValue as error:
                    raise PreflightError("task.data_path must be configured.") from error
                if data_path not in (None, "???"):
                    config.task.data_path = str(resolve_from_root(data_path, root_kind="data"))
            for field in ("debug_dir", "input_report"):
                if field in config.task and config.task[field] not in (None, "???"):
                    config.task[field] = str(resolve_from_root(config.task[field], root_kind="output"))
    seed_everything(int(config.seed))
    return config


def activate_runtime_roots(config: DictConfig) -> None:
    """Restore configured roots after a spawned worker imports the package.

    Args:
        config: Runtime-normalized configuration containing absolute roots.

    Returns:
        None.

    Raises:
        PreflightError: If a worker receives incomplete or invalid roots.
    """
    try:
        configure_runtime_roots(
            asset_root=config.asset_root,
            data_root=config.data_root,
            output_root=config.output_root,
        )
    except (AttributeError, ValueError) as error:
        raise PreflightError(f"Worker received invalid runtime roots: {error}") from error


def _git_metadata() -> dict[str, object]:
    """Read git commit and dirty state without mutating the checkout.

    Returns:
        Commit and dirty-state dictionary, or unavailable markers.
    """
    checkout = source_checkout_root()
    if checkout is None:
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=checkout,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _dependency_versions() -> dict[str, str | None]:
    """Collect installed dependency versions with a Pinocchio module fallback.

    Returns:
        Mapping from stable dependency names to discovered versions or ``None``.
    """
    distributions = {
        "numpy": "numpy",
        "scipy": "scipy",
        "torch": "torch",
        "mujoco": "mujoco",
        "hydra-core": "hydra-core",
        "pytorch-kinematics": "pytorch-kinematics",
        "mingrui-utils-python": "mingrui-utils-python",
    }
    versions: dict[str, str | None] = {}
    for key, distribution in distributions.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None

    try:
        versions["pinocchio"] = importlib.metadata.version("pin")
    except importlib.metadata.PackageNotFoundError:
        try:
            versions["pinocchio"] = str(importlib.import_module("pinocchio").__version__)
        except (AttributeError, ImportError):
            versions["pinocchio"] = None
    return versions


def _dependency_origins() -> dict[str, str | None]:
    """Record where critical Python modules were actually imported from.

    Version strings alone do not distinguish conda and wheel builds that can
    produce observably different closed-loop floating trajectories.

    Returns:
        Mapping from module names to normalized source or extension paths.
    """
    origins: dict[str, str | None] = {}
    for module_name in (
        "ada_grasp_ctrl",
        "numpy",
        "scipy",
        "torch",
        "mujoco",
        "pinocchio",
        "pytorch_kinematics",
        "mr_utils",
    ):
        try:
            module_path = getattr(importlib.import_module(module_name), "__file__", None)
        except ImportError:
            module_path = None
        origins[module_name] = str(Path(module_path).resolve(strict=False)) if module_path else None
    return origins


def write_run_manifest(config: DictConfig, input_paths: Iterable[str] = ()) -> Path:
    """Write configuration and environment metadata for reproducibility.

    Args:
        config: Normalized task configuration.
        input_paths: Deterministically ordered task inputs, if already known.

    Returns:
        Path to the written YAML manifest.
    """
    with warnings.catch_warnings():
        # Some containerized NVIDIA drivers expose CUDA but not NVML. Hardware
        # metadata remains best-effort and should not pollute successful runs.
        warnings.filterwarnings("ignore", message="Can't initialize NVML")
        cuda_available = torch.cuda.is_available()
        gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]

    manifest = {
        "config": OmegaConf.to_container(config, resolve=True),
        "roots": {
            "asset_root": str(Path(config.asset_root).resolve(strict=False)),
            "data_root": str(Path(config.data_root).resolve(strict=False)),
            "output_root": str(Path(config.output_root).resolve(strict=False)),
        },
        "git": _git_metadata(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _dependency_versions(),
            "package_origins": _dependency_origins(),
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "gpu_names": gpu_names,
        },
        "inputs": sorted(str(Path(path).resolve(strict=False)) for path in input_paths),
    }
    output_path = Path(config.log_dir) / "run_manifest.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(manifest, file_obj, sort_keys=False)
    return output_path
