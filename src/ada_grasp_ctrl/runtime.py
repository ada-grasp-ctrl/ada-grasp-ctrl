"""Runtime configuration, reproducibility, and run-manifest helpers."""

import importlib.metadata
import os
from pathlib import Path
import platform
import random
import subprocess
from typing import Iterable
import warnings

import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import yaml

from .errors import PreflightError
from .paths import project_root, resolve_from_root

PATH_FIELDS = (
    "asset_root",
    "data_root",
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
    with open_dict(config):
        config.project_root = str(project_root())
        config.n_worker = resolve_worker_count(config.n_worker)
        for field in PATH_FIELDS:
            if field in config:
                config[field] = str(resolve_from_root(config[field]))
        if "hand" in config and "xml_path" in config.hand:
            config.hand.xml_path = str(resolve_from_root(config.hand.xml_path))
        if "task" in config:
            for field in ("data_path", "debug_dir"):
                if field in config.task and config.task[field] not in (None, "???"):
                    config.task[field] = str(resolve_from_root(config.task[field]))
    seed_everything(int(config.seed))
    return config


def _git_metadata() -> dict[str, object]:
    """Read git commit and dirty state without mutating the checkout.

    Returns:
        Commit and dirty-state dictionary, or unavailable markers.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root(),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def write_run_manifest(config: DictConfig, input_paths: Iterable[str] = ()) -> Path:
    """Write configuration and environment metadata for reproducibility.

    Args:
        config: Normalized task configuration.
        input_paths: Deterministically ordered task inputs, if already known.

    Returns:
        Path to the written YAML manifest.
    """
    packages = ["numpy", "scipy", "torch", "mujoco", "pin", "hydra-core"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    with warnings.catch_warnings():
        # Some containerized NVIDIA drivers expose CUDA but not NVML. Hardware
        # metadata remains best-effort and should not pollute successful runs.
        warnings.filterwarnings("ignore", message="Can't initialize NVML")
        cuda_available = torch.cuda.is_available()
        gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]

    manifest = {
        "config": OmegaConf.to_container(config, resolve=True),
        "git": _git_metadata(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": versions,
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
