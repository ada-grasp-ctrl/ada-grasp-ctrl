"""Explicit external-root resolution independent of package layout and CWD."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Union

PathLike = Union[str, Path]
RootKind = Literal["asset", "data", "output"]

ROOT_ENVIRONMENT = {
    "asset": "ADA_GRASP_CTRL_ASSET_ROOT",
    "data": "ADA_GRASP_CTRL_DATA_ROOT",
    "output": "ADA_GRASP_CTRL_OUTPUT_ROOT",
}
SOURCE_DEFAULTS = {
    "asset": Path("assets"),
    "data": Path("."),
    "output": Path("output"),
}

_RUNTIME_ROOTS: dict[RootKind, Path] = {}


def source_checkout_root(start: Path | None = None) -> Path | None:
    """Find a source checkout by stable repository markers.

    Args:
        start: Optional file or directory from which to search upward.

    Returns:
        Absolute checkout root, or ``None`` for an installed wheel.
    """
    candidate = (start or Path(__file__)).resolve(strict=False)
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file() and (directory / "src" / "ada_grasp_ctrl").is_dir():
            return directory
    return None


def project_root() -> Path:
    """Return the detected source-checkout root.

    Returns:
        Absolute repository root containing ``pyproject.toml``.

    Raises:
        RuntimeError: If the package is running from a wheel installation.
    """
    root = source_checkout_root()
    if root is None:
        raise RuntimeError("No ada-grasp-ctrl source checkout is available in this installation.")
    return root


def resolve_external_root(kind: RootKind, configured: object = None) -> Path:
    """Resolve one external root by config, environment, then checkout default.

    Args:
        kind: Asset, data, or output root category.
        configured: Optional Hydra/CLI-configured path.

    Returns:
        Absolute external root.

    Raises:
        ValueError: If the root is missing or relative without a checkout anchor.
    """
    if kind not in ROOT_ENVIRONMENT:
        raise ValueError(f"Unsupported root kind: {kind}")
    checkout = source_checkout_root()
    raw_value = configured
    if raw_value in (None, "", "???"):
        raw_value = os.environ.get(ROOT_ENVIRONMENT[kind])
    if raw_value not in (None, "", "???"):
        candidate = Path(str(raw_value)).expanduser()
        if not candidate.is_absolute():
            if checkout is None:
                raise ValueError(
                    f"{kind}_root must be absolute outside a source checkout; got {candidate}. "
                    f"Set it explicitly or use {ROOT_ENVIRONMENT[kind]}."
                )
            candidate = checkout / candidate
        return candidate.resolve(strict=False)
    if checkout is not None:
        return (checkout / SOURCE_DEFAULTS[kind]).resolve(strict=False)
    raise ValueError(f"Missing {kind}_root for wheel execution. Set the Hydra/CLI field or {ROOT_ENVIRONMENT[kind]}.")


def configure_runtime_roots(*, asset_root: PathLike, data_root: PathLike, output_root: PathLike) -> None:
    """Activate absolute roots for legacy robot and record path helpers.

    Args:
        asset_root: Directory containing the external ``hand`` asset tree.
        data_root: Directory anchoring relative input and record paths.
        output_root: Directory anchoring generated outputs.

    Returns:
        None.

    Raises:
        ValueError: If any activated root is not absolute.
    """
    roots = {
        "asset": Path(asset_root).expanduser(),
        "data": Path(data_root).expanduser(),
        "output": Path(output_root).expanduser(),
    }
    for kind, root in roots.items():
        if not root.is_absolute():
            raise ValueError(f"Runtime {kind}_root must be absolute, got {root}.")
        _RUNTIME_ROOTS[kind] = root.resolve(strict=False)


def reset_runtime_roots() -> None:
    """Clear process-local roots so tests or embedded callers can reconfigure.

    Returns:
        None.
    """
    _RUNTIME_ROOTS.clear()


def runtime_root(kind: RootKind) -> Path:
    """Return an activated root or its source-checkout default.

    Args:
        kind: Asset, data, or output root category.

    Returns:
        Absolute root path.
    """
    configured = _RUNTIME_ROOTS.get(kind)
    if configured is not None:
        return configured
    return resolve_external_root(kind)


def resolve_from_root(path: PathLike, root_kind: RootKind = "data") -> Path:
    """Resolve a configured or stored path against its explicit root.

    Legacy ``assets/`` and ``output/`` prefixes are stripped only when the
    corresponding root kind is requested. Other relative paths retain every
    component below ``root_kind``.

    Args:
        path: Absolute path or root-relative path.
        root_kind: Root used for a generic relative path.

    Returns:
        Normalized absolute path.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    parts = candidate.parts
    if root_kind == "asset" and parts and parts[0] == "assets":
        return runtime_root("asset").joinpath(*parts[1:]).resolve(strict=False)
    if root_kind == "output" and parts and parts[0] == "output":
        return runtime_root("output").joinpath(*parts[1:]).resolve(strict=False)
    return (runtime_root(root_kind) / candidate).resolve(strict=False)


def project_path(*parts: str) -> str:
    """Build an absolute data-root path for compatibility APIs.

    Args:
        parts: Relative path components.

    Returns:
        Absolute path encoded as a string for third-party APIs.
    """
    return str(runtime_root("data").joinpath(*parts).resolve(strict=False))


def map_path(source: PathLike, source_root: PathLike, target_root: PathLike) -> Path:
    """Map a source file to an equivalent path below another root.

    Args:
        source: Input file path.
        source_root: Root that must contain the input file.
        target_root: Root below which the relative path is recreated.

    Returns:
        Mapped output path.

    Raises:
        ValueError: If the input file is outside ``source_root``.
    """
    source_path = Path(source).resolve(strict=False)
    root_path = Path(source_root).resolve(strict=False)
    return Path(target_root).resolve(strict=False) / source_path.relative_to(root_path)
