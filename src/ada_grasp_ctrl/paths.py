"""Path helpers that make execution independent of the current directory."""

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def project_root() -> Path:
    """Return the source-checkout root.

    Returns:
        Absolute repository root containing ``pyproject.toml``.
    """
    return Path(__file__).resolve().parents[2]


def resolve_from_root(path: PathLike) -> Path:
    """Resolve a configured path against the repository root.

    Args:
        path: Absolute path or repository-relative path.

    Returns:
        Normalized absolute path.
    """
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root() / candidate
    return candidate.resolve(strict=False)


def project_path(*parts: str) -> str:
    """Build an absolute path below the repository root.

    Args:
        parts: Relative path components.

    Returns:
        Absolute path encoded as a string for third-party APIs.
    """
    return str(project_root().joinpath(*parts).resolve(strict=False))


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
