"""Generic checksum and scientific-difference helpers for trajectory evidence."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Iterable

import numpy as np

if __package__:
    from .compare_golden import _ignored, classify, compare_values
else:
    from compare_golden import _ignored, classify, compare_values


def sha256_file(path: Path) -> str:
    """Hash one file without loading the entire payload into memory.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_scientific_digest(digest: Any, value: Any) -> None:
    """Feed a nested NumPy record into a stable scientific-content digest.

    Args:
        digest: Hash object receiving canonical bytes.
        value: Nested mapping, sequence, array, scalar, or null value.

    Returns:
        None.
    """
    if isinstance(value, dict):
        digest.update(b"mapping{")
        keys = sorted((key for key in value if not _ignored(key)), key=str)
        for key in keys:
            _update_scientific_digest(digest, key)
            _update_scientific_digest(digest, value[key])
        digest.update(b"}")
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence[")
        for item in value:
            _update_scientific_digest(digest, item)
        digest.update(b"]")
        return
    if isinstance(value, np.ndarray):
        digest.update(b"array:")
        digest.update(value.dtype.str.encode("utf-8"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        if value.dtype.hasobject:
            for item in value.flat:
                _update_scientific_digest(digest, item)
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
        return
    if isinstance(value, np.generic):
        _update_scientific_digest(digest, value.item())
        return
    if isinstance(value, bool):
        digest.update(b"bool:1" if value else b"bool:0")
        return
    if isinstance(value, int):
        digest.update(f"int:{value}".encode("ascii"))
        return
    if isinstance(value, float):
        digest.update(b"float:")
        digest.update(struct.pack(">d", value))
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"str:{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
        return
    if value is None:
        digest.update(b"none")
        return
    raise TypeError(f"Unsupported trajectory value type: {type(value).__name__}")


def scientific_sha256(record: dict[str, Any]) -> str:
    """Hash scientific record content while excluding approved metadata.

    Args:
        record: Loaded control trajectory mapping.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    _update_scientific_digest(digest, record)
    return digest.hexdigest()


def inventory_files(paths: Iterable[Path], root: Path, include_scientific: bool = False) -> list[dict[str, Any]]:
    """Build sorted checksum entries for explicit files.

    Args:
        paths: Files to include.
        root: Root used for portable relative paths.
        include_scientific: Whether NPY records also receive scientific digests and classifications.

    Returns:
        JSON-serializable checksum entries sorted by relative path.
    """
    entries = []
    for path in sorted(paths):
        entry: dict[str, Any] = {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if include_scientific:
            record = np.load(path, allow_pickle=True).item()
            entry["scientific_sha256"] = scientific_sha256(record)
            entry["classification"] = classify(record)
        entries.append(entry)
    return entries


def inventory_tree(root: Path, include_scientific: bool = False) -> list[dict[str, Any]]:
    """Inventory every NPY file under a tree.

    Args:
        root: Tree containing NPY files.
        include_scientific: Whether to hash and classify loaded records.

    Returns:
        Sorted checksum entries relative to ``root``.
    """
    return inventory_files(root.rglob("*.npy"), root, include_scientific=include_scientific)


def inventory_digest(entries: list[dict[str, Any]], field: str) -> str:
    """Aggregate a portable tree digest from per-file entries.

    Args:
        entries: File inventory entries.
        field: Digest field to aggregate, such as ``sha256`` or ``scientific_sha256``.

    Returns:
        Lowercase hexadecimal SHA-256 digest for paths and selected digests.
    """
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry[field].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _mismatch_category(message: str) -> str:
    """Classify one human-readable mismatch into a stable audit category.

    Args:
        message: Mismatch description from ``compare_values``.

    Returns:
        Stable mismatch category name.
    """
    if "missing keys " in message:
        return "missing_keys"
    if "unexpected keys " in message:
        return "unexpected_keys"
    if ": shape " in message:
        return "shape"
    if "sequence length " in message:
        return "sequence_length"
    if "numeric mismatch " in message:
        return "numeric"
    if "array values differ" in message:
        return "array_values"
    if "expected mapping" in message:
        return "type"
    return "scalar_value"


def record_difference(label: str, historical_path: Path, current_path: Path) -> dict[str, Any] | None:
    """Summarize scientific differences for one trajectory pair.

    Args:
        label: Portable logical path for the pair.
        historical_path: Previous trajectory record.
        current_path: Current trajectory record.

    Returns:
        Difference entry, or ``None`` when the scientific records match.
    """
    historical = np.load(historical_path, allow_pickle=True).item()
    current = np.load(current_path, allow_pickle=True).item()
    failures: list[str] = []
    compare_values(historical, current, label, failures)
    if not failures:
        return None
    categories = Counter(_mismatch_category(failure) for failure in failures)
    return {
        "path": label,
        "historical_classification": classify(historical),
        "current_classification": classify(current),
        "classification_changed": classify(historical) != classify(current),
        "historical_sha256": sha256_file(historical_path),
        "current_sha256": sha256_file(current_path),
        "historical_scientific_sha256": scientific_sha256(historical),
        "current_scientific_sha256": scientific_sha256(current),
        "mismatch_count": len(failures),
        "mismatch_categories": dict(sorted(categories.items())),
    }


def difference_report(pairs: Iterable[tuple[str, Path, Path]]) -> dict[str, Any]:
    """Aggregate pairwise trajectory differences.

    Args:
        pairs: Logical label, historical path, and current path triples.

    Returns:
        Compact report with per-file and aggregate mismatch counts.
    """
    entries = []
    categories: Counter[str] = Counter()
    for label, historical_path, current_path in pairs:
        entry = record_difference(label, historical_path, current_path)
        if entry is None:
            continue
        entries.append(entry)
        categories.update(entry["mismatch_categories"])
    return {
        "changed_file_count": len(entries),
        "classification_change_count": sum(entry["classification_changed"] for entry in entries),
        "mismatch_count": sum(entry["mismatch_count"] for entry in entries),
        "mismatch_categories": dict(sorted(categories.items())),
        "files": entries,
    }


def verify_inventory(
    expected: list[dict[str, Any]],
    root: Path,
    digest_field: str,
) -> list[str]:
    """Verify a current trajectory tree against a stored portable inventory.

    Args:
        expected: Stored inventory entries.
        root: Current tree root.
        digest_field: ``sha256`` or ``scientific_sha256``.

    Returns:
        Verification failures; an empty list means acceptance.
    """
    include_scientific = digest_field == "scientific_sha256"
    actual = inventory_tree(root, include_scientific=include_scientific)
    expected_by_path = {entry["path"]: entry for entry in expected}
    actual_by_path = {entry["path"]: entry for entry in actual}
    failures = []
    missing = sorted(expected_by_path.keys() - actual_by_path.keys())
    extra = sorted(actual_by_path.keys() - expected_by_path.keys())
    if missing:
        failures.append(f"Missing files: {missing}")
    if extra:
        failures.append(f"Unexpected files: {extra}")
    for relative in sorted(expected_by_path.keys() & actual_by_path.keys()):
        if expected_by_path[relative][digest_field] != actual_by_path[relative][digest_field]:
            failures.append(f"{relative}: {digest_field} differs")
        elif (
            include_scientific
            and expected_by_path[relative]["classification"] != actual_by_path[relative]["classification"]
        ):
            failures.append(f"{relative}: classification differs")
    return failures
