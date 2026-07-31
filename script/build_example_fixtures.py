"""Build the checked-in three-hand fixtures from the fixed golden sample."""

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECT_ID = "core_bottle_15787789482f045d8add95bf56d3d2fa"
SAMPLE_RELATIVE = Path(OBJECT_ID) / "tabletop_ur10e" / "scale006_pose004_0" / "partial_pc_00_6.npy"
OBJECT_RELATIVE = Path("examples/assets/object") / OBJECT_ID


def _load(path: Path) -> dict:
    """Load one NumPy mapping.

    Args:
        path: Source record.

    Returns:
        Mutable mapping copy.
    """
    return dict(np.load(path, allow_pickle=True).item())


def _save(path: Path, record: dict) -> None:
    """Save one version-one fixture.

    Args:
        path: Destination record.
        record: Mapping to serialize.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    current = dict(record)
    current["schema_version"] = 1
    np.save(path, current)


def build_scene() -> Path:
    """Write the shared Learning-converter scene record.

    Returns:
        Scene path.
    """
    scene_path = PROJECT_ROOT / "examples/data/scene.npy"
    scene = {
        "task": {"obj_name": "target"},
        "scene": {
            "target": {
                "file_path": f"../assets/object/{OBJECT_ID}/mesh/simplified.obj",
                "pose": np.array([0.71100098, 0.04075842, 0.02034247, 0.58787514, -0.37770758, 0.61379698, 0.36741404]),
                "scale": [0.06],
            }
        },
    }
    np.save(scene_path, scene)
    return scene_path


def build_hand(hand: str) -> None:
    """Write raw, formatted, and dummy-arm fixtures for one hand.

    Args:
        hand: ``shadow``, ``allegro``, or ``leap_tac3d``.

    Returns:
        None.
    """
    formatted_source = PROJECT_ROOT / f"output/learn_{hand}/graspdata" / SAMPLE_RELATIVE
    dummy_source = PROJECT_ROOT / f"output/learn_dummy_arm_{hand}/graspdata" / SAMPLE_RELATIVE
    formatted = _load(formatted_source)
    dummy = _load(dummy_source)
    for record in (formatted, dummy):
        record["obj_path"] = str(OBJECT_RELATIVE)

    hand_root = PROJECT_ROOT / "examples/data" / hand
    _save(hand_root / "formatted/grasp.npy", formatted)
    _save(hand_root / "dummy_arm/grasp.npy", dummy)
    raw = {
        "schema_version": 1,
        "scene_path": "../../scene.npy",
        "pregrasp_qpos": formatted["pregrasp_qpos"],
        "grasp_qpos": formatted["grasp_qpos"],
        "squeeze_qpos": formatted["squeeze_qpos"],
    }
    _save(hand_root / "raw/learning.npy", raw)


def main() -> None:
    """Build all checked-in example records.

    Returns:
        None.
    """
    build_scene()
    for hand in ("shadow", "allegro", "leap_tac3d"):
        build_hand(hand)


if __name__ == "__main__":
    main()
