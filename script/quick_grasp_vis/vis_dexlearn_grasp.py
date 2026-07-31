"""Visualize one formatted grasp without editing source-code paths."""

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh as tm

from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.schema import load_npy_record, validate_grasp_record
from ada_grasp_ctrl.utils.robots.base import RobotFactory
from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW

from trimesh_visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    """Parse the static-grasp visualization command line.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", required=True, choices=("shadow", "allegro", "leap_tac3d"))
    parser.add_argument("--grasp", required=True, type=Path, help="Formatted grasp .npy path.")
    parser.add_argument(
        "--object-root",
        type=Path,
        default=None,
        help="Optional object directory overriding the record's obj_path.",
    )
    parser.add_argument("--export", type=Path, default=None, help="Export a scene file instead of opening a window.")
    return parser.parse_args()


def build_scene(hand: str, grasp_path: Path, object_root: Path | None = None) -> tm.Scene:
    """Build a Trimesh scene for pregrasp, grasp, squeeze, and object meshes.

    Args:
        hand: Supported hand name.
        grasp_path: Formatted version-zero or version-one grasp record.
        object_root: Optional object directory override.

    Returns:
        Trimesh scene ready for display or export.
    """
    grasp = validate_grasp_record(load_npy_record(grasp_path), grasp_path)
    prefix = "" if hand == "allegro" else "rh_"
    robot = RobotFactory.create_robot(hand, prefix=prefix)
    visualizer = Visualizer(robot_mjcf_path=robot.get_file_path("mjcf"))
    joint_names = grasp.get("joint_names")
    meshes = []
    for field, color in (
        ("pregrasp_qpos", [30, 119, 179]),
        ("grasp_qpos", [255, 127, 13]),
        ("squeeze_qpos", [44, 160, 44]),
    ):
        visualizer.set_robot_parameters(
            torch.as_tensor(grasp[field], dtype=torch.float32).unsqueeze(0),
            joint_names=joint_names,
        )
        meshes.append(visualizer.get_robot_trimesh_data(index=0, color=color))

    object_dir = resolve_from_root(object_root or grasp["obj_path"])
    object_mesh_path = object_dir / "mesh" / "simplified.obj"
    if not object_mesh_path.is_file():
        raise FileNotFoundError(f"Object visualization mesh is missing: {object_mesh_path}")
    object_mesh = tm.load_mesh(object_mesh_path, process=False)
    object_mesh.apply_scale(float(grasp["obj_scale"]))
    object_pose = np.asarray(grasp["obj_pose"])
    object_transform = posQuat2Isometry3d(object_pose[:3], quatWXYZ2XYZW(object_pose[3:]))
    object_mesh.apply_transform(object_transform)
    meshes.extend([object_mesh, tm.creation.axis(origin_size=0.01, axis_length=0.2)])
    return tm.Scene(geometry=meshes)


def main() -> None:
    """Display or export the requested grasp scene.

    Returns:
        None.
    """
    args = parse_args()
    scene = build_scene(args.hand, args.grasp.resolve(), args.object_root)
    if args.export is not None:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        scene.export(args.export)
        print(f"Exported scene to {args.export}")
    else:
        scene.show(smooth=False)


if __name__ == "__main__":
    main()
