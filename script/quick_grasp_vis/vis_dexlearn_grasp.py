import numpy as np

import torch
import trimesh as tm
import yaml
import os
import re
import json

import sys
from pathlib import Path

# Get parent of parent
parent_parent = Path(__file__).resolve().parents[2]
sys.path.append(str(parent_parent))

from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW
from src.util.robots.base import Robot, RobotFactory
from trimesh_visualizer import Visualizer

"""
Visualize the generated grasps from DexLearn via Trimesh.
"""


if __name__ == "__main__":
    hand = "shadow"
    robot = RobotFactory.create_robot(hand, prefix="rh_")
    robot_mjcf_path = robot.get_file_path("mjcf")
    pc_centering = False

    visualizer = Visualizer(robot_mjcf_path=robot_mjcf_path)

    object_path = "assets/object/DGN_2k"
    pc_path = "vision_data/azure_kinect_dk"
    object_pc_folder = os.path.join(object_path, pc_path)
    prefix = (
        "../DexLearn/output/bodex_tabletop_shadow_nflow_debug0/tests/"
    ) # change to your path
    for i in range(10):
        grasp_file_path = (
            f"step_045000/ddg_gd_rubber_duck_poisson_001/tabletop_ur10e/scale006_pose000_0/partial_pc_00_{i}.npy"
        ) # change to your path
        grasp_data = np.load(os.path.join(prefix, grasp_file_path), allow_pickle=True).item()
        scene_path = grasp_data["scene_path"]
        scene_data = np.load(scene_path, allow_pickle=True).item()
        pc_path = os.path.join(
            object_pc_folder, scene_data["scene_id"], re.sub(r"_\d+\.npy$", ".npy", os.path.basename(grasp_file_path))
        )

        obj_name = scene_data["task"]["obj_name"]
        obj_pose = scene_data["scene"][obj_name]["pose"]
        obj_scale = scene_data["scene"][obj_name]["scale"]
        obj_mesh_path = scene_data["scene"][obj_name]["file_path"]
        obj_mesh_path = os.path.abspath(os.path.join(os.path.dirname(scene_path), obj_mesh_path))

        # pointcloud mesh
        pc = np.load(pc_path).reshape(-1, 3)

        # move to centroid
        if pc_centering:
            pc_centroid = np.mean(pc, axis=0, keepdims=True)
            pc = pc - pc_centroid
        colors = np.tile([0, 0, 255, 255], (pc.shape[0], 1))  # Blue in RGBA
        pc = tm.points.PointCloud(pc, colors=colors)

        # object mesh
        obj_transform = posQuat2Isometry3d(obj_pose[:3], quatWXYZ2XYZW(obj_pose[3:]))
        if pc_centering:
            obj_transform[:3, 3] += -pc_centroid.reshape(3)
        obj_mesh = tm.load_mesh(obj_mesh_path, process=False)
        obj_mesh = obj_mesh.copy().apply_scale(obj_scale)
        obj_mesh.apply_transform(obj_transform)

        pregrasp_qpos = grasp_data["pregrasp_qpos"]
        grasp_qpos = grasp_data["grasp_qpos"]
        squeeze_qpos = grasp_data["squeeze_qpos"]

        # # enlarge the pregrasp finger configuration
        # pregrasp_qpos[7:] += 2 * (pregrasp_qpos[7:] - grasp_qpos[7:])

        visualizer.set_robot_parameters(torch.tensor(pregrasp_qpos).unsqueeze(0))
        robot_mesh_0 = visualizer.get_robot_trimesh_data(i=0, color=[30, 119, 179])

        grasp_qpos = grasp_data["grasp_qpos"]
        visualizer.set_robot_parameters(torch.tensor(grasp_qpos).unsqueeze(0))
        robot_mesh_1 = visualizer.get_robot_trimesh_data(i=0, color=[255, 127, 13])

        visualizer.set_robot_parameters(torch.tensor(squeeze_qpos).unsqueeze(0))
        robot_mesh_2 = visualizer.get_robot_trimesh_data(i=0, color=[44, 160, 44])

        axis = tm.creation.axis(origin_size=0.01, axis_length=1.0)
        scene = tm.Scene(geometry=[robot_mesh_0, robot_mesh_1, robot_mesh_2, pc, obj_mesh])
        scene.show(smooth=False)
