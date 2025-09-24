import os
from glob import glob
import logging
import multiprocessing

import numpy as np
from transforms3d import quaternions as tq
import torch

from util.rot_util import torch_quaternion_to_matrix, torch_matrix_to_quaternion


def load_scene_cfg(scene_path):
    scene_cfg = np.load(scene_path, allow_pickle=True).item()

    def update_relative_path(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                update_relative_path(v)
            elif k.endswith("_path") and isinstance(v, str):
                d[k] = os.path.join(os.path.dirname(scene_path), v)
        return

    update_relative_path(scene_cfg["scene"])

    return scene_cfg


def BODex(params):
    data_file, configs = params[0], params[1]

    raw_data = np.load(data_file, allow_pickle=True).item()
    robot_pose = raw_data["robot_pose"][0]
    new_data = {}

    scene_path = raw_data["scene_path"][0].split("src/curobo/content/")[1]
    scene_cfg = load_scene_cfg(scene_path)
    obj_name = scene_cfg["task"]["obj_name"]
    new_data["obj_scale"] = scene_cfg["scene"][obj_name]["scale"][0]
    new_data["obj_pose"] = scene_cfg["scene"][obj_name]["pose"]
    new_data["obj_path"] = os.path.dirname(os.path.dirname(scene_cfg["scene"][obj_name]["file_path"]))
    new_data["scene_path"] = scene_path

    if configs.hand_name == "shadow":
        # Change qpos order of thumb
        new_data["joint_names"] = raw_data["joint_names"][5:] + raw_data["joint_names"][0:5]
        robot_pose = np.concatenate(
            [robot_pose[:, :, :7], robot_pose[:, :, 12:], robot_pose[:, :, 7:12]],
            axis=-1,
        )
        # Add a translation bias of palm which is included in XML but ignored in URDF
        tmp_rot = torch_quaternion_to_matrix(torch.tensor(robot_pose[:, :, 3:7], dtype=torch.float32))
        robot_pose[:, :, :3] -= (tmp_rot @ torch.tensor([0, 0, 0.034]).view(1, 1, 3, 1)).squeeze(-1).numpy()
    elif configs.hand_name == "allegro":
        # Add a rotation bias of palm which is included in XML but ignored in URDF
        tmp_rot = torch_quaternion_to_matrix(torch.tensor(robot_pose[:, :, 3:7]))
        delta_rot = torch_quaternion_to_matrix(torch.tensor([0, 1, 0, 1]).view(1, 1, 4))
        robot_pose[:, :, 3:7] = torch_matrix_to_quaternion(tmp_rot @ delta_rot.transpose(-1, -2))
    elif configs.hand_name == "ur10e_shadow":
        new_data["joint_names"] = (
            raw_data["joint_names"][:8] + raw_data["joint_names"][13:] + raw_data["joint_names"][8:13]
        )
        robot_pose = np.concatenate(
            [robot_pose[:, :, :8], robot_pose[:, :, 13:], robot_pose[:, :, 8:13]],
            axis=-1,
        )
    elif configs.hand_name == "leap":
        # Add a translation and rotation bias of palm which is included in XML but ignored in URDF
        tmp_rot = torch_quaternion_to_matrix(torch.tensor(robot_pose[:, :, 3:7]))
        delta_rot = torch_quaternion_to_matrix(torch.tensor([0, 1, 0, 0]).view(1, 1, 4))
        tmp_rot = tmp_rot @ delta_rot.transpose(-1, -2)
        robot_pose[:, :, 3:7] = torch_matrix_to_quaternion(tmp_rot)
        robot_pose[:, :, :3] -= (tmp_rot @ torch.tensor([0, 0, 0.1])).numpy()
        pass
    elif configs.hand_name == "leap_tac3d":
        # no need to convert base pose
        # no need to convert joint order
        pass
    else:
        raise NotImplementedError

    for i in range(len(robot_pose)):
        if configs.hand.mocap:
            new_data["pregrasp_qpos"] = robot_pose[i, 0]
            new_data["grasp_qpos"] = robot_pose[i, 1]
            new_data["squeeze_qpos"] = robot_pose[i, 2]
        else:
            new_data["approach_qpos"] = robot_pose[i, :-4]
            new_data["pregrasp_qpos"] = robot_pose[i, -4]
            new_data["grasp_qpos"] = robot_pose[i, -3]
            new_data["squeeze_qpos"] = robot_pose[i, -2]
            new_data["lift_qpos"] = robot_pose[i, -1]
        save_path = (
            data_file.replace(configs.task.data_path, configs.grasp_dir)
            .replace("_grasp.npy", f"/{i}_grasp.npy")
            .replace("_mogen.npy", f"/{i}_mogen.npy")
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, new_data)

    # # Additionally save all grasps in a single npy (more convenient for batch IK)
    # if configs.hand.mocap:
    #     new_data["pregrasp_qpos"] = robot_pose[:, 0]
    #     new_data["grasp_qpos"] = robot_pose[:, 1]
    #     new_data["squeeze_qpos"] = robot_pose[:, 2]
    # else:
    #     new_data["approach_qpos"] = robot_pose[:, :-4]
    #     new_data["pregrasp_qpos"] = robot_pose[:, -4]
    #     new_data["grasp_qpos"] = robot_pose[:, -3]
    #     new_data["squeeze_qpos"] = robot_pose[:, -2]
    #     new_data["lift_qpos"] = robot_pose[:, -1]
    # save_path = data_file.replace(configs.task.data_path, configs.grasp_dir).replace("_grasp.npy", "_allgrasp.npy")
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # np.save(save_path, new_data)

    return


def Learning(params):
    data_file, configs = params[0], params[1]
    raw_data = np.load(data_file, allow_pickle=True).item()
    scene_cfg = load_scene_cfg(raw_data["scene_path"])
    target_obj = scene_cfg["task"]["obj_name"]
    new_data = {}
    new_data["obj_path"] = os.path.dirname(os.path.dirname(scene_cfg["scene"][target_obj]["file_path"]))
    new_data["obj_pose"] = scene_cfg["scene"][target_obj]["pose"]
    new_data["obj_scale"] = scene_cfg["scene"][target_obj]["scale"][0]
    save_path = data_file.replace(configs.task.data_path, configs.grasp_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    new_data["grasp_qpos"] = raw_data["grasp_qpos"]
    new_data["pregrasp_qpos"] = raw_data["pregrasp_qpos"]
    new_data["squeeze_qpos"] = raw_data["squeeze_qpos"]
    np.save(save_path, new_data)
    return


def Batched(params):
    data_file, configs = params[0], params[1]
    raw_data = np.load(data_file, allow_pickle=True).item()
    scene_cfg = load_scene_cfg(raw_data["scene_path"])
    target_obj = scene_cfg["task"]["obj_name"]
    new_data = {}
    new_data["obj_path"] = os.path.dirname(os.path.dirname(scene_cfg["scene"][target_obj]["file_path"]))
    new_data["obj_pose"] = scene_cfg["scene"][target_obj]["pose"]
    obj_scale_in_scene = scene_cfg["scene"][target_obj]["scale"][0]
    save_path = data_file.replace(configs.task.data_path, configs.grasp_dir)
    for i in range(raw_data["grasp_qpos"].shape[0]):
        save_path = os.path.join(save_path.split(".npy")[0], f"{i}.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        new_data["obj_scale"] = obj_scale_in_scene * raw_data["scene_scale"][i]
        new_data["grasp_qpos"] = raw_data["grasp_qpos"][i]
        new_data["pregrasp_qpos"] = raw_data["pregrasp_qpos"][i]
        new_data["squeeze_qpos"] = raw_data["squeeze_qpos"][i]
        np.save(save_path, new_data)
    return


def task_format(configs):
    if configs.task.data_name == "BODex":
        if configs.hand.mocap:
            raw_data_struct = ["**", "*_grasp.npy"]
        else:
            raw_data_struct = ["**", "*_mogen.npy"]
    else:
        raw_data_struct = ["**", "*.npy"]
    raw_data_path_lst = glob(os.path.join(configs.task.data_path, *raw_data_struct), recursive=True)
    raw_file_num = len(raw_data_path_lst)
    if configs.task.max_num > 0:
        raw_data_path_lst = np.random.permutation(sorted(raw_data_path_lst))[: configs.task.max_num]
        # raw_data_path_lst = np.random.permutation(sorted(raw_data_path_lst))[1000:6000] # TEMP
    logging.info(
        f"Find {raw_file_num} raw files for {os.path.join(configs.task.data_path, *raw_data_struct)}, use {len(raw_data_path_lst)}"
    )

    if len(raw_data_path_lst) == 0:
        return

    iterable_params = zip(raw_data_path_lst, [configs] * len(raw_data_path_lst))

    with multiprocessing.Pool(processes=configs.n_worker) as pool:
        result_iter = pool.imap_unordered(eval(configs.task.data_name), iterable_params)
        results = list(result_iter)

    grasp_lst = glob(os.path.join(configs.grasp_dir, "**/*.npy"), recursive=True)
    logging.info(f"Get {len(grasp_lst)} grasp data in {configs.save_dir}")
    logging.info("Finish format conversion")
    return
