import numpy as np
import os
from glob import glob
import multiprocessing
import logging
from pathlib import Path
from tqdm import tqdm
import yaml

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc

from util.rot_util import torch_quaternion_to_matrix, torch_matrix_to_axis_angle
from util.grasp_controller import GraspController
from util.robot_adaptor import RobotAdaptor
from util.pin_helper import PinocchioHelper
from util.robots.base import RobotFactory, Robot, ArmHand

from mr_utils.utils_torch import quaternion_angular_error
from mr_utils.utils_calc import quatWXYZ2XYZW, sciR


def read_data(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    return data


def read_data_with_index(args):
    idx, npy_path = args
    return idx, read_data(npy_path)


def get_control_results(data_lst, configs):
    robot_prefix = "rh_" if "allegro" not in configs.hand_name else ""
    robot: ArmHand = RobotFactory.create_robot(robot_type=configs.hand_name, prefix=robot_prefix)

    robot_file_path = robot.get_file_path("mjcf")
    dof_names = robot.dof_names
    doa_names = robot.doa_names
    doa2dof_matrix = robot.doa2dof_matrix

    robot_model = PinocchioHelper(robot_file_path=robot_file_path, robot_file_type="mjcf")
    robot_adaptor = RobotAdaptor(
        robot_model=robot_model,
        dof_names=dof_names,
        doa_names=doa_names,
        doa2dof_matrix=doa2dof_matrix,
    )
    grasp_ctrl = GraspController(configs=None, robot=robot, robot_adaptor=robot_adaptor)

    lift_height = configs.task.lift_height
    n_s = configs.task.n_terminal_steps

    err_obj_pos_all = []
    err_obj_angle_all = []
    sum_cf_all = []
    wrench_all = []
    normalized_wrench_all = []
    success_cases = []
    failure_cases = []
    invalid_cases = []

    for i, r_data in enumerate(tqdm(data_lst, desc="Computing results")):
        if r_data["obj_pose"] == []:
            invalid_cases.append(i)
            continue

        seq_obj_pose = np.asarray(r_data["obj_pose"])
        # obj poses
        init_obj_pose = seq_obj_pose[0, :]
        target_obj_pose = init_obj_pose.copy()
        target_obj_pose[2] += lift_height
        final_obj_pose = seq_obj_pose[-1, :]
        err_obj_pos = np.linalg.norm(target_obj_pose[:3] - final_obj_pose[:3])
        err_obj_pos_z = np.linalg.norm(target_obj_pose[2] - final_obj_pose[2])
        err_obj_angle = (
            sciR.from_quat(quatWXYZ2XYZW(target_obj_pose[3:])).inv() * sciR.from_quat(quatWXYZ2XYZW(final_obj_pose[3:]))
        ).magnitude()
        err_obj_angle = np.rad2deg(err_obj_angle)

        # contact forces
        T = len(r_data["contacts"])
        seq_wrench = np.zeros((T, 6))
        seq_normalized_wrench = np.zeros((T, 6))
        seq_sum_cf = np.zeros((T))
        for t in range(T):
            contacts = r_data["contacts"][t]
            n_con = len(contacts)
            if n_con > 0:
                grasp_matrix = grasp_ctrl.compute_grasp_matrix(contacts)
                cf_all = np.concatenate([c["contact_force"][:3] for c in contacts], axis=0)
                wrench = grasp_matrix @ cf_all.reshape(-1, 1)
                normalized_wrench = grasp_ctrl.compute_normalized_wrench(grasp_matrix, cf_all)
                sum_cf = np.sum(cf_all.reshape(-1, 3)[:, 0])
                seq_wrench[t, :] = wrench.reshape(-1)
                seq_normalized_wrench[t, :] = normalized_wrench.reshape(-1)
                seq_sum_cf[t] = sum_cf

        if err_obj_pos_z < lift_height / 2.0:  # regard as successful grasp
            success_cases.append(i)
            err_obj_pos_all.append(err_obj_pos)
            err_obj_angle_all.append(err_obj_angle)
            sum_cf_all.append(np.mean(seq_sum_cf[-n_s:]))
            wrench_all.append(np.mean(np.linalg.norm(seq_wrench[-n_s:], axis=-1)))
            normalized_wrench_all.append(np.mean(np.linalg.norm(seq_normalized_wrench[-n_s:], axis=-1)))
        else:
            failure_cases.append(i)

    err_obj_pos_all = np.asarray(err_obj_pos_all)
    err_obj_angle_all = np.asarray(err_obj_angle_all)
    sum_cf_all = np.asarray(sum_cf_all)
    wrench_all = np.asarray(wrench_all)
    normalized_wrench_all = np.asarray(normalized_wrench_all)

    success_rate = len(success_cases) / (len(success_cases) + len(failure_cases))
    print(f"success rate: {success_rate}")
    print(f"ave obj pos err: {np.mean(err_obj_pos_all)} +- {np.std(err_obj_pos_all)}")
    print(f"ave obj angle err: {np.mean(err_obj_angle_all)} +- {np.std(err_obj_angle_all)}")
    print(f"ave sum_cf_all: {np.mean(sum_cf_all)} +- {np.std(sum_cf_all)}")
    print(f"ave wrench_all: {np.mean(wrench_all)} +- {np.std(wrench_all)}")
    print(f"ave normalized_wrench_all: {np.mean(normalized_wrench_all)} +- {np.std(normalized_wrench_all)}")
    print(f"failure cases: {failure_cases}")
    print(f"num of invalid cases: {len(invalid_cases)}")

    stat_results = {
        "success_rate": success_rate,
        "ave_obj_pos_err": {
            "mean": float(np.mean(err_obj_pos_all)),
            "std": float(np.std(err_obj_pos_all)),
        },
        "ave_obj_angle_err": {
            "mean": float(np.mean(err_obj_angle_all)),
            "std": float(np.std(err_obj_angle_all)),
        },
        "ave_sum_cf_all": {
            "mean": float(np.mean(sum_cf_all)),
            "std": float(np.std(sum_cf_all)),
        },
        "ave_wrench_all": {
            "mean": float(np.mean(wrench_all)),
            "std": float(np.std(wrench_all)),
        },
        "ave_normalized_wrench_all": {
            "mean": float(np.mean(normalized_wrench_all)),
            "std": float(np.std(normalized_wrench_all)),
        },
        "failure_cases": failure_cases,
        "num_invalid_cases": len(invalid_cases),
        "num_valid_cases": (len(success_cases) + len(failure_cases)),
    }

    # save yaml
    method = configs.task.method
    ablation_name = configs.task.ablation_name
    if method == "ours":
        method = f"{method}_{ablation_name}"

    setting_name = configs.task.setting_name
    save_dir = os.path.join(os.path.dirname(configs.control_dir), "control_stat_res")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{setting_name}_{method}.yaml")
    with open(save_path, "w") as f:
        yaml.safe_dump(stat_results, f, sort_keys=False)

    # index = success_cases[np.argmax(err_obj_angle_all)]
    # grasp_id = index // 8
    # pos_id = index % 8
    # print(f"max obj rot err: {np.max(err_obj_angle_all)}, grasp_id: {grasp_id}, pos_id: {pos_id}")


def task_control_stat(configs):
    control_lst = glob(os.path.join(configs.control_dir, "**/*.npy"), recursive=True)

    # the control results by the method
    method = configs.task.method
    ablation_name = configs.task.ablation_name
    if method == "ours":
        method = f"{method}_{ablation_name}"
    setting_name = configs.task.setting_name

    control_lst = [p for p in control_lst if Path(p).match(f"*/{method}/*.npy") and setting_name in p]
    # control_lst = [x for x in control_lst if method in x and "pos_0" not in x]
    control_lst = sorted(control_lst)
    logging.info(f"Find {len(control_lst)} grasp data using control method '{method}' in {configs.control_dir}.")

    with multiprocessing.Pool(processes=configs.n_worker) as pool:
        result_iter = pool.imap_unordered(read_data_with_index, enumerate(control_lst))
        data_lst = [None] * len(control_lst)
        for idx, data in result_iter:  # keep the original order
            data_lst[idx] = data

    get_control_results(data_lst, configs)

    logging.info("Finish statistics")
