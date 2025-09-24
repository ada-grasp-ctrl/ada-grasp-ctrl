import os
from glob import glob
import numpy as np
from itertools import islice
import torch
from typing import Optional
from tqdm import tqdm
import time

from mr_utils.utils_torch import (
    quaternion_to_matrix,
)


def chunk_iterable(iterable, size):
    """Yield successive chunks from an iterable, each of length `size`."""
    iterable = iter(iterable)
    while True:
        chunk = list(islice(iterable, size))
        if not chunk:
            break
        yield chunk


def solve_ik_for_arm_qpos(
    kin,
    b_poses: np.ndarray,
    n_arm_dof: int,
    batch_size: int = 128,
    ref_config: Optional[np.ndarray] = None,
    use_seed_config: bool = False,
    return_seeds: int = 1,
    device="cuda:0",
) -> np.ndarray:
    """
    Solves IK for a batch of base poses in fixed-size chunks (with padding) to work with CUDA graphs.

    Args:
        b_poses: np.ndarray of shape (..., 7), format: (x, y, z, qw, qx, qy, qz), defined in the world frame
        n_arm_dof: number of DOFs in the robot arm
        batch_size: size of each fixed-shape CUDA-compatible batch
        ref_config: np.ndarray of shape (..., n_dof) as reference joint config
        use_seed_config: whether to use reference config as seed for IK solver
        return_seeds: number of IK seeds to try

    Returns:
        arm_qpos: np.ndarray of shape (..., n_arm_dof)
    """
    if ref_config is not None:
        assert b_poses.shape[:-1] == ref_config.shape[:-1], "Shape mismatch between b_poses and ref_config"

    _b_poses = torch.from_numpy(b_poses).float().to(device)
    original_shape = _b_poses.shape[:-1]  # (...), 7
    _b_poses = _b_poses.view(-1, 7)  # (B, 7)
    total = _b_poses.shape[0]

    ref_config_tensor = None
    if ref_config is not None:
        ref_config_tensor = torch.from_numpy(ref_config).float().to(device).view(total, -1)

    all_qpos = []

    for i in tqdm(range(0, total, batch_size), desc="Batch IK solving"):
        t1 = time.time()

        batch = _b_poses[i : i + batch_size]
        actual_size = batch.shape[0]
        batch_ref = ref_config_tensor[i : i + batch_size] if ref_config is not None else None

        # Padding if last batch is smaller than batch_size
        if actual_size < batch_size:
            pad_n = batch_size - actual_size
            pad = batch[-1:].repeat(pad_n, 1)
            batch = torch.cat([batch, pad], dim=0)
            if batch_ref is not None:
                pad_ref = batch_ref[-1:].repeat(pad_n, 1)
                batch_ref = torch.cat([batch_ref, pad_ref], dim=0)

        pos = batch[:, :3]
        quat = batch[:, 3:7]
        matrix = torch.eye(4, device=device).repeat(batch_size, 1, 1)
        matrix[:, :3, 3] = pos
        matrix[:, :3, :3] = quaternion_to_matrix(quat)

        # Retry until success or max tries
        for attempt in range(10):
            res = kin.solve_ik_batch(
                matrix,
                ref_configs=batch_ref,
                use_ref_as_init=use_seed_config,
            )
            qpos = res["q"][..., :n_arm_dof]

            success_mask = res["success"].any(dim=-1)
            if not success_mask.all():
                tqdm.write(f"[Warn] Some IKs failed. Retrying batch {i // batch_size} (attempt {attempt + 1})")
                continue

            if ref_config is not None:
                dists = torch.norm(qpos - batch_ref[:, :n_arm_dof].unsqueeze(1), dim=-1)
                min_dists, _ = dists.min(dim=-1)
                max_dist = min_dists.max()
                tqdm.write(f"[Info] Max distances to refs in batch: {max_dist}.")
                if max_dist > 3.14:
                    idx = torch.argmax(min_dists)
                    tqdm.write(f"[Warn] Max ref mismatch > 3.14 rad: {max_dist:.4f}. Retrying.")
                    tqdm.write(f"qpos[{idx}]: {qpos[idx]}")
                    tqdm.write(f"ref [{idx}]: {batch_ref[idx, :n_arm_dof]}")
                    continue
            break
        else:
            raise RuntimeError(f"IK failed for batch {i // batch_size} after retries.")

        # Remove padding if needed
        qpos = qpos[:actual_size]
        all_qpos.append(qpos)
        tqdm.write(f"[Info] Batch {i // batch_size} solved in {time.time() - t1:.2f}s.")

    all_qpos = torch.cat(all_qpos, dim=0).view(*original_shape, n_arm_dof)
    return all_qpos.cpu().numpy()


def compute_dummy_arm_qpos(params, device, robot, kin):
    input_npy_path_list, configs = params[0], params[1]

    graspdata_list = []
    pregrasp_qpos = []
    grasp_qpos = []
    squeeze_qpos = []

    # concatenate grasps together for batched IK
    for path in input_npy_path_list:
        graspdata = np.load(path, allow_pickle=True).item()
        graspdata_list.append(graspdata)
        pregrasp_qpos.append(graspdata["pregrasp_qpos"])
        grasp_qpos.append(graspdata["grasp_qpos"])
        squeeze_qpos.append(graspdata["squeeze_qpos"])

    pregrasp_qpos_array = np.asarray(pregrasp_qpos)  # pose (x y z qw qx qy qz) + hand qpos
    grasp_qpos_array = np.asarray(grasp_qpos)
    squeeze_qpos_array = np.asarray(squeeze_qpos)
    n_grasp = pregrasp_qpos_array.shape[0]
    n_arm_dof = robot.arm.n_dof
    n_hand_dof = robot.hand.n_dof

    # Start IK solving
    kin.create_ik_solver(num_retries=10, regularlization=1e-3)

    # Step 1: pre-grasp poses
    zero_arm_qpos = np.zeros((n_grasp, n_arm_dof))
    zero_hand_qpos = np.zeros((n_grasp, n_hand_dof))
    pregrasp_ref_config = np.concatenate([zero_arm_qpos, zero_hand_qpos], axis=-1)

    pregrasp_arm_qpos = solve_ik_for_arm_qpos(
        kin,
        b_poses=pregrasp_qpos_array[:, 0:7],
        n_arm_dof=n_arm_dof,
        batch_size=n_grasp,
        ref_config=pregrasp_ref_config,
        use_seed_config=False,
        device=device,
    )

    # Step 2: in-grasp poses
    initial_noise_std = torch.zeros(len(kin.robot_joint_names), device=device)
    initial_noise_std[3:6] = 0.1  # noise only for arm rotation DOFs
    kin.create_ik_solver(num_retries=10, regularlization=1e-3, initial_noise_std=initial_noise_std)

    ingrasp_arm_qpos = solve_ik_for_arm_qpos(
        kin,
        b_poses=grasp_qpos_array[:, 0:7],
        n_arm_dof=n_arm_dof,
        batch_size=n_grasp,
        ref_config=np.concatenate([pregrasp_arm_qpos, zero_hand_qpos], axis=-1),
        use_seed_config=True,
        device=device,
    )

    # Step 3: squeeze poses
    squeeze_arm_qpos = solve_ik_for_arm_qpos(
        kin,
        b_poses=squeeze_qpos_array[:, 0:7],
        n_arm_dof=n_arm_dof,
        batch_size=n_grasp,
        ref_config=np.concatenate([ingrasp_arm_qpos, zero_hand_qpos], axis=-1),
        use_seed_config=True,
        device=device,
    )

    # Concatenate arm qpos and hand qpos
    pregrasp_qpos_array = np.concatenate([pregrasp_arm_qpos, pregrasp_qpos_array[:, 7:]], axis=-1)
    grasp_qpos_array = np.concatenate([ingrasp_arm_qpos, grasp_qpos_array[:, 7:]], axis=-1)
    squeeze_qpos_array = np.concatenate([squeeze_arm_qpos, squeeze_qpos_array[:, 7:]], axis=-1)

    # Save the IK solution to the original npy file
    for i in range(len(input_npy_path_list)):
        path = input_npy_path_list[i]
        graspdata = graspdata_list[i]  # keep the other saved info
        if "joint_names" in graspdata.keys():
            graspdata["joint_names"] = robot.arm.dof_names + graspdata["joint_names"]
        else:
            graspdata["joint_names"] = robot.dof_names  # need check
        graspdata["pregrasp_qpos"] = pregrasp_qpos_array[i, :]
        graspdata["grasp_qpos"] = grasp_qpos_array[i, :]
        graspdata["squeeze_qpos"] = squeeze_qpos_array[i, :]
        graspdata["approach_qpos"] = pregrasp_qpos_array[i, :].reshape(1, -1)
        path = path.replace(configs.hand_name, f"dummy_arm_{configs.hand_name}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, graspdata)


def task_dummy_arm_qpos(configs):
    input_path_lst = glob(os.path.join(configs.grasp_dir, "**/*.npy"), recursive=True)
    init_num = len(input_path_lst)

    if configs.skip:
        eval_path_lst = glob(os.path.join(configs.eval_dir, "**/*.npy"), recursive=True)
        eval_path_lst = [p.replace(configs.eval_dir, configs.grasp_dir) for p in eval_path_lst]
        input_path_lst = list(set(input_path_lst).difference(set(eval_path_lst)))
    skip_num = init_num - len(input_path_lst)
    input_path_lst = sorted(input_path_lst)
    if configs.task.max_num > 0:
        input_path_lst = np.random.permutation(input_path_lst)[: configs.task.max_num]

    if len(input_path_lst) == 0:
        return

    chunks = list(chunk_iterable(input_path_lst, 8192)) # batch size

    iterable_params = zip(chunks, [configs] * len(chunks))

    from util.robots import RobotFactory
    from util.pk_helper import PytorchKinematicsHelper

    device = configs.task.device
    robot_prefix = "rh_" if "allegro" not in configs.hand_name else ""
    robot = RobotFactory.create_robot(robot_type=f"dummy_arm_{configs.hand_name}", prefix=robot_prefix)
    kin = PytorchKinematicsHelper(robot=robot, device=device)
    kin.create_serial_chain(ee_name=robot.arm.body_names[-1])  # dummy_RotZB

    for ip in iterable_params:
        compute_dummy_arm_qpos(ip, device, robot, kin)
