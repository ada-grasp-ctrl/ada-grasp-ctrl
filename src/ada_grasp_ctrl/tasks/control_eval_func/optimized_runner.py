"""Shared optimized approach/squeeze runner for ours and BS2 policies."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class WrenchControlPolicy:
    """Policy differences for the shared wrench-control episode."""

    use_approaching_arm_motion: bool


def run_wrench_control_episode(evaluator, pregrasp_qpos, grasp_qpos, squeeze_qpos, policy):
    """Run the common optimized controller without changing algorithm weights.

    Args:
        evaluator: Initialized :class:`BaseEval` subclass instance.
        pregrasp_qpos: Prepared pregrasp qpos retained for API symmetry.
        grasp_qpos: Target in-grasp qpos.
        squeeze_qpos: Target squeezed qpos.
        policy: Only the method-specific approaching-arm decision.

    Returns:
        None.
    """
    del pregrasp_qpos  # MuJoCo has already been initialized at this pose.
    control_config = evaluator.configs.task.control
    evaluator.desired_sum_force_as_traj = control_config.desired_sum_force_as_traj
    evaluator.use_desired_sum_force_ub = control_config.use_desired_sum_force_ub

    ctrl_freq = evaluator.ctrl_freq
    action_dt = evaluator.action_dt
    sim_step_per_action = evaluator.sim_step_per_action
    debug = evaluator.configs.task.debug_viewer
    robot = evaluator.robot
    grasp_ctrl = evaluator.grasp_ctrl

    current_qpos_f = evaluator.mj_ho.get_qpos_f(names=robot.dof_names)
    evaluator.mj_ho.ctrl_qpos_a(robot.doa_names, evaluator.robot_adaptor._dof2doa(current_qpos_f))
    current_qpos_f = evaluator.mj_ho.get_qpos_f(names=robot.dof_names)
    grasp_qpos_f = evaluator._dof_data2user(grasp_qpos)
    squeeze_qpos_f = evaluator._dof_data2user(squeeze_qpos)
    path_approach = grasp_ctrl.interplote_qpos(current_qpos_f, grasp_qpos_f, step=ctrl_freq * 2)
    path_squeeze = grasp_ctrl.interplote_qpos(grasp_qpos_f, squeeze_qpos_f, step=ctrl_freq * 2)
    qpos_path = np.concatenate([path_approach, path_squeeze], axis=0)

    final_sum_force = grasp_ctrl.final_sum_force
    desired_sum_force_upper = final_sum_force + 0.2
    force_increment = final_sum_force / path_squeeze.shape[0]
    last_delta_q = np.zeros(robot.n_doa)
    max_steps = qpos_path.shape[0] * 2
    stage = 1
    step = 0
    waypoint_index = 0
    # This value is overwritten on every ordinary Stage-1 first step. It only
    # defines the formerly broken edge case where the first solve enters Stage 2.
    desired_sum_force = grasp_ctrl.stage1_force_thres

    while step < max_steps:
        target_qpos_f = qpos_path[waypoint_index]
        current_qpos_f = evaluator.mj_ho.get_qpos_f(names=robot.dof_names)
        current_qpos_a = evaluator.mj_ho.get_qpos_a()
        contacts = evaluator.mj_ho.get_curr_contact_info()
        object_pose = evaluator.mj_ho.get_obj_pose()
        contact_forces = np.array([contact["contact_force"][:3] for contact in contacts]).reshape(-1, 3)
        current_sum_force = np.sum(contact_forces[:, 0])
        grasp_matrix = grasp_ctrl.compute_grasp_matrix(contacts)

        if step >= qpos_path.shape[0] and current_sum_force > final_sum_force:
            break

        start = time.time()
        balance_metric, _ = grasp_ctrl.check_wrench_balance(grasp_matrix, b_print_opt_details=False)
        balance_elapsed = time.time() - start
        if debug:
            print(f"--------------- {step} step ---------------")
            for contact in contacts:
                print(
                    f"{step} step, body1_name: {contact['body1_name']}, "
                    f"body2_name: {contact['body2_name']}, contact_force: {contact['contact_force']}"
                )
            print(f"curr_sum_force: {current_sum_force}")
            print(f"check_wrench_balance() time cost: {balance_elapsed}")
            print(f"balance_metric: {balance_metric}")

        if balance_metric < grasp_ctrl.balance_thres or (
            control_config.stage2_after_full_path and step > qpos_path.shape[0]
        ):
            stage = 2
        elif control_config.free_stage_switch:
            stage = 1

        if stage == 1:
            desired_sum_force = max(grasp_ctrl.stage1_force_thres, current_sum_force - force_increment)
        elif evaluator.desired_sum_force_as_traj:
            desired_sum_force = max(current_sum_force, desired_sum_force) + force_increment
            if evaluator.use_desired_sum_force_ub:
                desired_sum_force = min(desired_sum_force, desired_sum_force_upper)
        else:
            desired_sum_force = min(current_sum_force, final_sum_force) + force_increment

        start = time.time()
        result = grasp_ctrl.ctrl_opt(
            stage=stage,
            dt=action_dt,
            curr_q_a=current_qpos_a,
            curr_q_f=current_qpos_f,
            target_q_f=target_qpos_f,
            desired_sum_force=desired_sum_force,
            last_dq_a=last_delta_q,
            ho_contacts=contacts,
            grasp_matrix=grasp_matrix,
            b_use_arm_motion=policy.use_approaching_arm_motion,
            b_print_opt_details=debug,
        )
        control_elapsed = time.time() - start
        last_delta_q = result["dq_a"]

        assert sim_step_per_action % 5 == 0
        evaluator.mj_ho.ctrl_qpos_a_with_interp(
            current_qpos_a,
            result["q_a"],
            names=robot.doa_names,
            step_outer=sim_step_per_action // 5,
            step_inner=5,
        )
        step += 1
        waypoint_index = min(waypoint_index + 1, len(qpos_path) - 1)

        grasp_ctrl.r_data["obj_pose"].append(object_pose)
        grasp_ctrl.r_data["dof"].append(current_qpos_f)
        grasp_ctrl.r_data["doa"].append(current_qpos_a)
        grasp_ctrl.r_data["contacts"].append(contacts)
        grasp_ctrl.r_data["planned_dof"].append(target_qpos_f)
        grasp_ctrl.r_data["balance_metric"].append(balance_metric)
        grasp_ctrl.r_data["t_check_balance"].append(balance_elapsed)
        grasp_ctrl.r_data["t_ctrl_opt"].append(control_elapsed)
        grasp_ctrl.r_data["stage"].append(stage)
        grasp_ctrl.r_data["opt_res"].append(result)
        grasp_ctrl.r_data["desired_sum_force"].append(desired_sum_force)
