import os
import sys
import time
import copy
import warnings
import numpy as np
from scipy.linalg import block_diag

from .base import BaseEval
from util.robots.base import RobotFactory, Robot, ArmHand
from util.pin_helper import PinocchioHelper
from util.robot_adaptor import RobotAdaptor
from util.grasp_controller import GraspController

warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
Baseline1:
    Position + Force feedback control.
    Each finger is controlled independently.
    Before contact, position control; After contact, force control.
    The desired contact force of each finger is set to the same pre-defined value.
"""


class tabletopDummyArmBS1Eval(BaseEval):
    def _initialize(self):
        self.method_name = "bs1"
        robot_name = self.configs.hand_name
        robot_prefix = "rh_" if "allegro" not in robot_name else ""
        robot: ArmHand = RobotFactory.create_robot(robot_type=robot_name, prefix=robot_prefix)

        robot_file_path = robot.get_file_path("mjcf")
        dof_names = robot.dof_names
        doa_names = robot.doa_names
        doa2dof_matrix = robot.doa2dof_matrix

        self.robot = robot
        self.robot_model = PinocchioHelper(robot_file_path=robot_file_path, robot_file_type="mjcf")
        self.robot_adaptor = RobotAdaptor(
            robot_model=self.robot_model,
            dof_names=dof_names,
            doa_names=doa_names,
            doa2dof_matrix=doa2dof_matrix,
        )
        self.grasp_ctrl = GraspController(
            configs=self.configs.task.control, robot=self.robot, robot_adaptor=self.robot_adaptor
        )
        self.dof_data2user_indices = [self.grasp_data["joint_names"].index(name) for name in dof_names]

    def _dof_data2user(self, q):
        return q[..., self.dof_data2user_indices].copy()

    def damped_pinv(self, J):
        lambd = 1.0
        I_mat = np.eye(J.shape[0])
        J_inv = J.T @ np.linalg.inv(J @ J.T + lambd**2 * I_mat)
        return J_inv

    def _simulate_under_extforce_details(self, pregrasp_qpos, grasp_qpos, squeeze_qpos):
        # Pre-calculated parameters
        ctrl_freq = self.ctrl_freq
        sim_step_per_action = self.sim_step_per_action
        b_debug = self.configs.task.debug_viewer
        arm_ndoa = self.robot.arm.n_doa
        hand_ndoa = self.robot.hand.n_doa
        Kp_inv = self.grasp_ctrl.Kp_inv[-hand_ndoa:, -hand_ndoa:]

        # initialize actuated qpos
        curr_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
        qpos_a = self.robot_adaptor._dof2doa(curr_qpos_f)
        self.mj_ho.ctrl_qpos_a(self.robot.doa_names, qpos_a)
        init_qpos_a = qpos_a.copy()

        # compute full path via interpolation
        curr_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
        grasp_qpos_f = self._dof_data2user(grasp_qpos)
        squeeze_qpos_f = self._dof_data2user(squeeze_qpos)
        qpos_f_path_1 = self.grasp_ctrl.interplote_qpos(curr_qpos_f, grasp_qpos_f, step=ctrl_freq * 2)
        qpos_f_path_2 = self.grasp_ctrl.interplote_qpos(grasp_qpos_f, squeeze_qpos_f, step=ctrl_freq * 2)
        qpos_f_path = np.concatenate([qpos_f_path_1, qpos_f_path_2], axis=0)

        if "shadow" in self.robot.name:
            final_single_force = 5.0 # slightly larger than F_ub / N_finger to improve the grasp success rate.
        elif "allegro" in self.robot.name:
            final_single_force = 3.0
        elif "leap" in self.robot.name:
            final_single_force = 2.5
        else:
            raise NotImplementedError
        max_steps = int(qpos_f_path.shape[0] * 1.2)
        step = 0
        waypoint_idx = 0

        while step < max_steps:
            target_qpos_f = qpos_f_path[waypoint_idx]
            # get state
            curr_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
            curr_qpos_a = self.mj_ho.get_qpos_a()
            ho_contacts = self.mj_ho.get_curr_contact_info()
            obj_pose = self.mj_ho.get_obj_pose()  # pos + quat(w,x,y,z)

            # compute some variables
            contact_force_all = np.array([contact["contact_force"][:3] for contact in ho_contacts]).reshape(-1, 3)
            curr_sum_force = np.sum(contact_force_all[:, 0])

            if b_debug:
                print(f"--------------- {step} step ---------------")
                for contact in ho_contacts:
                    print(
                        f"{step} step, body1_name: {contact['body1_name']}, body2_name: {contact['body2_name']}, "
                        + f"contact_force: {contact['contact_force']}"
                    )
                print(f"curr_sum_force: {curr_sum_force}")

            # compute qpos error
            curr_hand_qpos_a = curr_qpos_a[-hand_ndoa:]
            target_qpos_a = self.robot_adaptor._dof2doa(target_qpos_f)
            target_hand_qpos_a = target_qpos_a[-hand_ndoa:]
            hand_qpos_err = (target_hand_qpos_a - curr_hand_qpos_a).reshape(-1, 1)
            w_q = np.ones_like(curr_hand_qpos_a)
            gain_q = np.eye(hand_ndoa)

            n_con = len(ho_contacts)
            if n_con:
                # obtain in-contact joints and contact force errors
                updated_contacts, stacked = self.grasp_ctrl.Ks(curr_qpos_a, curr_qpos_f, ho_contacts)
                contact_force_all = np.concatenate(
                    [c["contact_force"][:3] for c in updated_contacts], axis=0
                ).reshape(-1, 1)
                contact_jaco_all = stacked["jaco_hf"]

                target_contact_force_all = np.tile(np.array([final_single_force, 0, 0]), (n_con, 1)).reshape(-1, 1)
                contact_force_err = target_contact_force_all - contact_force_all
                in_contact_q_indices = np.any(contact_jaco_all != 0, axis=0)
                w_q[in_contact_q_indices] = 0

            # position control command
            w_q = np.diag(w_q)
            delta_hand_q_a = w_q @ gain_q @ hand_qpos_err

            if n_con:
                # force control command
                gain_f = min(0.8, 1.0 / (len(qpos_f_path) - waypoint_idx))  # step size
                force_control_input = gain_f * Kp_inv @ contact_jaco_all.T @ contact_force_err

                delta_hand_q_a += force_control_input
                if b_debug:
                    print(f"pos_control_input: {(w_q @ gain_q @ hand_qpos_err).reshape(-1)}")
                    print(f"force_control_input: {force_control_input.reshape(-1)}")

            opt_hand_q_a = curr_hand_qpos_a + delta_hand_q_a.reshape(-1)
            opt_q_a = np.concatenate([init_qpos_a[:arm_ndoa], opt_hand_q_a], axis=0)

            assert sim_step_per_action % 5 == 0
            self.mj_ho.ctrl_qpos_a_with_interp(
                curr_qpos_a, opt_q_a, names=self.robot.doa_names, step_outer=sim_step_per_action // 5, step_inner=5
            )

            step += 1  # next step
            waypoint_idx = min(waypoint_idx + 1, len(qpos_f_path) - 1)

            self.grasp_ctrl.r_data["obj_pose"].append(obj_pose)
            self.grasp_ctrl.r_data["dof"].append(curr_qpos_f)
            self.grasp_ctrl.r_data["doa"].append(curr_qpos_a)
            self.grasp_ctrl.r_data["contacts"].append(ho_contacts)
            self.grasp_ctrl.r_data["planned_dof"].append(target_qpos_f)

        return
