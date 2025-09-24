import os
import sys

import numpy as np

from .base import BaseEval
from util.rot_util import np_get_delta_qpos
from util.robots.base import RobotFactory, Robot, ArmHand
from util.pin_helper import PinocchioHelper
from util.robot_adaptor import RobotAdaptor
from util.grasp_controller import GraspController


class tabletopDummyArmOpEval(BaseEval):
    def _initialize(self):
        self.method_name = "op"
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
        self.grasp_ctrl = GraspController(robot=self.robot, robot_adaptor=self.robot_adaptor)

        self.dof_data2user_indices = [self.grasp_data["joint_names"].index(name) for name in dof_names]

    def _dof_data2user(self, q):
        return q[..., self.dof_data2user_indices].copy()

    def _simulate_under_extforce_details(self, pre_obj_qpos, lift_height):
        self._initialize()

        # 1. Set object gravity
        external_force_direction = np.array([0.0, 0, -1, 0, 0, 0])
        self.mj_ho.set_ext_force_on_obj(10 * external_force_direction * self.configs.task.obj_mass)

        sim_dt = self.mj_ho.spec.option.timestep
        ctrl_freq = 10
        action_dt = 1.0 / ctrl_freq
        sim_step_per_action = action_dt / sim_dt
        assert sim_step_per_action == int(sim_step_per_action)
        sim_step_per_action = int(sim_step_per_action)

        # --------------------------------------------------------------------

        # initialize actuated qpos
        curr_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
        qpos_a = self.robot_adaptor._dof2doa(curr_qpos_f)
        self.mj_ho.ctrl_qpos_a(self.robot.doa_names, qpos_a)

        # ho_contacts = self.mj_ho.get_curr_contact_info()
        # contact_force_all = np.array([contact["contact_force"][:3] for contact in ho_contacts]).reshape(-1, 3)
        # curr_sum_force = np.sum(contact_force_all[:, 0])
        # if curr_sum_force > 5:
        #     return

        # compute full path via interpolation
        curr_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
        grasp_qpos_f = self._dof_data2user(self.grasp_data["grasp_qpos"])
        squeeze_qpos_f = self._dof_data2user(self.grasp_data["squeeze_qpos"])
        qpos_f_path_1 = self.grasp_ctrl.interplote_qpos(curr_qpos_f, grasp_qpos_f, step=ctrl_freq * 2)
        qpos_f_path_2 = self.grasp_ctrl.interplote_qpos(grasp_qpos_f, squeeze_qpos_f, step=ctrl_freq * 2)
        qpos_f_path = np.concatenate([qpos_f_path_1, qpos_f_path_2], axis=0)

        for i, target_qpos_f in enumerate(qpos_f_path):
            # get state
            curr_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
            curr_qpos_a = self.mj_ho.get_qpos_a()

            # control command
            target_qpos_a = self.robot_adaptor._dof2doa(target_qpos_f)

            assert sim_step_per_action % 5 == 0
            self.mj_ho.ctrl_qpos_a_with_interp(
                curr_qpos_a,
                target_qpos_a,
                names=self.robot.doa_names,
                step_outer=sim_step_per_action // 5,
                step_inner=5,
            )

        # -------------------------------------------------
        # Lift the object
        # -------------------------------------------------
        curr_qpos_a = self.mj_ho.get_qpos_a()
        lift_qpos_a = curr_qpos_a.copy()
        lift_qpos_a[2] += lift_height  # lift, by IK
        path = self.grasp_ctrl.interplote_qpos(curr_qpos_a, lift_qpos_a, step=2 * ctrl_freq)
        for q_a in path:
            curr_qpos_a = self.mj_ho.get_qpos_a()
            obj_pose = self.mj_ho.get_obj_pose()  # pos + quat(w,x,y,z)
            self.mj_ho.ctrl_qpos_a_with_interp(
                curr_qpos_a, q_a, names=self.robot.doa_names, step_outer=sim_step_per_action // 5, step_inner=5
            )
            self.grasp_ctrl.r_data["obj_pose"].append(obj_pose)

        # -------------------------------------------------
        # Test grasp under disturbances
        # -------------------------------------------------
        external_force_direction = np.array(
            [
                [-1.0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        )
        for i in range(len(external_force_direction)):
            self.mj_ho.set_ext_force_on_obj(10 * external_force_direction[i] * self.configs.task.obj_mass)

            # Wait for 2 seconds
            for _ in range(10):
                self.mj_ho.control_hand_step(step_inner=50)

                # Early stop
                latter_obj_qpos = self.mj_ho.get_obj_pose()
                delta_pos, delta_angle = np_get_delta_qpos(pre_obj_qpos, latter_obj_qpos)
                succ_flag = (delta_pos < self.configs.task.simulation_metrics.trans_thre) & (
                    delta_angle < self.configs.task.simulation_metrics.angle_thre
                )
                if not succ_flag:
                    break

            if not succ_flag:
                break

        return
