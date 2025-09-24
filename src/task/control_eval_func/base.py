import os
import sys
from copy import deepcopy
import logging

import numpy as np
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from util.rot_util import (
    np_get_delta_qpos,
)
from util.hand_util import MjHO
from util.file_util import load_json


class BaseEval:
    def __init__(self, input_npy_path, configs):
        self.input_npy_path = input_npy_path
        self.configs = configs
        self.grasp_data = np.load(input_npy_path, allow_pickle=True).item()

        # Fix object mass by setting density
        obj_info = load_json(os.path.join(self.grasp_data["obj_path"], "info/simplified.json"))
        obj_coef = obj_info["mass"] / (obj_info["density"] * (obj_info["scale"] ** 3))
        new_obj_density = configs.task.obj_mass / (obj_coef * (self.grasp_data["obj_scale"] ** 3))

        # Build mj_spec
        self.mj_ho = MjHO(
            obj_path=self.grasp_data["obj_path"],
            obj_scale=self.grasp_data["obj_scale"],
            has_floor_z0=configs.setting == "tabletop",
            obj_density=new_obj_density,
            hand_xml_path=configs.hand.xml_path,
            hand_mocap=configs.hand.mocap,
            exclude_table_contact=configs.hand.exclude_table_contact,
            friction_coef=configs.task.miu_coef,
            debug_render=configs.task.debug_render,
            debug_viewer=configs.task.debug_viewer,
        )

        if self.configs.task.debug_viewer or self.configs.task.debug_render:
            with open("debug.xml", "w") as f:
                f.write(self.mj_ho.spec.to_xml())

        return

    def _simulate_under_extforce_details(self, pre_obj_qpos):
        raise NotImplementedError

    def _eval_simulate_under_extforce(self, obj_pose, file_suffix):
        eval_config = self.configs.task.simulation_metrics

        # initialize grasp controller
        self.mj_ho.reset()
        self._initialize()
        pregrasp_qpos = self.grasp_data["pregrasp_qpos"].copy()
        grasp_qpos = self.grasp_data["grasp_qpos"].copy()
        squeeze_qpos = self.grasp_data["squeeze_qpos"].copy()

        # set the arm qpos of pregrasp_qpos to be the same as grasp_qpos
        if self.configs.task.arm_pregrasp_is_grasp:
            n_arm_dof = 6
            pregrasp_qpos[:n_arm_dof] = grasp_qpos[:n_arm_dof]

        # adjust (larger) pregrasp hand qpos and (tighter) hand squeeze qpos
        pregrasp_hand_qpos = pregrasp_qpos[n_arm_dof:]
        grasp_hand_qpos = grasp_qpos[n_arm_dof:]
        squeeze_hand_qpos = squeeze_qpos[n_arm_dof:]
        t = self.configs.task.graspdata.pregrasp_t
        pregrasp_qpos[n_arm_dof:] += t * (pregrasp_hand_qpos - grasp_hand_qpos)
        t = self.configs.task.graspdata.squeeze_t
        squeeze_qpos[n_arm_dof:] += t * (squeeze_hand_qpos - grasp_hand_qpos)

        # reset object and hand
        init_qpos = pregrasp_qpos
        init_obj_pose = obj_pose.copy()
        ho_contact, hh_contact = self.mj_ho.get_contact_info(init_qpos, init_obj_pose)
        self.mj_ho.udpate_debug_viewer()

        # compute target final obj qpos
        lift_height = 0.2
        pre_obj_qpos = deepcopy(self.mj_ho.get_obj_pose())
        if self.configs.setting == "tabletop":
            pre_obj_qpos[2] += lift_height

        # Filter out bad initialization with severe penetration
        ho_dist = min([c["contact_dist"] for c in ho_contact]) if len(ho_contact) > 0 else 0
        hh_dist = min([c["contact_dist"] for c in hh_contact]) if len(hh_contact) > 0 else 0
        curr_ho_contact = self.mj_ho.get_curr_contact_info()
        contact_force = max([np.linalg.norm(c["contact_force"]) for c in curr_ho_contact]) if len(ho_contact) > 0 else 0
        if ho_dist < -eval_config.max_pene or hh_dist < -eval_config.max_pene or contact_force > eval_config.max_force:
            if self.configs.task.debug_viewer or self.configs.task.debug_render:
                print(
                    f"Severe penetration larger than {eval_config.max_pene}. ho_dist: {ho_dist}, hh_dist: {hh_dist}, contact_force: {contact_force}"
                )
        else:
            # Set object gravity
            external_force_direction = np.array([0.0, 0, -1, 0, 0, 0])
            self.mj_ho.set_ext_force_on_obj(10 * external_force_direction * self.configs.task.obj_mass)

            # Shared parameters
            self.ctrl_freq = 10
            sim_dt = self.mj_ho.spec.option.timestep
            self.action_dt = 1.0 / self.ctrl_freq
            sim_step_per_action = self.action_dt / sim_dt
            assert sim_step_per_action == int(sim_step_per_action)
            self.sim_step_per_action = int(sim_step_per_action)

            # Detailed simulation methods for testing
            self._simulate_under_extforce_details(pregrasp_qpos, grasp_qpos, squeeze_qpos)

            # Lift the object
            curr_qpos_a = self.mj_ho.get_qpos_a()
            lift_qpos_a = curr_qpos_a.copy()
            lift_qpos_a[2] += lift_height  # lift, by IK
            path = self.grasp_ctrl.interplote_qpos(curr_qpos_a, lift_qpos_a, step=2 * self.ctrl_freq)
            for q_a in path:
                curr_qpos_a = self.mj_ho.get_qpos_a()
                obj_pose = self.mj_ho.get_obj_pose()  # pos + quat(w,x,y,z)
                self.grasp_ctrl.r_data["obj_pose"].append(obj_pose)
                self.mj_ho.ctrl_qpos_a_with_interp(
                    curr_qpos_a, q_a, names=self.robot.doa_names, step_outer=self.sim_step_per_action // 5, step_inner=5
                )

            # terminal state
            obj_pose = self.mj_ho.get_obj_pose()
            self.grasp_ctrl.r_data["obj_pose"].append(obj_pose)

        input_dir = self.configs[self.configs.task.input_data]
        dirname = os.path.dirname(self.input_npy_path)
        filename = os.path.basename(self.input_npy_path)
        self.ablation_name = self.configs.task.control.ablation_name
        method_name = f"{self.method_name}_{self.ablation_name}" if self.method_name == "ours" else self.method_name

        new_dirname = os.path.join(dirname.replace(input_dir, self.configs.control_dir), f"{method_name}")
        new_filename = filename.replace(".npy", f"{file_suffix}.npy")
        save_path = os.path.join(new_dirname, new_filename)
        self.grasp_ctrl.save_recorded_data(path=save_path)

        # -------------------------------------------------------------------------------
        # Compare the resulted object pose
        latter_obj_qpos = self.mj_ho.get_obj_pose()
        delta_pos, delta_angle = np_get_delta_qpos(pre_obj_qpos, latter_obj_qpos)

        if self.configs.task.debug_viewer or self.configs.task.debug_render:
            print(delta_pos, delta_angle)
            # Save rendered video
            if self.configs.task.debug_render and len(self.mj_ho.debug_images) > 10:
                # save mp4
                debug_path = self.input_npy_path.replace(input_dir, self.configs.task.debug_dir).replace(
                    ".npy", f"{method_name}_{file_suffix}.mp4"
                )
                os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                # Extract every other frame from a 50 Hz image sequence to convert it into 25 Hz.
                frames = self.mj_ho.debug_images[::2]
                with imageio.get_writer(debug_path, fps=25, codec="libx264", quality=8) as writer:
                    for img in frames:
                        writer.append_data(img)
                print("Save MP4 (25Hz) to ", debug_path)


    def run(self):
        # with object position uncertainty (8 planar directions)
        directions = np.array(
            [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]],
            dtype=float,
        )
        directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)

        for obj_offset_dist in self.configs.task.offsets:
            # compute object offsets
            if obj_offset_dist == 0:  # no object position uncertainty
                shifted_obj_poses = np.tile(self.grasp_data["obj_pose"].copy(), (1, 1))
            else:  # perturb the object position
                shifted_obj_poses = np.tile(self.grasp_data["obj_pose"].copy(), (len(directions), 1))
                shifted_obj_poses[:, 0:2] += obj_offset_dist * directions

            for i in range(shifted_obj_poses.shape[0]):
                file_suffix = f"_dist_{str(int(100 * obj_offset_dist))}_pos_{i}"
                self._eval_simulate_under_extforce(obj_pose=shifted_obj_poses[i, :], file_suffix=file_suffix)

        self.mj_ho.close_view_and_render()
        return
