import os
from copy import deepcopy
import logging

import numpy as np
import imageio

from ada_grasp_ctrl.batch import SampleStatus
from ada_grasp_ctrl.errors import ControlSolveEpisodeAbort
from ada_grasp_ctrl.paths import map_path, resolve_from_root
from ada_grasp_ctrl.schema import load_npy_record, validate_grasp_record
from ada_grasp_ctrl.utils.rot_util import (
    np_get_delta_qpos,
)
from ada_grasp_ctrl.utils.hand_util import MjHO
from ada_grasp_ctrl.utils.file_util import load_json
from ada_grasp_ctrl.utils.grasp_controller import GraspController
from ada_grasp_ctrl.utils.robots.base import RobotFactory


def control_output_path(input_npy_path, configs, file_suffix, method_name=None):
    """Build the historical control path using explicit configured roots.

    Args:
        input_npy_path: Source grasp record.
        configs: Composed application configuration.
        file_suffix: Offset/position suffix including its leading underscore.
        method_name: Optional method override; defaults to the configured method.

    Returns:
        Absolute destination path.
    """
    input_root = configs[configs.task.input_data]
    mapped = map_path(input_npy_path, input_root, configs.control_dir)
    method = method_name or configs.task.method
    if method == "ours":
        method = f"{method}_{configs.task.control.ablation_name}"
    return mapped.parent / method / f"{mapped.stem}{file_suffix}.npy"


class BaseEval:
    """Common MuJoCo episode lifecycle shared by all control policies."""

    def __init__(self, input_npy_path, configs, debug_viewer_session=None):
        """Create an evaluator and validate its grasp/object inputs.

        Args:
            input_npy_path: Version-zero or version-one grasp record.
            configs: Composed application configuration.
            debug_viewer_session: Optional task-owned persistent viewer.

        Returns:
            None.
        """
        self.input_npy_path = input_npy_path
        self.configs = configs
        robot_name = self.configs.hand_name
        robot_prefix = "rh_" if "allegro" not in robot_name else ""
        self.robot = RobotFactory.create_robot(robot_type=robot_name, prefix=robot_prefix)
        self.grasp_data = validate_grasp_record(
            load_npy_record(input_npy_path),
            input_npy_path,
            expected_joint_dim=self.robot.n_dof,
            expected_joint_names=self.robot.dof_names,
            require_joint_names=True,
        )
        self.grasp_data["obj_path"] = str(resolve_from_root(self.grasp_data["obj_path"], root_kind="data"))
        validate_grasp_record(
            self.grasp_data,
            input_npy_path,
            expected_joint_dim=self.robot.n_dof,
            expected_joint_names=self.robot.dof_names,
            require_joint_names=True,
            require_assets=True,
        )

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
            debug_viewer_session=debug_viewer_session,
        )

        if self.configs.task.debug_viewer or self.configs.task.debug_render:
            with open("debug.xml", "w") as f:
                f.write(self.mj_ho.spec.to_xml())

        return

    def _initialize_controller(self, method_name):
        """Create episode-local controller state from cached read-only models.

        Args:
            method_name: Public control method identifier.

        Returns:
            None.
        """
        self.method_name = method_name
        # Import Pinocchio only when an episode starts. The CLI runtime loads
        # Torch first, which selects the Conda C++ runtime required by Pinocchio.
        from ada_grasp_ctrl.utils.pin_helper import PinocchioHelper
        from ada_grasp_ctrl.utils.robot_adaptor import RobotAdaptor

        self.robot_model = PinocchioHelper(
            robot_file_path=self.robot.get_file_path("mjcf"),
            robot_file_type="mjcf",
        )
        self.robot_adaptor = RobotAdaptor(
            robot_model=self.robot_model,
            dof_names=self.robot.dof_names,
            doa_names=self.robot.doa_names,
            doa2dof_matrix=self.robot.doa2dof_matrix,
        )
        self.grasp_ctrl = GraspController(
            configs=self.configs.task.control,
            robot=self.robot,
            robot_adaptor=self.robot_adaptor,
        )
        self.dof_data2user_indices = [self.grasp_data["joint_names"].index(name) for name in self.robot.dof_names]

    def _dof_data2user(self, qpos):
        """Reorder stored qpos into the robot model's public joint order.

        Args:
            qpos: Stored qpos array ending in the data joint dimension.

        Returns:
            Reordered qpos copy.
        """
        return qpos[..., self.dof_data2user_indices].copy()

    def _simulate_under_extforce_details(self, pre_obj_qpos):
        """Execute method-specific approach and squeeze control.

        Args:
            pre_obj_qpos: Prepared grasp trajectory arguments supplied by subclasses.

        Returns:
            None.
        """
        raise NotImplementedError

    def _eval_simulate_under_extforce(self, obj_pose, file_suffix):
        """Execute, lift, classify, and save one perturbed episode.

        Args:
            obj_pose: Initial object WXYZ pose.
            file_suffix: Output suffix identifying the perturbation.

        Returns:
            Pair of output path and structured sample status.
        """
        eval_config = self.configs.task.simulation_metrics

        # initialize grasp controller
        self.mj_ho.reset()
        self._initialize()
        pregrasp_qpos = self.grasp_data["pregrasp_qpos"].copy()
        grasp_qpos = self.grasp_data["grasp_qpos"].copy()
        squeeze_qpos = self.grasp_data["squeeze_qpos"].copy()
        n_arm_dof = self.robot.arm.n_dof

        # set the arm qpos of pregrasp_qpos to be the same as grasp_qpos
        if self.configs.task.arm_pregrasp_is_grasp:
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
        contact_force = (
            max(np.linalg.norm(contact["contact_force"]) for contact in curr_ho_contact) if curr_ho_contact else 0
        )
        invalid_initialization = (
            ho_dist < -eval_config.max_pene or hh_dist < -eval_config.max_pene or contact_force > eval_config.max_force
        )
        episode_aborted = False
        if invalid_initialization:
            if self.configs.task.debug_viewer or self.configs.task.debug_render:
                print(
                    f"Severe penetration larger than {eval_config.max_pene}. "
                    f"ho_dist: {ho_dist}, hh_dist: {hh_dist}, contact_force: {contact_force}"
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
            try:
                self._simulate_under_extforce_details(pregrasp_qpos, grasp_qpos, squeeze_qpos)
            except ControlSolveEpisodeAbort as error:
                episode_aborted = True
                logging.warning("Aborted control episode %s: %s", file_suffix, error)

            if not episode_aborted:
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
                        curr_qpos_a,
                        q_a,
                        names=self.robot.doa_names,
                        step_outer=self.sim_step_per_action // 5,
                        step_inner=5,
                    )

                # terminal state
                obj_pose = self.mj_ho.get_obj_pose()
                self.grasp_ctrl.r_data["obj_pose"].append(obj_pose)

        if invalid_initialization:
            status = SampleStatus.INVALID_INITIALIZATION
        elif self.grasp_ctrl.solver_degraded:
            status = SampleStatus.SOLVER_DEGRADED
        else:
            status = SampleStatus.COMPLETED
        save_path = control_output_path(
            self.input_npy_path,
            self.configs,
            file_suffix,
            method_name=self.method_name,
        )
        self.grasp_ctrl.save_recorded_data(path=str(save_path), episode_status=status.value)

        # -------------------------------------------------------------------------------
        # Compare the resulted object pose
        latter_obj_qpos = self.mj_ho.get_obj_pose()
        delta_pos, delta_angle = np_get_delta_qpos(pre_obj_qpos, latter_obj_qpos)

        if self.configs.task.debug_viewer or self.configs.task.debug_render:
            print(delta_pos, delta_angle)
            # Save rendered video
            if self.configs.task.debug_render and len(self.mj_ho.debug_images) > 10:
                input_root = self.configs[self.configs.task.input_data]
                debug_source = map_path(self.input_npy_path, input_root, self.configs.task.debug_dir)
                debug_path = debug_source.with_name(f"{debug_source.stem}_{self.method_name}{file_suffix}.mp4")
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                # Extract every other frame from a 50 Hz image sequence to convert it into 25 Hz.
                frames = self.mj_ho.debug_images[::2]
                with imageio.get_writer(str(debug_path), fps=25, codec="libx264", quality=8) as writer:
                    for img in frames:
                        writer.append_data(img)
                print("Save MP4 (25Hz) to ", debug_path)
        return str(save_path.resolve(strict=False)), status

    def run(self):
        """Run every configured object-offset episode for one grasp.

        Returns:
            Output paths and per-output statuses.
        """
        # with object position uncertainty (8 planar directions)
        directions = np.array(
            [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]],
            dtype=float,
        )
        directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)

        output_paths = []
        statuses = []
        try:
            for obj_offset_dist in self.configs.task.offsets:
                # A zero offset has one deterministic pose; nonzero offsets use
                # the eight normalized planar perturbation directions.
                if obj_offset_dist == 0:
                    shifted_obj_poses = np.tile(self.grasp_data["obj_pose"].copy(), (1, 1))
                else:
                    shifted_obj_poses = np.tile(self.grasp_data["obj_pose"].copy(), (len(directions), 1))
                    shifted_obj_poses[:, 0:2] += obj_offset_dist * directions

                for index in range(shifted_obj_poses.shape[0]):
                    file_suffix = f"_dist_{str(int(100 * obj_offset_dist))}_pos_{index}"
                    output_path, status = self._eval_simulate_under_extforce(
                        obj_pose=shifted_obj_poses[index, :], file_suffix=file_suffix
                    )
                    output_paths.append(output_path)
                    statuses.append(status)
        finally:
            self.mj_ho.close_view_and_render()
        return output_paths, statuses
