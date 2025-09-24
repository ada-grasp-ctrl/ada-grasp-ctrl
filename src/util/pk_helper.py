import torch
import pytorch_kinematics as pk
from typing import Optional, List, Union, Dict
import re
import os

from mr_utils.utils_torch import quaternion_xyzw2wxyz
from .robots.base import Robot


class PytorchKinematicsHelper:
    def __init__(self, robot: Robot, device):
        self.robot = robot
        self.device = device

        # base & world frame
        self.robot_joint_names = self.robot.dof_names
        bpos = torch.tensor(self.robot.base_pose[:3], device=device)
        bquat = quaternion_xyzw2wxyz(torch.tensor(self.robot.base_pose[3:7], device=device))
        self.base_tf_in_world = pk.Transform3d(pos=bpos, rot=bquat, device=device)

        # full chain
        file_path = self.robot.get_file_path(type="mjcf")
        mjcf_string = open(file_path).read()
        match = re.search(r'meshdir="([^"]+)"', mjcf_string)
        if match:  # convert relative path of meshdir to absolute path
            meshdir = match.group(1)
            if not os.path.isabs(meshdir):
                meshdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(file_path)), meshdir))
                mjcf_string = re.sub(r'meshdir="([^"]+)"', f'meshdir="{meshdir}"', mjcf_string)
        self.chain = pk.build_chain_from_mjcf(mjcf_string)
        self.chain = self.chain.to(device=device, dtype=torch.float64)  # dtype seems necessary

    def tf_world_to_base(self, tf_in_w: pk.Transform3d):
        return (
            self.base_tf_in_world.inverse()
            .compose(tf_in_w.to(self.device, dtype=self.base_tf_in_world.dtype))
            .to(self.device, dtype=tf_in_w.dtype)
        )

    def tf_base_to_world(self, tf_in_b: pk.Transform3d):
        return self.base_tf_in_world.compose(
            tf_in_b.to(self.device, dtype=self.base_tf_in_world.dtype),
        ).to(self.device, dtype=tf_in_b.dtype)

    def fk(self, q: torch.Tensor, link_name: Optional[str] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform forward kinematics for a given joint configuration.

        Args:
            q (torch.Tensor): Joint positions, shape (B, n_dof)
            link_name (Optional[str]): If specified, return FK of this link only.
                                    Otherwise, return FK of all links.

        Returns:
            - torch.Tensor: (B, 4, 4) transformation matrix if link_name is specified. Defind in world.
            - Dict[str, torch.Tensor]: Dictionary of (B, 4, 4) matrices for all links otherwise.
        """
        if q.shape[-1] != len(self.robot_joint_names):
            raise ValueError(f"Expected q to have last dimension {len(self.robot_joint_names)}, got {q.shape[-1]}")

        # Convert joint tensor to a dictionary format expected by the kinematic chain
        q_dict = {name: q[..., j] for j, name in enumerate(self.robot_joint_names)}

        # Compute forward kinematics
        transforms = self.chain.forward_kinematics(q_dict)

        if link_name is not None:
            if link_name not in transforms:
                raise KeyError(f"Link '{link_name}' not found in kinematic chain.")

            return self.tf_base_to_world(transforms[link_name]).get_matrix()  # shape (B, 4, 4)
        else:
            return {name: self.tf_base_to_world(tf).get_matrix() for name, tf in transforms.items()}

    def create_serial_chain(self, ee_name):
        self.serial_chain = pk.SerialChain(self.chain, ee_name).to(device=self.device, dtype=torch.float64)

    def create_ik_solver(
        self,
        num_retries=10,
        regularlization=1e-3,
        initial_noise_std: Optional[torch.tensor] = None,
    ):
        if not hasattr(self, "serial_chain"):
            raise NameError("Have not created the serial_chain.")

        self.n_ik_retries = num_retries
        self.initial_noise_std = initial_noise_std

        if initial_noise_std is not None:
            assert self.initial_noise_std.shape[0] == len(self.robot_joint_names), (
                "initial_noise_std must match n_robot_joints"
            )

        joint_lims = torch.tensor(self.serial_chain.get_joint_limits(), device=self.device)

        self.ik = pk.PseudoInverseIK(
            self.serial_chain,
            pos_tolerance=1e-3,
            rot_tolerance=1e-2,
            max_iterations=50,
            retry_configs=None,
            num_retries=num_retries,
            joint_limits=joint_lims.T,
            early_stopping_any_converged=True,
            early_stopping_no_improvement="all",
            debug=False,
            lr=0.2,
            regularlization=regularlization,
        )

    def solve_ik_batch(
        self,
        matrix: torch.Tensor,
        ref_configs: Optional[torch.Tensor] = None,
        use_ref_as_init: bool = True,
    ):
        """
        Solve a batch of inverse kinematics problems.

        Args:
            matrix (torch.Tensor): target poses of shape (B, 4, 4), defined in the world frame
            ref_configs (Optional[torch.Tensor]): reference full joint configs (B, DOF), optional

        Returns:
            Dict[str, torch.Tensor]: dictionary with keys:
                - "q": (B, DOF) full joint configuration
                - "success": (B, n_seeds) convergence flags
                - "err_pos": (B, n_seeds) position error
                - "err_rot": (B, n_seeds) rotation error
        """
        if not hasattr(self, "ik"):
            raise NameError("IK solver not initialized.")

        B = matrix.shape[0]
        serial_joint_names = self.serial_chain.get_joint_parameter_names()
        serial_indices = [self.robot_joint_names.index(name) for name in serial_joint_names]
        num_joints = len(self.robot_joint_names)
        joint_lims = torch.tensor(self.serial_chain.get_joint_limits(), device=self.device)

        # Prepare full_q and serial_ref_configs if needed
        noised_serial_ref_configs = None
        if ref_configs is not None:
            assert ref_configs.shape[0] == B, "ref_configs must match batch size"
            full_q = ref_configs.clone()
            serial_ref_configs = ref_configs[:, serial_indices]
            serial_ref_configs = serial_ref_configs.unsqueeze(1).repeat(1, self.n_ik_retries, 1)

            if use_ref_as_init:
                noise_std = (
                    self.initial_noise_std[serial_indices].view(1, 1, -1) if self.initial_noise_std is not None else 0
                )  # add some noises to the initial values
                noised_serial_ref_configs = serial_ref_configs + torch.randn_like(serial_ref_configs) * noise_std
                # clamp into the joint limits
                joint_mins = joint_lims[0, :].view(1, 1, -1)  # shape (1, 1, D)
                joint_maxs = joint_lims[1, :].view(1, 1, -1)  # shape (1, 1, D)
                noised_serial_ref_configs = torch.clamp(noised_serial_ref_configs, min=joint_mins, max=joint_maxs)
        else:
            full_q = torch.zeros((B, num_joints), dtype=matrix.dtype, device=matrix.device)
            # re-sample initial configs (default: uniform sampling)
            self.ik.sample_configs(num_configs=self.n_ik_retries)

        # Solve IK
        goal_tf_in_world = pk.Transform3d(matrix=matrix, device=self.device)
        goal_tf_in_base = self.tf_world_to_base(goal_tf_in_world)
        goal_tf_in_base = goal_tf_in_base.to(self.device, dtype=torch.float64)  # convert from float32 to float64
        sol = self.ik.solve(goal_tf_in_base, ref_configs=noised_serial_ref_configs)

        # check if the solutions are within joint limits
        within_lims = self.check_in_joint_limits(sol.solutions[sol.converged, :], joint_lims)
        if not within_lims.all():
            raise RuntimeError("Some converged IK solutions are out of joint limits.")

        # Select best IK solution
        if ref_configs is not None:
            # Use reference config to find the closest solution (among successful ones)
            ref_q = serial_ref_configs[:, 0:1, :]  # shape: (B, 1, D)
            diff = sol.solutions - ref_q
            dist = torch.norm(diff, dim=-1)  # (B, n_seeds)
            dist[~sol.converged] = 1e6
            best_idx = torch.argmin(dist, dim=1)  # (B,)
        else:
            # Use first converged solution (fallback if no ref config)
            best_idx = torch.argmax(sol.converged.to(torch.int), dim=1)  # (B,)

        batch_idx = torch.arange(B, device=matrix.device)
        serial_q = sol.solutions[batch_idx, best_idx, :].to(matrix.dtype)  # (B, serial_dof)
        full_q[:, serial_indices] = serial_q  # Fill in full joint vector

        # return the best IK solution among the retries for each item
        return {
            "q": full_q,
            "success": sol.converged[batch_idx, best_idx],
            "err_pos": sol.err_pos[batch_idx, best_idx],
            "err_rot": sol.err_rot[batch_idx, best_idx],
        }

    def check_in_joint_limits(self, qpos: torch.Tensor, joint_lims: torch.Tensor):
        """
        Args:
            qpos: shape (B, n_dof)
            joint_lims: shape (2, n_dof)
        """
        lower = joint_lims[0]  # shape: (D,)
        upper = joint_lims[1]  # shape: (D,)

        # Reshape limits to match solution shape
        while lower.dim() < qpos.dim():
            lower = lower.unsqueeze(0)
            upper = upper.unsqueeze(0)

        # Check if each solution is within the limits
        within_lower = qpos >= lower
        within_upper = qpos <= upper
        within_limits = within_lower & within_upper  # shape: (B, D) or (B, N, D)

        # All joints in each solution must be within limits
        is_valid = within_limits.all(dim=-1)  # shape: (B,) or (B, N)

        return is_valid
