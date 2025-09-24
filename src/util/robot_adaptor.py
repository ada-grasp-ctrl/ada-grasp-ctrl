from abc import abstractmethod
from typing import List
import numpy as np
import torch

try:
    from .pin_helper import PinocchioHelper
except:
    from pin_helper import PinocchioHelper


class RobotAdaptor:
    def __init__(
        self,
        robot_model: PinocchioHelper,
        dof_names: List[str],
        doa_names: List[str],
        doa2dof_matrix: np.ndarray,
    ):
        self.robot_model = robot_model
        self.dof_names = dof_names
        self.doa_names = doa_names
        self.doa2dof_matrix = doa2dof_matrix

        self._dof_m2u_indices = [self.robot_model.dof_joint_names.index(name) for name in dof_names]
        self._dof_u2m_indices = [dof_names.index(name) for name in self.robot_model.dof_joint_names]

    @property
    def doa(self) -> int:
        return len(self.doa_names)

    def check_doa(self, q):
        assert len(q) == self.doa

    def _doa2dof(self, q_a):
        if isinstance(q_a, torch.Tensor):
            q_f = torch.tensor(self.doa2dof_matrix, dtype=q_a.dtype) @ q_a.reshape(-1, 1)
        else:
            q_f = self.doa2dof_matrix @ q_a.reshape(-1, 1)
        return q_f.reshape(-1)

    def _dof2doa(self, q_f):
        q_a = np.linalg.pinv(self.doa2dof_matrix) @ q_f.reshape(-1, 1)
        return q_a.reshape(-1)

    def _dof_u2m(self, q):
        """
        Joint order converter: user to robot_model.
        """
        return q[self._dof_u2m_indices]

    def _dof_m2u(self, q):
        return q[self._dof_m2u_indices]

    def compute_fk_a(self, q_a):
        q_f = self._doa2dof(q_a)
        self.compute_fk_f(q_f)

    def compute_fk_f(self, q_f):
        q_fm = self._dof_u2m(q_f)
        self.robot_model.compute_forward_kinematics(qpos=q_fm)

    def compute_jaco_a(self, q_a):
        q_fm = self._dof_u2m(self._doa2dof(q_a))
        self.robot_model.compute_jacobians(qpos=q_fm)

    def get_frame_pose(self, frame_name):
        return self.robot_model.get_frame_pose(frame_name=frame_name)

    def get_frame_space_jaco(self, frame_name):
        """
        Return the jacobian w.r.t. doa.
        """
        jaco_fm = self.robot_model.get_frame_space_jacobian(frame_name=frame_name)
        jaco_fu = jaco_fm[:, self._dof_m2u_indices]
        jaco_a = jaco_fu @ self.doa2dof_matrix
        return jaco_a

    def get_frame_jaco(self, frame_name, type="space"):
        """
        Return the jacobian w.r.t. doa.
        """
        jaco_fm = self.robot_model.get_frame_jacobian(frame_name=frame_name, type=type)
        jaco_fu = jaco_fm[:, self._dof_m2u_indices]
        jaco_a = jaco_fu @ self.doa2dof_matrix
        return jaco_a

    @property
    def joint_limits_f(self):
        jl_fm = self.robot_model.joint_limits.copy()
        jl_fu = np.zeros_like(jl_fm)
        for i in range(jl_fm.shape[0]):
            jl_fu[i, :] = self._dof_m2u(jl_fm[i, :])
        return jl_fu


if __name__ == "__main__":
    from robots.base import Robot, RobotFactory

    robot: Robot = RobotFactory.create_robot(robot_type="dummy_arm_shadow", prefix="rh_")
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

    doa = robot_adaptor.joint_limits_f
