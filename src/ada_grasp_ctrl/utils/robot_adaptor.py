from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .pin_helper import PinocchioHelper


class RobotAdaptor:
    def __init__(
        self,
        robot_model: PinocchioHelper,
        dof_names: Sequence[str],
        doa_names: Sequence[str],
        doa2dof_matrix: np.ndarray,
    ):
        """Map user joint/actuator order to a Pinocchio model.

        Args:
            robot_model: Prepared Pinocchio helper.
            dof_names: User-facing degree-of-freedom names and order.
            doa_names: User-facing degree-of-actuation names and order.
            doa2dof_matrix: Linear map shaped ``(num_dof, num_doa)``.

        Returns:
            None.

        Raises:
            ValueError: If names or the actuation map are ambiguous or incompatible.
        """
        self.robot_model = robot_model
        self.dof_names = self._validate_names(dof_names, "dof_names")
        self.doa_names = self._validate_names(doa_names, "doa_names")
        model_names = self._validate_names(self.robot_model.dof_joint_names, "robot_model.dof_joint_names")
        if set(model_names) != set(self.dof_names):
            missing = sorted(set(model_names) - set(self.dof_names))
            unexpected = sorted(set(self.dof_names) - set(model_names))
            raise ValueError(
                f"dof_names must contain exactly the model joints; missing={missing}, unexpected={unexpected}."
            )

        matrix = np.asarray(doa2dof_matrix, dtype=float)
        expected_shape = (len(self.dof_names), len(self.doa_names))
        if matrix.shape != expected_shape:
            raise ValueError(f"doa2dof_matrix must have shape {expected_shape}; got {matrix.shape}.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("doa2dof_matrix must contain only finite values.")
        self.doa2dof_matrix = matrix.copy()

        self._dof_m2u_indices = [model_names.index(name) for name in self.dof_names]
        self._dof_u2m_indices = [self.dof_names.index(name) for name in model_names]

    @staticmethod
    def _validate_names(names: Sequence[str], field: str) -> list[str]:
        """Validate a unique, nonempty sequence of names.

        Args:
            names: Candidate ordered names.
            field: Field label used in errors.

        Returns:
            Validated names as a list.

        Raises:
            ValueError: If names are empty, duplicated, or not strings.
        """
        if isinstance(names, (str, bytes)):
            raise ValueError(f"{field} must be a sequence of unique nonempty strings.")
        values = list(names)
        if not values or any(not isinstance(name, str) or not name for name in values):
            raise ValueError(f"{field} must be a sequence of unique nonempty strings.")
        if len(set(values)) != len(values):
            raise ValueError(f"{field} must contain unique names.")
        return values

    @property
    def doa(self) -> int:
        """Return the number of independent actuators."""
        return len(self.doa_names)

    def check_doa(self, q):
        """Validate one actuator vector.

        Args:
            q: NumPy or Torch actuator vector.

        Returns:
            None.

        Raises:
            ValueError: If the vector is not one-dimensional with exact DOA length.
        """
        shape = tuple(q.shape) if hasattr(q, "shape") else np.asarray(q).shape
        if shape != (self.doa,):
            raise ValueError(f"Expected a DOA vector with shape ({self.doa},); got {shape}.")

    def _check_dof(self, q) -> None:
        """Validate one full-joint vector.

        Args:
            q: NumPy or Torch degree-of-freedom vector.

        Returns:
            None.

        Raises:
            ValueError: If the vector is not one-dimensional with exact DOF length.
        """
        shape = tuple(q.shape) if hasattr(q, "shape") else np.asarray(q).shape
        expected = (len(self.dof_names),)
        if shape != expected:
            raise ValueError(f"Expected a DOF vector with shape {expected}; got {shape}.")

    def _doa2dof(self, q_a):
        """Project an actuator vector into user-ordered joint coordinates.

        Args:
            q_a: NumPy or Torch actuator vector.

        Returns:
            Joint vector using the same array family, dtype, and device.
        """
        self.check_doa(q_a)
        if isinstance(q_a, torch.Tensor):
            matrix = torch.as_tensor(self.doa2dof_matrix, dtype=q_a.dtype, device=q_a.device)
            q_f = matrix @ q_a.reshape(-1, 1)
        else:
            q_f = self.doa2dof_matrix @ np.asarray(q_a).reshape(-1, 1)
        return q_f.reshape(-1)

    def _dof2doa(self, q_f):
        """Compute the least-squares actuator vector for user-ordered joints.

        Args:
            q_f: NumPy or Torch user-ordered joint vector.

        Returns:
            Minimum-norm actuator vector using the input array family.
        """
        self._check_dof(q_f)
        if isinstance(q_f, torch.Tensor):
            matrix = torch.as_tensor(self.doa2dof_matrix, dtype=q_f.dtype, device=q_f.device)
            q_a = torch.linalg.pinv(matrix) @ q_f.reshape(-1, 1)
        else:
            q_a = np.linalg.pinv(self.doa2dof_matrix) @ np.asarray(q_f).reshape(-1, 1)
        return q_a.reshape(-1)

    def _dof_u2m(self, q):
        """
        Joint order converter: user to robot_model.
        """
        self._check_dof(q)
        return q[self._dof_u2m_indices]

    def _dof_m2u(self, q):
        self._check_dof(q)
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
        if jl_fm.shape != (2, len(self.dof_names)):
            raise ValueError(f"Expected model joint limits with shape {(2, len(self.dof_names))}; got {jl_fm.shape}.")
        return jl_fm[:, self._dof_m2u_indices]


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
