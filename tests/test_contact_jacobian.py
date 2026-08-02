"""Finite-difference contracts for contact-point Jacobian frame transport."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.utils.grasp_controller import GraspController
from ada_grasp_ctrl.utils.pin_helper import PinocchioHelper
from ada_grasp_ctrl.utils.robot_adaptor import RobotAdaptor
from ada_grasp_ctrl.utils.robots import RobotFactory


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ContactJacobianContractTest(unittest.TestCase):
    """Keep actual and desired contact Jacobians in one current contact frame."""

    ROBOT_CASES = (
        ("dummy_arm_shadow", "rh_", "rh_ffdistal"),
        ("dummy_arm_allegro", "", "ff_distal"),
        ("dummy_arm_leap_tac3d", "rh_", "rh_fingertip_base"),
    )

    @staticmethod
    def _controller(robot_type: str, prefix: str) -> tuple[GraspController, RobotAdaptor]:
        """Construct one maintained arm-hand controller and Pinocchio adaptor.

        Args:
            robot_type: Registered dummy-arm hand identifier.
            prefix: Joint and body prefix used by the maintained MJCF.

        Returns:
            Configured controller and its robot adaptor.
        """
        robot = RobotFactory.create_robot(robot_type=robot_type, prefix=prefix)
        robot_model = PinocchioHelper(robot_file_path=robot.get_file_path("mjcf"), robot_file_type="mjcf")
        adaptor = RobotAdaptor(
            robot_model=robot_model,
            dof_names=robot.dof_names,
            doa_names=robot.doa_names,
            doa2dof_matrix=robot.doa2dof_matrix,
        )
        task_defaults = OmegaConf.load(
            PROJECT_ROOT / "src" / "ada_grasp_ctrl" / "config" / "task" / "control_eval.yaml"
        )
        task_defaults.control.final_sum_force = 10.0
        return GraspController(task_defaults.control, robot, adaptor), adaptor

    @staticmethod
    def _test_configurations(adaptor: RobotAdaptor) -> tuple[np.ndarray, np.ndarray]:
        """Return deterministic actual and desired configurations within limits.

        Args:
            adaptor: Robot adaptor providing ordered joint limits.

        Returns:
            Actual and desired actuator vectors.
        """
        limits = adaptor.joint_limits_f
        lower, upper = limits[0], limits[1]
        actual = np.clip(np.zeros(adaptor.doa), lower + 0.25 * (upper - lower), upper - 0.25 * (upper - lower))
        desired = actual.copy()
        desired[:6] += np.array([0.01, -0.015, 0.02, 0.20, -0.16, 0.12])
        hand_delta = np.linspace(0.04, 0.10, adaptor.doa - 6)
        desired[6:] += hand_delta
        desired = np.clip(desired, lower + 1e-3, upper - 1e-3)
        return actual, desired

    @staticmethod
    def _point_position(adaptor: RobotAdaptor, qpos: np.ndarray, body_name: str, point_local: np.ndarray) -> np.ndarray:
        """Evaluate one body-fixed material point in the world frame.

        Args:
            adaptor: Robot adaptor used for forward kinematics.
            qpos: Actuator configuration.
            body_name: Contact body frame name.
            point_local: Contact point in body coordinates.

        Returns:
            Three-dimensional world point.
        """
        adaptor.compute_fk_a(qpos)
        pose = adaptor.get_frame_pose(frame_name=body_name)
        return pose[:3, :3] @ point_local + pose[:3, 3]

    def _finite_difference_jacobian(
        self,
        adaptor: RobotAdaptor,
        qpos: np.ndarray,
        body_name: str,
        point_local: np.ndarray,
        current_contact_rotation: np.ndarray,
    ) -> np.ndarray:
        """Differentiate a material contact point in the current contact frame.

        Args:
            adaptor: Robot adaptor used for forward kinematics.
            qpos: Linearization configuration.
            body_name: Contact body frame name.
            point_local: Contact point in body coordinates.
            current_contact_rotation: Contact-to-world rotation at the actual pose.

        Returns:
            Contact-frame translational Jacobian.
        """
        epsilon = 1e-7
        jacobian = np.zeros((3, qpos.size))
        for column in range(qpos.size):
            positive = qpos.copy()
            negative = qpos.copy()
            positive[column] += epsilon
            negative[column] -= epsilon
            world_delta = self._point_position(adaptor, positive, body_name, point_local) - self._point_position(
                adaptor,
                negative,
                body_name,
                point_local,
            )
            jacobian[:, column] = current_contact_rotation.T @ world_delta / (2.0 * epsilon)
        return jacobian

    def test_actual_and_desired_jacobians_match_contact_frame_finite_differences(self):
        """Transport J(qd) through the body frames before the contact rotation."""
        point_local = np.array([0.012, -0.007, 0.018])
        contact_in_body = Rotation.from_rotvec(np.array([0.4, -0.3, 0.2])).as_matrix()

        for robot_type, prefix, body_name in self.ROBOT_CASES:
            with self.subTest(robot_type=robot_type):
                controller, adaptor = self._controller(robot_type, prefix)
                actual, desired = self._test_configurations(adaptor)
                adaptor.compute_fk_a(actual)
                current_body_rotation = adaptor.get_frame_pose(frame_name=body_name)[:3, :3]
                current_contact_rotation = current_body_rotation @ contact_in_body
                contacts = [
                    {
                        "body1_name": body_name,
                        "contact_pos_local": point_local.reshape(3, 1),
                        "contact_frame_local": contact_in_body,
                    }
                ]

                updated_contacts, _ = controller.Ks(q_a=desired, q_f=actual, contacts=contacts)
                actual_numeric = self._finite_difference_jacobian(
                    adaptor,
                    actual,
                    body_name,
                    point_local,
                    current_contact_rotation,
                )
                desired_numeric = self._finite_difference_jacobian(
                    adaptor,
                    desired,
                    body_name,
                    point_local,
                    current_contact_rotation,
                )

                np.testing.assert_allclose(updated_contacts[0]["jaco_f"], actual_numeric, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(updated_contacts[0]["jaco_a"], desired_numeric, rtol=1e-6, atol=1e-7)

    def test_equal_actual_and_desired_poses_produce_identical_contact_jacobians(self):
        """Retain the identical-frame behavior when qd equals q."""
        point_local = np.array([0.01, 0.003, -0.006])
        contact_in_body = Rotation.from_rotvec(np.array([-0.2, 0.35, 0.1])).as_matrix()

        for robot_type, prefix, body_name in self.ROBOT_CASES:
            with self.subTest(robot_type=robot_type):
                controller, adaptor = self._controller(robot_type, prefix)
                actual, _ = self._test_configurations(adaptor)
                contacts = [
                    {
                        "body1_name": body_name,
                        "contact_pos_local": point_local.reshape(3, 1),
                        "contact_frame_local": contact_in_body,
                    }
                ]

                updated_contacts, _ = controller.Ks(q_a=actual, q_f=actual, contacts=contacts)

                np.testing.assert_allclose(updated_contacts[0]["jaco_a"], updated_contacts[0]["jaco_f"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
