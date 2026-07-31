"""Numerical contracts for rotations, grasp wrenches, and joint adaptation."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from ada_grasp_ctrl.utils.grasp_controller import GraspController
from ada_grasp_ctrl.utils.robot_adaptor import RobotAdaptor
from ada_grasp_ctrl.utils.rot_util import (
    np_get_delta_qpos,
    torch_matrix_to_quaternion,
    torch_quaternion_to_matrix,
)


class RotationContractTest(unittest.TestCase):
    """Keep quaternion conventions and shortest-angle semantics explicit."""

    def test_quaternion_matrix_round_trip_is_batched_and_sign_standardized(self):
        """Round-trip normalized WXYZ quaternions through rotation matrices."""
        quaternions = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
                [-0.5, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float64,
        )
        matrices = torch_quaternion_to_matrix(quaternions)
        recovered = torch_matrix_to_quaternion(matrices)

        self.assertEqual(matrices.shape, (3, 3, 3))
        self.assertTrue(torch.all(recovered[:, 0] >= 0))
        torch.testing.assert_close(
            torch_quaternion_to_matrix(recovered),
            matrices,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_zero_quaternion_is_rejected_before_nan_propagation(self):
        """Reject an undefined rotation instead of returning a NaN matrix."""
        with self.assertRaisesRegex(ValueError, "nonzero"):
            torch_quaternion_to_matrix(torch.zeros(4))

    def test_pose_delta_treats_opposite_quaternion_signs_as_same_rotation(self):
        """Use the shortest rotation because q and -q encode one orientation."""
        pose = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        sign_flipped = pose.copy()
        sign_flipped[3:7] *= -1

        delta_position, delta_angle = np_get_delta_qpos(pose, sign_flipped)

        self.assertEqual(delta_position, 0.0)
        self.assertAlmostEqual(delta_angle, 0.0)


class GraspWrenchContractTest(unittest.TestCase):
    """Verify contact-frame column layout and normalized wrench behavior."""

    def setUp(self):
        """Create a controller shell for methods that do not use instance state.

        Returns:
            None.
        """
        self.controller = GraspController.__new__(GraspController)

    def test_grasp_matrix_matches_independent_per_contact_construction(self):
        """Concatenate each contact's force and centroid-relative torque map."""
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        contacts = [
            {"contact_pos": np.array([0.01, 0.02, 0.03]), "contact_frame": np.eye(3)},
            {"contact_pos": np.array([-0.03, 0.01, 0.05]), "contact_frame": rotation},
        ]

        centroid = np.mean([contact["contact_pos"] for contact in contacts], axis=0)
        expected_blocks = []
        for contact in contacts:
            frame = contact["contact_frame"]
            radius_cm = (contact["contact_pos"] - centroid) * 100.0
            torque = np.column_stack([np.cross(radius_cm, frame[:, axis]) for axis in range(3)])
            expected_blocks.append(np.vstack([frame, torque]))
        expected = np.concatenate(expected_blocks, axis=1)

        actual = self.controller.compute_grasp_matrix(contacts)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_normalized_wrench_is_scale_invariant_and_finite_at_zero_force(self):
        """Normalize force and torque contributions without divide-by-zero output."""
        contacts = [
            {"contact_pos": np.array([0.02, 0.0, 0.0]), "contact_frame": np.eye(3)},
            {"contact_pos": np.array([-0.02, 0.0, 0.0]), "contact_frame": np.eye(3)},
        ]
        grasp_matrix = self.controller.compute_grasp_matrix(contacts)
        forces = np.array([[2.0, 0.4, -0.2], [1.5, -0.1, 0.3]])

        normalized = self.controller.compute_normalized_wrench(grasp_matrix, forces)
        scaled = self.controller.compute_normalized_wrench(grasp_matrix, 7.0 * forces)
        zero = self.controller.compute_normalized_wrench(grasp_matrix, np.zeros_like(forces))

        np.testing.assert_allclose(normalized, scaled, rtol=1e-8, atol=1e-8)
        np.testing.assert_array_equal(zero, np.zeros(6))
        self.assertTrue(np.all(np.isfinite(zero)))


class _FakeRobotModel:
    """Minimal Pinocchio-like model with a deliberately different joint order."""

    dof_joint_names = ["joint_c", "joint_a", "joint_d", "joint_b"]
    joint_limits = np.array(
        [
            [-3.0, -1.0, -4.0, -2.0],
            [3.0, 1.0, 4.0, 2.0],
        ]
    )

    def __init__(self):
        """Initialize captured calls and deterministic Jacobians.

        Returns:
            None.
        """
        self.forward_qpos = None
        self.jacobian_qpos = None
        self.space_jacobian = np.arange(24.0).reshape(6, 4)
        self.body_jacobian = self.space_jacobian + 100.0

    def compute_forward_kinematics(self, qpos):
        """Capture model-ordered forward-kinematics input.

        Args:
            qpos: Model-ordered joint vector.

        Returns:
            None.
        """
        self.forward_qpos = np.asarray(qpos).copy()

    def compute_jacobians(self, qpos):
        """Capture model-ordered Jacobian input.

        Args:
            qpos: Model-ordered joint vector.

        Returns:
            None.
        """
        self.jacobian_qpos = np.asarray(qpos).copy()

    def get_frame_space_jacobian(self, frame_name):
        """Return a deterministic model-ordered space Jacobian.

        Args:
            frame_name: Ignored test frame identifier.

        Returns:
            Six-by-four model-ordered Jacobian.
        """
        return self.space_jacobian.copy()

    def get_frame_jacobian(self, frame_name, type="space"):
        """Return a deterministic model-ordered general Jacobian.

        Args:
            frame_name: Ignored test frame identifier.
            type: Ignored reference-frame selector.

        Returns:
            Six-by-four model-ordered Jacobian.
        """
        return self.body_jacobian.copy()

    def get_frame_pose(self, frame_name):
        """Return an identity pose for delegation coverage.

        Args:
            frame_name: Ignored test frame identifier.

        Returns:
            Four-by-four identity transform.
        """
        return np.eye(4)


class RobotAdaptorContractTest(unittest.TestCase):
    """Lock user/model order conversion and actuation projection semantics."""

    def setUp(self):
        """Create a nontrivial two-actuator, four-joint adaptor.

        Returns:
            None.
        """
        self.model = _FakeRobotModel()
        self.user_names = ["joint_a", "joint_b", "joint_c", "joint_d"]
        self.matrix = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, -1.0],
            ]
        )
        self.adaptor = RobotAdaptor(
            robot_model=self.model,
            dof_names=self.user_names,
            doa_names=["actuator_0", "actuator_1"],
            doa2dof_matrix=self.matrix,
        )

    def test_actuation_projection_and_model_order_are_exact(self):
        """Project DOA to user DOF and reorder before Pinocchio calls."""
        actuator_qpos = np.array([2.0, -1.0])
        expected_user = self.matrix @ actuator_qpos
        expected_model = expected_user[[2, 0, 3, 1]]

        self.adaptor.compute_fk_a(actuator_qpos)
        self.adaptor.compute_jaco_a(actuator_qpos)

        np.testing.assert_array_equal(self.model.forward_qpos, expected_model)
        np.testing.assert_array_equal(self.model.jacobian_qpos, expected_model)
        np.testing.assert_allclose(self.adaptor._dof2doa(expected_user), actuator_qpos, atol=1e-12)

    def test_jacobians_and_limits_return_user_joint_order(self):
        """Reorder model columns before applying the DOA projection matrix."""
        model_to_user = [1, 3, 0, 2]
        expected_space = self.model.space_jacobian[:, model_to_user] @ self.matrix
        expected_body = self.model.body_jacobian[:, model_to_user] @ self.matrix

        np.testing.assert_array_equal(self.adaptor.get_frame_space_jaco("tip"), expected_space)
        np.testing.assert_array_equal(self.adaptor.get_frame_jaco("tip", type="body"), expected_body)
        np.testing.assert_array_equal(self.adaptor.joint_limits_f, self.model.joint_limits[:, model_to_user])

    def test_constructor_rejects_ambiguous_names_and_matrix_shapes(self):
        """Fail early on duplicate joints or a malformed actuation map."""
        with self.assertRaisesRegex(ValueError, "unique"):
            RobotAdaptor(self.model, ["joint_a"] * 4, ["a", "b"], self.matrix)
        with self.assertRaisesRegex(ValueError, "shape"):
            RobotAdaptor(self.model, self.user_names, ["a", "b"], np.eye(4))

    def test_torch_projection_preserves_dtype_and_device(self):
        """Construct the projection tensor beside its input tensor."""
        actuator_qpos = torch.tensor([2.0, -1.0], dtype=torch.float64)

        projected = self.adaptor._doa2dof(actuator_qpos)

        self.assertEqual(projected.dtype, actuator_qpos.dtype)
        self.assertEqual(projected.device, actuator_qpos.device)
        torch.testing.assert_close(projected, torch.tensor([2.0, -1.0, 1.0, 5.0], dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
