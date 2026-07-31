"""Finite-difference and safety tests for shared SLSQP helpers."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ada_grasp_ctrl.optimization import (
    diagnose_slsqp_result,
    friction_cone_jacobian,
    friction_cone_slack,
    select_control_solution,
)
from ada_grasp_ctrl.utils.grasp_controller import GraspController, _ControlProblemContext


class OptimizationHelperTest(unittest.TestCase):
    """Prove analytic gradients and rejected-solution fallback semantics."""

    def test_friction_cone_gradient_matches_central_finite_difference(self):
        """Match every analytic entry, including the normal derivative at fx=0."""
        mu = 0.3
        forces = np.array([[0.0, 0.4, -0.2], [2.0, -0.3, 0.7]])
        analytic = friction_cone_jacobian(forces, mu)
        epsilon = 1e-7
        numeric = np.zeros_like(analytic)
        flat = forces.reshape(-1)
        for column in range(flat.size):
            positive = flat.copy()
            negative = flat.copy()
            positive[column] += epsilon
            negative[column] -= epsilon
            numeric[:, column] = (
                friction_cone_slack(positive.reshape(-1, 3), mu) - friction_cone_slack(negative.reshape(-1, 3), mu)
            ) / (2 * epsilon)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-8)
        self.assertEqual(analytic[0, 0], mu)

    def test_tangential_origin_uses_zero_subgradient_and_constant_normal_gradient(self):
        """Use the symmetric subgradient at the cone tip without losing d/d fx."""
        gradient = friction_cone_jacobian(np.zeros((1, 3)), 0.6)
        np.testing.assert_array_equal(gradient, np.array([[0.6, 0.0, 0.0]]))

    @staticmethod
    def _result(x, *, success=True, fun=1.0, status=0):
        """Build a minimal SciPy-like result for validator tests.

        Args:
            x: Candidate variables.
            success: Solver convergence flag.
            fun: Objective value.
            status: Solver status code.

        Returns:
            Namespace exposing expected SciPy result attributes.
        """
        return SimpleNamespace(
            x=np.asarray(x),
            success=success,
            fun=fun,
            status=status,
            message="test",
            nit=3,
        )

    def test_solver_diagnostics_cover_success_and_rejection_modes(self):
        """Reject nonconvergence, nonfinite values, constraint, and bound violations."""
        constraints = [
            {"type": "eq", "fun": lambda x: np.array([x[0] - 1.0])},
            {"type": "ineq", "fun": lambda x: np.array([x[1]])},
        ]
        bounds = [(0.0, 2.0), (0.0, 1.0)]
        accepted = diagnose_slsqp_result(self._result([1.0, 0.5]), constraints, bounds)
        self.assertTrue(accepted["accepted"])

        nonconverged = diagnose_slsqp_result(self._result([1.0, 0.5], success=False, status=9), constraints, bounds)
        self.assertFalse(nonconverged["accepted"])
        nonfinite = diagnose_slsqp_result(self._result([np.nan, 0.5], fun=np.nan), constraints, bounds)
        self.assertFalse(nonfinite["finite"])
        violated_constraint = diagnose_slsqp_result(self._result([1.0, -2e-5]), constraints, bounds)
        self.assertFalse(violated_constraint["accepted"])
        violated_bound = diagnose_slsqp_result(self._result([1.0, 1.0 + 2e-8]), constraints, bounds)
        self.assertFalse(violated_bound["accepted"])

    def test_rejected_control_solution_holds_qpos_and_zeros_delta_history(self):
        """Never apply rejected qpos/contact-force candidates."""
        current_qpos = np.array([0.2, -0.4])
        measured_force = np.array([3.0, 0.1, -0.2])
        candidate = np.array([0.5, 0.6, 99.0, 98.0, 97.0])
        qpos, delta, force = select_control_solution(
            current_qpos,
            measured_force,
            candidate,
            2,
            {"accepted": False},
        )
        np.testing.assert_array_equal(qpos, current_qpos)
        np.testing.assert_array_equal(delta, np.zeros(2))
        np.testing.assert_array_equal(force, measured_force)


class SharedControlProblemTest(unittest.TestCase):
    """Cover the shared coordinated/equal-contact optimization path."""

    @staticmethod
    def _controller() -> GraspController:
        """Build a minimal controller without loading robot assets.

        Returns:
            Controller exposing all configuration flags used by the builder.
        """
        controller = object.__new__(GraspController)
        controller.mu = 0.3
        controller.tan_motion_pen_weight = 2.0
        controller.stage2_incontact_force_only = False
        controller.stage2_penalize_contact_qda = False
        controller.stage2_penalize_tan_motion = False
        controller.stage2_ctrl_tan_force = True
        controller.stage2_tan_force_constraint = False
        controller.stage2_equal_joint_force_cost = False
        controller.stage2_increase_force = False
        return controller

    @staticmethod
    def _context() -> _ControlProblemContext:
        """Build a two-DOF, one-contact linearized problem fixture.

        Returns:
            Deterministic shared optimization context.
        """
        return _ControlProblemContext(
            num_arm_dof=1,
            num_hand_dof=1,
            num_dof=2,
            num_contacts=1,
            doa_to_dof=np.eye(2),
            joint_limits=np.array([[-2.0, -2.0], [2.0, 2.0]]),
            max_delta_qpos=np.array([0.5, 0.5]),
            contact_forces=np.array([1.2, 0.1, -0.2]),
            contact_jacobian=np.array([[0.2, -0.1], [0.3, 0.4], [-0.2, 0.5]]),
            stiffness_jacobian=np.array([[0.7, -0.2], [0.1, 0.6], [-0.4, 0.3]]),
            hand_base_name="hand_base",
            target_hand_base_position=np.zeros(3),
            target_hand_base_orientation=None,
        )

    def _build_stage2_problem(self, policy, controller=None):
        """Build one stage-2 problem for callback and constraint tests.

        Args:
            policy: Coordinated or equal-contact policy identifier.
            controller: Optional configured controller fixture.

        Returns:
            Controller and its shared problem definition.
        """
        controller = controller or self._controller()
        problem = controller._build_control_problem(
            policy=policy,
            context=self._context(),
            stage=2,
            dt=0.2,
            current_qpos_a=np.array([0.1, -0.2]),
            target_qpos_f=np.array([0.0, 0.3]),
            last_delta_qpos_a=np.array([0.02, -0.01]),
            desired_sum_force=2.0,
            desired_forces=np.array([[1.5, 0.0, 0.0]]),
            contacts=[{}],
            grasp_matrix=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.1, 0.2, 0.0],
                    [0.0, -0.1, 0.3],
                    [0.2, 0.0, -0.2],
                ]
            ),
            use_arm_motion=True,
        )
        return controller, problem

    def test_public_wrappers_delegate_to_the_same_optimizer(self):
        """Keep both compatibility APIs on one shared optimization path."""
        controller = self._controller()
        expected = {"q_a": np.zeros(2), "dq_a": np.zeros(2), "cf": np.zeros(0)}
        common = dict(
            stage=1,
            dt=0.1,
            curr_q_a=np.zeros(2),
            curr_q_f=np.zeros(2),
            target_q_f=np.zeros(2),
            last_dq_a=np.zeros(2),
        )
        with patch.object(controller, "_optimize_control", return_value=expected) as optimize:
            self.assertIs(controller.ctrl_opt(desired_sum_force=1.0, **common), expected)
            coordinated = optimize.call_args.kwargs
            optimize.reset_mock()
            self.assertIs(
                controller.ctrl_opt_bs3(desired_sum_force=1.0, desired_forces=None, **common),
                expected,
            )
            equal_contact = optimize.call_args.kwargs

        self.assertEqual(coordinated["policy"], "coordinated")
        self.assertEqual(coordinated["solver_name"], "control")
        self.assertEqual(equal_contact["policy"], "equal_contact")
        self.assertEqual(equal_contact["solver_name"], "control_bs3")

    def test_policy_constraint_sets_retain_intentional_differences(self):
        """Keep coordinated safety constraints out of equal-contact control."""
        coordinated_controller = self._controller()
        coordinated_controller.stage2_increase_force = True
        _, coordinated = self._build_stage2_problem("coordinated", coordinated_controller)
        _, equal_contact = self._build_stage2_problem("equal_contact")

        self.assertEqual(
            coordinated.constraint_names,
            (
                "q_limits",
                "arm_doa",
                "contact_model",
                "friction_cone",
                "force_magnitude",
                "increase_normal_force",
            ),
        )
        self.assertEqual(equal_contact.constraint_names, ("q_limits", "arm_doa", "contact_model"))

    def test_optional_stage2_cost_gradient_matches_finite_difference(self):
        """Differentiate optional tangential costs under the full contact model."""
        controller = self._controller()
        controller.stage2_penalize_tan_motion = True
        controller.stage2_tan_force_constraint = True
        _, problem = self._build_stage2_problem("coordinated", controller)
        variables = np.array([0.04, -0.03, 1.1, 0.2, -0.1])
        epsilon = 1e-7
        numeric = np.zeros_like(variables)
        for index in range(variables.size):
            positive = variables.copy()
            negative = variables.copy()
            positive[index] += epsilon
            negative[index] -= epsilon
            numeric[index] = (problem.objective(positive) - problem.objective(negative)) / (2 * epsilon)
        problem.objective(variables)
        analytic = problem.jacobian(variables)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-5)

    def test_shared_solver_rejects_failed_candidate(self):
        """Keep the hold-last fallback after moving both APIs to one solver."""
        controller = self._controller()
        controller.r_data = {"solver_diagnostics": []}
        controller.solver_degraded = False
        failed = SimpleNamespace(
            x=np.array([0.4, 0.5, 9.0, 8.0, 7.0]),
            success=False,
            fun=1.0,
            status=9,
            message="iteration limit",
            nit=200,
        )
        with patch("ada_grasp_ctrl.utils.grasp_controller.minimize", return_value=failed):
            result = controller._solve_control_problem(
                solver_name="test",
                stage=2,
                objective=lambda x: float(x @ x),
                jacobian=lambda x: 2 * x,
                constraints=[],
                bounds=[(-1.0, 1.0)] * 2 + [(0.0, 10.0)] * 3,
                initial_variables=np.zeros(5),
                joint_limit_constraint=lambda x: np.ones(4),
                current_qpos_a=np.array([0.2, -0.4]),
                current_contact_forces=np.array([1.0, 0.1, -0.2]),
                num_dof=2,
                print_details=False,
            )
        np.testing.assert_array_equal(result["q_a"], np.array([0.2, -0.4]))
        np.testing.assert_array_equal(result["dq_a"], np.zeros(2))
        np.testing.assert_array_equal(result["cf"], np.array([1.0, 0.1, -0.2]))
        self.assertTrue(controller.solver_degraded)


if __name__ == "__main__":
    unittest.main()
