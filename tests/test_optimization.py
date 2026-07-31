"""Finite-difference and safety tests for shared SLSQP helpers."""

from types import SimpleNamespace
import unittest

import numpy as np

from ada_grasp_ctrl.optimization import (
    diagnose_slsqp_result,
    friction_cone_jacobian,
    friction_cone_slack,
    select_control_solution,
)


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


if __name__ == "__main__":
    unittest.main()
