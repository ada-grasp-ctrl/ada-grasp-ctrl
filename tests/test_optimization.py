"""Finite-difference and safety tests for shared SLSQP helpers."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from omegaconf import OmegaConf

from ada_grasp_ctrl.optimization import (
    SOLVER_FAILURE_POLICIES,
    diagnose_slsqp_result,
    friction_cone_jacobian,
    friction_cone_slack,
    select_control_solution,
    solve_linear_system,
)
from ada_grasp_ctrl.errors import ControlSolveEpisodeAbort
from ada_grasp_ctrl.utils.grasp_controller import (
    GraspController,
    GraspControllerParameters,
    _ControlProblemContext,
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

    def test_linear_system_uses_direct_solve_when_well_conditioned(self):
        """Return the exact multi-RHS solution without a degradation marker."""
        matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
        rhs = np.array([[1.0, 0.0], [0.0, 1.0]])
        solution, diagnostics = solve_linear_system(matrix, rhs)
        np.testing.assert_allclose(matrix @ solution, rhs, rtol=1e-12, atol=1e-12)
        self.assertTrue(diagnostics["accepted"])
        self.assertEqual(diagnostics["method"], "solve")

    def test_singular_linear_system_uses_diagnosed_least_squares_fallback(self):
        """Keep a finite minimum-norm result and expose numerical degradation."""
        matrix = np.array([[1.0, 1.0], [2.0, 2.0]])
        rhs = np.array([1.0, 2.0])
        solution, diagnostics = solve_linear_system(matrix, rhs)
        np.testing.assert_allclose(matrix @ solution, rhs, rtol=1e-12, atol=1e-12)
        self.assertFalse(diagnostics["accepted"])
        self.assertTrue(diagnostics["success"])
        self.assertEqual(diagnostics["method"], "lstsq")
        self.assertGreater(diagnostics["condition_number"], diagnostics["condition_limit"])

    def test_failed_linear_fallback_returns_finite_zero_solution(self):
        """Prevent nonfinite control data when both direct and fallback solves fail."""
        matrix = np.array([[1.0, 1.0], [2.0, 2.0]])
        rhs = np.array([1.0, 2.0])
        with patch(
            "ada_grasp_ctrl.optimization.np.linalg.lstsq",
            side_effect=np.linalg.LinAlgError("test failure"),
        ):
            solution, diagnostics = solve_linear_system(matrix, rhs)
        np.testing.assert_array_equal(solution, np.zeros(2))
        self.assertFalse(diagnostics["accepted"])
        self.assertEqual(diagnostics["method"], "zero")

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
        self.assertTrue(accepted["candidate_applicable"])

        nonconverged = diagnose_slsqp_result(self._result([1.0, 0.5], success=False, status=9), constraints, bounds)
        self.assertFalse(nonconverged["accepted"])
        self.assertTrue(nonconverged["candidate_applicable"])
        nonfinite = diagnose_slsqp_result(self._result([np.nan, 0.5], fun=np.nan), constraints, bounds)
        self.assertFalse(nonfinite["finite"])
        self.assertFalse(nonfinite["candidate_applicable"])
        violated_equality = diagnose_slsqp_result(self._result([1.0 + 2e-5, 0.5]), constraints, bounds)
        self.assertFalse(violated_equality["accepted"])
        self.assertTrue(violated_equality["candidate_applicable"])
        violated_inequality = diagnose_slsqp_result(self._result([1.0, -2e-5]), constraints, bounds)
        self.assertFalse(violated_inequality["accepted"])
        self.assertTrue(violated_inequality["candidate_applicable"])
        violated_bound = diagnose_slsqp_result(self._result([1.0, 1.0 + 2e-8]), constraints, bounds)
        self.assertFalse(violated_bound["accepted"])
        self.assertTrue(violated_bound["candidate_applicable"])
        violated_joint_limit = diagnose_slsqp_result(
            self._result([1.0, 0.5]),
            constraints,
            bounds,
            joint_limit_constraint=lambda x: np.array([-2e-8]),
        )
        self.assertFalse(violated_joint_limit["accepted"])
        self.assertTrue(violated_joint_limit["candidate_applicable"])
        rejected_candidates = (
            (violated_equality, np.array([1.0 + 2e-5, 0.5])),
            (violated_inequality, np.array([1.0, -2e-5])),
            (violated_bound, np.array([1.0, 1.0 + 2e-8])),
            (violated_joint_limit, np.array([1.0, 0.5])),
        )
        for diagnostics, candidate in rejected_candidates:
            decision = select_control_solution(
                np.zeros(2),
                np.zeros(0),
                candidate,
                2,
                diagnostics,
                failure_policy="apply_candidate",
            )
            self.assertEqual(decision.decision, "apply_candidate")
            self.assertFalse(diagnostics["accepted"])

    def test_control_solution_policy_matrix(self):
        """Apply, hold, or abort one finite rejected command as configured."""
        current_qpos = np.array([0.2, -0.4])
        measured_force = np.array([3.0, 0.1, -0.2])
        candidate = np.array([0.5, 0.6, 99.0, 98.0, 97.0])
        diagnostics = {"accepted": False, "candidate_applicable": True}

        applied = select_control_solution(
            current_qpos,
            measured_force,
            candidate,
            2,
            diagnostics,
            failure_policy="apply_candidate",
        )
        np.testing.assert_allclose(applied.qpos, np.array([0.7, 0.2]), rtol=0, atol=1e-15)
        np.testing.assert_array_equal(applied.delta_qpos, candidate[:2])
        np.testing.assert_array_equal(applied.contact_forces, candidate[2:])
        self.assertEqual(applied.decision, "apply_candidate")
        self.assertTrue(applied.action_applied)
        self.assertFalse(applied.episode_aborted)

        default_applied = select_control_solution(current_qpos, measured_force, candidate, 2, diagnostics)
        self.assertEqual(default_applied.decision, "apply_candidate")
        np.testing.assert_array_equal(default_applied.qpos, applied.qpos)

        held = select_control_solution(
            current_qpos,
            measured_force,
            candidate,
            2,
            diagnostics,
            failure_policy="hold_current",
        )
        np.testing.assert_array_equal(held.qpos, current_qpos)
        np.testing.assert_array_equal(held.delta_qpos, np.zeros(2))
        np.testing.assert_array_equal(held.contact_forces, measured_force)
        self.assertEqual(held.decision, "hold_current")
        self.assertTrue(held.action_applied)
        self.assertFalse(held.episode_aborted)

        aborted = select_control_solution(
            current_qpos,
            measured_force,
            candidate,
            2,
            diagnostics,
            failure_policy="fail_episode",
        )
        self.assertIsNone(aborted.qpos)
        self.assertEqual(aborted.decision, "abort_episode")
        self.assertFalse(aborted.action_applied)
        self.assertTrue(aborted.episode_aborted)

    def test_accepted_control_solution_is_policy_invariant(self):
        """Apply an accepted candidate identically under every failure policy."""
        decisions = [
            select_control_solution(
                np.array([0.2, -0.4]),
                np.array([3.0, 0.1, -0.2]),
                np.array([0.5, 0.6, 9.0, 8.0, 7.0]),
                2,
                {"accepted": True, "candidate_applicable": True},
                failure_policy=policy,
            )
            for policy in SOLVER_FAILURE_POLICIES
        ]
        for decision in decisions:
            np.testing.assert_array_equal(decision.qpos, decisions[0].qpos)
            np.testing.assert_array_equal(decision.delta_qpos, decisions[0].delta_qpos)
            np.testing.assert_array_equal(decision.contact_forces, decisions[0].contact_forces)
            self.assertEqual(decision.decision, "apply_accepted")
            self.assertFalse(decision.episode_aborted)

    def test_nonapplicable_candidates_never_reach_an_action(self):
        """Abort apply/fail policies and hold safely for malformed candidates."""
        malformed_candidates = (
            None,
            "not numeric",
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2, np.nan, 0.0, 0.0]),
            np.array([0.1, 0.2, np.inf, 0.0, 0.0]),
        )
        for candidate in malformed_candidates:
            with self.subTest(candidate=repr(candidate)):
                for policy in ("apply_candidate", "fail_episode"):
                    decision = select_control_solution(
                        np.array([0.2, -0.4]),
                        np.array([3.0, 0.1, -0.2]),
                        candidate,
                        2,
                        {"accepted": False, "candidate_applicable": False},
                        failure_policy=policy,
                    )
                    self.assertTrue(decision.episode_aborted)
                    self.assertFalse(decision.action_applied)
                    self.assertIsNone(decision.qpos)

                held = select_control_solution(
                    np.array([0.2, -0.4]),
                    np.array([3.0, 0.1, -0.2]),
                    candidate,
                    2,
                    {"accepted": False, "candidate_applicable": False},
                    failure_policy="hold_current",
                )
                np.testing.assert_array_equal(held.qpos, np.array([0.2, -0.4]))
                self.assertTrue(np.all(np.isfinite(held.qpos)))


class SharedControlProblemTest(unittest.TestCase):
    """Cover the shared coordinated/equal-contact optimization path."""

    @staticmethod
    def _controller() -> GraspController:
        """Build a minimal controller without loading robot assets.

        Returns:
            Controller exposing all configuration flags used by the builder.
        """
        controller = object.__new__(GraspController)
        controller.parameters = GraspControllerParameters.from_config(
            OmegaConf.create(
                {
                    "balance_thres": 0.2,
                    "friction_cone_mu": 0.3,
                    "final_sum_force": 15.0,
                    "kp_clip": {"min": 0.0, "max": 1e3},
                    "objective_weights": {
                        "hand_base_pose": [0.0, 0.0, 100.0, 10.0, 10.0, 10.0],
                        "hand_joint_position": 1.0,
                        "joint_velocity": 0.01,
                        "joint_acceleration": 0.001,
                        "coordinated_tangential_force": 0.1,
                        "equal_contact_force": 1.0,
                        "control_wrench": [1.0] * 6,
                        "equal_joint_force": 0.01,
                        "in_contact_joint_velocity_multiplier": 100.0,
                    },
                    "wrench_balance": {
                        "normal_force_sum": 1.0,
                        "wrench_weights": [1.0] * 6,
                        "normal_force_bounds": [0.0, 10.0],
                        "tangential_force_bounds": [-10.0, 10.0],
                        "ftol": 1e-6,
                        "maxiter": 200,
                    },
                    "command_solver": {
                        "normal_force_bounds": [0.0, 100.0],
                        "tangential_force_bounds": [-50.0, 50.0],
                        "ftol": 1e-6,
                        "maxiter": 200,
                    },
                }
            )
        )
        controller.mu = 0.3
        controller.tan_motion_pen_weight = 2.0
        controller.stage2_incontact_force_only = False
        controller.stage2_penalize_contact_qda = False
        controller.stage2_penalize_tan_motion = False
        controller.stage2_ctrl_tan_force = True
        controller.stage1_tan_force_constraint = False
        controller.stage2_tan_force_constraint = False
        controller.stage2_equal_joint_force_cost = False
        controller.stage2_increase_force = False
        controller.solver_failure_policy = "hold_current"
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

    def test_constructor_uses_hydra_parameters_without_robot_name_dispatch(self):
        """Read Hydra parameters and preserve the legacy shared contact-model switch."""
        project_root = Path(__file__).resolve().parents[1]
        task_defaults = OmegaConf.load(
            project_root / "src" / "ada_grasp_ctrl" / "config" / "task" / "control_eval.yaml"
        )
        composed = OmegaConf.create(
            {
                "hand": {"controller": {"final_sum_force": 12.0}},
                "task": OmegaConf.to_container(task_defaults, resolve=False),
            }
        )
        composed.task.control.kp_clip.min = 2.0
        composed.task.control.kp_clip.max = 5.0
        robot = SimpleNamespace(doa_kp=np.array([1.0, 10.0]))

        controller = GraspController(
            configs=composed.task.control,
            robot=robot,
            robot_adaptor=object(),
        )

        self.assertEqual(controller.final_sum_force, 12.0)
        self.assertEqual(controller.balance_thres, 0.2)
        self.assertEqual(controller.mu, 0.3)
        self.assertTrue(controller.stage1_tan_force_constraint)
        self.assertFalse(controller.stage2_tan_force_constraint)
        np.testing.assert_array_equal(controller.Kp, np.diag([2.0, 5.0]))
        np.testing.assert_array_equal(controller.Kp_inv, np.diag([0.5, 0.2]))

        del composed.task.control.stage1_tan_force_constraint
        composed.task.control.stage2_tan_force_constraint = True
        legacy_controller = GraspController(
            configs=composed.task.control,
            robot=robot,
            robot_adaptor=object(),
        )
        self.assertTrue(legacy_controller.stage1_tan_force_constraint)
        self.assertTrue(legacy_controller.stage2_tan_force_constraint)

    def test_stage_specific_tangential_contact_model_constraints_are_independent(self):
        """Toggle the full contact model independently in Stage 1 and Stage 2."""
        controller = self._controller()
        context = self._context()
        common = dict(
            policy="coordinated",
            context=context,
            dt=0.2,
            current_qpos_a=np.array([0.1, -0.2]),
            target_qpos_f=np.array([0.0, 0.3]),
            last_delta_qpos_a=np.array([0.02, -0.01]),
            desired_sum_force=2.0,
            desired_forces=None,
            contacts=[{}],
            grasp_matrix=np.zeros((6, 3)),
            use_arm_motion=True,
        )

        def contact_model_dimension(stage: int) -> int:
            """Return the equality dimension for one stage's contact model.

            Args:
                stage: Controller stage whose contact-model constraint is inspected.
            """
            problem = controller._build_control_problem(stage=stage, **common)
            constraint_index = problem.constraint_names.index("contact_model")
            residual = problem.constraints[constraint_index]["fun"](problem.initial_variables)
            return np.asarray(residual).size

        self.assertEqual(contact_model_dimension(stage=1), 1)
        self.assertEqual(contact_model_dimension(stage=2), 1)

        controller.stage1_tan_force_constraint = True
        self.assertEqual(contact_model_dimension(stage=1), 3)
        self.assertEqual(contact_model_dimension(stage=2), 1)

        controller.stage1_tan_force_constraint = False
        controller.stage2_tan_force_constraint = True
        self.assertEqual(contact_model_dimension(stage=1), 1)
        self.assertEqual(contact_model_dimension(stage=2), 3)

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
        self.assertEqual(
            coordinated.bounds[-3:],
            [(0.0, 100.0), (-50.0, 50.0), (-50.0, 50.0)],
        )

    def test_wrench_balance_uses_hydra_bounds_and_solver_options(self):
        """Pass the configured balance normalization, bounds, and SLSQP options."""
        controller = self._controller()
        controller.balance_use_normalized = True
        controller.r_data = {"solver_diagnostics": []}
        controller.solver_degraded = False
        accepted = SimpleNamespace(
            x=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            success=True,
            fun=0.0,
            status=0,
            message="accepted",
            nit=1,
        )

        with patch("ada_grasp_ctrl.utils.grasp_controller.minimize", return_value=accepted) as minimize_mock:
            metric, forces = controller.check_wrench_balance(np.zeros((6, 6)))

        self.assertEqual(metric, 0.0)
        np.testing.assert_array_equal(forces, accepted.x)
        self.assertEqual(
            minimize_mock.call_args.kwargs["bounds"],
            [(0.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)] * 2,
        )
        self.assertEqual(
            minimize_mock.call_args.kwargs["options"],
            {"ftol": 1e-6, "disp": False, "maxiter": 200},
        )

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

    def test_shared_solver_holds_failed_candidate_and_preserves_warnings(self):
        """Keep the explicit hold behavior and scoped warning handling."""
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

        def clipped_minimize(**kwargs):
            """Emit SciPy's documented clipping warning and return a failure.

            Args:
                kwargs: Minimize arguments, unused by the deterministic fixture.

            Returns:
                Failed SciPy-like result.
            """
            del kwargs
            warnings.warn(
                "Values in x were outside bounds during a minimize step, clipping to bounds",
                RuntimeWarning,
                stacklevel=2,
            )
            warnings.warn("unexpected optimizer warning", UserWarning, stacklevel=2)
            return failed

        with warnings.catch_warnings(record=True) as observed_warnings:
            warnings.simplefilter("always")
            with patch(
                "ada_grasp_ctrl.utils.grasp_controller.minimize",
                side_effect=clipped_minimize,
            ) as minimize_mock:
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
        self.assertEqual(len(observed_warnings), 1)
        self.assertEqual(str(observed_warnings[0].message), "unexpected optimizer warning")
        diagnostic = controller.r_data["solver_diagnostics"][-1]
        self.assertEqual(diagnostic["bound_clipping_warning_count"], 1)
        self.assertEqual(diagnostic["failure_policy"], "hold_current")
        self.assertEqual(diagnostic["decision"], "hold_current")
        self.assertTrue(diagnostic["action_applied"])
        self.assertFalse(diagnostic["episode_aborted"])
        self.assertEqual(
            minimize_mock.call_args.kwargs["options"],
            {"ftol": 1e-6, "disp": False, "maxiter": 200},
        )

    def test_shared_solver_applies_rejected_candidate_when_configured(self):
        """Apply a finite rejected command without rewriting solver acceptance."""
        controller = self._controller()
        controller.r_data = {"solver_diagnostics": []}
        controller.solver_degraded = False
        controller.solver_failure_policy = "apply_candidate"
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
                solver_name="control",
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

        np.testing.assert_allclose(result["q_a"], np.array([0.6, 0.1]), rtol=0, atol=1e-15)
        np.testing.assert_array_equal(result["dq_a"], np.array([0.4, 0.5]))
        np.testing.assert_array_equal(result["cf"], np.array([9.0, 8.0, 7.0]))
        diagnostic = controller.r_data["solver_diagnostics"][-1]
        self.assertFalse(diagnostic["accepted"])
        self.assertTrue(diagnostic["candidate_applicable"])
        self.assertEqual(diagnostic["decision"], "apply_candidate")
        self.assertTrue(controller.solver_degraded)

    def test_shared_solver_aborts_rejected_or_malformed_candidate(self):
        """Raise the domain abort only after recording complete diagnostics."""
        cases = (
            ("fail_episode", np.zeros(5), True),
            ("apply_candidate", None, False),
        )
        for policy, candidate, applicable in cases:
            with self.subTest(policy=policy, candidate=repr(candidate)):
                controller = self._controller()
                controller.r_data = {"solver_diagnostics": []}
                controller.solver_degraded = False
                controller.solver_failure_policy = policy
                result = SimpleNamespace(
                    success=False,
                    fun=1.0,
                    status=9,
                    message="failed",
                    nit=200,
                )
                if candidate is not None:
                    result.x = candidate

                with patch("ada_grasp_ctrl.utils.grasp_controller.minimize", return_value=result):
                    with self.assertRaises(ControlSolveEpisodeAbort):
                        controller._solve_control_problem(
                            solver_name="control",
                            stage=1,
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

                diagnostic = controller.r_data["solver_diagnostics"][-1]
                self.assertEqual(diagnostic["candidate_applicable"], applicable)
                self.assertEqual(diagnostic["decision"], "abort_episode")
                self.assertFalse(diagnostic["action_applied"])
                self.assertTrue(diagnostic["episode_aborted"])
                self.assertTrue(controller.solver_degraded)


if __name__ == "__main__":
    unittest.main()
