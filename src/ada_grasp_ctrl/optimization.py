"""Shared analytic constraints and acceptance diagnostics for SLSQP solves."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


SolverFailurePolicy = Literal["apply_candidate", "hold_current", "fail_episode"]
ControlSolutionDecisionName = Literal["apply_accepted", "apply_candidate", "hold_current", "abort_episode"]
DEFAULT_SOLVER_FAILURE_POLICY: SolverFailurePolicy = "apply_candidate"
SOLVER_FAILURE_POLICIES: tuple[SolverFailurePolicy, ...] = (
    "apply_candidate",
    "hold_current",
    "fail_episode",
)


@dataclass(frozen=True)
class ControlSolutionDecision:
    """Policy decision for one diagnosed command-producing solver result.

    Args:
        qpos: Next actuator target, or ``None`` when the episode must abort.
        delta_qpos: Applied actuator delta/history, or ``None`` on abort.
        contact_forces: Candidate or preserved forces, or ``None`` on abort.
        decision: Stable diagnostic decision identifier.
        action_applied: Whether an actuator command should be sent.
        episode_aborted: Whether the current offset episode must stop.
    """

    qpos: np.ndarray | None
    delta_qpos: np.ndarray | None
    contact_forces: np.ndarray | None
    decision: ControlSolutionDecisionName
    action_applied: bool
    episode_aborted: bool


def _solver_candidate(candidate: Any, expected_dimension: int) -> tuple[np.ndarray, bool]:
    """Normalize an optimizer candidate defensively.

    Args:
        candidate: Raw optimizer candidate value.
        expected_dimension: Required flattened candidate dimension.

    Returns:
        Normalized candidate and whether it is runtime-applicable.
    """
    try:
        variables = np.asarray(candidate, dtype=float).reshape(-1)
    except (OverflowError, TypeError, ValueError):
        return np.asarray([], dtype=float), False
    applicable = bool(variables.size == expected_dimension and np.all(np.isfinite(variables)))
    return variables, applicable


def solve_linear_system(
    matrix: np.ndarray,
    right_hand_side: np.ndarray,
    *,
    condition_limit: float = 1e12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a square system with condition diagnostics and a safe fallback.

    A well-conditioned system uses :func:`numpy.linalg.solve`. Singular or
    ill-conditioned systems use the minimum-norm least-squares solution and
    are explicitly marked degraded. If even that fallback is nonfinite or
    fails, a zero solution is returned so invalid values cannot reach control.

    Args:
        matrix: Square coefficient matrix ``A``.
        right_hand_side: Vector or matrix ``B`` in ``A @ X = B``.
        condition_limit: Largest condition number accepted for direct solve.

    Returns:
        Solution and a JSON/NumPy-serializable diagnostic dictionary.

    Raises:
        ValueError: If shapes are incompatible or ``condition_limit`` is invalid.
    """
    coefficients = np.asarray(matrix)
    rhs = np.asarray(right_hand_side)
    if coefficients.ndim != 2 or coefficients.shape[0] != coefficients.shape[1]:
        raise ValueError(f"matrix must be square, got shape {coefficients.shape}")
    if rhs.ndim not in {1, 2} or rhs.shape[0] != coefficients.shape[0]:
        raise ValueError(
            "right_hand_side must be a vector or matrix with leading dimension "
            f"{coefficients.shape[0]}, got shape {rhs.shape}"
        )
    if not np.isfinite(condition_limit) or condition_limit <= 0:
        raise ValueError(f"condition_limit must be positive and finite, got {condition_limit}")

    try:
        condition_number = float(np.linalg.cond(coefficients))
    except np.linalg.LinAlgError:
        condition_number = np.inf

    method = "solve"
    message = "direct solve accepted"
    degraded = not np.isfinite(condition_number) or condition_number > condition_limit
    solution = None
    if not degraded:
        try:
            solution = np.linalg.solve(coefficients, rhs)
            if not np.all(np.isfinite(solution)):
                degraded = True
                message = "direct solve returned nonfinite values"
        except np.linalg.LinAlgError as error:
            degraded = True
            message = f"direct solve failed: {error}"
    else:
        message = f"condition number {condition_number:.6g} exceeds limit {condition_limit:.6g}"

    fallback_failed = False
    if degraded:
        method = "lstsq"
        try:
            solution = np.linalg.lstsq(coefficients, rhs, rcond=None)[0]
            if not np.all(np.isfinite(solution)):
                fallback_failed = True
                message = f"{message}; least-squares fallback returned nonfinite values"
        except np.linalg.LinAlgError as error:
            fallback_failed = True
            message = f"{message}; least-squares fallback failed: {error}"
        if fallback_failed:
            method = "zero"
            solution_shape = rhs.shape
            solution = np.zeros(solution_shape, dtype=np.result_type(coefficients, rhs, float))

    residual_norm = float(np.linalg.norm(coefficients @ solution - rhs))
    finite = bool(np.all(np.isfinite(solution)) and np.isfinite(residual_norm))
    diagnostics = {
        "accepted": bool(not degraded and finite),
        "success": finite,
        "method": method,
        "message": message,
        "condition_number": (condition_number if np.isfinite(condition_number) else None),
        "condition_limit": float(condition_limit),
        "residual_norm": (residual_norm if np.isfinite(residual_norm) else None),
        "finite": finite,
        "matrix_shape": tuple(int(value) for value in coefficients.shape),
        "rhs_shape": tuple(int(value) for value in rhs.shape),
    }
    return solution, diagnostics


def friction_cone_slack(contact_forces: np.ndarray, friction_coefficient: float) -> np.ndarray:
    """Evaluate circular Coulomb friction-cone slack.

    Args:
        contact_forces: Array shaped ``(..., 3)`` with normal force first.
        friction_coefficient: Nonnegative Coulomb coefficient.

    Returns:
        Per-contact slack ``mu * fx - sqrt(fy^2 + fz^2)``.
    """
    forces = np.asarray(contact_forces, dtype=float).reshape(-1, 3)
    return friction_coefficient * forces[:, 0] - np.linalg.norm(forces[:, 1:], axis=1)


def friction_cone_jacobian(contact_forces: np.ndarray, friction_coefficient: float) -> np.ndarray:
    """Return the block Jacobian of circular friction-cone slack.

    The normal derivative is always ``mu``; it does not depend on the sign of
    ``fx``. At the nondifferentiable tangential origin, the zero subgradient is
    used for ``fy`` and ``fz`` so an all-zero SLSQP initial point remains valid.

    Args:
        contact_forces: Array shaped ``(N, 3)`` with normal force first.
        friction_coefficient: Nonnegative Coulomb coefficient.

    Returns:
        Dense Jacobian shaped ``(N, 3N)``.
    """
    forces = np.asarray(contact_forces, dtype=float).reshape(-1, 3)
    count = forces.shape[0]
    jacobian = np.zeros((count, 3 * count), dtype=float)
    tangential_norm = np.linalg.norm(forces[:, 1:], axis=1)
    nonzero = tangential_norm > 0
    rows = np.arange(count)
    jacobian[rows, 3 * rows] = friction_coefficient
    jacobian[rows[nonzero], 3 * rows[nonzero] + 1] = -forces[nonzero, 1] / tangential_norm[nonzero]
    jacobian[rows[nonzero], 3 * rows[nonzero] + 2] = -forces[nonzero, 2] / tangential_norm[nonzero]
    return jacobian


def _constraint_values(
    constraints: Sequence[dict[str, Any]],
    variables: np.ndarray,
) -> tuple[float, float, bool]:
    """Evaluate equality residuals and inequality slacks defensively.

    Args:
        constraints: SciPy-style constraint dictionaries.
        variables: Candidate solution vector.

    Returns:
        Maximum equality residual, minimum inequality slack, and finiteness flag.
    """
    equality_residual = 0.0
    inequality_slack = np.inf
    finite = True
    for constraint in constraints:
        try:
            values = np.asarray(constraint["fun"](variables), dtype=float).reshape(-1)
        except Exception:
            return np.inf, -np.inf, False
        if not np.all(np.isfinite(values)):
            finite = False
        if constraint["type"] == "eq" and values.size:
            equality_residual = max(equality_residual, float(np.max(np.abs(values))))
        elif constraint["type"] == "ineq" and values.size:
            inequality_slack = min(inequality_slack, float(np.min(values)))
    return equality_residual, inequality_slack, finite


def _bound_violation(variables: np.ndarray, bounds: Sequence[tuple[float | None, float | None]]) -> float:
    """Compute the largest scalar bound violation.

    Args:
        variables: Candidate solution vector.
        bounds: SciPy-style lower/upper bounds.

    Returns:
        Nonnegative maximum violation.
    """
    violation = 0.0
    for value, (lower, upper) in zip(variables, bounds):
        if lower is not None:
            violation = max(violation, float(lower - value))
        if upper is not None:
            violation = max(violation, float(value - upper))
    return max(0.0, violation)


def diagnose_slsqp_result(
    result: Any,
    constraints: Sequence[dict[str, Any]],
    bounds: Sequence[tuple[float | None, float | None]],
    *,
    joint_limit_constraint: Callable[[np.ndarray], np.ndarray] | None = None,
    equality_tolerance: float = 1e-5,
    inequality_tolerance: float = 1e-5,
    bound_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Diagnose a SciPy SLSQP result before the configured policy acts.

    Args:
        result: SciPy-like object with ``x``, ``success``, and solver metadata.
        constraints: Constraints passed to SLSQP.
        bounds: Bounds passed to SLSQP.
        joint_limit_constraint: Optional nonnegative joint-limit slack function.
        equality_tolerance: Maximum accepted absolute equality residual.
        inequality_tolerance: Magnitude of accepted negative inequality slack.
        bound_tolerance: Maximum accepted bound/joint-limit violation.

    Returns:
        JSON/NumPy-serializable diagnostic dictionary containing ``accepted``.
    """
    variables, candidate_applicable = _solver_candidate(getattr(result, "x", None), len(bounds))
    objective = getattr(result, "fun", np.nan)
    try:
        objective_array = np.asarray(objective, dtype=float)
        objective_finite = objective_array.size == 1 and np.isfinite(objective_array).all()
    except (OverflowError, TypeError, ValueError):
        objective_finite = False
    if candidate_applicable:
        equality_residual, inequality_slack, constraints_finite = _constraint_values(constraints, variables)
        bound_violation = _bound_violation(variables, bounds)
    else:
        equality_residual, inequality_slack, constraints_finite = np.inf, -np.inf, False
        bound_violation = np.inf

    joint_limit_violation = 0.0
    if joint_limit_constraint is not None and candidate_applicable:
        try:
            joint_slack = np.asarray(joint_limit_constraint(variables), dtype=float).reshape(-1)
            if not np.all(np.isfinite(joint_slack)):
                joint_limit_violation = np.inf
            elif joint_slack.size:
                joint_limit_violation = max(0.0, float(-np.min(joint_slack)))
        except Exception:
            joint_limit_violation = np.inf

    finite = bool(candidate_applicable and objective_finite and constraints_finite)
    accepted = bool(
        getattr(result, "success", False)
        and finite
        and equality_residual <= equality_tolerance
        and inequality_slack >= -inequality_tolerance
        and bound_violation <= bound_tolerance
        and joint_limit_violation <= bound_tolerance
    )
    return {
        "accepted": accepted,
        "success": bool(getattr(result, "success", False)),
        "status": int(getattr(result, "status", -1)),
        "message": str(getattr(result, "message", "")),
        "nit": int(getattr(result, "nit", -1)),
        "fun": float(objective_array.reshape(-1)[0]) if objective_finite else None,
        "finite": finite,
        "candidate_applicable": candidate_applicable,
        "max_equality_residual": float(equality_residual),
        "min_inequality_slack": (None if np.isposinf(inequality_slack) else float(inequality_slack)),
        "bound_violation": float(bound_violation),
        "joint_limit_violation": float(joint_limit_violation),
    }


def select_control_solution(
    current_qpos: np.ndarray,
    current_contact_forces: np.ndarray,
    candidate: Any,
    command_dimension: int,
    diagnostics: dict[str, Any],
    *,
    failure_policy: SolverFailurePolicy = DEFAULT_SOLVER_FAILURE_POLICY,
) -> ControlSolutionDecision:
    """Select an action or episode abort from one diagnosed solver result.

    Args:
        current_qpos: Currently commanded actuated positions.
        current_contact_forces: Measured local contact forces.
        candidate: Solver vector containing delta-q followed by forces, if available.
        command_dimension: Number of leading delta-q variables.
        diagnostics: Result from :func:`diagnose_slsqp_result`.
        failure_policy: Configured rejected-result behavior.

    Returns:
        Structured actuator/history/abort decision.

    Raises:
        ValueError: If ``failure_policy`` is unsupported or dimensions are invalid.
    """
    if failure_policy not in SOLVER_FAILURE_POLICIES:
        supported = ", ".join(SOLVER_FAILURE_POLICIES)
        raise ValueError(f"Unsupported solver failure policy '{failure_policy}'. Supported values: {supported}.")
    if command_dimension < 0:
        raise ValueError(f"command_dimension must be nonnegative, got {command_dimension}")
    current = np.asarray(current_qpos, dtype=float).reshape(-1)
    measured_forces = np.asarray(current_contact_forces, dtype=float).reshape(-1)
    if current.size != command_dimension:
        raise ValueError(f"current_qpos must contain {command_dimension} values, got {current.size}")
    variables, candidate_applicable = _solver_candidate(candidate, command_dimension + measured_forces.size)
    if diagnostics["accepted"]:
        if not candidate_applicable:
            return ControlSolutionDecision(None, None, None, "abort_episode", False, True)
        delta = variables[:command_dimension].copy()
        return ControlSolutionDecision(
            current + delta,
            delta,
            variables[command_dimension:].copy(),
            "apply_accepted",
            True,
            False,
        )
    if failure_policy == "apply_candidate" and candidate_applicable:
        delta = variables[:command_dimension].copy()
        return ControlSolutionDecision(
            current + delta,
            delta,
            variables[command_dimension:].copy(),
            "apply_candidate",
            True,
            False,
        )
    if failure_policy == "hold_current":
        # Zeroing delta/history makes the following acceleration penalty start
        # from the deliberately held command.
        return ControlSolutionDecision(
            current.copy(),
            np.zeros(command_dimension),
            measured_forces.copy(),
            "hold_current",
            True,
            False,
        )
    return ControlSolutionDecision(None, None, None, "abort_episode", False, True)
