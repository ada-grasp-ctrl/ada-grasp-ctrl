"""Shared analytic constraints and acceptance diagnostics for SLSQP solves."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


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
    """Validate a SciPy SLSQP result before its command can be applied.

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
    try:
        variables = np.asarray(result.x, dtype=float).reshape(-1)
    except Exception:
        variables = np.asarray([], dtype=float)
    objective = getattr(result, "fun", np.nan)
    objective_finite = np.asarray(objective).size == 1 and np.isfinite(objective).all()
    variables_finite = variables.size == len(bounds) and np.all(np.isfinite(variables))
    if variables_finite:
        equality_residual, inequality_slack, constraints_finite = _constraint_values(constraints, variables)
        bound_violation = _bound_violation(variables, bounds)
    else:
        equality_residual, inequality_slack, constraints_finite = np.inf, -np.inf, False
        bound_violation = np.inf

    joint_limit_violation = 0.0
    if joint_limit_constraint is not None and variables_finite:
        try:
            joint_slack = np.asarray(joint_limit_constraint(variables), dtype=float).reshape(-1)
            if not np.all(np.isfinite(joint_slack)):
                joint_limit_violation = np.inf
            elif joint_slack.size:
                joint_limit_violation = max(0.0, float(-np.min(joint_slack)))
        except Exception:
            joint_limit_violation = np.inf

    finite = bool(variables_finite and objective_finite and constraints_finite)
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
        "fun": float(objective) if objective_finite else None,
        "finite": finite,
        "max_equality_residual": float(equality_residual),
        "min_inequality_slack": (None if np.isposinf(inequality_slack) else float(inequality_slack)),
        "bound_violation": float(bound_violation),
        "joint_limit_violation": float(joint_limit_violation),
    }


def select_control_solution(
    current_qpos: np.ndarray,
    current_contact_forces: np.ndarray,
    candidate: np.ndarray,
    command_dimension: int,
    diagnostics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply an accepted candidate or return the hold-last safe fallback.

    Args:
        current_qpos: Currently commanded actuated positions.
        current_contact_forces: Measured local contact forces.
        candidate: Solver vector containing delta-q followed by forces.
        command_dimension: Number of leading delta-q variables.
        diagnostics: Result from :func:`diagnose_slsqp_result`.

    Returns:
        Next qpos, applied delta-q, and accepted/preserved contact forces.
    """
    current = np.asarray(current_qpos, dtype=float).reshape(-1)
    measured_forces = np.asarray(current_contact_forces, dtype=float).reshape(-1)
    if diagnostics["accepted"]:
        variables = np.asarray(candidate, dtype=float).reshape(-1)
        delta = variables[:command_dimension].copy()
        return current + delta, delta, variables[command_dimension:].copy()
    # A rejected solution never reaches MuJoCo. Zeroing both delta and history
    # makes the following acceleration penalty start from the held command.
    return current.copy(), np.zeros(command_dimension), measured_forces.copy()
