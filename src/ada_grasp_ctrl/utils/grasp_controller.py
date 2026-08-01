from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING
import warnings

from scipy.optimize import minimize
from scipy.linalg import block_diag
import numpy as np
import os
from mr_utils.utils_calc import (
    isometry3dToPosQuat,
    isometry3dToPosOri,
    sciR,
    skew,
)

if TYPE_CHECKING:
    from .robot_adaptor import RobotAdaptor
    from .robots.base import ArmHand
from ada_grasp_ctrl.schema import with_current_schema
from ada_grasp_ctrl.errors import ControlSolveEpisodeAbort
from ada_grasp_ctrl.optimization import (
    DEFAULT_SOLVER_FAILURE_POLICY,
    SOLVER_FAILURE_POLICIES,
    diagnose_slsqp_result,
    friction_cone_jacobian,
    friction_cone_slack,
    select_control_solution,
    solve_linear_system,
)


_SLSQP_BOUND_CLIPPING_WARNING = "Values in x were outside bounds during a minimize step, clipping to bounds"


@dataclass(frozen=True)
class _ControlProblemContext:
    """Shared dimensions, kinematics, contact model, and target-pose data."""

    num_arm_dof: int
    num_hand_dof: int
    num_dof: int
    num_contacts: int
    doa_to_dof: np.ndarray
    joint_limits: np.ndarray
    max_delta_qpos: np.ndarray
    contact_forces: np.ndarray
    contact_jacobian: np.ndarray
    stiffness_jacobian: np.ndarray
    hand_base_name: str
    target_hand_base_position: np.ndarray
    target_hand_base_orientation: Any


@dataclass(frozen=True)
class _ControlProblemDefinition:
    """Callbacks, constraints, bounds, and initial state for one SLSQP solve."""

    objective: Callable[[np.ndarray], float]
    jacobian: Callable[[np.ndarray], np.ndarray]
    constraints: list[dict[str, Any]]
    constraint_names: tuple[str, ...]
    bounds: list[tuple[float | None, float | None]]
    initial_variables: np.ndarray
    joint_limit_constraint: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class _ObjectiveWeights:
    """Hydra-managed weights used to assemble dimension-dependent objectives."""

    hand_base_pose: tuple[float, ...]
    hand_joint_position: float
    joint_velocity: float
    joint_acceleration: float
    coordinated_tangential_force: float
    equal_contact_force: float
    control_wrench: tuple[float, ...]
    equal_joint_force: float
    in_contact_joint_velocity_multiplier: float


@dataclass(frozen=True)
class _WrenchBalanceParameters:
    """Hydra-managed wrench-balance optimization parameters."""

    normal_force_sum: float
    wrench_weights: tuple[float, ...]
    normal_force_bounds: tuple[float, float]
    tangential_force_bounds: tuple[float, float]
    ftol: float
    maxiter: int


@dataclass(frozen=True)
class _CommandSolverParameters:
    """Hydra-managed command-producing SLSQP parameters."""

    normal_force_bounds: tuple[float, float]
    tangential_force_bounds: tuple[float, float]
    ftol: float
    maxiter: int


@dataclass(frozen=True)
class GraspControllerParameters:
    """Validated controller parameters parsed from ``task.control`` Hydra config."""

    balance_thres: float
    friction_cone_mu: float
    final_sum_force: float
    kp_clip_min: float
    kp_clip_max: float
    objective_weights: _ObjectiveWeights
    wrench_balance: _WrenchBalanceParameters
    command_solver: _CommandSolverParameters

    @classmethod
    def from_config(cls, configs: Any) -> GraspControllerParameters:
        """Parse and validate the configurable controller contract.

        Args:
            configs: Hydra ``task.control`` configuration.

        Returns:
            Immutable validated controller parameters.

        Raises:
            ValueError: If a field is missing, nonfinite, malformed, or outside
                its valid range.
        """
        if configs is None:
            raise ValueError("task.control configuration is required")

        kp_clip_min = _finite_float(_config_value(configs, "kp_clip.min"), "kp_clip.min", nonnegative=True)
        kp_clip_max = _finite_float(_config_value(configs, "kp_clip.max"), "kp_clip.max", nonnegative=True)
        if kp_clip_min >= kp_clip_max:
            raise ValueError("task.control.kp_clip.min must be less than task.control.kp_clip.max")

        objective_weights = _ObjectiveWeights(
            hand_base_pose=_float_tuple(
                _config_value(configs, "objective_weights.hand_base_pose"),
                "objective_weights.hand_base_pose",
                length=6,
                nonnegative=True,
            ),
            hand_joint_position=_finite_float(
                _config_value(configs, "objective_weights.hand_joint_position"),
                "objective_weights.hand_joint_position",
                nonnegative=True,
            ),
            joint_velocity=_finite_float(
                _config_value(configs, "objective_weights.joint_velocity"),
                "objective_weights.joint_velocity",
                nonnegative=True,
            ),
            joint_acceleration=_finite_float(
                _config_value(configs, "objective_weights.joint_acceleration"),
                "objective_weights.joint_acceleration",
                nonnegative=True,
            ),
            coordinated_tangential_force=_finite_float(
                _config_value(configs, "objective_weights.coordinated_tangential_force"),
                "objective_weights.coordinated_tangential_force",
                nonnegative=True,
            ),
            equal_contact_force=_finite_float(
                _config_value(configs, "objective_weights.equal_contact_force"),
                "objective_weights.equal_contact_force",
                nonnegative=True,
            ),
            control_wrench=_float_tuple(
                _config_value(configs, "objective_weights.control_wrench"),
                "objective_weights.control_wrench",
                length=6,
                nonnegative=True,
            ),
            equal_joint_force=_finite_float(
                _config_value(configs, "objective_weights.equal_joint_force"),
                "objective_weights.equal_joint_force",
                nonnegative=True,
            ),
            in_contact_joint_velocity_multiplier=_finite_float(
                _config_value(configs, "objective_weights.in_contact_joint_velocity_multiplier"),
                "objective_weights.in_contact_joint_velocity_multiplier",
                nonnegative=True,
            ),
        )
        wrench_balance = _WrenchBalanceParameters(
            normal_force_sum=_finite_float(
                _config_value(configs, "wrench_balance.normal_force_sum"),
                "wrench_balance.normal_force_sum",
                positive=True,
            ),
            wrench_weights=_float_tuple(
                _config_value(configs, "wrench_balance.wrench_weights"),
                "wrench_balance.wrench_weights",
                length=6,
                nonnegative=True,
            ),
            normal_force_bounds=_bounds(
                _config_value(configs, "wrench_balance.normal_force_bounds"),
                "wrench_balance.normal_force_bounds",
                nonnegative_lower=True,
            ),
            tangential_force_bounds=_bounds(
                _config_value(configs, "wrench_balance.tangential_force_bounds"),
                "wrench_balance.tangential_force_bounds",
            ),
            ftol=_finite_float(
                _config_value(configs, "wrench_balance.ftol"),
                "wrench_balance.ftol",
                positive=True,
            ),
            maxiter=_positive_int(
                _config_value(configs, "wrench_balance.maxiter"),
                "wrench_balance.maxiter",
            ),
        )
        command_solver = _CommandSolverParameters(
            normal_force_bounds=_bounds(
                _config_value(configs, "command_solver.normal_force_bounds"),
                "command_solver.normal_force_bounds",
                nonnegative_lower=True,
            ),
            tangential_force_bounds=_bounds(
                _config_value(configs, "command_solver.tangential_force_bounds"),
                "command_solver.tangential_force_bounds",
            ),
            ftol=_finite_float(
                _config_value(configs, "command_solver.ftol"),
                "command_solver.ftol",
                positive=True,
            ),
            maxiter=_positive_int(
                _config_value(configs, "command_solver.maxiter"),
                "command_solver.maxiter",
            ),
        )
        return cls(
            balance_thres=_finite_float(
                _config_value(configs, "balance_thres"),
                "balance_thres",
                nonnegative=True,
            ),
            friction_cone_mu=_finite_float(
                _config_value(configs, "friction_cone_mu"),
                "friction_cone_mu",
                positive=True,
            ),
            final_sum_force=_finite_float(
                _config_value(configs, "final_sum_force"),
                "final_sum_force",
                positive=True,
            ),
            kp_clip_min=kp_clip_min,
            kp_clip_max=kp_clip_max,
            objective_weights=objective_weights,
            wrench_balance=wrench_balance,
            command_solver=command_solver,
        )


def _config_value(configs: Any, path: str) -> Any:
    """Read one required dotted field from a mapping-like Hydra config."""
    current = configs
    for field_name in path.split("."):
        try:
            current = current[field_name]
        except Exception as error:
            raise ValueError(f"Missing or unresolved task.control.{path}") from error
    return current


def _finite_float(value: Any, path: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    """Convert one configuration scalar to a validated finite float."""
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"task.control.{path} must be a finite number") from error
    if not np.isfinite(converted):
        raise ValueError(f"task.control.{path} must be a finite number")
    if positive and converted <= 0:
        raise ValueError(f"task.control.{path} must be greater than zero")
    if nonnegative and converted < 0:
        raise ValueError(f"task.control.{path} must be nonnegative")
    return converted


def _float_tuple(
    value: Any,
    path: str,
    *,
    length: int,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    """Convert one fixed-length configuration sequence to finite floats."""
    if isinstance(value, (str, bytes)):
        raise ValueError(f"task.control.{path} must contain exactly {length} numbers")
    try:
        items = tuple(value)
    except TypeError as error:
        raise ValueError(f"task.control.{path} must contain exactly {length} numbers") from error
    if len(items) != length:
        raise ValueError(f"task.control.{path} must contain exactly {length} numbers")
    return tuple(_finite_float(item, f"{path}[{index}]", nonnegative=nonnegative) for index, item in enumerate(items))


def _bounds(value: Any, path: str, *, nonnegative_lower: bool = False) -> tuple[float, float]:
    """Validate one lower/upper bound pair."""
    lower, upper = _float_tuple(value, path, length=2)
    if lower >= upper:
        raise ValueError(f"task.control.{path}[0] must be less than task.control.{path}[1]")
    if nonnegative_lower and lower < 0:
        raise ValueError(f"task.control.{path}[0] must be nonnegative")
    return lower, upper


def _positive_int(value: Any, path: str) -> int:
    """Validate one strictly positive integer configuration field."""
    if isinstance(value, bool):
        raise ValueError(f"task.control.{path} must be a positive integer")
    try:
        converted = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"task.control.{path} must be a positive integer") from error
    if converted <= 0 or not np.isfinite(numeric) or numeric != converted:
        raise ValueError(f"task.control.{path} must be a positive integer")
    return converted


class GraspController:
    def __init__(self, configs, robot: ArmHand, robot_adaptor: RobotAdaptor):
        self.robot = robot
        self.robot_adaptor = robot_adaptor

        self.r_data = {
            "obj_pose": [],
            "dof": [],
            "doa": [],
            "contacts": [],
            "planned_dof": [],
            "balance_metric": [],
            "t_check_balance": [],
            "t_ctrl_opt": [],
            "t_step_cost": [],
            "stage": [],
            "opt_res": [],
            "desired_sum_force": [],
            "solver_diagnostics": [],
        }
        self.solver_degraded = False
        self.solver_failure_policy = DEFAULT_SOLVER_FAILURE_POLICY
        self.parameters: GraspControllerParameters | None = None

        # hyper-parameters
        if configs is not None:
            self.parameters = GraspControllerParameters.from_config(configs)
            self.solver_failure_policy = str(configs.get("solver_failure_policy", DEFAULT_SOLVER_FAILURE_POLICY))
            if self.solver_failure_policy not in SOLVER_FAILURE_POLICIES:
                supported = ", ".join(SOLVER_FAILURE_POLICIES)
                raise ValueError(
                    f"Unsupported solver failure policy '{self.solver_failure_policy}'. Supported values: {supported}."
                )
            self.stage2_incontact_force_only = configs.stage2_incontact_force_only
            self.stage2_Ks_hand_only = configs.stage2_Ks_hand_only
            self.stage2_penalize_tan_motion = configs.stage2_penalize_tan_motion
            self.balance_use_normalized = configs.balance_use_normalized

            self.stage2_equal_joint_force_cost = configs.stage2_equal_joint_force_cost

            self.stage2_ctrl_tan_force = configs.stage2_ctrl_tan_force
            self.stage2_tan_force_constraint = configs.stage2_tan_force_constraint
            self.stage1_tan_force_constraint = configs.get(
                "stage1_tan_force_constraint",
                self.stage2_tan_force_constraint,
            )

            self.stage2_increase_force = configs.stage2_increase_force

            self.Ke_scalar = configs.Ke_scalar
            self.stage1_force_thres = configs.stage1_force_thres

            self.Kp = np.diag(
                np.clip(
                    self.robot.doa_kp,
                    self.parameters.kp_clip_min,
                    self.parameters.kp_clip_max,
                )
            )
            self.Kp_inv = self._solve_linear_system(
                self.Kp,
                np.eye(self.Kp.shape[0]),
                system_name="joint_stiffness_inverse",
            )
            self.Ke = np.diag([self.Ke_scalar, self.Ke_scalar, self.Ke_scalar])  # x-axis is the contact normal

            self.tan_motion_pen_weight = configs.tan_motion_pen_weight
            self.use_multi_contact_model = configs.use_multi_contact_model
            self.stage2_penalize_contact_qda = configs.stage2_penalize_contact_qda
            self.jaco_reference_frame = configs.jaco_reference_frame
            self.balance_thres = self.parameters.balance_thres
            self.mu = self.parameters.friction_cone_mu
            self.final_sum_force = self.parameters.final_sum_force

    def _require_parameters(self) -> GraspControllerParameters:
        """Return configured parameters for methods unavailable in statistics-only mode.

        Returns:
            Validated controller parameters.

        Raises:
            RuntimeError: If the controller was created without ``task.control``.
        """
        if self.parameters is None:
            raise RuntimeError("Controller optimization requires task.control configuration")
        return self.parameters

    def _record_solver_diagnostic(self, solver_name, diagnostics, stage=None):
        """Append one solver diagnostic and latch episode degradation.

        Args:
            solver_name: Stable solver call identifier.
            diagnostics: Acceptance details from the shared validator.
            stage: Optional control stage.

        Returns:
            None.
        """
        record = dict(diagnostics)
        record["solver"] = solver_name
        if stage is not None:
            record["stage"] = int(stage)
        self.r_data["solver_diagnostics"].append(record)
        if not diagnostics["accepted"]:
            self.solver_degraded = True

    def save_recorded_data(self, path, episode_status="completed"):
        """Save one versioned control trajectory.

        Args:
            path: Destination ``.npy`` path.
            episode_status: Structured episode outcome.

        Returns:
            None.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        record = with_current_schema(self.r_data)
        record["episode_status"] = episode_status
        np.save(path, record, allow_pickle=True)
        print(f"Save recorded control data to {path}.")

    def interplote_qpos(self, qpos1: np.array, qpos2: np.array, step: int) -> np.array:
        return np.linspace(qpos1, qpos2, step + 1)[1:]

    def _solve_linear_system(
        self,
        matrix: np.ndarray,
        right_hand_side: np.ndarray,
        *,
        system_name: str,
    ) -> np.ndarray:
        """Solve a controller system and record any numerical degradation.

        Args:
            matrix: Square coefficient matrix.
            right_hand_side: Vector or matrix on the right side of the system.
            system_name: Stable diagnostic identifier for this physical system.

        Returns:
            Direct or safely degraded solution.
        """
        solution, diagnostics = solve_linear_system(matrix, right_hand_side)
        if not diagnostics["accepted"]:
            self._record_solver_diagnostic(f"linear:{system_name}", diagnostics)
        return solution

    def Ks(self, q_a, q_f, contacts):
        """
        Compute Ks.
        """
        I3 = np.eye(3)
        hand_ndoa = self.robot.hand.n_doa

        # compute J(q) and J(qd)
        body_name_lst = []
        for i, contact in enumerate(contacts):
            body_name_lst.append(contact["body1_name"])
        self.robot_adaptor.compute_jaco_a(q_a)  # J(qd)
        body_jaco_a_lst = [self.robot_adaptor.get_frame_jaco(frame_name=name, type="body") for name in body_name_lst]
        pose_a_lst = [self.robot_adaptor.get_frame_pose(frame_name=name) for name in body_name_lst]
        self.robot_adaptor.compute_jaco_a(q_f)  # J(q)
        body_jaco_f_lst = [self.robot_adaptor.get_frame_jaco(frame_name=name, type="body") for name in body_name_lst]
        pose_f_lst = [self.robot_adaptor.get_frame_pose(frame_name=name) for name in body_name_lst]

        # --- Compute per-contact Jacobians ---
        for i, c in enumerate(contacts):
            cp_local = c["contact_pos_local"].reshape(-1, 1)
            cf_local = c["contact_frame_local"].reshape(3, 3)
            Trans = np.block([[I3, -skew(cp_local)]])
            body_jaco_f = body_jaco_f_lst[i]
            body_jaco_a = body_jaco_a_lst[i]
            contact_jaco_f = cf_local.T @ Trans @ body_jaco_f
            contact_jaco_a = cf_local.T @ Trans @ body_jaco_a

            if self.jaco_reference_frame:
                # adjust J(qd) to be defined in the local contact frame of q;
                # otherwise, it is in the local contact frame of qd.
                delta_rot = pose_f_lst[i][:3, :3].T @ pose_a_lst[i][:3, :3]
                contact_jaco_a = delta_rot @ contact_jaco_a

            c["jaco_a"], c["jaco_f"] = contact_jaco_a, contact_jaco_f
            c["jaco_ha"], c["jaco_hf"] = contact_jaco_a[:, -hand_ndoa:], contact_jaco_f[:, -hand_ndoa:]
            contacts[i] = c

        # --- Compute Ks ---
        if self.use_multi_contact_model:  # if consider multiple contact on the same finger
            n_con = len(contacts)
            I_stack = np.eye(3 * n_con)
            Ke_stack = block_diag(*([self.Ke] * n_con))
            J_a_stack = np.concatenate([c["jaco_a"] for c in contacts], axis=0)
            J_f_stack = np.concatenate([c["jaco_f"] for c in contacts], axis=0)
            Kr_inv_stack = J_a_stack @ self.Kp_inv @ J_f_stack.T
            Ks_stack = self._solve_linear_system(
                I_stack + Ke_stack @ Kr_inv_stack,
                Ke_stack,
                system_name="multi_contact_stiffness",
            )

            J_ha_stack = np.concatenate([c["jaco_ha"] for c in contacts], axis=0)
            J_hf_stack = np.concatenate([c["jaco_hf"] for c in contacts], axis=0)
            Kr_h_inv_stack = J_ha_stack @ self.Kp_inv[-hand_ndoa:, -hand_ndoa:] @ J_hf_stack.T
            Ks_h_stack = self._solve_linear_system(
                I_stack + Ke_stack @ Kr_h_inv_stack,
                Ke_stack,
                system_name="multi_contact_hand_stiffness",
            )
        else:
            for i, contact in enumerate(contacts):
                contact_jaco_a = contact["jaco_a"]
                contact_jaco_f = contact["jaco_f"]
                Kr_inv = contact_jaco_a @ self.Kp_inv @ contact_jaco_f.T
                Ks = self._solve_linear_system(
                    I3 + self.Ke @ Kr_inv,
                    self.Ke,
                    system_name="contact_stiffness",
                )
                contact["Ks"] = Ks
                # only hand
                contact_jaco_ha = contact_jaco_a[:, -hand_ndoa:]
                contact_jaco_hf = contact_jaco_f[:, -hand_ndoa:]
                Kr_h_inv = contact_jaco_ha @ self.Kp_inv[-hand_ndoa:, -hand_ndoa:] @ contact_jaco_hf.T
                Ks_h = self._solve_linear_system(
                    I3 + self.Ke @ Kr_h_inv,
                    self.Ke,
                    system_name="contact_hand_stiffness",
                )
                contact["Ks_h"] = Ks_h
                contacts[i] = contact

            Ks_stack = block_diag(*[c["Ks"] for c in contacts])
            J_a_stack = np.concatenate([c["jaco_a"] for c in contacts], axis=0)
            J_f_stack = np.concatenate([c["jaco_f"] for c in contacts], axis=0)
            Ks_h_stack = block_diag(*[c["Ks_h"] for c in contacts])
            J_ha_stack = np.concatenate([c["jaco_ha"] for c in contacts], axis=0)
            J_hf_stack = np.concatenate([c["jaco_hf"] for c in contacts], axis=0)

        stacked = {
            "Ks": Ks_stack,
            "jaco_a": J_a_stack,
            "jaco_f": J_f_stack,
            "Ks_h": Ks_h_stack,
            "jaco_ha": J_ha_stack,
            "jaco_hf": J_hf_stack,
        }

        return contacts, stacked

    # def compute_grasp_matrix(self, ho_contacts) -> np.ndarray:
    #     n_con = len(ho_contacts)
    #     if n_con == 0:
    #         return None

    #     contact_frame = [contact["contact_frame"] for contact in ho_contacts]
    #     contact_pos_all = [contact["contact_pos"] for contact in ho_contacts]
    #     contact_pos_all = np.asarray(contact_pos_all).reshape(-1, 3)
    #     contact_centroid = contact_pos_all.mean(axis=0, keepdims=True)
    #     contact_r = contact_pos_all - contact_centroid
    #     contact_r = contact_r * 100.0  # unit from m to cm; then, the unit of torque is (N x cm)

    #     contact_G = []
    #     for i in range(len(ho_contacts)):
    #         r = contact_r[i, :]
    #         n, o, t = contact_frame[i][:, 0], contact_frame[i][:, 1], contact_frame[i][:, 2]
    #         G = np.block(
    #             [
    #                 [n.reshape(-1, 1), o.reshape(-1, 1), t.reshape(-1, 1)],
    #                 [np.cross(r, n).reshape(-1, 1), np.cross(r, o).reshape(-1, 1), np.cross(r, t).reshape(-1, 1)],
    #             ]
    #         )
    #         contact_G.append(G)
    #     contact_G = np.concatenate(contact_G, axis=1)

    #     return contact_G

    def compute_grasp_matrix(self, ho_contacts) -> np.ndarray:
        n_con = len(ho_contacts)
        if n_con == 0:
            return None

        # Extract positions and frames
        contact_frames = np.array([c["contact_frame"] for c in ho_contacts])  # (n, 3, 3)
        contact_pos = np.array([c["contact_pos"] for c in ho_contacts])  # (n, 3)

        # Compute centroid and relative positions (scaled to cm)
        centroid = contact_pos.mean(axis=0, keepdims=True)
        r_all = (contact_pos - centroid) * 100.0  # (n, 3); unit from m to cm; then, the unit of torque is (N x cm)

        # Split frame axes
        n_vecs = contact_frames[:, :, 0]  # (n, 3)
        o_vecs = contact_frames[:, :, 1]
        t_vecs = contact_frames[:, :, 2]

        # Compute torque components using broadcasting
        cross_n = np.cross(r_all, n_vecs)  # (n, 3)
        cross_o = np.cross(r_all, o_vecs)
        cross_t = np.cross(r_all, t_vecs)

        # Stack translational and rotational parts
        G_blocks = np.stack(
            [
                np.stack([n_vecs, o_vecs, t_vecs], axis=2),  # (n, 3, 3)
                np.stack([cross_n, cross_o, cross_t], axis=2),  # (n, 3, 3)
            ],
            axis=1,
        )  # (n, 2, 3, 3)

        # Reshape each block into (6, 3) and concatenate along columns
        contact_G = G_blocks.reshape(n_con, 6, 3).transpose(1, 0, 2).reshape(6, -1)

        return contact_G

    def compute_normalized_wrench(self, grasp_matrix: np.ndarray, contact_forces: np.ndarray):
        wrench = (grasp_matrix @ contact_forces.reshape(-1, 1)).reshape(-1)

        cf = contact_forces.reshape(-1, 3)
        G = grasp_matrix.reshape(6, -1, 3).transpose(1, 0, 2)
        fts = np.matmul(G, cf[:, :, None]).reshape(-1, 6)
        sum_forces_mag = np.sum(np.linalg.norm(fts[:, :3], axis=1))
        sum_torques_mag = np.sum(np.linalg.norm(fts[:, 3:], axis=1))

        wrench[:3] /= sum_forces_mag + 1e-8
        wrench[3:] /= sum_torques_mag + 1e-8
        return wrench

    def check_wrench_balance(self, grasp_matrix, b_print_opt_details=False):
        controller_parameters = self._require_parameters()
        if grasp_matrix is None:
            return 1.0, None

        contact_G = grasp_matrix.copy()
        n_con = contact_G.shape[1] // 3

        if n_con < 2:  # only one contact cannot be in wrench balance
            return 1.0, None

        parameters = controller_parameters.wrench_balance
        w_wrench = np.diag(parameters.wrench_weights)
        mu = self.mu
        gamma = parameters.normal_force_sum

        def objective(x):
            cf = x.copy()
            wrench = contact_G @ cf.reshape(-1, 1)
            cost = wrench.T @ w_wrench @ wrench
            grad = 2 * contact_G.T @ w_wrench @ contact_G @ cf.reshape(-1, 1)
            return cost.item(), grad

        def friction_cone_constraint(x):
            cf = x.copy().reshape(-1, 3)
            return friction_cone_slack(cf, mu)

        def friction_cone_constraint_grad(x):
            return friction_cone_jacobian(x.reshape(-1, 3), mu)

        def force_magnitude_constraint(x):
            cf = x.copy().reshape(-1, 3)
            constraint = np.sum(cf[:, 0]) - gamma  # == 0
            return constraint.reshape(-1)

        def force_magnitude_constraint_grad(x):
            cf = x.reshape(-1, 3)  # shape (n, 3)
            grad = np.zeros_like(cf)  # shape (n, 3)
            grad[:, 0] = 1.0
            return grad.reshape(-1)  # flatten to match x shape

        constraints_list = [
            dict(type="ineq", fun=friction_cone_constraint, jac=friction_cone_constraint_grad),
            dict(type="eq", fun=force_magnitude_constraint, jac=force_magnitude_constraint_grad),
        ]

        bounds = [
            parameters.normal_force_bounds,
            parameters.tangential_force_bounds,
            parameters.tangential_force_bounds,
        ]
        bounds *= n_con

        res = minimize(
            fun=objective,
            jac=True,
            constraints=constraints_list,
            x0=np.zeros((3 * n_con)),
            bounds=bounds,
            method="SLSQP",
            options={
                "ftol": parameters.ftol,
                "disp": b_print_opt_details,
                "maxiter": parameters.maxiter,
            },
        )

        diagnostics = diagnose_slsqp_result(res, constraints_list, bounds)
        self._record_solver_diagnostic("wrench_balance", diagnostics)
        if not diagnostics["accepted"]:
            # A failed balance solve cannot justify entering Stage 2 and its
            # contact-force vector must not be consumed by downstream logic.
            return 1.0, None
        cf = res.x.reshape(-1)

        if self.balance_use_normalized:
            metric = np.linalg.norm(self.compute_normalized_wrench(grasp_matrix, cf))
        else:
            metric = np.linalg.norm(grasp_matrix @ cf.reshape(-1, 1))

        return metric, cf

    def _prepare_control_problem(
        self,
        *,
        stage: int,
        dt: float,
        current_qpos_a: np.ndarray,
        current_qpos_f: np.ndarray,
        target_qpos_f: np.ndarray,
        contacts: list[dict[str, Any]] | None,
    ) -> _ControlProblemContext:
        """Build shared dimensions, contact linearization, and target pose.

        Args:
            stage: Current control stage.
            dt: Action interval in seconds.
            current_qpos_a: Current actuator vector.
            current_qpos_f: Current full-joint vector.
            target_qpos_f: Target full-joint vector.
            contacts: Current hand-object contacts.

        Returns:
            Immutable context consumed by either optimization policy.
        """
        num_arm_dof = self.robot.arm.n_dof
        num_hand_dof = self.robot.hand.n_dof
        num_dof = num_arm_dof + num_hand_dof
        contact_list = contacts or []
        num_contacts = len(contact_list)
        contact_jacobian = np.zeros((0, num_dof))
        stiffness_jacobian = np.zeros((0, num_dof))
        contact_forces = np.zeros(0)
        if num_contacts:
            updated_contacts, stacked = self.Ks(
                q_a=current_qpos_a,
                q_f=current_qpos_f,
                contacts=contact_list,
            )
            contact_forces = np.concatenate(
                [contact["contact_force"][:3] for contact in updated_contacts],
                axis=0,
            )
            contact_jacobian = stacked["jaco_a"]
            if stage == 2 and self.stage2_Ks_hand_only:
                stiffness = stacked["Ks_h"]
                contact_jacobian[:, :num_arm_dof] = 0
            else:
                stiffness = stacked["Ks"]
            stiffness_jacobian = stiffness @ contact_jacobian

        hand_base_name = self.robot.hand.base_name
        self.robot_adaptor.compute_fk_f(target_qpos_f)
        target_hand_base_pose = self.robot_adaptor.get_frame_pose(frame_name=hand_base_name)
        target_position, target_orientation = isometry3dToPosOri(target_hand_base_pose)
        return _ControlProblemContext(
            num_arm_dof=num_arm_dof,
            num_hand_dof=num_hand_dof,
            num_dof=num_dof,
            num_contacts=num_contacts,
            doa_to_dof=self.robot_adaptor.doa2dof_matrix,
            joint_limits=self.robot_adaptor.joint_limits_f,
            max_delta_qpos=np.asarray(self.robot.doa_max_vel) * dt,
            contact_forces=contact_forces,
            contact_jacobian=contact_jacobian,
            stiffness_jacobian=stiffness_jacobian,
            hand_base_name=hand_base_name,
            target_hand_base_position=target_position,
            target_hand_base_orientation=target_orientation,
        )

    def _solve_control_problem(
        self,
        *,
        solver_name: str,
        stage: int,
        objective: Callable[[np.ndarray], float],
        jacobian: Callable[[np.ndarray], np.ndarray],
        constraints: list[dict[str, Any]],
        bounds: Sequence[tuple[float | None, float | None]],
        initial_variables: np.ndarray,
        joint_limit_constraint: Callable[[np.ndarray], np.ndarray],
        current_qpos_a: np.ndarray,
        current_contact_forces: np.ndarray,
        num_dof: int,
        print_details: bool,
    ) -> dict[str, np.ndarray]:
        """Solve, diagnose, and safely select one control optimization result.

        Args:
            solver_name: Stable diagnostic identifier.
            stage: Current control stage.
            objective: Scalar objective callback.
            jacobian: Objective-gradient callback.
            constraints: SciPy SLSQP constraint dictionaries.
            bounds: Per-variable lower and upper bounds.
            initial_variables: Initial joint-delta/contact-force vector.
            joint_limit_constraint: Joint-limit residual callback for diagnostics.
            current_qpos_a: Current actuator vector used by rejection fallback.
            current_contact_forces: Measured force vector used by rejection fallback.
            num_dof: Leading joint-delta variable count.
            print_details: Whether SciPy prints solver progress.

        Returns:
            Policy-selected actuator, delta, and contact-force vectors.

        Raises:
            ControlSolveEpisodeAbort: If the selected policy aborts this offset.
        """
        solver_parameters = self._require_parameters().command_solver
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            result = minimize(
                fun=objective,
                jac=jacobian,
                constraints=constraints,
                x0=initial_variables,
                bounds=bounds,
                method="SLSQP",
                options={
                    "ftol": solver_parameters.ftol,
                    "disp": print_details,
                    "maxiter": solver_parameters.maxiter,
                },
            )
        bound_clipping_warning_count = 0
        for warning_message in caught_warnings:
            if (
                issubclass(warning_message.category, RuntimeWarning)
                and str(warning_message.message) == _SLSQP_BOUND_CLIPPING_WARNING
            ):
                bound_clipping_warning_count += 1
                continue
            # Only the documented SLSQP trial-step clipping warning is handled
            # locally. Every unrelated warning remains visible to callers.
            warnings.warn_explicit(
                warning_message.message,
                warning_message.category,
                warning_message.filename,
                warning_message.lineno,
            )
        diagnostics = diagnose_slsqp_result(
            result,
            constraints,
            bounds,
            joint_limit_constraint=joint_limit_constraint,
        )
        diagnostics["bound_clipping_warning_count"] = bound_clipping_warning_count
        failure_policy = getattr(self, "solver_failure_policy", DEFAULT_SOLVER_FAILURE_POLICY)
        decision = select_control_solution(
            current_qpos_a,
            current_contact_forces,
            getattr(result, "x", None),
            num_dof,
            diagnostics,
            failure_policy=failure_policy,
        )
        diagnostics.update(
            {
                "failure_policy": failure_policy,
                "decision": decision.decision,
                "action_applied": decision.action_applied,
                "episode_aborted": decision.episode_aborted,
            }
        )
        self._record_solver_diagnostic(solver_name, diagnostics, stage=stage)
        if decision.episode_aborted:
            raise ControlSolveEpisodeAbort(
                f"Command solver '{solver_name}' was rejected at stage {stage}; "
                f"policy '{failure_policy}' selected episode abort."
            )
        return {
            "q_a": decision.qpos,
            "dq_a": decision.delta_qpos,
            "cf": decision.contact_forces,
        }

    def _build_control_problem(
        self,
        *,
        policy: Literal["coordinated", "equal_contact"],
        context: _ControlProblemContext,
        stage: int,
        dt: float,
        current_qpos_a: np.ndarray,
        target_qpos_f: np.ndarray,
        last_delta_qpos_a: np.ndarray,
        desired_sum_force: float | None,
        desired_forces: np.ndarray | None,
        contacts: list[dict[str, Any]] | None,
        grasp_matrix: np.ndarray | None,
        use_arm_motion: bool,
    ) -> _ControlProblemDefinition:
        """Build one shared SLSQP problem with policy-specific force terms.

        Args:
            policy: Coordinated wrench control or equal-contact force control.
            context: Shared dimensions, kinematics, and contact linearization.
            stage: Current control stage, either 1 or 2.
            dt: Action interval in seconds.
            current_qpos_a: Current actuator vector.
            target_qpos_f: Target full-joint vector.
            last_delta_qpos_a: Previously applied actuator delta.
            desired_sum_force: Desired total normal force when constrained.
            desired_forces: Per-contact desired force vectors for equal-contact control.
            contacts: Current hand-object contacts.
            grasp_matrix: Optional precomputed grasp matrix.
            use_arm_motion: Whether stage 1 may move the arm.

        Returns:
            Complete solver callbacks, constraints, bounds, and initial state.

        Raises:
            ValueError: If ``policy`` or ``stage`` is unsupported.
        """
        if policy not in {"coordinated", "equal_contact"}:
            raise ValueError(f"Unsupported control optimization policy: {policy}")
        if stage not in {1, 2}:
            raise ValueError(f"Unsupported control optimization stage: {stage}")

        coordinated = policy == "coordinated"
        n_arm_dof = context.num_arm_dof
        n_hand_dof = context.num_hand_dof
        n_dof = context.num_dof
        n_con = context.num_contacts
        doa2dof_matrix = context.doa_to_dof
        contact_force_all = context.contact_forces
        contact_jaco_all = context.contact_jacobian
        contact_jaco_h = contact_jaco_all[:, -n_hand_dof:]
        stiffness_jaco = context.stiffness_jacobian

        contact_grasp_matrix = None
        if coordinated and n_con:
            contact_grasp_matrix = self.compute_grasp_matrix(contacts) if grasp_matrix is None else grasp_matrix

        controller_parameters = self._require_parameters()
        weights = controller_parameters.objective_weights
        w_hb_pose = np.diag(weights.hand_base_pose)
        w_q_hand = weights.hand_joint_position * np.eye(n_hand_dof)
        w_dqa = weights.joint_velocity * np.eye(n_dof)
        w_ddqa = weights.joint_acceleration * np.eye(n_dof)
        single_contact_motion_weight = np.diag([0.0, self.tan_motion_pen_weight, self.tan_motion_pen_weight])
        w_cp = block_diag(*[single_contact_motion_weight for _ in range(n_con)])
        single_contact_force_weight = (
            np.diag([0.0, weights.coordinated_tangential_force, weights.coordinated_tangential_force])
            if coordinated
            else weights.equal_contact_force * np.eye(3)
        )
        w_cf = block_diag(*[single_contact_force_weight for _ in range(n_con)])
        w_wrench = np.diag(weights.control_wrench)
        w_equal_joint_force = weights.equal_joint_force * np.eye(n_hand_dof)

        if stage == 2 and n_con:
            in_contact_q_indices = contact_jaco_all.any(axis=0)
            in_contact_qh_indices = contact_jaco_h.any(axis=0)
            if self.stage2_incontact_force_only:
                w_q_hand[in_contact_qh_indices, in_contact_qh_indices] = 0
            if self.stage2_penalize_contact_qda:
                w_dqa[in_contact_q_indices, in_contact_q_indices] *= weights.in_contact_joint_velocity_multiplier

        def objective(x: np.ndarray) -> float:
            """Evaluate the common objective plus policy-specific force costs.

            Args:
                x: Joint-delta variables followed by contact-force variables.

            Returns:
                Scalar weighted objective value.
            """
            dq_a = x[:n_dof].copy()
            cf = x[n_dof:].copy()
            q_a = current_qpos_a + dq_a
            q_f = doa2dof_matrix @ q_a
            q_hand = q_f[n_arm_dof:]

            target_q_hand = target_qpos_f[n_arm_dof:]
            self.err_q_hand = err_q_hand = (q_hand - target_q_hand).reshape(-1, 1)
            cost_q_hand = err_q_hand.T @ w_q_hand @ err_q_hand

            self.err_dqa = err_dqa = (dq_a / dt).reshape(-1, 1)
            cost_dqa = err_dqa.T @ w_dqa @ err_dqa

            self.err_ddqa = err_ddqa = ((dq_a - last_delta_qpos_a) / dt**2).reshape(-1, 1)
            cost_ddqa = err_ddqa.T @ w_ddqa @ err_ddqa

            cost_tan_motion = 0
            cost_policy_force = 0
            cost_tan_cf = 0
            cost_equal_joint_force = 0
            if n_con:
                penalize_tan_motion = stage == 1 or (coordinated and self.stage2_penalize_tan_motion)
                if penalize_tan_motion:
                    self.err_cp = err_cp = contact_jaco_all @ dq_a.reshape(-1, 1)
                    cost_tan_motion = err_cp.T @ w_cp @ err_cp

                if stage == 2 and coordinated:
                    self.wrench = wrench = contact_grasp_matrix @ cf.reshape(-1, 1)
                    cost_policy_force = wrench.T @ w_wrench @ wrench
                    if self.stage2_ctrl_tan_force:
                        predicted_force = contact_force_all.reshape(-1, 1) + stiffness_jaco @ dq_a.reshape(-1, 1)
                        self.err_cf = err_cf = cf.reshape(-1, 1) - predicted_force
                        cost_tan_cf = err_cf.T @ w_cf @ err_cf
                    if self.stage2_equal_joint_force_cost:
                        normal_indices = np.arange(0, n_con * 3, 3)
                        normal_jacobian = contact_jaco_h[normal_indices, :]
                        normal_forces = cf.reshape(-1, 3)[:, 0].reshape(-1, 1)
                        self.err_ef = tau_normal = normal_jacobian.T @ normal_forces
                        cost_equal_joint_force = tau_normal.T @ w_equal_joint_force @ tau_normal
                elif stage == 2:
                    self.err_cf = err_cf = (cf.reshape(-1, 3) - desired_forces).reshape(-1, 1)
                    cost_policy_force = err_cf.T @ w_cf @ err_cf

            cost_hb_pose = 0
            if stage == 1 and use_arm_motion:
                self.robot_adaptor.compute_fk_a(q_a)
                hand_base_pose = self.robot_adaptor.get_frame_pose(frame_name=context.hand_base_name)
                hand_base_position, hand_base_quaternion = isometry3dToPosQuat(hand_base_pose)
                position_error = hand_base_position - context.target_hand_base_position
                hand_base_orientation = sciR.from_quat(hand_base_quaternion)
                orientation_error = (hand_base_orientation * context.target_hand_base_orientation.inv()).as_rotvec()
                self.err_hb_pose = err_hb_pose = np.concatenate([position_error, orientation_error], axis=0).reshape(
                    -1, 1
                )
                cost_hb_pose = err_hb_pose.T @ w_hb_pose @ err_hb_pose

            total_cost = (
                cost_dqa
                + cost_ddqa
                + cost_q_hand
                + cost_tan_motion
                + cost_policy_force
                + cost_tan_cf
                + cost_equal_joint_force
                + cost_hb_pose
            )
            return total_cost.item()

        def jacobian(x: np.ndarray) -> np.ndarray:
            """Evaluate the analytic gradient matching :func:`objective`.

            Args:
                x: Joint-delta variables followed by contact-force variables.

            Returns:
                Dense objective gradient aligned with ``x``.
            """
            dq_a = x[:n_dof].copy()
            q_a = current_qpos_a + dq_a
            gradient = np.zeros(x.shape[0])

            grad_dqa = 2.0 / dt * w_dqa @ self.err_dqa
            gradient[:n_dof] += grad_dqa.reshape(-1)
            grad_ddqa = 2.0 / dt**2 * w_ddqa @ self.err_ddqa
            gradient[:n_dof] += grad_ddqa.reshape(-1)
            hand_mapping = doa2dof_matrix[n_arm_dof:, :]
            grad_q_hand = 2.0 * hand_mapping.T @ w_q_hand @ self.err_q_hand
            gradient[:n_dof] += grad_q_hand.reshape(-1)

            if n_con:
                penalize_tan_motion = stage == 1 or (coordinated and self.stage2_penalize_tan_motion)
                if penalize_tan_motion:
                    grad_tan_motion = 2.0 * contact_jaco_all.T @ w_cp @ self.err_cp
                    gradient[:n_dof] += grad_tan_motion.reshape(-1)

                if stage == 2 and coordinated:
                    grad_wrench = 2 * contact_grasp_matrix.T @ w_wrench @ self.wrench
                    gradient[n_dof:] += grad_wrench.reshape(-1)
                    if self.stage2_ctrl_tan_force:
                        grad_tan_dqa = -2 * stiffness_jaco.T @ w_cf @ self.err_cf
                        grad_tan_cf = 2 * w_cf @ self.err_cf
                        gradient[:n_dof] += grad_tan_dqa.reshape(-1)
                        gradient[n_dof:] += grad_tan_cf.reshape(-1)
                    if self.stage2_equal_joint_force_cost:
                        normal_indices = np.arange(n_con) * 3
                        normal_jacobian = contact_jaco_h[normal_indices, :]
                        grad_equal_force = 2 * normal_jacobian @ w_equal_joint_force @ self.err_ef
                        gradient[n_dof + normal_indices] += grad_equal_force.reshape(-1)
                elif stage == 2:
                    gradient[n_dof:] += (2 * w_cf @ self.err_cf).reshape(-1)

            if stage == 1 and use_arm_motion:
                self.robot_adaptor.compute_jaco_a(q_a)
                hand_base_jacobian = self.robot_adaptor.get_frame_jaco(
                    frame_name=context.hand_base_name,
                    type="space",
                )
                grad_hb_pose = 2.0 * hand_base_jacobian.T @ w_hb_pose @ self.err_hb_pose
                gradient[:n_dof] += grad_hb_pose.reshape(-1)
            return gradient

        if not coordinated:
            full_contact_model = True
        elif stage == 1:
            full_contact_model = self.stage1_tan_force_constraint
        else:
            full_contact_model = self.stage2_tan_force_constraint

        def contact_model_constraint(x: np.ndarray) -> np.ndarray:
            """Match optimized forces to the linearized contact stiffness model.

            Args:
                x: Joint-delta variables followed by contact-force variables.

            Returns:
                Equality residual for full forces or normal forces only.
            """
            dq_a = x[:n_dof].copy()
            force_delta = x[n_dof:].copy() - contact_force_all
            error = force_delta.reshape(-1, 1) - stiffness_jaco @ dq_a.reshape(-1, 1)
            if full_contact_model:
                return error.reshape(-1)
            return error.reshape(-1, 3)[:, 0].reshape(-1)

        def contact_model_constraint_grad(x: np.ndarray) -> np.ndarray:
            """Return the constant contact-model constraint Jacobian.

            Args:
                x: Solver variables, unused because the model is linearized.

            Returns:
                Dense constraint Jacobian aligned with ``x``.
            """
            del x
            if full_contact_model:
                grad_cf = np.eye(3 * n_con)
                grad_dq_a = -stiffness_jaco
            else:
                normal_indices = np.arange(0, n_con * 3, 3)
                grad_cf = np.zeros((n_con, 3 * n_con))
                grad_cf[np.arange(n_con), normal_indices] = 1.0
                grad_dq_a = -stiffness_jaco[normal_indices, :]
            return np.hstack([grad_dq_a, grad_cf])

        def increase_normal_force_constraint(x: np.ndarray) -> np.ndarray:
            """Require nonnegative predicted normal-force increments.

            Args:
                x: Joint-delta variables followed by contact-force variables.

            Returns:
                Per-contact predicted normal-force increments.
            """
            force_delta = stiffness_jaco @ x[:n_dof].reshape(-1, 1)
            return force_delta.reshape(-1, 3)[:, 0].reshape(-1)

        def increase_normal_force_constraint_grad(x: np.ndarray) -> np.ndarray:
            """Return the constant normal-force-increase Jacobian.

            Args:
                x: Solver variables, unused because the model is linearized.

            Returns:
                Dense constraint Jacobian aligned with ``x``.
            """
            del x
            normal_indices = np.arange(0, n_con * 3, 3)
            grad_dq_a = stiffness_jaco[normal_indices, :]
            grad_cf = np.zeros((n_con, 3 * n_con))
            return np.hstack([grad_dq_a, grad_cf])

        def q_limits_constraint(x: np.ndarray) -> np.ndarray:
            """Return nonnegative lower- and upper-joint-limit slack.

            Args:
                x: Joint-delta variables followed by contact-force variables.

            Returns:
                Lower- then upper-limit slack for every full joint.
            """
            q_a = current_qpos_a + x[:n_dof]
            q_f = doa2dof_matrix @ q_a.reshape(-1, 1)
            signs = np.concatenate([-np.eye(n_dof), np.eye(n_dof)], axis=0)
            offsets = np.concatenate(
                [context.joint_limits[0, :], -context.joint_limits[1, :]],
                axis=0,
            ).reshape(-1, 1)
            return -(signs @ q_f + offsets).reshape(-1)

        def q_limits_constraint_grad(x: np.ndarray) -> np.ndarray:
            """Return the constant joint-limit constraint Jacobian.

            Args:
                x: Solver variables used to size the dense Jacobian.

            Returns:
                Dense joint-limit Jacobian aligned with ``x``.
            """
            gradient = np.zeros((2 * n_dof, len(x)))
            signs = np.concatenate([-np.eye(n_dof), np.eye(n_dof)], axis=0)
            gradient[:, :n_dof] = -signs @ doa2dof_matrix
            return gradient

        def friction_cone_constraint(x: np.ndarray) -> np.ndarray:
            """Return nonnegative circular Coulomb friction-cone slack.

            Args:
                x: Joint-delta variables followed by contact-force variables.

            Returns:
                Per-contact friction-cone slack.
            """
            return friction_cone_slack(x[n_dof:].reshape(-1, 3), self.mu)

        def friction_cone_constraint_grad(x: np.ndarray) -> np.ndarray:
            """Return the regularized friction-cone Jacobian used by control.

            Args:
                x: Joint-delta variables followed by contact-force variables.

            Returns:
                Dense friction-cone Jacobian aligned with ``x``.
            """
            contact_forces = x[n_dof:].reshape(-1, 3)
            fy, fz = contact_forces[:, 1], contact_forces[:, 2]
            tangential_norm = np.sqrt(fy**2 + fz**2) + 1e-8
            gradient = np.zeros((n_con, x.shape[0]))
            indices = np.arange(n_con)
            gradient[indices, n_dof + 3 * indices] = self.mu
            gradient[indices, n_dof + 3 * indices + 1] = -fy / tangential_norm
            gradient[indices, n_dof + 3 * indices + 2] = -fz / tangential_norm
            return gradient

        def force_magnitude_constraint(x: np.ndarray) -> np.ndarray:
            """Return the desired total-normal-force residual.

            Args:
                x: Joint-delta variables followed by contact-force variables.

            Returns:
                Single total-normal-force residual.
            """
            contact_forces = x[n_dof:].reshape(-1, 3)
            return np.asarray(desired_sum_force - np.sum(contact_forces[:, 0])).reshape(-1)

        def force_magnitude_constraint_grad(x: np.ndarray) -> np.ndarray:
            """Return the total-normal-force constraint Jacobian.

            Args:
                x: Solver variables used to size the dense Jacobian.

            Returns:
                One-row force-magnitude Jacobian.
            """
            gradient = np.zeros((1, x.shape[0]))
            normal_indices = np.arange(n_con) * 3
            gradient[0, n_dof + normal_indices] = -1.0
            return gradient

        def arm_doa_constraint(x: np.ndarray) -> np.ndarray:
            """Require zero arm actuator delta.

            Args:
                x: Joint-delta variables followed by contact-force variables.

            Returns:
                Arm actuator deltas as equality residuals.
            """
            return x[:n_arm_dof].copy().reshape(-1)

        def arm_doa_constraint_grad(x: np.ndarray) -> np.ndarray:
            """Return the constant frozen-arm constraint Jacobian.

            Args:
                x: Solver variables used to size the dense Jacobian.

            Returns:
                Dense frozen-arm Jacobian aligned with ``x``.
            """
            gradient = np.zeros((n_arm_dof, x.shape[0]))
            gradient[:, :n_arm_dof] = np.eye(n_arm_dof)
            return gradient

        constraints: list[dict[str, Any]] = []
        constraint_names: list[str] = []

        def add_constraint(
            name: str,
            constraint_type: Literal["eq", "ineq"],
            function: Callable[[np.ndarray], np.ndarray],
            gradient: Callable[[np.ndarray], np.ndarray],
        ) -> None:
            """Append one named SciPy constraint while retaining solver order.

            Args:
                name: Stable test and diagnostic name.
                constraint_type: SciPy equality or inequality type.
                function: Constraint residual callback.
                gradient: Constraint Jacobian callback.

            Returns:
                None.
            """
            constraint_names.append(name)
            constraints.append(dict(type=constraint_type, fun=function, jac=gradient))

        add_constraint("q_limits", "ineq", q_limits_constraint, q_limits_constraint_grad)
        if stage == 1:
            if n_con:
                add_constraint("contact_model", "eq", contact_model_constraint, contact_model_constraint_grad)
                add_constraint("force_magnitude", "ineq", force_magnitude_constraint, force_magnitude_constraint_grad)
            if coordinated and not use_arm_motion:
                add_constraint("arm_doa", "eq", arm_doa_constraint, arm_doa_constraint_grad)
        else:
            add_constraint("arm_doa", "eq", arm_doa_constraint, arm_doa_constraint_grad)
            if n_con:
                add_constraint("contact_model", "eq", contact_model_constraint, contact_model_constraint_grad)
                if coordinated:
                    add_constraint("friction_cone", "ineq", friction_cone_constraint, friction_cone_constraint_grad)
                    add_constraint("force_magnitude", "eq", force_magnitude_constraint, force_magnitude_constraint_grad)
                    if self.stage2_increase_force:
                        add_constraint(
                            "increase_normal_force",
                            "ineq",
                            increase_normal_force_constraint,
                            increase_normal_force_constraint_grad,
                        )

        bounds_dq = [(-limit, limit) for limit in context.max_delta_qpos]
        solver_parameters = controller_parameters.command_solver
        bounds_cf = [
            solver_parameters.normal_force_bounds,
            solver_parameters.tangential_force_bounds,
            solver_parameters.tangential_force_bounds,
        ] * n_con
        return _ControlProblemDefinition(
            objective=objective,
            jacobian=jacobian,
            constraints=constraints,
            constraint_names=tuple(constraint_names),
            bounds=bounds_dq + bounds_cf,
            initial_variables=np.concatenate([np.zeros(n_dof), contact_force_all], axis=0),
            joint_limit_constraint=q_limits_constraint,
        )

    def _optimize_control(
        self,
        *,
        policy: Literal["coordinated", "equal_contact"],
        solver_name: str,
        stage: int,
        dt: float,
        current_qpos_a: np.ndarray,
        current_qpos_f: np.ndarray,
        target_qpos_f: np.ndarray,
        last_delta_qpos_a: np.ndarray,
        desired_sum_force: float | None,
        desired_forces: np.ndarray | None,
        contacts: list[dict[str, Any]] | None,
        grasp_matrix: np.ndarray | None,
        use_arm_motion: bool,
        print_details: bool,
    ) -> dict[str, np.ndarray]:
        """Prepare, build, solve, and validate either control policy.

        Args:
            policy: Coordinated wrench control or equal-contact force control.
            solver_name: Stable diagnostic identifier.
            stage: Current control stage.
            dt: Action interval in seconds.
            current_qpos_a: Current actuator vector.
            current_qpos_f: Current full-joint vector.
            target_qpos_f: Target full-joint vector.
            last_delta_qpos_a: Previously applied actuator delta.
            desired_sum_force: Desired total normal force when constrained.
            desired_forces: Per-contact desired forces for equal-contact control.
            contacts: Current hand-object contacts.
            grasp_matrix: Optional precomputed grasp matrix.
            use_arm_motion: Whether stage 1 may move the arm.
            print_details: Whether SciPy prints solver progress.

        Returns:
            Policy-selected actuator, delta, and contact-force vectors.

        Raises:
            ControlSolveEpisodeAbort: If the selected policy aborts this offset.
        """
        context = self._prepare_control_problem(
            stage=stage,
            dt=dt,
            current_qpos_a=current_qpos_a,
            current_qpos_f=current_qpos_f,
            target_qpos_f=target_qpos_f,
            contacts=contacts,
        )
        problem = self._build_control_problem(
            policy=policy,
            context=context,
            stage=stage,
            dt=dt,
            current_qpos_a=current_qpos_a,
            target_qpos_f=target_qpos_f,
            last_delta_qpos_a=last_delta_qpos_a,
            desired_sum_force=desired_sum_force,
            desired_forces=desired_forces,
            contacts=contacts,
            grasp_matrix=grasp_matrix,
            use_arm_motion=use_arm_motion,
        )
        return self._solve_control_problem(
            solver_name=solver_name,
            stage=stage,
            objective=problem.objective,
            jacobian=problem.jacobian,
            constraints=problem.constraints,
            bounds=problem.bounds,
            initial_variables=problem.initial_variables,
            joint_limit_constraint=problem.joint_limit_constraint,
            current_qpos_a=current_qpos_a,
            current_contact_forces=context.contact_forces,
            num_dof=context.num_dof,
            print_details=print_details,
        )

    def ctrl_opt(
        self,
        stage: int,
        dt: float,
        curr_q_a: np.ndarray,
        curr_q_f: np.ndarray,
        target_q_f: np.ndarray,
        desired_sum_force: float,
        last_dq_a: np.ndarray,
        ho_contacts: list[dict[str, Any]] | None = None,
        grasp_matrix: np.ndarray | None = None,
        b_use_arm_motion: bool = True,
        b_print_opt_details: bool = False,
    ) -> dict[str, np.ndarray]:
        """Run coordinated wrench control through the shared optimizer.

        Args:
            stage: Current control stage.
            dt: Action interval in seconds.
            curr_q_a: Current actuator vector.
            curr_q_f: Current full-joint vector.
            target_q_f: Target full-joint vector.
            desired_sum_force: Desired total normal force.
            last_dq_a: Previously applied actuator delta.
            ho_contacts: Current hand-object contacts.
            grasp_matrix: Optional precomputed grasp matrix.
            b_use_arm_motion: Whether stage 1 may move the arm.
            b_print_opt_details: Whether SciPy prints solver progress.

        Returns:
            Policy-selected actuator, delta, and contact-force vectors.

        Raises:
            ControlSolveEpisodeAbort: If the selected policy aborts this offset.
        """
        return self._optimize_control(
            policy="coordinated",
            solver_name="control",
            stage=stage,
            dt=dt,
            current_qpos_a=curr_q_a,
            current_qpos_f=curr_q_f,
            target_qpos_f=target_q_f,
            last_delta_qpos_a=last_dq_a,
            desired_sum_force=desired_sum_force,
            desired_forces=None,
            contacts=ho_contacts,
            grasp_matrix=grasp_matrix,
            use_arm_motion=b_use_arm_motion,
            print_details=b_print_opt_details,
        )

    def ctrl_opt_bs3(
        self,
        stage: int,
        dt: float,
        curr_q_a: np.ndarray,
        curr_q_f: np.ndarray,
        target_q_f: np.ndarray,
        last_dq_a: np.ndarray,
        desired_sum_force: float | None = None,
        desired_forces: np.ndarray | None = None,
        ho_contacts: list[dict[str, Any]] | None = None,
        grasp_matrix: np.ndarray | None = None,
        b_print_opt_details: bool = False,
    ) -> dict[str, np.ndarray]:
        """Run equal-contact force control through the shared optimizer.

        Args:
            stage: Current control stage.
            dt: Action interval in seconds.
            curr_q_a: Current actuator vector.
            curr_q_f: Current full-joint vector.
            target_q_f: Target full-joint vector.
            last_dq_a: Previously applied actuator delta.
            desired_sum_force: Desired stage-1 total normal force.
            desired_forces: Per-contact desired stage-2 force vectors.
            ho_contacts: Current hand-object contacts.
            grasp_matrix: Accepted for API symmetry; not used by this policy.
            b_print_opt_details: Whether SciPy prints solver progress.

        Returns:
            Policy-selected actuator, delta, and contact-force vectors.

        Raises:
            ControlSolveEpisodeAbort: If the selected policy aborts this offset.
        """
        return self._optimize_control(
            policy="equal_contact",
            solver_name="control_bs3",
            stage=stage,
            dt=dt,
            current_qpos_a=curr_q_a,
            current_qpos_f=curr_q_f,
            target_qpos_f=target_q_f,
            last_delta_qpos_a=last_dq_a,
            desired_sum_force=desired_sum_force,
            desired_forces=desired_forces,
            contacts=ho_contacts,
            grasp_matrix=grasp_matrix,
            use_arm_motion=True,
            print_details=b_print_opt_details,
        )
