import numpy as np

from .base import BaseEval
from .episode_runner import (
    DummyArmEpisodeRunner,
    EpisodeStep,
    StepControl,
    final_single_contact_force,
    run_dummy_arm_episode,
)

"""
Baseline1:
    Position + Force feedback control.
    Each finger is controlled independently.
    Before contact, position control; After contact, force control.
    The desired contact force of each finger is set to the same pre-defined value.
"""


class _IndependentForceFeedbackPolicy:
    """Combine per-joint position and independent contact-force feedback."""

    def initialize(self, runner: DummyArmEpisodeRunner) -> None:
        """Cache published gains and the fixed arm target.

        Args:
            runner: Prepared common episode runner.

        Returns:
            None.
        """
        self.debug = runner.evaluator.configs.task.debug_viewer
        self.arm_ndoa = runner.robot.arm.n_doa
        self.hand_ndoa = runner.robot.hand.n_doa
        self.hand_kp_inverse = runner.grasp_ctrl.Kp_inv[-self.hand_ndoa :, -self.hand_ndoa :]
        self.initial_qpos_a = runner.initial_qpos_a
        self.final_single_force = final_single_contact_force(runner.robot.name)

    def max_steps(self, runner: DummyArmEpisodeRunner) -> int:
        """Retain the baseline's 20-percent path overrun.

        Args:
            runner: Prepared common episode runner.

        Returns:
            Maximum number of control actions.
        """
        return int(len(runner.qpos_path) * 1.2)

    def should_stop(self, runner: DummyArmEpisodeRunner, step: EpisodeStep) -> bool:
        """Run until the common maximum because BS1 has no early stop.

        Args:
            runner: Prepared common episode runner.
            step: Current sampled state.

        Returns:
            Always ``False``.
        """
        del runner, step
        return False

    def control(self, runner: DummyArmEpisodeRunner, step: EpisodeStep) -> StepControl:
        """Compute the historical independent position/force command.

        Args:
            runner: Prepared common episode runner.
            step: Current sampled state.

        Returns:
            Full actuator target without extra diagnostics.
        """
        if self.debug:
            runner.print_step_contacts(step)

        current_hand_qpos_a = step.current_qpos_a[-self.hand_ndoa :]
        target_qpos_a = runner.robot_adaptor._dof2doa(step.target_qpos_f)
        target_hand_qpos_a = target_qpos_a[-self.hand_ndoa :]
        hand_qpos_error = (target_hand_qpos_a - current_hand_qpos_a).reshape(-1, 1)
        joint_weights = np.ones_like(current_hand_qpos_a)
        position_gain = np.eye(self.hand_ndoa)

        num_contacts = len(step.contacts)
        if num_contacts:
            updated_contacts, stacked = runner.grasp_ctrl.Ks(
                step.current_qpos_a,
                step.current_qpos_f,
                step.contacts,
            )
            contact_forces = np.concatenate(
                [contact["contact_force"][:3] for contact in updated_contacts],
                axis=0,
            ).reshape(-1, 1)
            contact_jacobian = stacked["jaco_hf"]
            target_contact_forces = np.tile(
                np.array([self.final_single_force, 0, 0]),
                (num_contacts, 1),
            ).reshape(-1, 1)
            contact_force_error = target_contact_forces - contact_forces
            in_contact_joint_indices = np.any(contact_jacobian != 0, axis=0)
            joint_weights[in_contact_joint_indices] = 0

        joint_weight_matrix = np.diag(joint_weights)
        delta_hand_qpos_a = joint_weight_matrix @ position_gain @ hand_qpos_error
        if num_contacts:
            force_gain = min(0.8, 1.0 / (len(runner.qpos_path) - step.waypoint_index))
            force_control_input = force_gain * self.hand_kp_inverse @ contact_jacobian.T @ contact_force_error
            delta_hand_qpos_a += force_control_input
            if self.debug:
                print(f"pos_control_input: {(joint_weight_matrix @ position_gain @ hand_qpos_error).reshape(-1)}")
                print(f"force_control_input: {force_control_input.reshape(-1)}")

        optimized_hand_qpos_a = current_hand_qpos_a + delta_hand_qpos_a.reshape(-1)
        optimized_qpos_a = np.concatenate(
            [self.initial_qpos_a[: self.arm_ndoa], optimized_hand_qpos_a],
            axis=0,
        )
        return StepControl(optimized_qpos_a)


class tabletopDummyArmBS1Eval(BaseEval):
    def _initialize(self):
        """Initialize the BS1 controller.

        Returns:
            None.
        """
        self._initialize_controller("bs1")

    def _simulate_under_extforce_details(self, pregrasp_qpos, grasp_qpos, squeeze_qpos):
        """Run BS1 inside the common episode lifecycle.

        Args:
            pregrasp_qpos: Initial qpos.
            grasp_qpos: In-grasp target qpos.
            squeeze_qpos: Squeezed target qpos.

        Returns:
            None.
        """
        run_dummy_arm_episode(
            self,
            pregrasp_qpos,
            grasp_qpos,
            squeeze_qpos,
            _IndependentForceFeedbackPolicy(),
        )
