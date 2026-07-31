import time
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
Baseline2:
    Use equal desired forces for each contact in grasping phase. Other settings are the same as those of ours.
"""


class _EqualContactForcePolicy:
    """Use the shared optimization with an equal target per active contact."""

    def initialize(self, runner: DummyArmEpisodeRunner) -> None:
        """Initialize force targets, stage state, and previous motion.

        Args:
            runner: Prepared common episode runner.

        Returns:
            None.
        """
        self.debug = runner.evaluator.configs.task.debug_viewer
        self.control_config = runner.evaluator.configs.task.control
        self.action_dt = runner.evaluator.action_dt
        self.final_single_force = final_single_contact_force(runner.robot.name)
        self.force_increment = self.final_single_force / len(runner.path_squeeze)
        self.last_delta_qpos_a = np.zeros(runner.robot.n_doa)
        self.stage = 1

    def max_steps(self, runner: DummyArmEpisodeRunner) -> int:
        """Retain the baseline's 20-percent path overrun.

        Args:
            runner: Prepared common episode runner.

        Returns:
            Maximum number of control actions.
        """
        return int(len(runner.qpos_path) * 1.2)

    def should_stop(self, runner: DummyArmEpisodeRunner, step: EpisodeStep) -> bool:
        """Stop after the path when every active contact exceeds its target.

        Args:
            runner: Prepared common episode runner.
            step: Current sampled state.

        Returns:
            Whether the historical BS3 terminal criterion is satisfied.
        """
        return step.index >= len(runner.qpos_path) and bool(np.all(step.contact_forces[:, 0] > self.final_single_force))

    def control(self, runner: DummyArmEpisodeRunner, step: EpisodeStep) -> StepControl:
        """Compute one equal-contact-force optimization command.

        Args:
            runner: Prepared common episode runner.
            step: Current sampled state.

        Returns:
            Optimized actuator target and historical timing diagnostics.
        """
        grasp_matrix = runner.grasp_ctrl.compute_grasp_matrix(step.contacts)
        start = time.time()
        balance_metric, _ = runner.grasp_ctrl.check_wrench_balance(
            grasp_matrix,
            b_print_opt_details=False,
        )
        balance_elapsed = time.time() - start

        if self.debug:
            runner.print_step_contacts(step)
            print(f"check_wrench_balance() time cost: {balance_elapsed}")
            print(f"balance_metric: {balance_metric}")

        if balance_metric < runner.grasp_ctrl.balance_thres or (
            self.control_config.stage2_after_full_path and step.index > len(runner.qpos_path)
        ):
            self.stage = 2
        elif self.control_config.free_stage_switch:
            self.stage = 1

        if self.stage == 1:
            desired_sum_force = max(
                runner.grasp_ctrl.stage1_force_thres,
                step.current_sum_force - self.force_increment,
            )
            desired_forces = None
        else:
            desired_sum_force = None
            num_contacts = step.contact_forces.shape[0]
            final_desired_forces = np.tile(
                np.array([self.final_single_force, 0, 0]),
                [num_contacts, 1],
            )
            step_size = min(0.8, 1.0 / (len(runner.qpos_path) - step.waypoint_index))
            desired_forces = step.contact_forces + step_size * (final_desired_forces - step.contact_forces)

        start = time.time()
        result = runner.grasp_ctrl.ctrl_opt_bs3(
            stage=self.stage,
            dt=self.action_dt,
            curr_q_a=step.current_qpos_a,
            curr_q_f=step.current_qpos_f,
            target_q_f=step.target_qpos_f,
            desired_sum_force=desired_sum_force,
            desired_forces=desired_forces,
            last_dq_a=self.last_delta_qpos_a,
            ho_contacts=step.contacts,
            grasp_matrix=grasp_matrix,
            b_print_opt_details=self.debug,
        )
        control_elapsed = time.time() - start
        self.last_delta_qpos_a = result["dq_a"]
        return StepControl(
            result["q_a"],
            diagnostics={
                "balance_metric": balance_metric,
                "t_check_balance": balance_elapsed,
                "t_ctrl_opt": control_elapsed,
            },
        )


class tabletopDummyArmBS3Eval(BaseEval):
    def _initialize(self):
        """Initialize the BS3 controller.

        Returns:
            None.
        """
        self._initialize_controller("bs3")

    def _simulate_under_extforce_details(self, pregrasp_qpos, grasp_qpos, squeeze_qpos):
        """Run BS3 inside the common episode lifecycle.

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
            _EqualContactForcePolicy(),
        )
