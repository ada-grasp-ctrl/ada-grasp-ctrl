"""Shared optimized approach/squeeze runner for ours and BS2 policies."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .episode_runner import (
    DummyArmEpisodeRunner,
    EpisodeStep,
    StepControl,
    run_dummy_arm_episode,
)


@dataclass(frozen=True)
class WrenchControlPolicy:
    """Policy differences for the shared wrench-control episode."""

    use_approaching_arm_motion: bool


class _CoordinatedWrenchPolicy:
    """Implement ours/BS2 as one strategy with an arm-motion flag."""

    def __init__(self, settings: WrenchControlPolicy):
        """Store the sole public policy difference.

        Args:
            settings: Coordinated wrench-control policy settings.

        Returns:
            None.
        """
        self.settings = settings

    def initialize(self, runner: DummyArmEpisodeRunner) -> None:
        """Initialize stage and force-trajectory state.

        Args:
            runner: Prepared common episode runner.

        Returns:
            None.
        """
        evaluator = runner.evaluator
        self.control_config = evaluator.configs.task.control
        evaluator.desired_sum_force_as_traj = self.control_config.desired_sum_force_as_traj
        evaluator.use_desired_sum_force_ub = self.control_config.use_desired_sum_force_ub
        self.desired_sum_force_as_traj = evaluator.desired_sum_force_as_traj
        self.use_desired_sum_force_ub = evaluator.use_desired_sum_force_ub
        self.action_dt = evaluator.action_dt
        self.debug = evaluator.configs.task.debug_viewer
        self.final_sum_force = runner.grasp_ctrl.final_sum_force
        self.desired_sum_force_upper = self.final_sum_force + 0.2
        self.force_increment = self.final_sum_force / len(runner.path_squeeze)
        self.last_delta_qpos_a = np.zeros(runner.robot.n_doa)
        self.stage = 1
        # Preserve the historical first-step Stage-2 edge case.
        self.desired_sum_force = runner.grasp_ctrl.stage1_force_thres

    def max_steps(self, runner: DummyArmEpisodeRunner) -> int:
        """Allow at most twice the interpolated path length.

        Args:
            runner: Prepared common episode runner.

        Returns:
            Maximum number of control actions.
        """
        return len(runner.qpos_path) * 2

    def should_stop(self, runner: DummyArmEpisodeRunner, step: EpisodeStep) -> bool:
        """Stop after the path once total normal force exceeds its target.

        Args:
            runner: Prepared common episode runner.
            step: Current sampled state.

        Returns:
            Whether the published terminal criterion is satisfied.
        """
        return step.index >= len(runner.qpos_path) and step.current_sum_force > self.final_sum_force

    def control(self, runner: DummyArmEpisodeRunner, step: EpisodeStep) -> StepControl:
        """Compute one coordinated wrench-control optimization command.

        Args:
            runner: Prepared common episode runner.
            step: Current sampled state.

        Returns:
            Optimized actuator target and full historical diagnostics.
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
            self.desired_sum_force = max(
                runner.grasp_ctrl.stage1_force_thres,
                step.current_sum_force - self.force_increment,
            )
        elif self.desired_sum_force_as_traj:
            self.desired_sum_force = max(step.current_sum_force, self.desired_sum_force) + self.force_increment
            if self.use_desired_sum_force_ub:
                self.desired_sum_force = min(
                    self.desired_sum_force,
                    self.desired_sum_force_upper,
                )
        else:
            self.desired_sum_force = min(step.current_sum_force, self.final_sum_force) + self.force_increment

        start = time.time()
        result = runner.grasp_ctrl.ctrl_opt(
            stage=self.stage,
            dt=self.action_dt,
            curr_q_a=step.current_qpos_a,
            curr_q_f=step.current_qpos_f,
            target_q_f=step.target_qpos_f,
            desired_sum_force=self.desired_sum_force,
            last_dq_a=self.last_delta_qpos_a,
            ho_contacts=step.contacts,
            grasp_matrix=grasp_matrix,
            b_use_arm_motion=self.settings.use_approaching_arm_motion,
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
                "stage": self.stage,
                "opt_res": result,
                "desired_sum_force": self.desired_sum_force,
            },
        )


def run_wrench_control_episode(evaluator, pregrasp_qpos, grasp_qpos, squeeze_qpos, policy):
    """Run the common optimized strategy inside the shared episode lifecycle.

    Args:
        evaluator: Initialized :class:`BaseEval` subclass instance.
        pregrasp_qpos: Prepared pregrasp qpos retained for API symmetry.
        grasp_qpos: Target in-grasp qpos.
        squeeze_qpos: Target squeezed qpos.
        policy: Only the method-specific approaching-arm decision.

    Returns:
        None.
    """
    run_dummy_arm_episode(
        evaluator,
        pregrasp_qpos,
        grasp_qpos,
        squeeze_qpos,
        _CoordinatedWrenchPolicy(policy),
    )
