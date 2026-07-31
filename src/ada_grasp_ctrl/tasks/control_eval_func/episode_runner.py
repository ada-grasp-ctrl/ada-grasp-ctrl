"""Shared dummy-arm approach and squeeze episode lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class EpisodeStep:
    """State sampled immediately before one control action."""

    index: int
    waypoint_index: int
    target_qpos_f: np.ndarray
    current_qpos_f: np.ndarray
    current_qpos_a: np.ndarray
    contacts: list[dict[str, Any]]
    object_pose: np.ndarray
    contact_forces: np.ndarray
    current_sum_force: float


@dataclass(frozen=True)
class StepControl:
    """One policy command plus diagnostic values recorded for that step."""

    target_qpos_a: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


class EpisodePolicy(Protocol):
    """Strategy hooks consumed by :class:`DummyArmEpisodeRunner`."""

    def initialize(self, runner: "DummyArmEpisodeRunner") -> None:
        """Initialize policy state from a prepared runner."""

    def max_steps(self, runner: "DummyArmEpisodeRunner") -> int:
        """Return the maximum number of control actions."""

    def should_stop(self, runner: "DummyArmEpisodeRunner", step: EpisodeStep) -> bool:
        """Return whether the episode should stop before the current action."""

    def control(self, runner: "DummyArmEpisodeRunner", step: EpisodeStep) -> StepControl:
        """Compute the next actuator target and diagnostics."""


class DummyArmEpisodeRunner:
    """Own common initialization, interpolation, sampling, stepping, and recording."""

    def __init__(self, evaluator: Any, grasp_qpos: np.ndarray, squeeze_qpos: np.ndarray):
        """Prepare the shared approach and squeeze trajectory.

        Args:
            evaluator: Initialized :class:`BaseEval` policy evaluator.
            grasp_qpos: Stored in-grasp target qpos.
            squeeze_qpos: Stored squeezed target qpos.

        Returns:
            None.
        """
        self.evaluator = evaluator
        self.mj_ho = evaluator.mj_ho
        self.robot = evaluator.robot
        self.robot_adaptor = evaluator.robot_adaptor
        self.grasp_ctrl = evaluator.grasp_ctrl
        self.sim_step_per_action = evaluator.sim_step_per_action

        current_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
        initial_qpos_a = self.robot_adaptor._dof2doa(current_qpos_f)
        self.mj_ho.ctrl_qpos_a(self.robot.doa_names, initial_qpos_a)
        self.initial_qpos_a = initial_qpos_a.copy()

        current_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
        grasp_qpos_f = evaluator._dof_data2user(grasp_qpos)
        squeeze_qpos_f = evaluator._dof_data2user(squeeze_qpos)
        self.path_approach = self.grasp_ctrl.interplote_qpos(
            current_qpos_f,
            grasp_qpos_f,
            step=evaluator.ctrl_freq * 2,
        )
        self.path_squeeze = self.grasp_ctrl.interplote_qpos(
            grasp_qpos_f,
            squeeze_qpos_f,
            step=evaluator.ctrl_freq * 2,
        )
        self.qpos_path = np.concatenate([self.path_approach, self.path_squeeze], axis=0)

    def sample_step(self, index: int, waypoint_index: int) -> EpisodeStep:
        """Sample simulator state for one policy decision.

        Args:
            index: Zero-based action index.
            waypoint_index: Current path waypoint index.

        Returns:
            Immutable step state.
        """
        contacts = self.mj_ho.get_curr_contact_info()
        contact_forces = np.array([contact["contact_force"][:3] for contact in contacts]).reshape(-1, 3)
        return EpisodeStep(
            index=index,
            waypoint_index=waypoint_index,
            target_qpos_f=self.qpos_path[waypoint_index],
            current_qpos_f=self.mj_ho.get_qpos_f(names=self.robot.dof_names),
            current_qpos_a=self.mj_ho.get_qpos_a(),
            contacts=contacts,
            object_pose=self.mj_ho.get_obj_pose(),
            contact_forces=contact_forces,
            current_sum_force=float(np.sum(contact_forces[:, 0])),
        )

    def apply_control(self, step: EpisodeStep, control: StepControl) -> None:
        """Interpolate one actuator command and record its pre-action state.

        Args:
            step: State captured before the action.
            control: Policy actuator target and diagnostic values.

        Returns:
            None.
        """
        if self.sim_step_per_action % 5 != 0:
            raise ValueError("sim_step_per_action must be divisible by 5.")
        self.mj_ho.ctrl_qpos_a_with_interp(
            step.current_qpos_a,
            control.target_qpos_a,
            names=self.robot.doa_names,
            step_outer=self.sim_step_per_action // 5,
            step_inner=5,
        )

        record = self.grasp_ctrl.r_data
        record["obj_pose"].append(step.object_pose)
        record["dof"].append(step.current_qpos_f)
        record["doa"].append(step.current_qpos_a)
        record["contacts"].append(step.contacts)
        record["planned_dof"].append(step.target_qpos_f)
        for field_name, value in control.diagnostics.items():
            if field_name not in record:
                raise KeyError(f"Unknown control diagnostic field: {field_name}.")
            record[field_name].append(value)

    def print_step_contacts(self, step: EpisodeStep) -> None:
        """Print one debug snapshot using the historical contact details.

        Args:
            step: State captured before the action.

        Returns:
            None.
        """
        print(f"--------------- {step.index} step ---------------")
        for contact in step.contacts:
            print(
                f"{step.index} step, body1_name: {contact['body1_name']}, "
                f"body2_name: {contact['body2_name']}, contact_force: {contact['contact_force']}"
            )
        print(f"curr_sum_force: {step.current_sum_force}")

    def run(self, policy: EpisodePolicy) -> None:
        """Execute a policy inside the common per-action lifecycle.

        Args:
            policy: Method-specific stopping and control hooks.

        Returns:
            None.
        """
        policy.initialize(self)
        step_index = 0
        waypoint_index = 0
        max_steps = policy.max_steps(self)
        while step_index < max_steps:
            step = self.sample_step(step_index, waypoint_index)
            if policy.should_stop(self, step):
                break
            control = policy.control(self, step)
            self.apply_control(step, control)
            step_index += 1
            waypoint_index = min(waypoint_index + 1, len(self.qpos_path) - 1)


def run_dummy_arm_episode(
    evaluator: Any,
    pregrasp_qpos: np.ndarray,
    grasp_qpos: np.ndarray,
    squeeze_qpos: np.ndarray,
    policy: EpisodePolicy,
) -> None:
    """Prepare and execute one method-specific approach/squeeze policy.

    Args:
        evaluator: Initialized :class:`BaseEval` policy evaluator.
        pregrasp_qpos: Initial qpos already applied by :class:`BaseEval`.
        grasp_qpos: Stored in-grasp target qpos.
        squeeze_qpos: Stored squeezed target qpos.
        policy: Method-specific stopping and control hooks.

    Returns:
        None.
    """
    del pregrasp_qpos
    DummyArmEpisodeRunner(evaluator, grasp_qpos, squeeze_qpos).run(policy)


def final_single_contact_force(robot_name: str) -> float:
    """Return the published per-contact force target for one hand.

    Args:
        robot_name: Registered robot name containing the hand family.

    Returns:
        Target normal force in newtons.

    Raises:
        NotImplementedError: If the hand has no published baseline target.
    """
    if "shadow" in robot_name:
        return 5.0
    if "allegro" in robot_name:
        return 3.0
    if "leap" in robot_name:
        return 2.5
    raise NotImplementedError(f"No per-contact force target for robot '{robot_name}'.")
