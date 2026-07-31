"""Unit contracts for the shared dummy-arm episode lifecycle."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from ada_grasp_ctrl.tasks.control_eval_func.episode_runner import (
    DummyArmEpisodeRunner,
    EpisodeStep,
    StepControl,
)


class _FakeAdaptor:
    """Identity actuator/joint projection used by the runner fixture."""

    def _dof2doa(self, qpos):
        """Return an independent actuator-vector copy.

        Args:
            qpos: Full-joint vector.

        Returns:
            Copied actuator vector.
        """
        return np.asarray(qpos).copy()


class _FakeController:
    """Provide interpolation and trajectory storage without optimization."""

    def __init__(self):
        """Initialize every field touched by the common runner.

        Returns:
            None.
        """
        self.r_data = {
            "obj_pose": [],
            "dof": [],
            "doa": [],
            "contacts": [],
            "planned_dof": [],
            "balance_metric": [],
        }

    def interplote_qpos(self, qpos1, qpos2, step):
        """Match the production endpoint-excluding-start interpolation.

        Args:
            qpos1: Starting vector.
            qpos2: Target vector.
            step: Number of returned waypoints.

        Returns:
            Interpolated path without the starting vector.
        """
        return np.linspace(qpos1, qpos2, step + 1)[1:]


class _FakeMjHO:
    """Minimal simulator facade that records applied actuator commands."""

    def __init__(self):
        """Initialize a two-joint state and command history.

        Returns:
            None.
        """
        self.qpos_f = np.zeros(2)
        self.qpos_a = np.zeros(2)
        self.commands = []

    def get_qpos_f(self, names):
        """Return the current full-joint state.

        Args:
            names: Expected joint names.

        Returns:
            Copied full-joint vector.
        """
        del names
        return self.qpos_f.copy()

    def get_qpos_a(self):
        """Return the current actuator state."""
        return self.qpos_a.copy()

    def ctrl_qpos_a(self, names, qpos):
        """Apply the runner's initial direct actuator synchronization.

        Args:
            names: Expected actuator names.
            qpos: Initial actuator vector.

        Returns:
            None.
        """
        del names
        self.qpos_a = np.asarray(qpos).copy()
        self.qpos_f = np.asarray(qpos).copy()

    def get_curr_contact_info(self):
        """Return one deterministic contact."""
        return [
            {
                "body1_name": "finger",
                "body2_name": "object",
                "contact_force": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            }
        ]

    def get_obj_pose(self):
        """Return a deterministic object pose."""
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def ctrl_qpos_a_with_interp(self, current, target, names, step_outer, step_inner):
        """Capture one interpolated action and update simulator state.

        Args:
            current: Pre-action actuator vector.
            target: Target actuator vector.
            names: Expected actuator names.
            step_outer: Number of outer interpolation steps.
            step_inner: Number of MuJoCo steps per outer step.

        Returns:
            None.
        """
        del names
        self.commands.append((np.asarray(current).copy(), np.asarray(target).copy(), step_outer, step_inner))
        self.qpos_a = np.asarray(target).copy()
        self.qpos_f = np.asarray(target).copy()


class _TwoStepPolicy:
    """Simple policy that proves hook order and diagnostic recording."""

    def initialize(self, runner: DummyArmEpisodeRunner) -> None:
        """Record that initialization occurred after path preparation.

        Args:
            runner: Prepared common episode runner.

        Returns:
            None.
        """
        self.initialized_path_length = len(runner.qpos_path)

    def max_steps(self, runner: DummyArmEpisodeRunner) -> int:
        """Limit the fixture to two actions.

        Args:
            runner: Prepared common episode runner.

        Returns:
            Two actions.
        """
        del runner
        return 2

    def should_stop(self, runner: DummyArmEpisodeRunner, step: EpisodeStep) -> bool:
        """Allow both configured actions.

        Args:
            runner: Prepared common episode runner.
            step: Current sampled state.

        Returns:
            Always ``False``.
        """
        del runner, step
        return False

    def control(self, runner: DummyArmEpisodeRunner, step: EpisodeStep) -> StepControl:
        """Offset each waypoint and emit one diagnostic.

        Args:
            runner: Prepared common episode runner.
            step: Current sampled state.

        Returns:
            Offset target plus the step index as a balance metric.
        """
        del runner
        return StepControl(
            step.target_qpos_f + 1.0,
            diagnostics={"balance_metric": float(step.index)},
        )


class EpisodeRunnerContractTest(unittest.TestCase):
    """Verify the runner owns setup, sampling, stepping, and base recording."""

    def test_common_lifecycle_prepares_path_and_records_pre_action_state(self):
        """Execute policy hooks while keeping trajectories exactly aligned."""
        mj_ho = _FakeMjHO()
        controller = _FakeController()
        evaluator = SimpleNamespace(
            mj_ho=mj_ho,
            robot=SimpleNamespace(dof_names=["j0", "j1"], doa_names=["a0", "a1"]),
            robot_adaptor=_FakeAdaptor(),
            grasp_ctrl=controller,
            sim_step_per_action=10,
            ctrl_freq=1,
            _dof_data2user=lambda value: np.asarray(value).copy(),
        )
        policy = _TwoStepPolicy()
        runner = DummyArmEpisodeRunner(
            evaluator,
            grasp_qpos=np.ones(2),
            squeeze_qpos=np.full(2, 2.0),
        )

        runner.run(policy)

        self.assertEqual(policy.initialized_path_length, 4)
        self.assertEqual(len(mj_ho.commands), 2)
        np.testing.assert_array_equal(controller.r_data["planned_dof"][0], np.full(2, 0.5))
        np.testing.assert_array_equal(controller.r_data["planned_dof"][1], np.ones(2))
        np.testing.assert_array_equal(controller.r_data["dof"][0], np.zeros(2))
        np.testing.assert_array_equal(controller.r_data["dof"][1], np.full(2, 1.5))
        self.assertEqual(controller.r_data["balance_metric"], [0.0, 1.0])
        self.assertEqual(mj_ho.commands[0][2:], (2, 5))


if __name__ == "__main__":
    unittest.main()
