"""Unit contracts for the shared dummy-arm episode lifecycle."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ada_grasp_ctrl.batch import SampleStatus
from ada_grasp_ctrl.errors import ControlSolveEpisodeAbort
from ada_grasp_ctrl.tasks.control_eval_func.base import BaseEval
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


class _AbortController:
    """Record a partial trajectory and prove lift interpolation was skipped."""

    def __init__(self):
        """Initialize the fields used by :class:`BaseEval`."""
        self.r_data = {
            "obj_pose": [],
            "dof": [],
            "doa": [],
            "contacts": [],
            "planned_dof": [],
            "solver_diagnostics": [],
        }
        self.solver_degraded = False
        self.lift_interpolation_calls = 0

    def interplote_qpos(self, qpos1, qpos2, step):
        """Record an unexpected lift request.

        Args:
            qpos1: Starting actuator vector.
            qpos2: Lift target vector.
            step: Interpolation step count.

        Returns:
            One target waypoint.
        """
        del qpos1, step
        self.lift_interpolation_calls += 1
        return np.asarray([qpos2])

    def save_recorded_data(self, path, episode_status="completed"):
        """Persist the partial record for status and alignment assertions.

        Args:
            path: Destination NPY path.
            episode_status: Structured episode outcome.

        Returns:
            None.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = dict(self.r_data)
        record["episode_status"] = episode_status
        np.save(destination, record, allow_pickle=True)


class _AbortMjHO:
    """Minimal simulator facade for the per-offset abort lifecycle."""

    def __init__(self):
        """Initialize a finite state and actuator call counters."""
        self.spec = SimpleNamespace(option=SimpleNamespace(timestep=0.01))
        self.qpos_a = np.zeros(3)
        self.control_calls = 0

    def reset(self):
        """Reset the fake simulator."""

    def get_contact_info(self, qpos, obj_pose):
        """Return a penetration-free initialization.

        Args:
            qpos: Initial robot qpos.
            obj_pose: Initial object pose.

        Returns:
            Empty hand-object and hand-hand contacts.
        """
        del qpos, obj_pose
        return [], []

    def udpate_debug_viewer(self):
        """Match the production compatibility spelling without side effects."""

    def get_obj_pose(self):
        """Return one finite WXYZ object pose."""
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def get_curr_contact_info(self):
        """Return no initial contacts."""
        return []

    def set_ext_force_on_obj(self, force):
        """Accept the configured gravity force.

        Args:
            force: Six-dimensional external wrench.
        """
        del force

    def get_qpos_a(self):
        """Return the current actuator state."""
        return self.qpos_a.copy()

    def ctrl_qpos_a_with_interp(self, current, target, names, step_outer, step_inner):
        """Record any actuator command, which abort mode must avoid.

        Args:
            current: Current actuator vector.
            target: Target actuator vector.
            names: Actuator names.
            step_outer: Outer interpolation count.
            step_inner: Inner simulation count.
        """
        del current, target, names, step_outer, step_inner
        self.control_calls += 1

    def close_view_and_render(self):
        """Close the fake simulator."""


class _AbortEval(BaseEval):
    """Inject a command-solver abort into the real per-offset lifecycle."""

    def __init__(self, root: Path):
        """Build only the fields consumed by ``_eval_simulate_under_extforce``.

        Args:
            root: Temporary output root.
        """
        self.input_npy_path = str(root / "grasp" / "sample.npy")
        self.configs = SimpleNamespace(
            setting="tabletop",
            task=SimpleNamespace(
                simulation_metrics=SimpleNamespace(max_pene=0.005, max_force=1.0),
                arm_pregrasp_is_grasp=True,
                graspdata=SimpleNamespace(pregrasp_t=0, squeeze_t=0),
                debug_viewer=False,
                debug_render=False,
                obj_mass=0.1,
            ),
        )
        self.grasp_data = {
            "pregrasp_qpos": np.zeros(3),
            "grasp_qpos": np.zeros(3),
            "squeeze_qpos": np.zeros(3),
        }
        self.robot = SimpleNamespace(arm=SimpleNamespace(n_dof=1), doa_names=["a0", "a1", "a2"])
        self.mj_ho = _AbortMjHO()
        self.controller = _AbortController()
        self.save_path = root / "control" / "sample.npy"

    def _initialize(self):
        """Install the deterministic controller for one offset."""
        self.method_name = "ours"
        self.grasp_ctrl = self.controller

    def _simulate_under_extforce_details(self, pregrasp_qpos, grasp_qpos, squeeze_qpos):
        """Latch degradation and abort before any actuator action.

        Args:
            pregrasp_qpos: Prepared pregrasp qpos.
            grasp_qpos: Prepared grasp qpos.
            squeeze_qpos: Prepared squeeze qpos.
        """
        del pregrasp_qpos, grasp_qpos, squeeze_qpos
        self.controller.solver_degraded = True
        self.controller.r_data["solver_diagnostics"].append(
            {
                "accepted": False,
                "failure_policy": "fail_episode",
                "decision": "abort_episode",
                "action_applied": False,
                "episode_aborted": True,
            }
        )
        raise ControlSolveEpisodeAbort("injected failure")


class _OffsetContinuationEval(BaseEval):
    """Prove one degraded offset does not stop later offsets."""

    def __init__(self):
        """Initialize one zero and one nonzero perturbation group."""
        self.configs = SimpleNamespace(task=SimpleNamespace(offsets=[0.0, 0.01]))
        self.grasp_data = {"obj_pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])}
        self.mj_ho = _AbortMjHO()
        self.calls = []

    def _eval_simulate_under_extforce(self, obj_pose, file_suffix):
        """Return one degradation followed by completed offsets.

        Args:
            obj_pose: Shifted object pose.
            file_suffix: Deterministic offset identifier.

        Returns:
            Synthetic output path and status.
        """
        del obj_pose
        self.calls.append(file_suffix)
        status = SampleStatus.SOLVER_DEGRADED if len(self.calls) == 1 else SampleStatus.COMPLETED
        return f"{file_suffix}.npy", status


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

    def test_fail_episode_saves_partial_record_and_skips_lift(self):
        """Catch the domain abort per offset without applying or lifting."""
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = _AbortEval(Path(temporary))
            with patch(
                "ada_grasp_ctrl.tasks.control_eval_func.base.control_output_path",
                return_value=evaluator.save_path,
            ):
                output_path, status = evaluator._eval_simulate_under_extforce(
                    np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                    "_dist_0_pos_0",
                )

            self.assertEqual(status, SampleStatus.SOLVER_DEGRADED)
            self.assertEqual(Path(output_path), evaluator.save_path)
            self.assertEqual(evaluator.mj_ho.control_calls, 0)
            self.assertEqual(evaluator.controller.lift_interpolation_calls, 0)
            record = np.load(evaluator.save_path, allow_pickle=True).item()
            self.assertEqual(record["episode_status"], "solver_degraded")
            self.assertEqual(record["solver_diagnostics"][-1]["decision"], "abort_episode")
            for field in ("obj_pose", "dof", "doa", "contacts", "planned_dof"):
                self.assertEqual(record[field], [])

    def test_degraded_offset_does_not_stop_later_offsets(self):
        """Continue all eight nonzero perturbations after one degraded result."""
        evaluator = _OffsetContinuationEval()

        output_paths, statuses = evaluator.run()

        self.assertEqual(len(evaluator.calls), 9)
        self.assertEqual(len(output_paths), 9)
        self.assertEqual(statuses[0], SampleStatus.SOLVER_DEGRADED)
        self.assertEqual(statuses[1:], [SampleStatus.COMPLETED] * 8)


if __name__ == "__main__":
    unittest.main()
