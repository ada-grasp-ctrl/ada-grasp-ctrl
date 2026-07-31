"""Open-loop position-control baseline."""

from .base import BaseEval
from .episode_runner import DummyArmEpisodeRunner, EpisodeStep, StepControl, run_dummy_arm_episode


class _OpenLoopPolicy:
    """Follow every interpolated waypoint without feedback."""

    def initialize(self, runner: DummyArmEpisodeRunner) -> None:
        """Accept the prepared runner without additional state.

        Args:
            runner: Prepared common episode runner.

        Returns:
            None.
        """
        del runner

    def max_steps(self, runner: DummyArmEpisodeRunner) -> int:
        """Return one action per interpolated waypoint.

        Args:
            runner: Prepared common episode runner.

        Returns:
            Number of path waypoints.
        """
        return len(runner.qpos_path)

    def should_stop(self, runner: DummyArmEpisodeRunner, step: EpisodeStep) -> bool:
        """Keep running until the common maximum is reached.

        Args:
            runner: Prepared common episode runner.
            step: Current sampled state.

        Returns:
            Always ``False``.
        """
        del runner, step
        return False

    def control(self, runner: DummyArmEpisodeRunner, step: EpisodeStep) -> StepControl:
        """Convert the current full-joint waypoint to actuator coordinates.

        Args:
            runner: Prepared common episode runner.
            step: Current sampled state.

        Returns:
            Actuator target without extra diagnostics.
        """
        return StepControl(runner.robot_adaptor._dof2doa(step.target_qpos_f))


class tabletopDummyArmOpEval(BaseEval):
    def _initialize(self):
        """Initialize the open-loop controller.

        Returns:
            None.
        """
        self._initialize_controller("op")

    def _simulate_under_extforce_details(self, pregrasp_qpos, grasp_qpos, squeeze_qpos):
        """Run the open-loop strategy inside the common episode lifecycle.

        Args:
            pregrasp_qpos: Initial qpos.
            grasp_qpos: In-grasp target qpos.
            squeeze_qpos: Squeezed target qpos.

        Returns:
            None.
        """
        run_dummy_arm_episode(self, pregrasp_qpos, grasp_qpos, squeeze_qpos, _OpenLoopPolicy())
