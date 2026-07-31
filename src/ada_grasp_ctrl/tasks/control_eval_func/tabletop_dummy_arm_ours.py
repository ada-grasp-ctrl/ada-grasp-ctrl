"""Our coordinated wrench-control policy."""

from .base import BaseEval
from .optimized_runner import WrenchControlPolicy, run_wrench_control_episode


class tabletopDummyArmOursEval(BaseEval):
    """Evaluate ours with arm motion enabled during Stage 1."""

    def _initialize(self):
        """Initialize the common controller for ours.

        Returns:
            None.
        """
        self._initialize_controller("ours")

    def _simulate_under_extforce_details(self, pregrasp_qpos, grasp_qpos, squeeze_qpos):
        """Delegate the trajectory to the shared optimized runner.

        Args:
            pregrasp_qpos: Initial qpos.
            grasp_qpos: In-grasp target qpos.
            squeeze_qpos: Squeezed target qpos.

        Returns:
            None.
        """
        run_wrench_control_episode(
            self,
            pregrasp_qpos,
            grasp_qpos,
            squeeze_qpos,
            WrenchControlPolicy(use_approaching_arm_motion=True),
        )
