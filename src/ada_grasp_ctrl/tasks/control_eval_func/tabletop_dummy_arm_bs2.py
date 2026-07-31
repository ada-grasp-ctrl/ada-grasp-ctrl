"""BS2 policy: ours without approaching-arm motion."""

from .base import BaseEval
from .optimized_runner import WrenchControlPolicy, run_wrench_control_episode


class tabletopDummyArmBS2Eval(BaseEval):
    """Evaluate BS2 using the exact shared ours path with one policy flag changed."""

    def _initialize(self):
        """Initialize the common controller for BS2.

        Returns:
            None.
        """
        self._initialize_controller("bs2")

    def _simulate_under_extforce_details(self, pregrasp_qpos, grasp_qpos, squeeze_qpos):
        """Delegate the trajectory with Stage-1 arm motion disabled.

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
            WrenchControlPolicy(use_approaching_arm_motion=False),
        )
