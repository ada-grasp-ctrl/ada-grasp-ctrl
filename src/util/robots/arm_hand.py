from .base import Robot, ArmHand, Arm, Hand, register_robot, RobotFactory
from abc import ABC, abstractmethod


@register_robot("dummy_arm_shadow")
class DummyArmShadow(ArmHand):
    def __init__(self, prefix):
        super().__init__(prefix)

        self.name = "dummy_arm_shadow"

        arm_type = "dummy_arm"
        hand_type = "shadow"
        self.arm: Arm = RobotFactory.create_robot(robot_type=arm_type, prefix=prefix)
        self.hand: Hand = RobotFactory.create_robot(robot_type=hand_type, prefix=prefix)

        self.side = "rh"  # TODO
        assert self.side == "rh" or self.side == "lh"

        if self.side == "rh":
            # self._urdf_path = "assets/robots/dummy_arm_shadow/dummy_arm_shadow.urdf"
            self._mjcf_path = "assets/hand/dummy_arm_shadow/right_no_tendon.xml"
            # self._cfg_path = "dexgrasp_rl/cfg/robots/dummy_arm_shadow.yml"
        else:
            raise NotImplementedError()

        self._base_pose = [0.0, 0.0, 0.0, 0, 0.0, 0, 1.0]  # (xyz, xyzw), base pose in the world frame
        assert len(self._base_pose) == 7


@register_robot("dummy_arm_allegro")
class DummyArmAllegro(ArmHand):
    def __init__(self, prefix):
        super().__init__(prefix)

        self.name = "dummy_arm_allegro"

        arm_type = "dummy_arm"
        hand_type = "allegro"
        self.arm: Arm = RobotFactory.create_robot(robot_type=arm_type, prefix=prefix)
        self.hand: Hand = RobotFactory.create_robot(robot_type=hand_type, prefix=prefix)

        self.side = "rh"  # TODO
        assert self.side == "rh" or self.side == "lh"

        if self.side == "rh":
            self._mjcf_path = "assets/hand/dummy_arm_allegro/right.xml"
        else:
            raise NotImplementedError()

        self._base_pose = [0.0, 0.0, 0.0, 0, 0.0, 0, 1.0]  # (xyz, xyzw), base pose in the world frame
        assert len(self._base_pose) == 7


@register_robot("dummy_arm_leap_tac3d")
class DummyArmLeapTac3d(ArmHand):
    def __init__(self, prefix):
        super().__init__(prefix)

        self.name = "dummy_arm_leap_tac3d"

        arm_type = "dummy_arm"
        hand_type = "leap_tac3d"
        self.arm: Arm = RobotFactory.create_robot(robot_type=arm_type, prefix=prefix)
        self.hand: Hand = RobotFactory.create_robot(robot_type=hand_type, prefix=prefix)

        self.side = "rh"  # TODO
        assert self.side == "rh" or self.side == "lh"

        if self.side == "rh":
            self._mjcf_path = "assets/hand/dummy_arm_leap_tac3d/leap_tac3d.xml"
        else:
            raise NotImplementedError()

        self._base_pose = [0.0, 0.0, 0.0, 0, 0.0, 0, 1.0]  # (xyz, xyzw), base pose in the world frame
        assert len(self._base_pose) == 7
