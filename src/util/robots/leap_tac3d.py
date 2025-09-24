from .base import Hand, register_robot
from abc import ABC, abstractmethod
import numpy as np
import copy


@register_robot("leap_tac3d")
class LeapTac3d(Hand):
    def __init__(self, prefix):
        super().__init__(prefix)
        self.name = "leap_tac3d"

        # these names does not contain prefix
        self._dof_names = [
            "joint_1",
            "joint_0",
            "joint_2",
            "joint_3",
            "joint_5",
            "joint_4",
            "joint_6",
            "joint_7",
            "joint_9",
            "joint_8",
            "joint_10",
            "joint_11",
            "joint_12",
            "joint_13",
            "joint_14",
            "joint_15",
        ]
        self._doa_names = [
            "actuator_1",
            "actuator_0",
            "actuator_2",
            "actuator_3",
            "actuator_5",
            "actuator_4",
            "actuator_6",
            "actuator_7",
            "actuator_9",
            "actuator_8",
            "actuator_10",
            "actuator_11",
            "actuator_12",
            "actuator_13",
            "actuator_14",
            "actuator_15",
        ]
        self._doa2dof_matrix = np.eye(len(self._dof_names))
        self._doa_kp = [5] * len(self._doa_names)
        self._body_names = []

        self._doa_max_vel = [0.3] * len(self._doa_names)
        self._base_name = "palm_lower"
        self._base_pose = [0, 0, 0, 0, 0, 0, 1]  # (xyz, xyzw)

        self.side = "rh"  # TODO
        assert self.side == "rh" or self.side == "lh"

        if self.side == "rh":
            self._mjcf_path = "assets/hand/leap_tac3d/leap_tac3d.xml"
        else:
            raise NotImplementedError()
