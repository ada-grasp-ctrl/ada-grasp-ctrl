from .base import Arm, register_robot
from abc import ABC, abstractmethod
import numpy as np


@register_robot("dummy_arm")
class DummyArm(Arm):
    def __init__(self, prefix):
        super().__init__(prefix)
        self.name = "dummy_arm"

        # these names does not contain prefix
        self._dof_names = [
            "dummy_TransXJ",
            "dummy_TransYJ",
            "dummy_TransZJ",
            "dummy_RotXJ",
            "dummy_RotYJ",
            "dummy_RotZJ",
        ]
        self._doa_names = [
            "dummy_TransXA",
            "dummy_TransYA",
            "dummy_TransZA",
            "dummy_RotXA",
            "dummy_RotYA",
            "dummy_RotZA",
        ]

        # self._doa_names = self._dof_names
        self._doa2dof_matrix = np.eye(len(self._dof_names))

        self._doa_kp = [10000, 10000, 10000, 100, 100, 100]

        self._doa_max_vel = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5]
        self._body_names = [
            "dummy_TransXB",
            "dummy_TransYB",
            "dummy_TransZB",
            "dummy_RotXB",
            "dummy_RotYB",
            "dummy_RotZB",
        ]
        self._base_pose = [0, 0, 0, 0, 0, 0, 0]  # (xyz, xyzw)
        self._base_name = "dummy_base"
