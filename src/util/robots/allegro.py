from .base import Hand, register_robot
from abc import ABC, abstractmethod
import numpy as np
import copy


@register_robot("allegro")
class Allegro(Hand):
    def __init__(self, prefix):
        super().__init__(prefix)
        self.name = "allegro"

        # these names does not contain prefix
        self._dof_names = [
            "ffj0",
            "ffj1",
            "ffj2",
            "ffj3",
            "mfj0",
            "mfj1",
            "mfj2",
            "mfj3",
            "rfj0",
            "rfj1",
            "rfj2",
            "rfj3",
            "thj0",
            "thj1",
            "thj2",
            "thj3",
        ]
        self._doa_names = [
            "ffa0",
            "ffa1",
            "ffa2",
            "ffa3",
            "mfa0",
            "mfa1",
            "mfa2",
            "mfa3",
            "rfa0",
            "rfa1",
            "rfa2",
            "rfa3",
            "tha0",
            "tha1",
            "tha2",
            "tha3",
        ]
        self._doa2dof_matrix = np.eye(len(self._dof_names))
        self._doa_kp = [5] * len(self._doa_names)
        self._body_names = []

        self._doa_max_vel = [0.3] * len(self._doa_names)
        self._base_name = "palm"
        self._base_pose = [0, 0, 0, 0, 0, 0, 1]  # (xyz, xyzw)

        self.side = "rh"  # TODO
        assert self.side == "rh" or self.side == "lh"

        if self.side == "rh":
            self._mjcf_path = "assets/hand/allegro/right_hand.xml"
        else:
            raise NotImplementedError()
