from .base import Hand, register_robot
from abc import ABC, abstractmethod
import numpy as np


@register_robot("shadow")
class Shadow(Hand):
    def __init__(self, prefix):
        super().__init__(prefix)
        self.name = "shadow"

        # these names does not contain prefix
        self._dof_names = [
            "FFJ4",
            "FFJ3",
            "FFJ2",
            "FFJ1",
            "MFJ4",
            "MFJ3",
            "MFJ2",
            "MFJ1",
            "RFJ4",
            "RFJ3",
            "RFJ2",
            "RFJ1",
            "LFJ5",
            "LFJ4",
            "LFJ3",
            "LFJ2",
            "LFJ1",
            "THJ5",
            "THJ4",
            "THJ3",
            "THJ2",
            "THJ1",
        ]
        self._doa_names = [
            "A_FFJ4",
            "A_FFJ3",
            "A_FFJ2",
            "A_FFJ1",
            "A_MFJ4",
            "A_MFJ3",
            "A_MFJ2",
            "A_MFJ1",
            "A_RFJ4",
            "A_RFJ3",
            "A_RFJ2",
            "A_RFJ1",
            "A_LFJ5",
            "A_LFJ4",
            "A_LFJ3",
            "A_LFJ2",
            "A_LFJ1",
            "A_THJ5",
            "A_THJ4",
            "A_THJ3",
            "A_THJ2",
            "A_THJ1",
        ]
        # self._doa_names = self._dof_names
        self._doa2dof_matrix = np.eye(len(self._dof_names))

        self._doa_kp = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

        self._body_names = [
            "palm",
            "ffknuckle",
            "ffproximal",
            "ffmiddle",
            "ffdistal",
            "fftip",
            "lfmetacarpal",
            "lfknuckle",
            "lfproximal",
            "lfmiddle",
            "lfdistal",
            "lftip",
            "mfknuckle",
            "mfproximal",
            "mfmiddle",
            "mfdistal",
            "mftip",
            "rfknuckle",
            "rfproximal",
            "rfmiddle",
            "rfdistal",
            "rftip",
            "thbase",
            "thproximal",
            "thhub",
            "thmiddle",
            "thdistal",
            "thtip",
        ]

        # self._fingertip_names = ["thtip", "fftip", "mftip", "rftip", "lftip"]
        # self._mano2dex_mapping = {
        #     "wrist": ["palm"],
        #     "thumb_proximal": ["thbase", "thproximal"],  # one-to-many mapping
        #     "thumb_intermediate": ["thhub", "thmiddle"],
        #     "thumb_distal": ["thdistal"],
        #     "thumb_tip": ["thtip"],
        #     "index_proximal": ["ffknuckle", "ffproximal"],
        #     "index_intermediate": ["ffmiddle"],
        #     "index_distal": ["ffdistal"],
        #     "index_tip": ["fftip"],
        #     "middle_proximal": ["mfknuckle", "mfproximal"],
        #     "middle_intermediate": ["mfmiddle"],
        #     "middle_distal": ["mfdistal"],
        #     "middle_tip": ["mftip"],
        #     "ring_proximal": ["rfknuckle", "rfproximal"],
        #     "ring_intermediate": ["rfmiddle"],
        #     "ring_distal": ["rfdistal"],
        #     "ring_tip": ["rftip"],
        #     "pinky_proximal": ["lfmetacarpal", "lfknuckle", "lfproximal"],
        #     "pinky_intermediate": ["lfmiddle"],
        #     "pinky_distal": ["lfdistal"],
        #     "pinky_tip": ["lftip"],
        # }

        self._doa_max_vel = [0.3] * len(self._doa_names)
        self._base_name = "palm"
        self._base_pose = [0, 0, 0, 0, 0, 0, 1]  # (xyz, xyzw)

        self.side = "rh"  # TODO
        assert self.side == "rh" or self.side == "lh"

        if self.side == "rh":
            self._mjcf_path = "assets/hand/shadow/right_hand.xml"
        else:
            raise NotImplementedError()
