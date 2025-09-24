import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from util.rot_util import np_normal_to_rot


def build_grasp_matrix(pos, normal):
    rot = np_normal_to_rot(normal)
    axis_0, axis_1, axis_2 = rot[..., 0], rot[..., 1], rot[..., 2]

    # Normalize contact position
    relative_pos = pos - pos.mean(axis=0)[None]
    relative_pos /= np.linalg.norm(relative_pos, axis=-1).mean(axis=0) + 1e-6

    grasp_matrix = np.zeros((pos.shape[0], 6, 3))
    grasp_matrix[:, :3, 0] = axis_0
    grasp_matrix[:, :3, 1] = axis_1
    grasp_matrix[:, :3, 2] = axis_2
    grasp_matrix[:, 3:, 0] = np.cross(relative_pos, axis_0, axis=-1)
    grasp_matrix[:, 3:, 1] = np.cross(relative_pos, axis_1, axis=-1)
    grasp_matrix[:, 3:, 2] = np.cross(relative_pos, axis_2, axis=-1)
    return grasp_matrix
