import numpy as np
import scipy

from .base import build_grasp_matrix


def calcu_q1_metric(contact_pos, contact_normal, miu_coef, num_friction_approx=8):
    grasp_matrix = build_grasp_matrix(contact_pos, contact_normal)
    aranged_angles = (
        np.arange(num_friction_approx).astype(np.float32)
        * 2
        * np.pi
        / num_friction_approx
    )[None]

    f = np.stack(
        [
            aranged_angles * 0 + 1,
            miu_coef[0] * np.sin(aranged_angles),
            miu_coef[0] * np.cos(aranged_angles),
        ],
        axis=-1,
    )
    corner_point = (f @ grasp_matrix.transpose(0, 2, 1)).reshape(-1, 6)
    corner_point = np.concatenate([corner_point, corner_point[0:1, :] * 0.0], axis=0)

    try:
        q1_metric = 2
        wrench_space = scipy.spatial.ConvexHull(corner_point)
        for equation in wrench_space.equations:
            q1_metric = np.minimum(
                q1_metric, np.abs(equation[6]) / np.linalg.norm(equation[:6])
            )
    except:
        q1_metric = 0

    return 2 - q1_metric
