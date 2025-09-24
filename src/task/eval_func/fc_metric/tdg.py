import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from util.rot_util import np_normalize_vector, random_sample_points_on_sphere
from .base import build_grasp_matrix


def calcu_tdg_metric(contact_pos, contact_normal, miu_coef, enable_density=False):
    tdg_energy = TDGEnergy(miu_coef, enable_density)
    tdg_metric = tdg_energy.forward(contact_pos, contact_normal)
    return tdg_metric


class TDGEnergy:

    def __init__(self, miu_coef, enable_density):
        self.miu_coef = miu_coef
        self.enable_density = enable_density

        # NOTE: This TWS is only for force closure.
        self.direction_num = 1000
        direction_3D = random_sample_points_on_sphere(3, self.direction_num)
        self.target_direction_6D = np.concatenate(
            [direction_3D, direction_3D * 0.0], axis=-1
        )  # [P, 3]
        self.F_center_direction = np.array([1, 0, 0])[None, None]

        return

    def GWS(self, G_matrix, normal):
        """Approximate the GWS boundary by dense samples.

        Returns
        ----------
        w: [b, P, 6]
        """
        # Solve q_W(u): q_W(u) equals to G * q_F(G^T @ u), so first solve q_F(u')
        direction_F = np_normalize_vector(
            (self.target_direction_6D[None] @ G_matrix).transpose(1, 0, 2)
        )  # G^T @ u: [P, n, 3]
        proj_on_cn = (direction_F * self.F_center_direction).sum(axis=-1)[
            ..., None
        ]  # [P, n, 1]
        perp_to_cn = direction_F - proj_on_cn * self.F_center_direction  # [P, n, 3]

        angles = np.arccos(np.clip(proj_on_cn, a_min=-1, a_max=1))  # [P, n, 1]
        bottom_length = self.miu_coef[0]
        bottom_angle = np.arctan(bottom_length)

        region1 = angles <= bottom_angle
        region2 = (angles > bottom_angle) & (angles <= np.pi / 2)
        region3 = angles > np.pi / 2
        perp_norm = np.linalg.norm(perp_to_cn, axis=-1)[..., None]

        # A more continuous approximation
        help3 = perp_norm / (
            perp_norm - 2 * bottom_length * np.clip(proj_on_cn, a_min=-100, a_max=0)
        )
        help2 = self.F_center_direction + bottom_length * np_normalize_vector(
            perp_to_cn
        )
        argmin_3D_on_normalized_cone = (
            region1
            * (
                self.F_center_direction
                + perp_to_cn
                / np.clip(proj_on_cn, a_min=np.cos(bottom_angle) / 2, a_max=100)
            )
            + region2 * help2
            + region3 * help3 * help2
        )  # [P, n, 3]

        # Get q_W(u) = G * q_F(G^T @ u)
        w = (G_matrix[None] @ argmin_3D_on_normalized_cone[..., None]).squeeze(
            -1
        )  # [P, n, 6]

        # NOTE: use density to change the force bound. It can help to synthesize more human-like pose, i.e. four fingers on one side and the thumb finger on another.
        if self.enable_density:
            cos_theta = (normal[:, None, :] * normal[:, :, None]).sum(
                axis=-1
            )  # [b, n, n]
            density = 1 / np.clip(
                np.clip(cos_theta, a_min=0, a_max=100).sum(axis=-1),
                a_min=1e-4,
                a_max=100,
            )
            final_w = (w * density[None, :, None]).sum(axis=1)  # [P, 6]
        else:
            final_w = w.sum(axis=1)
        return final_w

    def forward(self, pos, normal):
        # G: F \in R^3 (or R^4) -> W \in R^6
        G_matrix = build_grasp_matrix(pos, normal)
        w = self.GWS(G_matrix, normal)
        cos_wt = (np_normalize_vector(w) * self.target_direction_6D).sum(axis=-1)
        gras_energy = (1 - cos_wt).mean(axis=-1)
        return gras_energy
