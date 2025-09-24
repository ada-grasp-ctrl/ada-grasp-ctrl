"""
Most codes in this file are from pytorch3d:

https://github.com/facebookresearch/pytorch3d/tree/main

"""

import torch
import numpy as np
import transforms3d.quaternions as tq
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import pdb


def interplote_pose(pose1: np.array, pose2: np.array, step: int) -> np.array:
    trans1, quat1 = pose1[:3], pose1[3:7]
    trans2, quat2 = pose2[:3], pose2[3:7]

    slerp = Slerp([0, 1], R.from_quat([quat1, quat2], scalar_first=True))
    trans_interp = np.linspace(trans1, trans2, step + 1)[1:]
    quat_interp = slerp(np.linspace(0, 1, step + 1))[1:].as_quat(scalar_first=True)
    return np.concatenate([trans_interp, quat_interp], axis=1)


def interplote_qpos(qpos1: np.array, qpos2: np.array, step: int) -> np.array:
    return np.linspace(qpos1, qpos2, step + 1)[1:]


def np_normalize_vector(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def np_normal_to_rot(axis_0, rot_base1=np.array([[0, 1, 0]]), rot_base2=np.array([[0, 0, 1]])):
    proj_xy = np.abs(np.sum(axis_0 * rot_base1, axis=-1, keepdims=True))
    axis_1 = np.where(proj_xy > 0.99, rot_base2, rot_base1)

    axis_1 = np_normalize_vector(axis_1 - np.sum(axis_1 * axis_0, axis=-1, keepdims=True) * axis_0)
    axis_2 = np.cross(axis_0, axis_1, axis=-1)

    return np.stack([axis_0, axis_1, axis_2], axis=-1)


def np_get_delta_qpos(qpos1, qpos2):
    # qpos: [x, y, z, qw, qx, qy, qz]
    delta_pos = np.linalg.norm(qpos1[:3] - qpos2[:3])  # (1)
    q1_inv = tq.qinverse(qpos1[3:])
    q_rel = tq.qmult(qpos2[3:], q1_inv)
    if np.abs(q_rel[0]) > 1:
        q_rel[0] = 1
    angle = 2 * np.arccos(q_rel[0])
    angle_degrees = np.degrees(angle)
    return delta_pos, angle_degrees


def even_sample_points_on_sphere(dim_num, delta_angle=45):
    """
    The method comes from https://stackoverflow.com/a/62754601
    Sample angles evenly in each dimension and finally normalize to sphere.
    """
    assert 90 % delta_angle == 0
    point_per_dim = 90 // delta_angle + 1
    point_num = point_per_dim ** (dim_num - 1) * dim_num * 2
    # print(f"Start to generate {point_num} points (with duplication) on S^{dim_num-1}!")

    comb = np.arange(point_per_dim ** (dim_num - 1))
    comb_lst = []
    for i in range(dim_num - 1):
        comb_lst.append(comb % point_per_dim)
        comb = comb // point_per_dim
    comb_array = np.stack(comb_lst, axis=-1)  # [p, d-1]

    # used to remove duplicated points!
    has_one = ((comb_array == point_per_dim - 1) | (comb_array == 0)) * np.arange(start=1, stop=dim_num)
    has_one = np.where(has_one == 0, dim_num, has_one)
    has_one = has_one.min(axis=-1)

    points_lst = []
    angle_array = (comb_array * delta_angle - 45) * np.pi / 180
    points_part = np.tan(angle_array)
    np_ones = np.ones_like(points_part[:, 0:1])  # [p, 1]
    for i in range(dim_num):
        pp1 = points_part[np.where(i < has_one)[0], :]  # remove duplicated points!
        points = np.concatenate(
            [
                np.concatenate([pp1[:, :i], np_ones[: pp1.shape[0]]], axis=-1),
                pp1[:, i:],
            ],
            axis=-1,
        )
        points_lst.append(points)

        pp2 = points_part[np.where(i < has_one)[0], :]  # remove duplicated points!
        points2 = np.concatenate(
            [
                np.concatenate([pp2[:, :i], -np_ones[: pp2.shape[0]]], axis=-1),
                pp2[:, i:],
            ],
            axis=-1,
        )
        points_lst.append(points2)

    points_array = np.concatenate(points_lst, axis=0)  # [P, d]
    points_array = np_normalize_vector(points_array)
    # print(f"Finish generating! Got {points_array.shape[0]} points (without duplication) on S^{dim_num-1}!")
    return points_array


def random_sample_points_on_sphere(dim_num, point_num):
    points = np.random.randn(point_num, dim_num)
    points = np_normalize_vector(points)
    return points


def torch_normalize_vector(v: torch.Tensor):
    return v / torch.clamp(v.norm(dim=-1, p=2, keepdim=True), min=1e-12)


def axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def torch_quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first, as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    quaternions = torch.as_tensor(quaternions)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def torch_normal_to_rot(axis_0, rot_base1=torch.tensor([0, 0, 1.0]), rot_base2=torch.tensor([0, 1.0, 0])):
    tmp_rot_base1 = rot_base1.view([1] * (len(axis_0.shape) - 1) + [3]).to(axis_0.device)
    tmp_rot_base2 = rot_base2.view([1] * (len(axis_0.shape) - 1) + [3]).to(axis_0.device)

    proj_xy = (axis_0 * tmp_rot_base1).sum(dim=-1, keepdim=True).abs()
    axis_1 = torch.where(proj_xy > 0.99, tmp_rot_base2, tmp_rot_base1)  # avoid normal prependicular to axis_y1
    axis_1 = torch_normalize_vector(axis_1 - (axis_1 * axis_0).sum(dim=-1, keepdim=True) * axis_0)
    axis_2 = torch.cross(axis_0, axis_1, dim=-1)
    return torch.stack([axis_0, axis_1, axis_2], dim=-1)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def torch_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )
    return standardize_quaternion(out)


def torch_quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return quaternions[..., 1:] / sin_half_angles_over_angles


def torch_matrix_to_axis_angle(matrix):
    return torch_quaternion_to_axis_angle(torch_matrix_to_quaternion(matrix))
