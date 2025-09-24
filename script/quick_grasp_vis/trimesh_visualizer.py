import xml.etree.ElementTree as ET
import os
import trimesh as tm
import torch
from mr_utils.pytorch3d.rotation_conversions import quaternion_to_matrix
from mr_utils.utils_calc import sciR, transformPositions, posQuat2Isometry3d
import pytorch_kinematics as pk
import numpy as np


def get_trimesh_from_mjmodel_mesh(model, mesh_id):
    """
    Convert a MuJoCo mesh (by mesh_id) to a trimesh.Trimesh object.
    """
    v_start = model.mesh_vertadr[mesh_id]
    v_count = model.mesh_vertnum[mesh_id]

    f_start = model.mesh_faceadr[mesh_id]
    f_count = model.mesh_facenum[mesh_id]

    vertices = model.mesh_vert[v_start : v_start + v_count, :]
    faces = model.mesh_face[f_start : f_start + f_count, :]

    mesh = tm.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def extract_mesh_info_from_mjcf(mjcf_path):
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    link_mesh_dict = {}

    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue

        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        mesh_id = model.geom_dataid[geom_id]
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)

        mesh_scale = model.mesh_scale[mesh_id].tolist()
        pos = model.geom_pos[geom_id].tolist()

        quat = model.geom_quat[geom_id]  # (w, x, y, z)
        quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]  # reorder for scipy
        rpy = sciR.from_quat(quat_xyzw).as_euler("xyz", degrees=False).tolist()

        link_mesh_dict[body_name] = {
            "mesh_id": mesh_id,
            "scale": mesh_scale,  # not used
            "xyz": pos,
            "rpy": rpy,
        }
    return model, link_mesh_dict


class Visualizer:
    def __init__(self, robot_mjcf_path, device="cpu"):
        self.chain = pk.build_chain_from_mjcf(
            data=open(robot_mjcf_path).read(),
        ).to(dtype=torch.float, device=device)

        # Extract mesh info and Mujoco model object
        model, mesh_dict = extract_mesh_info_from_mjcf(robot_mjcf_path)

        self.robot_mesh = {}

        for link_name, info in mesh_dict.items():
            mesh_id = info["mesh_id"]
            # Get trimesh directly from Mujoco model
            trimesh_mesh = get_trimesh_from_mjmodel_mesh(model, mesh_id)
            vertices = np.asarray(trimesh_mesh.vertices)

            # Apply mesh offset transformation
            xyz = info["xyz"]
            rpy = info["rpy"]
            quat = sciR.from_euler("xyz", rpy, degrees=False).as_quat()
            mesh_pose_in_link = posQuat2Isometry3d(xyz, quat)
            vertices = transformPositions(vertices, target_frame_pose_inv=mesh_pose_in_link)

            vertices = torch.tensor(vertices, dtype=torch.float, device=device)
            faces = torch.tensor(trimesh_mesh.faces, dtype=torch.long, device=device)

            self.robot_mesh[link_name] = {"vertices": vertices, "faces": faces}

        self.hand_doa_pose = None
        self.global_translation = None
        self.global_rotation = None  # (n_batch, 3x3 matrix)
        self.current_status = None

    def set_robot_parameters(self, hand_pose, joint_names=None):
        """
        Set translation, rotation, joint angles, and contact points of grasps

        Parameters
        ----------
        hand_pose: (B, 3+4+`n_doas`) torch.FloatTensor
            translation, quaternion (w, x, y, z), and joint angles
        """
        self.global_translation = hand_pose[:, 0:3]
        self.global_rotation = quaternion_to_matrix(hand_pose[:, 3:7])

        # re-order the joints
        qpos = hand_pose[:, 7:]
        chain_joint_names = self.chain.get_joint_parameter_names().copy()
        if joint_names is None:
            joint_names = chain_joint_names.copy()
        external_to_chain = [joint_names.index(name) for name in chain_joint_names]
        chain_qpos = qpos[:, external_to_chain]

        self.current_status = self.chain.forward_kinematics(chain_qpos)

    def get_robot_trimesh_data(self, i, color=None):
        """
        Get full mesh

        Returns
        -------
        data: trimesh.Trimesh
        """
        data = tm.Trimesh()
        for link_name in self.robot_mesh:
            v = self.current_status[link_name].transform_points(self.robot_mesh[link_name]["vertices"])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.robot_mesh[link_name]["faces"].detach().cpu()
            data += tm.Trimesh(vertices=v, faces=f, face_colors=color)
        return data


if __name__ == "__main__":
    robot_urdf_path = "src/curobo/content/assets/robot/leap_description/leap_tac3d_v0.urdf"
    mesh_dir_path = "src/curobo/content/assets/robot/leap_description/"

    visualize = Visualizer(robot_urdf_path=robot_urdf_path, mesh_dir_path=mesh_dir_path)

    hand_pose = torch.zeros((1, 3 + 4 + 16))
    hand_pose[:, 3] = 1.0
    visualize.set_robot_parameters(hand_pose)

    robot_mesh = visualize.get_robot_trimesh_data(i=0)

    scene = tm.Scene(geometry=[robot_mesh])
    scene.show()
