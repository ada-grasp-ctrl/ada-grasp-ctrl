import os
import pdb
import copy

import trimesh
import numpy as np
import mujoco
import mujoco.viewer
import transforms3d.quaternions as tq
import time

from .rot_util import interplote_pose, interplote_qpos


class MjHO:
    hand_prefix: str = "child-"

    def __init__(
        self,
        obj_path,
        obj_scale,
        obj_density,
        hand_xml_path,
        hand_mocap,
        exclude_table_contact,
        friction_coef,
        has_floor_z0,
        debug_render=False,
        debug_viewer=False,
    ):
        self.hand_mocap = hand_mocap
        self.spec = mujoco.MjSpec()
        self.spec.meshdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.spec.option.timestep = 0.004
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY

        if debug_render or debug_viewer:
            self.spec.add_texture(
                type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                rgb1=[1.0, 1.0, 1.0],
                rgb2=[1.0, 1.0, 1.0],  # white background
                width=512,
                height=512,
            )
            self.spec.worldbody.add_light(
                name="direction_light1",
                pos=[0, 0, 1.5],
                dir=[0, -1, -1],
                castshadow=False,
            )
            self.spec.worldbody.add_light(
                name="direction_light2",
                pos=[0, 0, 1.5],
                dir=[0, 1, -1],
                castshadow=False,
            )
            self.spec.worldbody.add_light(
                name="direction_light3",
                pos=[0, 0, 1.5],
                dir=[-1, 0, -1],
                castshadow=False,
            )
            self.spec.worldbody.add_light(
                name="direction_light4",
                pos=[0, 0, 1.5],
                dir=[1, 0, -1],
                castshadow=False,
            )
            self.spec.worldbody.add_camera(name="closeup", pos=[0.75, 1.0, 1.0], xyaxes=[-1, 0, 0, 0, -1, 1])

        self._add_hand(hand_xml_path, hand_mocap)
        self._add_object(obj_path, obj_scale, obj_density, has_floor_z0)
        self._set_friction(friction_coef)
        self.spec.add_key()
        if exclude_table_contact is not None:
            for body_name in exclude_table_contact:
                self.spec.add_exclude(bodyname1="world", bodyname2=f"{self.hand_prefix}{body_name}")

        # exclude all contacts between hand and table
        body_names = [b.name for b in self.spec.bodies]
        for body_name in body_names:
            if ("object" not in body_name) and ("world" not in body_name):
                self.spec.add_exclude(bodyname1="world", bodyname2=body_name)

        # Get ready for simulation
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # mujoco.mj_forward(self.model, self.data)

        # For ctrl
        qpos2ctrl_matrix = np.zeros((self.model.nu, self.model.nv))
        mujoco.mju_sparse2dense(
            qpos2ctrl_matrix,
            self.data.actuator_moment,
            self.data.moment_rownnz,
            self.data.moment_rowadr,
            self.data.moment_colind,
        )
        self._qpos2ctrl_matrix = qpos2ctrl_matrix[..., :-6]

        self.ext_force_on_obj = None
        self.target_qpos_a = np.zeros((self.model.nu))

        self.debug_viewer = None
        self.debug_render = None
        if debug_viewer:
            self.debug_viewer = mujoco.viewer.launch_passive(self.model, self.data)

            self.debug_viewer.cam.lookat[:] = [0.7, 0, 0.3]
            self.debug_viewer.cam.distance = 0.55
            self.debug_viewer.cam.azimuth = 0
            self.debug_viewer.cam.elevation = -20

            self.debug_viewer.sync()

        if debug_render:
            self.debug_render = mujoco.Renderer(self.model, 480, 640)
            self.debug_options = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.debug_options)
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            self.debug_images = []

            # # 初始化视角参数
            self.cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.cam)

            self.cam.lookat[:] = [0.7, 0, 0.2]
            self.cam.distance = 0.6
            self.cam.azimuth = 25
            self.cam.elevation = 0
        return

    def reset(self):
        self.ext_force_on_obj = None
        self.target_qpos_a = np.zeros((self.model.nu))
        self.debug_images = []

    def _add_hand(self, xml_path, mocap_base):
        # Read hand xml
        child_spec = mujoco.MjSpec.from_file(xml_path)
        for m in child_spec.meshes:
            m.file = os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
        child_spec.meshdir = self.spec.meshdir

        for g in child_spec.geoms:
            # This solimp and solref comes from the Shadow Hand xml
            # They can generate larger force with smaller penetration
            # The body will be more "rigid" and less "soft"
            g.solimp[:3] = [0.5, 0.99, 0.0001]
            g.solref[:2] = [0.005, 1]

        attach_frame = self.spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(child_spec.worldbody, self.hand_prefix, "")
        # Add freejoint and mocap of hand root
        if mocap_base:
            child_world.add_freejoint(name="hand_freejoint")
            self.spec.worldbody.add_body(name="mocap_body", mocap=True)
            self.spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name1="mocap_body",
                name2=f"{self.hand_prefix}world",
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                solimp=[0.9, 0.95, 0.001, 0.5, 2],
                data=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            )
        return

    def _add_object(self, obj_path, obj_scale, obj_density, has_floor_z0):
        if has_floor_z0:
            floor_geom = self.spec.worldbody.add_geom(
                name="object_collision_floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=[0, 0, 0],
                size=[0, 0, 1.0],
                rgba=[1.0, 1.0, 1.0, 0.0],  # transparent
            )

        obj_body = self.spec.worldbody.add_body(name="object")
        obj_body.add_freejoint(name="obj_freejoint")
        parts_folder = os.path.join(obj_path, "urdf/meshes")
        for file in os.listdir(parts_folder):
            file_path = os.path.join(parts_folder, file)
            mesh_name = file.replace(".obj", "")
            mesh_id = mesh_name.replace("convex_piece_", "")

            self.spec.add_mesh(
                name=mesh_name,
                file=file_path,
                scale=[obj_scale, obj_scale, obj_scale],
            )
            obj_body.add_geom(
                name=f"object_visual_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=0,
                contype=0,
                conaffinity=0,
            )
            obj_body.add_geom(
                name=f"object_collision_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=obj_density,
                rgba=[0.925, 0.7, 0.42, 1.0],  # yellow
            )

        return

    def _set_friction(self, test_friction):
        self.spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.spec.option.noslip_iterations = 2
        self.spec.option.impratio = 10
        for g in self.spec.geoms:
            g.friction[:2] = test_friction
            g.condim = 4
        return

    def _qpos2ctrl(self, hand_qpos):
        if self.hand_mocap:
            return self._qpos2ctrl_matrix[:, 6:] @ hand_qpos[7:]
        else:
            return self._qpos2ctrl_matrix @ hand_qpos

    def get_obj_pose(self):
        return self.data.qpos[-7:].copy()

    def get_contact_info(self, hand_qpos, obj_pose, obj_margin=0):
        # Set margin and gap to detect contact
        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = obj_margin

        # Set pose and qpos for hand and object
        self.reset_pose_qpos(hand_qpos, obj_pose)

        object_id = self.model.nbody - 1
        hand_id = self.model.nbody - 2
        world_id = -1 if self.hand_mocap else 0

        # Processing all contact information
        ho_contact = []
        hh_contact = []
        for contact in self.data.contact:
            body1_id = self.model.geom(contact.geom1).bodyid
            body2_id = self.model.geom(contact.geom2).bodyid
            body1_name = self.model.body(self.model.geom(contact.geom1).bodyid).name
            body2_name = self.model.body(self.model.geom(contact.geom2).bodyid).name
            # hand and object
            if (body1_id > world_id and body1_id <= hand_id and body2_id == object_id) or (
                body2_id > world_id and body2_id <= hand_id and body1_id == object_id
            ):
                # keep body1=hand and body2=object
                if body2_id == object_id:
                    contact_normal = contact.frame[0:3]
                    hand_body_name = body1_name.removeprefix(self.hand_prefix)
                    obj_body_name = body2_name
                else:
                    contact_normal = -contact.frame[0:3]
                    hand_body_name = body2_name.removeprefix(self.hand_prefix)
                    obj_body_name = body1_name
                ho_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,
                        "contact_normal": contact_normal,
                        "body1_name": hand_body_name,
                        "body2_name": obj_body_name,
                    }
                )
            # hand and hand
            elif body1_id > world_id and body1_id < hand_id and body2_id > world_id and body2_id < hand_id:
                hh_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,
                        "contact_normal": contact.frame[0:3],
                        "body1_name": body1_name,
                        "body2_name": body2_name,
                    }
                )

        # Set margin and gap back
        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = 0
        return ho_contact, hh_contact

    def get_curr_contact_info(self):
        object_id = self.model.nbody - 1
        hand_id = self.model.nbody - 2
        world_id = -1 if self.hand_mocap else 0

        # Processing all contact information
        ho_contact = []
        for contact_id, contact in enumerate(self.data.contact):
            body1_id = self.model.geom(contact.geom1).bodyid
            body2_id = self.model.geom(contact.geom2).bodyid
            body1_name = self.model.body(self.model.geom(contact.geom1).bodyid).name
            body2_name = self.model.body(self.model.geom(contact.geom2).bodyid).name
            # hand and object
            if (body1_id > world_id and body1_id <= hand_id and body2_id == object_id) or (
                body2_id > world_id and body2_id <= hand_id and body1_id == object_id
            ):
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, contact_id, contact_force)  # in contact local frame

                # keep body1=hand and body2=object
                if body2_id == object_id:
                    contact_frame = contact.frame
                    hand_body_name = body1_name.removeprefix(self.hand_prefix)
                    hand_body_id = body1_id
                    obj_body_name = body2_name
                else:
                    contact_frame = -contact.frame
                    hand_body_name = body2_name.removeprefix(self.hand_prefix)
                    hand_body_id = body2_id
                    obj_body_name = body1_name

                # calculate contact pos in in-contact body frame
                pos_global = contact.pos.copy().reshape(-1, 1)
                body_pos = self.data.xpos[hand_body_id].reshape(-1, 1)
                body_mat = self.data.xmat[hand_body_id].reshape(3, 3)
                pos_local = body_mat.T @ (pos_global - body_pos)

                # calucate contact frame in in-contact body frame
                # note that the contact.frame is stored in transposed form (row-major)
                contact_frame = contact_frame.reshape(3, 3).T  # global orientation of the contact [X; Y; Z]
                contact_frame_local = body_mat.T @ contact_frame

                ho_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,  # in world frame
                        "contact_pos_local": pos_local,  # in body frame
                        "contact_force": contact_force,  # in contact local frame (from hand to object)
                        "contact_frame": contact_frame,  # in world frame
                        "contact_frame_local": contact_frame_local,  # in body frame
                        "body1_name": hand_body_name,
                        "body2_name": obj_body_name,
                    }
                )

        return copy.deepcopy(ho_contact)

    def set_ext_force_on_obj_single_step(self, ext_force):
        """
        Only valid for the next simulation step.
        """
        self.data.xfrc_applied[-1] = ext_force
        return

    def set_ext_force_on_obj(self, ext_force):
        """
        The force will be constantly kept.
        """
        self.ext_force_on_obj = ext_force

    def reset_pose_qpos(self, hand_qpos, obj_pose):
        # set key frame
        self.model.key_qpos[0] = np.concatenate([hand_qpos, obj_pose], axis=0)
        self.model.key_ctrl[0] = self._qpos2ctrl(hand_qpos)
        self.model.key_qvel[0] = 0
        self.model.key_act[0] = 0
        if self.hand_mocap:
            self.model.key_mpos[0] = hand_qpos[:3]
            self.model.key_mquat[0] = hand_qpos[3:7]

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        return

    def control_hand_with_interp(self, hand_qpos1, hand_qpos2, step_outer=10, step_inner=10):
        if self.hand_mocap:
            pose_interp = interplote_pose(hand_qpos1[:7], hand_qpos2[:7], step_outer)
        qpos_interp = interplote_qpos(self._qpos2ctrl(hand_qpos1), self._qpos2ctrl(hand_qpos2), step_outer)
        for j in range(step_outer):
            if self.hand_mocap:
                self.data.mocap_pos[0] = pose_interp[j, :3]
                self.data.mocap_quat[0] = pose_interp[j, 3:7]
            self.data.ctrl[:] = qpos_interp[j]
            mujoco.mj_forward(self.model, self.data)
            self.control_hand_step(step_inner)
        return

    def ctrl_qpos_a_with_interp(self, hand_qpos1, hand_qpos2, names, step_outer=10, step_inner=10):
        qpos_interp = interplote_qpos(hand_qpos1, hand_qpos2, step_outer)
        for j in range(step_outer):
            self.ctrl_qpos_a(names, qpos_interp[j])
            mujoco.mj_forward(self.model, self.data)
            self.control_hand_step(step_inner)
        return

    def control_hand_step(self, step_inner):
        for _ in range(step_inner):
            if self.ext_force_on_obj is not None:
                self.set_ext_force_on_obj_single_step(ext_force=self.ext_force_on_obj)
            mujoco.mj_step(self.model, self.data)

        if self.debug_render is not None:
            self.debug_render.update_scene(self.data, camera=self.cam, scene_option=self.debug_options)
            pixels = self.debug_render.render()
            self.debug_images.append(pixels)

        if self.debug_viewer is not None:
            self.debug_viewer.sync()
            time.sleep(self.spec.option.timestep)

        return

    def udpate_debug_viewer(self):
        if self.debug_viewer is not None:
            self.debug_viewer.sync()

    def close_view_and_render(self):
        if self.debug_viewer is not None:
            self.debug_viewer.close()
        if self.debug_render is not None:
            self.debug_render.close()

    def get_qpos_f(self, names):
        return np.array([self.data.joint(self.hand_prefix + name).qpos[0] for name in names])

    def get_qpos_a(self, names=None):
        return self.target_qpos_a.copy()

    def ctrl_qpos_a(self, names, q_a):
        for i, name in enumerate(names):
            self.data.actuator(self.hand_prefix + name).ctrl = q_a[i]
        self.target_qpos_a[:] = np.asarray(q_a)[:]

    def get_obj_mass(self):
        object_id = self.model.nbody - 1
        mass = self.model.body_mass[object_id]
        return mass


class RobotKinematics:
    def __init__(self, xml_path):
        spec = mujoco.MjSpec.from_file(xml_path)
        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)

        self.mesh_geom_info = {}
        for i in range(self.mj_model.ngeom):
            geom = self.mj_model.geom(i)
            mesh_id = geom.dataid
            if mesh_id != -1:
                mjm = self.mj_model.mesh(mesh_id)
                vert = self.mj_model.mesh_vert[mjm.vertadr[0] : mjm.vertadr[0] + mjm.vertnum[0]]
                face = self.mj_model.mesh_face[mjm.faceadr[0] : mjm.faceadr[0] + mjm.facenum[0]]
                body_name = self.mj_model.body(geom.bodyid).name
                mesh_name = mjm.name
                self.mesh_geom_info[f"{body_name}_{mesh_name}"] = {
                    "vert": vert,
                    "face": face,
                    "geom_id": i,
                }

        return

    def forward_kinematics(self, q):
        self.mj_data.qpos = q
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        return

    def get_init_meshes(self):
        init_mesh_lst = []
        mesh_name_lst = []
        for k, v in self.mesh_geom_info.items():
            mesh_name_lst.append(k)
            init_mesh_lst.append(trimesh.Trimesh(vertices=v["vert"], faces=v["face"]))
        return mesh_name_lst, init_mesh_lst

    def get_poses(self, root_pose):
        geom_poses = np.zeros((len(self.mesh_geom_info), 7))
        root_rot = tq.quat2mat(root_pose[3:])
        root_trans = root_pose[:3]
        for i, v in enumerate(self.mesh_geom_info.values()):
            geom_trans = self.mj_data.geom_xpos[v["geom_id"]]
            geom_rot = self.mj_data.geom_xmat[v["geom_id"]].reshape(3, 3)
            geom_poses[i, :3] = root_rot @ geom_trans + root_trans
            geom_poses[i, 3:] = tq.mat2quat(root_rot @ geom_rot)
        return geom_poses

    def get_posed_meshes(self, root_pose):
        root_rot = tq.quat2mat(root_pose[3:])
        root_trans = root_pose[:3]
        full_tm = []
        for k, v in self.mesh_geom_info.items():
            geom_rot = self.mj_data.geom_xmat[v["geom_id"]].reshape(3, 3)
            geom_trans = self.mj_data.geom_xpos[v["geom_id"]]
            posed_vert = (v["vert"] @ geom_rot.T + geom_trans) @ root_rot.T + root_trans
            posed_tm = trimesh.Trimesh(vertices=posed_vert, faces=v["face"])
            full_tm.append(posed_tm)
        full_tm = trimesh.util.concatenate(full_tm)
        return full_tm


if __name__ == "__main__":
    xml_path = os.path.join(os.path.dirname(__file__), "../../assets/hand/shadow/customized.xml")
    kinematic = RobotKinematics(xml_path)
    hand_qpos = np.zeros((22))
    kinematic.forward_kinematics(hand_qpos)
    visual_mesh = kinematic.get_posed_meshes()
    visual_mesh.export("debug_hand.obj")
