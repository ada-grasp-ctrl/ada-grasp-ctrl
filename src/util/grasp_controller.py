from scipy.optimize import minimize
from scipy.linalg import block_diag
import numpy as np
import os
from mr_utils.utils_calc import (
    isometry3dToPosQuat,
    isometry3dToPosOri,
    sciR,
    skew,
)
from .robot_adaptor import RobotAdaptor
from .robots.base import RobotFactory, Robot, ArmHand


class GraspController:
    def __init__(self, configs, robot: ArmHand, robot_adaptor: RobotAdaptor):
        self.robot = robot
        self.robot_adaptor = robot_adaptor

        self.r_data = {
            "obj_pose": [],
            "dof": [],
            "doa": [],
            "contacts": [],
            "planned_dof": [],
            "balance_metric": [],
            "t_check_balance": [],
            "t_ctrl_opt": [],
            "t_step_cost": [],
            "stage": [],
            "opt_res": [],
            "desired_sum_force": [],
        }

        # hyper-parameters
        if configs:
            self.stage2_incontact_force_only = configs.stage2_incontact_force_only
            self.stage2_Ks_hand_only = configs.stage2_Ks_hand_only
            self.stage2_penalize_tan_motion = configs.stage2_penalize_tan_motion
            self.balance_use_normalized = configs.balance_use_normalized

            self.stage2_equal_joint_force_cost = configs.stage2_equal_joint_force_cost

            self.stage2_ctrl_tan_force = configs.stage2_ctrl_tan_force
            self.stage2_tan_force_constraint = configs.stage2_tan_force_constraint

            self.stage2_increase_force = configs.stage2_increase_force

            self.Ke_scalar = configs.Ke_scalar
            self.stage1_force_thres = configs.stage1_force_thres

            self.Kp = np.diag(np.clip(self.robot.doa_kp, 0, 1e3))
            self.Kp_inv = np.linalg.inv(self.Kp)
            self.Ke = np.diag([self.Ke_scalar, self.Ke_scalar, self.Ke_scalar])  # x-axis is the contact normal

            self.tan_motion_pen_weight = configs.tan_motion_pen_weight
            self.use_multi_contact_model = configs.use_multi_contact_model
            self.stage2_penalize_contact_qda = configs.stage2_penalize_contact_qda
            self.jaco_reference_frame = configs.jaco_reference_frame

        self.balance_thres = 0.2
        self.mu = 0.3  # friction coef
        if "shadow" in self.robot.name:
            self.final_sum_force = 15.0
        elif "allegro" in self.robot.name:
            self.final_sum_force = 10.0
        elif "leap" in self.robot.name:
            self.final_sum_force = 8.0
        else:
            raise NotImplementedError()

    def save_recorded_data(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.r_data, allow_pickle=True)
        print(f"Save recorded control data to {path}.")

    def interplote_qpos(self, qpos1: np.array, qpos2: np.array, step: int) -> np.array:
        return np.linspace(qpos1, qpos2, step + 1)[1:]

    def Ks(self, q_a, q_f, contacts):
        """
        Compute Ks.
        """
        I3 = np.eye(3)
        hand_ndoa = self.robot.hand.n_doa

        # compute J(q) and J(qd)
        body_name_lst = []
        for i, contact in enumerate(contacts):
            body_name_lst.append(contact["body1_name"])
        self.robot_adaptor.compute_jaco_a(q_a)  # J(qd)
        body_jaco_a_lst = [self.robot_adaptor.get_frame_jaco(frame_name=name, type="body") for name in body_name_lst]
        pose_a_lst = [self.robot_adaptor.get_frame_pose(frame_name=name) for name in body_name_lst]
        self.robot_adaptor.compute_jaco_a(q_f)  # J(q)
        body_jaco_f_lst = [self.robot_adaptor.get_frame_jaco(frame_name=name, type="body") for name in body_name_lst]
        pose_f_lst = [self.robot_adaptor.get_frame_pose(frame_name=name) for name in body_name_lst]

        # --- Compute per-contact Jacobians ---
        for i, c in enumerate(contacts):
            cp_local = c["contact_pos_local"].reshape(-1, 1)
            cf_local = c["contact_frame_local"].reshape(3, 3)
            Trans = np.block([[I3, -skew(cp_local)]])
            body_jaco_f = body_jaco_f_lst[i]
            body_jaco_a = body_jaco_a_lst[i]
            contact_jaco_f = cf_local.T @ Trans @ body_jaco_f
            contact_jaco_a = cf_local.T @ Trans @ body_jaco_a

            if self.jaco_reference_frame:
                # adjust J(qd) to be defined in the local contact frame of q;
                # otherwise, it is in the local contact frame of qd.
                delta_rot = pose_f_lst[i][:3, :3].T @ pose_a_lst[i][:3, :3]
                contact_jaco_a = delta_rot @ contact_jaco_a

            c["jaco_a"], c["jaco_f"] = contact_jaco_a, contact_jaco_f
            c["jaco_ha"], c["jaco_hf"] = contact_jaco_a[:, -hand_ndoa:], contact_jaco_f[:, -hand_ndoa:]
            contacts[i] = c

        # --- Compute Ks ---
        if self.use_multi_contact_model:  # if consider multiple contact on the same finger
            n_con = len(contacts)
            I_stack = np.eye(3 * n_con)
            Ke_stack = block_diag(*([self.Ke] * n_con))
            J_a_stack = np.concatenate([c["jaco_a"] for c in contacts], axis=0)
            J_f_stack = np.concatenate([c["jaco_f"] for c in contacts], axis=0)
            Kr_inv_stack = J_a_stack @ self.Kp_inv @ J_f_stack.T
            Ks_stack = np.linalg.inv(I_stack + Ke_stack @ Kr_inv_stack) @ Ke_stack

            J_ha_stack = np.concatenate([c["jaco_ha"] for c in contacts], axis=0)
            J_hf_stack = np.concatenate([c["jaco_hf"] for c in contacts], axis=0)
            Kr_h_inv_stack = J_ha_stack @ self.Kp_inv[-hand_ndoa:, -hand_ndoa:] @ J_hf_stack.T
            Ks_h_stack = np.linalg.inv(I_stack + Ke_stack @ Kr_h_inv_stack) @ Ke_stack
        else:
            for i, contact in enumerate(contacts):
                contact_jaco_a = contact["jaco_a"]
                contact_jaco_f = contact["jaco_f"]
                Kr_inv = contact_jaco_a @ self.Kp_inv @ contact_jaco_f.T
                Ks = np.linalg.inv(I3 + self.Ke @ Kr_inv) @ self.Ke  # in contact local frame
                contact["Ks"] = Ks
                # only hand
                contact_jaco_ha = contact_jaco_a[:, -hand_ndoa:]
                contact_jaco_hf = contact_jaco_f[:, -hand_ndoa:]
                Kr_h_inv = contact_jaco_ha @ self.Kp_inv[-hand_ndoa:, -hand_ndoa:] @ contact_jaco_hf.T
                Ks_h = np.linalg.inv(I3 + self.Ke @ Kr_h_inv) @ self.Ke  # in contact local frame
                contact["Ks_h"] = Ks_h
                contacts[i] = contact

            Ks_stack = block_diag(*[c["Ks"] for c in contacts])
            J_a_stack = np.concatenate([c["jaco_a"] for c in contacts], axis=0)
            J_f_stack = np.concatenate([c["jaco_f"] for c in contacts], axis=0)
            Ks_h_stack = block_diag(*[c["Ks_h"] for c in contacts])
            J_ha_stack = np.concatenate([c["jaco_ha"] for c in contacts], axis=0)
            J_hf_stack = np.concatenate([c["jaco_hf"] for c in contacts], axis=0)

        stacked = {
            "Ks": Ks_stack,
            "jaco_a": J_a_stack,
            "jaco_f": J_f_stack,
            "Ks_h": Ks_h_stack,
            "jaco_ha": J_ha_stack,
            "jaco_hf": J_hf_stack,
        }

        return contacts, stacked

    # def compute_grasp_matrix(self, ho_contacts) -> np.ndarray:
    #     n_con = len(ho_contacts)
    #     if n_con == 0:
    #         return None

    #     contact_frame = [contact["contact_frame"] for contact in ho_contacts]
    #     contact_pos_all = [contact["contact_pos"] for contact in ho_contacts]
    #     contact_pos_all = np.asarray(contact_pos_all).reshape(-1, 3)
    #     contact_centroid = contact_pos_all.mean(axis=0, keepdims=True)
    #     contact_r = contact_pos_all - contact_centroid
    #     contact_r = contact_r * 100.0  # unit from m to cm; then, the unit of torque is (N x cm)

    #     contact_G = []
    #     for i in range(len(ho_contacts)):
    #         r = contact_r[i, :]
    #         n, o, t = contact_frame[i][:, 0], contact_frame[i][:, 1], contact_frame[i][:, 2]
    #         G = np.block(
    #             [
    #                 [n.reshape(-1, 1), o.reshape(-1, 1), t.reshape(-1, 1)],
    #                 [np.cross(r, n).reshape(-1, 1), np.cross(r, o).reshape(-1, 1), np.cross(r, t).reshape(-1, 1)],
    #             ]
    #         )
    #         contact_G.append(G)
    #     contact_G = np.concatenate(contact_G, axis=1)

    #     return contact_G

    def compute_grasp_matrix(self, ho_contacts) -> np.ndarray:
        n_con = len(ho_contacts)
        if n_con == 0:
            return None

        # Extract positions and frames
        contact_frames = np.array([c["contact_frame"] for c in ho_contacts])  # (n, 3, 3)
        contact_pos = np.array([c["contact_pos"] for c in ho_contacts])  # (n, 3)

        # Compute centroid and relative positions (scaled to cm)
        centroid = contact_pos.mean(axis=0, keepdims=True)
        r_all = (contact_pos - centroid) * 100.0  # (n, 3); unit from m to cm; then, the unit of torque is (N x cm)

        # Split frame axes
        n_vecs = contact_frames[:, :, 0]  # (n, 3)
        o_vecs = contact_frames[:, :, 1]
        t_vecs = contact_frames[:, :, 2]

        # Compute torque components using broadcasting
        cross_n = np.cross(r_all, n_vecs)  # (n, 3)
        cross_o = np.cross(r_all, o_vecs)
        cross_t = np.cross(r_all, t_vecs)

        # Stack translational and rotational parts
        G_blocks = np.stack(
            [
                np.stack([n_vecs, o_vecs, t_vecs], axis=2),  # (n, 3, 3)
                np.stack([cross_n, cross_o, cross_t], axis=2),  # (n, 3, 3)
            ],
            axis=1,
        )  # (n, 2, 3, 3)

        # Reshape each block into (6, 3) and concatenate along columns
        contact_G = G_blocks.reshape(n_con, 6, 3).transpose(1, 0, 2).reshape(6, -1)

        return contact_G

    def compute_normalized_wrench(self, grasp_matrix: np.ndarray, contact_forces: np.ndarray):
        wrench = (grasp_matrix @ contact_forces.reshape(-1, 1)).reshape(-1)

        cf = contact_forces.reshape(-1, 3)
        G = grasp_matrix.reshape(6, -1, 3).transpose(1, 0, 2)
        fts = np.matmul(G, cf[:, :, None]).reshape(-1, 6)
        sum_forces_mag = np.sum(np.linalg.norm(fts[:, :3], axis=1))
        sum_torques_mag = np.sum(np.linalg.norm(fts[:, 3:], axis=1))

        wrench[:3] /= sum_forces_mag + 1e-8
        wrench[3:] /= sum_torques_mag + 1e-8
        return wrench

    def check_wrench_balance(self, grasp_matrix, b_print_opt_details=False):
        if grasp_matrix is None:
            return 1.0, None

        contact_G = grasp_matrix.copy()
        n_con = contact_G.shape[1] // 3

        if n_con < 2:  # only one contact cannot be in wrench balance
            return 1.0, None

        # weights
        w_wrench = np.eye(6)
        mu = self.mu
        gamma = 1.0

        def objective(x):
            cf = x.copy()
            wrench = contact_G @ cf.reshape(-1, 1)
            cost = wrench.T @ w_wrench @ wrench
            grad = 2 * contact_G.T @ w_wrench @ contact_G @ cf.reshape(-1, 1)
            return cost.item(), grad

        def friction_cone_constraint(x):
            cf = x.copy().reshape(-1, 3)
            constraint = mu * cf[:, 0] - np.linalg.norm(cf[:, 1:], axis=-1)  # >= 0
            return constraint.reshape(-1)

        def friction_cone_constraint_grad(x):
            cf = x.reshape(-1, 3)  # shape (n, 3)
            n = cf.shape[0]
            grad = np.zeros((n, n * 3))

            epsilon = 1e-8
            for i in range(n):
                fx, fy, fz = cf[i]
                norm_yz = np.sqrt(fy**2 + fz**2) + epsilon  # avoid divide-by-zero

                grad[i, 3 * i + 0] = mu * np.sign(fx)
                grad[i, 3 * i + 1] = -fy / norm_yz
                grad[i, 3 * i + 2] = -fz / norm_yz

            return grad

        def force_magnitude_constraint(x):
            cf = x.copy().reshape(-1, 3)
            constraint = np.sum(cf[:, 0]) - gamma  # == 0
            return constraint.reshape(-1)

        def force_magnitude_constraint_grad(x):
            cf = x.reshape(-1, 3)  # shape (n, 3)
            grad = np.zeros_like(cf)  # shape (n, 3)
            grad[:, 0] = 1.0
            return grad.reshape(-1)  # flatten to match x shape

        constraints_list = [
            dict(type="ineq", fun=friction_cone_constraint, jac=friction_cone_constraint_grad),
            dict(type="eq", fun=force_magnitude_constraint, jac=force_magnitude_constraint_grad),
        ]

        bounds = [(0, 10), (-10, 10), (-10, 10)] * n_con

        res = minimize(
            fun=objective,
            jac=True,
            constraints=constraints_list,
            x0=np.zeros((3 * n_con)),
            bounds=bounds,
            method="SLSQP",
            options={"ftol": 1e-6, "disp": b_print_opt_details, "maxiter": 200},
        )

        cf = res.x.reshape(-1)

        if self.balance_use_normalized:
            metric = np.linalg.norm(self.compute_normalized_wrench(grasp_matrix, cf))
        else:
            metric = np.linalg.norm(grasp_matrix @ cf.reshape(-1, 1))

        return metric, cf

    def ctrl_opt(
        self,
        stage,
        dt,
        curr_q_a,
        curr_q_f,
        target_q_f,
        desired_sum_force,
        last_dq_a,
        ho_contacts=None,
        grasp_matrix=None,
        b_use_arm_motion=True,
        b_print_opt_details=False,
    ):
        # hyper-parameters
        mu = self.mu

        # variables for coding convenience
        n_arm_dof = self.robot.arm.n_dof
        n_hand_dof = self.robot.hand.n_dof
        n_dof = n_arm_dof + n_hand_dof
        doa2dof_matrix = self.robot_adaptor.doa2dof_matrix
        joint_limits_f = self.robot_adaptor.joint_limits_f
        q_step_max = np.asarray(self.robot.doa_max_vel) * dt
        n_con = len(ho_contacts) if ho_contacts else 0

        if n_con:
            # compute grasp matrix
            contact_G = self.compute_grasp_matrix(ho_contacts) if grasp_matrix is None else grasp_matrix
            # compute Ks and contact jacobian
            updated_contacts, stacked = self.Ks(q_a=curr_q_a, q_f=curr_q_f, contacts=ho_contacts)
            contact_force_all = np.concatenate([c["contact_force"][:3] for c in updated_contacts], axis=0)
            contact_jaco_all = stacked["jaco_a"]
            # whether use Ks_h
            if stage == 2 and self.stage2_Ks_hand_only:
                Ks_all = stacked["Ks_h"]
                contact_jaco_all[:, :n_arm_dof] = 0  # remove arm part
            else:
                Ks_all = stacked["Ks"]
            Ks_jaco = Ks_all @ contact_jaco_all
        else:
            contact_force_all = np.zeros((0))

        # compute target hand base pose
        hand_base_name = self.robot.hand.base_name
        self.robot_adaptor.compute_fk_f(target_q_f)
        target_hb_pose = self.robot_adaptor.get_frame_pose(frame_name=hand_base_name)
        target_hb_pos, target_hb_ori = isometry3dToPosOri(target_hb_pose)

        # weights
        w_hb_pose = np.diag([0, 0, 100.0, 10.0, 10.0, 10.0])
        w_q_hand = 1.0 * np.eye(n_hand_dof)
        w_dqa = 0.01 * np.eye(n_dof)  #  <= 0.01
        # w_ddqa = [0.00001] * n_arm_dof + [0.001] * n_hand_dof
        w_ddqa = [0.001] * n_arm_dof + [0.001] * n_hand_dof
        w_ddqa = np.diag(w_ddqa)
        w_cp = np.diag([0.0, self.tan_motion_pen_weight, self.tan_motion_pen_weight])
        w_cp = block_diag(*[w_cp for _ in range(n_con)])
        w_cf = np.diag([0.0, 0.1, 0.1])
        w_cf = block_diag(*[w_cf for _ in range(n_con)])
        w_wrench = np.diag([1.0, 1, 1, 1, 1, 1])
        w_efj = 0.01 * np.eye(n_hand_dof)

        if stage == 2 and n_con > 0:
            in_contact_q_indices = contact_jaco_all.any(axis=0)
            contact_jaco_h = contact_jaco_all[:, -n_hand_dof:]
            in_contact_qh_indices = contact_jaco_h.any(axis=0)
            if self.stage2_incontact_force_only:
                # in-contact joint, no position control
                w_q_hand[in_contact_qh_indices, in_contact_qh_indices] = 0
            if self.stage2_penalize_contact_qda:
                w_dqa[in_contact_q_indices, in_contact_q_indices] *= 100

        def objective(x):
            dq_a = x[:n_dof].copy()
            cf = x[n_dof:].copy()
            q_a = curr_q_a + dq_a
            q_f = doa2dof_matrix @ q_a
            _, q_hand = q_f[:n_arm_dof], q_f[n_arm_dof:]

            # cost for hand qpos
            target_q_hand = target_q_f[n_arm_dof:]
            self.err_q_hand = err_q_hand = (q_hand - target_q_hand).reshape(-1, 1)
            cost_q_hand = err_q_hand.T @ w_q_hand @ err_q_hand

            # cost for dqa
            self.err_dqa = err_dqa = (dq_a / dt).reshape(-1, 1)
            cost_dqa = err_dqa.T @ w_dqa @ err_dqa

            # cost for ddqa
            self.err_ddqa = err_ddqa = ((dq_a - last_dq_a) / dt**2).reshape(-1, 1)
            cost_ddqa = err_ddqa.T @ w_ddqa @ err_ddqa

            cost_wrench = 0
            cost_tan_motion = 0
            cost_tan_cf = 0
            cost_ef = 0
            if  n_con > 0:
                if stage == 1 or self.stage2_penalize_tan_motion:
                    # cost tangential motion (restrict the tangential motion of contacts)
                    dp = contact_jaco_all @ dq_a.reshape(-1, 1)
                    self.err_cp = err_cp = dp
                    cost_tan_motion = err_cp.T @ w_cp @ err_cp

                if stage == 2:
                    # cost wrench
                    self.wrench = wrench = contact_G @ cf.reshape(-1, 1)
                    cost_wrench = wrench.T @ w_wrench @ wrench

                    # cost tangential force
                    if self.stage2_ctrl_tan_force:
                        dcf = Ks_jaco @ dq_a.reshape(-1, 1)
                        pred_next_cf = contact_force_all.reshape(-1, 1) + dcf
                        self.err_cf = err_cf = cf.reshape(-1, 1) - pred_next_cf
                        cost_tan_cf = err_cf.T @ w_cf @ err_cf

                    # cost equal force
                    if self.stage2_equal_joint_force_cost:
                        idx_normal = np.arange(0, n_con * 3, 3)
                        J_n = contact_jaco_h[idx_normal, :]
                        self.err_ef = tau_n = J_n.T @ cf.reshape(-1, 3)[:, 0].reshape(-1, 1)
                        cost_ef = tau_n.T @ w_efj @ tau_n

            cost_hb_pose = 0
            if stage == 1 and b_use_arm_motion:
                self.robot_adaptor.compute_fk_a(q_a)
                hb_pose = self.robot_adaptor.get_frame_pose(frame_name=hand_base_name)
                hb_pos, hb_quat = isometry3dToPosQuat(hb_pose)
                err_hb_pos = hb_pos - target_hb_pos
                hb_ori = sciR.from_quat(hb_quat)
                err_hb_ori = (hb_ori * target_hb_ori.inv()).as_rotvec()
                self.err_hb_pose = err_hb_pose = np.concatenate([err_hb_pos, err_hb_ori], axis=0).reshape(-1, 1)  # 6D
                cost_hb_pose = err_hb_pose.T @ w_hb_pose @ err_hb_pose

            total_cost = (
                cost_dqa
                + cost_ddqa
                + cost_q_hand
                + cost_wrench
                + cost_tan_motion
                + cost_tan_cf
                + cost_hb_pose
                + cost_ef
            )
            return total_cost.item()

        def jacobian(x):
            dq_a = x[:n_dof].copy()
            q_a = curr_q_a + dq_a
            grad = np.zeros(x.shape[0])

            # grad of cost_dqa
            err_dqa = self.err_dqa
            grad_dqa = 2.0 / dt * w_dqa @ err_dqa
            grad[:n_dof] += grad_dqa.reshape(-1)

            # grad of cost_ddqa
            err_ddqa = self.err_ddqa
            grad_dqa = 2.0 / dt**2 * w_ddqa @ err_ddqa
            grad[:n_dof] += grad_dqa.reshape(-1)

            # grad of cost_q_hand
            J_qhand_dqa = doa2dof_matrix[n_arm_dof:, :]  # (n_hand_dof, n_dof)
            err_q_hand = self.err_q_hand
            grad_dqa = 2.0 * (J_qhand_dqa.T @ w_q_hand @ err_q_hand).reshape(-1)  # shape: (n_dof,)
            grad[:n_dof] += grad_dqa.reshape(-1)

            if n_con > 0:
                if stage == 1:
                    # grad of cost_tan_motion
                    err_cp = self.err_cp
                    grad_dqa = 2.0 * (contact_jaco_all.T @ w_cp @ err_cp).reshape(-1)  # shape (n_dof,)
                    grad[:n_dof] += grad_dqa.reshape(-1)

                if stage == 2:
                    # grad of cost_wrench
                    wrench = self.wrench
                    grad_cf = 2 * (contact_G.T @ w_wrench @ wrench)  # shape (n, 1)
                    grad[n_dof:] += grad_cf.reshape(-1)

                    if self.stage2_ctrl_tan_force and (not self.stage2_tan_force_constraint):
                        # grad of cost_tan_cf
                        err_cf = self.err_cf
                        grad_dqa = -2 * Ks_jaco.T @ w_cf @ err_cf  # shape: (n_dof, 1)
                        grad_cf = 2 * w_cf @ err_cf  # shape: (n_con * 3, 1)
                        grad[:n_dof] += grad_dqa.reshape(-1)
                        grad[n_dof:] += grad_cf.reshape(-1)

                    if self.stage2_equal_joint_force_cost:
                        err_ef = self.err_ef
                        idx = np.arange(n_con) * 3 + 0
                        J_n = contact_jaco_h[idx, :]
                        grad_ef = (2 * J_n @ w_efj @ err_ef).flatten()
                        grad[n_dof + idx] += grad_ef.reshape(-1)

            if stage == 1 and b_use_arm_motion:
                # grad of cost_hb_pose
                self.robot_adaptor.compute_jaco_a(q_a)
                hb_jaco = self.robot_adaptor.get_frame_jaco(frame_name=hand_base_name, type="space")
                err_hb_pose = self.err_hb_pose
                grad_dqa = 2.0 * hb_jaco.T @ w_hb_pose @ err_hb_pose
                grad[:n_dof] += grad_dqa.reshape(-1)

            return grad

        def contact_model_constraint(x):
            dq_a = x[:n_dof].copy()
            cf = x[n_dof:].copy()
            dcf = cf - contact_force_all

            err = dcf.reshape(-1, 1) - Ks_jaco @ dq_a.reshape(-1, 1)
            if self.stage2_tan_force_constraint:
                constraint = err.reshape(-1, 3)
            else:
                constraint = err.reshape(-1, 3)[:, 0]  # only constrain the normal forces
            return constraint.reshape(-1)  # == 0

        def contact_model_constraint_grad(x):
            if self.stage2_tan_force_constraint:
                grad_cf = np.eye(3 * n_con)
                grad_dq_a = -Ks_jaco
            else:
                idx_normal = np.arange(0, n_con * 3, 3)
                grad_cf = np.zeros((n_con, 3 * n_con))
                grad_cf[np.arange(n_con), idx_normal] = 1.0
                grad_dq_a = -Ks_jaco[idx_normal, :]

            jacobian = np.hstack([grad_dq_a, grad_cf])
            return jacobian  # shape: (n_contacts, x_dim)

        def increase_normal_force_constraint(x):
            dq_a = x[:n_dof].copy()
            dcf = Ks_jaco @ dq_a.reshape(-1, 1)
            constraint = dcf.reshape(-1, 3)[:, 0]
            return constraint.reshape(-1)  # >= 0

        def increase_normal_force_constraint_grad(x):
            idx_normal = np.arange(0, n_con * 3, 3)
            grad_dq_a = Ks_jaco[idx_normal, :]
            grad_cf = np.zeros((n_con, 3 * n_con))
            jacobian = np.hstack([grad_dq_a, grad_cf])
            return jacobian

        def q_limits_constraint(x):
            dq_a = x[:n_dof].copy()
            q_a = curr_q_a + dq_a
            q_f = doa2dof_matrix @ q_a.reshape(-1, 1)

            In = np.eye(n_dof)
            A = np.concatenate([-In, In], axis=0)
            b = np.concatenate([joint_limits_f[0, :], -joint_limits_f[1, :]], axis=0).reshape(-1, 1)
            constraint = A @ q_f + b
            return -constraint.reshape(-1)  # >= 0

        def q_limits_constraint_grad(x):
            grad = np.zeros((2 * n_dof, len(x)))  # shape: (2*n_dof, len(x))
            A = np.concatenate([-np.eye(n_dof), np.eye(n_dof)], axis=0)
            grad_wrt_dq_a = -A @ doa2dof_matrix
            grad[:, :n_dof] = grad_wrt_dq_a
            return grad

        def friction_cone_constraint(x):
            """
            Input normal forces must be positive.
            """
            cf = x[n_dof:].reshape(-1, 3)
            constraint = mu * cf[:, 0] - np.linalg.norm(cf[:, 1:], axis=-1)  # >= 0
            return constraint.reshape(-1)

        def friction_cone_constraint_grad(x):
            cf = x[n_dof:].reshape(-1, 3)
            fx, fy, fz = cf[:, 0], cf[:, 1], cf[:, 2]
            norm_yz = np.sqrt(fy**2 + fz**2) + 1e-8  # Avoid divide-by-zero

            grad = np.zeros((n_con, x.shape[0]))
            idx = np.arange(n_con)
            grad[idx, n_dof + 3 * idx + 0] = mu  # ∂f/∂fx
            grad[idx, n_dof + 3 * idx + 1] = -fy / norm_yz  # ∂f/∂fy
            grad[idx, n_dof + 3 * idx + 2] = -fz / norm_yz  # ∂f/∂fz

            return grad

        def force_magnitude_constraint(x):
            """
            Input normal forces must be positive.
            """
            cf = x[n_dof:].reshape(-1, 3)
            constraint = desired_sum_force - np.sum(cf[:, 0])  # == 0 / >= 0
            return constraint.reshape(-1)

        def force_magnitude_constraint_grad(x):
            n_vars = x.shape[0]
            grad = np.zeros((1, n_vars))  # shape: (1, len(x))
            idx = np.arange(n_con) * 3 + 0  # index of normal force in each contact
            grad[0, n_dof + idx] = -1.0
            return grad  # shape: (1, len(x))

        def arm_doa_constraint(x):
            dq_a_arm = x[:n_arm_dof].copy()
            constraint = dq_a_arm - 0  # == 0
            return constraint.reshape(-1)

        def arm_doa_constraint_grad(x):
            grad = np.zeros((n_arm_dof, x.shape[0]))
            grad[:, :n_arm_dof] = np.eye(n_arm_dof)
            return grad

        if stage == 1:
            if n_con == 0:
                constraints_list = [dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad)]
            else:
                constraints_list = [
                    dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad),
                    dict(type="eq", fun=contact_model_constraint, jac=contact_model_constraint_grad),
                    dict(type="ineq", fun=force_magnitude_constraint, jac=force_magnitude_constraint_grad),
                ]
            if not b_use_arm_motion:
                constraints_list.append(dict(type="eq", fun=arm_doa_constraint, jac=arm_doa_constraint_grad))
        elif stage == 2:
            if n_con == 0:
                constraints_list = [
                    dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad),
                    dict(type="eq", fun=arm_doa_constraint, jac=arm_doa_constraint_grad),
                ]
            else:
                constraints_list = [
                    dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad),
                    dict(type="eq", fun=arm_doa_constraint, jac=arm_doa_constraint_grad),
                    dict(type="eq", fun=contact_model_constraint, jac=contact_model_constraint_grad),
                    dict(type="ineq", fun=friction_cone_constraint, jac=friction_cone_constraint_grad),
                    dict(type="eq", fun=force_magnitude_constraint, jac=force_magnitude_constraint_grad),
                ]
                if self.stage2_increase_force:
                    constraints_list.append(
                        dict(
                            type="ineq", fun=increase_normal_force_constraint, jac=increase_normal_force_constraint_grad
                        )
                    )

        bounds_dq = [(-q_step_max[i], q_step_max[i]) for i in range(q_step_max.shape[0])]
        bounds_cf = [(0, 100), (-50, 50), (-50, 50)] * n_con
        bounds = bounds_dq + bounds_cf
        x0 = np.concatenate([np.zeros((n_dof)), contact_force_all], axis=0)

        res = minimize(
            fun=objective,
            jac=jacobian,
            constraints=constraints_list,
            x0=x0,
            bounds=bounds,
            method="SLSQP",
            options={"ftol": 1e-6, "disp": b_print_opt_details, "maxiter": 200},
        )

        res_var = res.x.reshape(-1)
        dq_a = res_var[:n_dof]
        cf = res_var[n_dof:]
        res = {}
        res["q_a"] = curr_q_a + dq_a
        res["dq_a"] = dq_a
        res["cf"] = cf

        return res

    def ctrl_opt_bs3(
        self,
        stage,
        dt,
        curr_q_a,
        curr_q_f,
        target_q_f,
        last_dq_a,
        desired_sum_force=None,
        desired_forces=None,
        ho_contacts=None,
        grasp_matrix=None,
        b_print_opt_details=False,
    ):
        # hyper-parameters
        mu = self.mu

        # variables for coding convenience
        n_arm_dof = self.robot.arm.n_dof
        n_hand_dof = self.robot.hand.n_dof
        n_dof = n_arm_dof + n_hand_dof
        doa2dof_matrix = self.robot_adaptor.doa2dof_matrix
        n_con = len(ho_contacts)
        joint_limits_f = self.robot_adaptor.joint_limits_f
        q_step_max = np.asarray(self.robot.doa_max_vel) * dt

        if n_con:
            # compute grasp matrix
            contact_G = self.compute_grasp_matrix(ho_contacts) if grasp_matrix is None else grasp_matrix
            # compute Ks and contact jacobian
            updated_contacts, stacked = self.Ks(q_a=curr_q_a, q_f=curr_q_f, contacts=ho_contacts)
            contact_force_all = np.concatenate([c["contact_force"][:3] for c in updated_contacts], axis=0)
            contact_jaco_all = stacked["jaco_a"]
            # whether use Ks_h
            if stage == 2 and self.stage2_Ks_hand_only:
                Ks_all = stacked["Ks_h"]
                contact_jaco_all[:, :n_arm_dof] = 0  # remove arm part
            else:
                Ks_all = stacked["Ks"]
            Ks_jaco = Ks_all @ contact_jaco_all
        else:
            contact_force_all = np.zeros((0))

        # compute target hand base pose
        hand_base_name = self.robot.hand.base_name
        self.robot_adaptor.compute_fk_f(target_q_f)
        target_hb_pose = self.robot_adaptor.get_frame_pose(frame_name=hand_base_name)
        target_hb_pos, target_hb_ori = isometry3dToPosOri(target_hb_pose)

        # compute target hand base pose
        hand_base_name = self.robot.hand.base_name
        self.robot_adaptor.compute_fk_f(target_q_f)
        target_hb_pose = self.robot_adaptor.get_frame_pose(frame_name=hand_base_name)
        target_hb_pos, target_hb_ori = isometry3dToPosOri(target_hb_pose)

        # weights
        w_hb_pose = np.diag([0, 0, 100.0, 10.0, 10.0, 10.0])
        w_q_hand = 1.0 * np.eye(n_hand_dof)
        w_dqa = 0.01 * np.eye(n_dof)  #  <= 0.01
        w_ddqa = [0.001] * n_arm_dof + [0.001] * n_hand_dof
        w_ddqa = np.diag(w_ddqa)
        w_cp = np.diag([0.0, self.tan_motion_pen_weight, self.tan_motion_pen_weight])
        w_cp = block_diag(*[w_cp for _ in range(n_con)])
        w_cf = np.diag([1.0, 1.0, 1.0])
        w_cf = block_diag(*[w_cf for _ in range(n_con)])

        # desired forces of each contact
        cf_d = desired_forces

        if stage == 2 and n_con > 0:
            in_contact_q_indices = contact_jaco_all.any(axis=0)
            contact_jaco_h = contact_jaco_all[:, -n_hand_dof:]
            in_contact_qh_indices = np.any(contact_jaco_h != 0, axis=0)
            if self.stage2_incontact_force_only:
                # in-contact joint, no position control
                w_q_hand[in_contact_qh_indices, in_contact_qh_indices] = 0
            if self.stage2_penalize_contact_qda:
                w_dqa[in_contact_q_indices, in_contact_q_indices] *= 100

        def objective(x):
            dq_a = x[:n_dof].copy()
            cf = x[n_dof:].copy()
            q_a = curr_q_a + dq_a
            q_f = doa2dof_matrix @ q_a
            _, q_hand = q_f[:n_arm_dof], q_f[n_arm_dof:]

            # cost for hand qpos
            target_q_hand = target_q_f[n_arm_dof:]
            self.err_q_hand = err_q_hand = (q_hand - target_q_hand).reshape(-1, 1)
            cost_q_hand = err_q_hand.T @ w_q_hand @ err_q_hand

            # cost for dqa
            self.err_dqa = err_dqa = (dq_a / dt).reshape(-1, 1)
            cost_dqa = err_dqa.T @ w_dqa @ err_dqa

            # cost for ddqa
            self.err_ddqa = err_ddqa = ((dq_a - last_dq_a) / dt**2).reshape(-1, 1)
            cost_ddqa = err_ddqa.T @ w_ddqa @ err_ddqa

            cost_tan_motion = 0
            cost_cf = 0
            if n_con > 0:
                if stage == 1:
                    # cost tangential motion (restrict the tangential motion of contacts)
                    dp = contact_jaco_all @ dq_a.reshape(-1, 1)
                    self.err_cp = err_cp = dp
                    cost_tan_motion = err_cp.T @ w_cp @ err_cp

                if stage == 2:
                    self.err_cf = err_cf = (cf.reshape(-1, 3) - cf_d).reshape(-1, 1)
                    cost_cf = err_cf.T @ w_cf @ err_cf

            cost_hb_pose = 0
            if stage == 1:
                self.robot_adaptor.compute_fk_a(q_a)
                hb_pose = self.robot_adaptor.get_frame_pose(frame_name=hand_base_name)
                hb_pos, hb_quat = isometry3dToPosQuat(hb_pose)
                err_hb_pos = hb_pos - target_hb_pos
                hb_ori = sciR.from_quat(hb_quat)
                err_hb_ori = (hb_ori * target_hb_ori.inv()).as_rotvec()
                self.err_hb_pose = err_hb_pose = np.concatenate([err_hb_pos, err_hb_ori], axis=0).reshape(-1, 1)  # 6D
                cost_hb_pose = err_hb_pose.T @ w_hb_pose @ err_hb_pose

            total_cost = cost_dqa + cost_ddqa + cost_q_hand + cost_tan_motion + cost_cf + cost_hb_pose
            return total_cost.item()

        def jacobian(x):
            dq_a = x[:n_dof].copy()
            q_a = curr_q_a + dq_a
            grad = np.zeros(x.shape[0])

            # grad of cost_dqa
            err_dqa = self.err_dqa
            grad_dqa = 2.0 / dt * w_dqa @ err_dqa
            grad[:n_dof] += grad_dqa.reshape(-1)

            # grad of cost_ddqa
            err_ddqa = self.err_ddqa
            grad_dqa = 2.0 / dt**2 * w_ddqa @ err_ddqa
            grad[:n_dof] += grad_dqa.reshape(-1)

            # grad of cost_q_hand
            J_qhand_dqa = doa2dof_matrix[n_arm_dof:, :]  # (n_hand_dof, n_dof)
            err_q_hand = self.err_q_hand
            grad_dqa = 2.0 * (J_qhand_dqa.T @ w_q_hand @ err_q_hand).reshape(-1)  # shape: (n_dof,)
            grad[:n_dof] += grad_dqa.reshape(-1)

            if n_con > 0:
                if stage == 1:
                    # grad of cost_tan_motion
                    err_cp = self.err_cp
                    grad_dqa = 2.0 * (contact_jaco_all.T @ w_cp @ err_cp).reshape(-1)  # shape (n_dof,)
                    grad[:n_dof] += grad_dqa.reshape(-1)

                if stage == 2:
                    # grad of cost_cf
                    err_cf = self.err_cf
                    grad_cf = 2 * w_cf @ err_cf  # shape: (n_con * 3, 1)
                    grad[n_dof:] += grad_cf.reshape(-1)

            if stage == 1:
                # grad of cost_hb_pose
                self.robot_adaptor.compute_jaco_a(q_a)
                hb_jaco = self.robot_adaptor.get_frame_jaco(frame_name=hand_base_name, type="space")
                err_hb_pose = self.err_hb_pose
                grad_dqa = 2.0 * hb_jaco.T @ w_hb_pose @ err_hb_pose
                grad[:n_dof] += grad_dqa.reshape(-1)

            return grad

        def contact_model_constraint(x):
            dq_a = x[:n_dof].copy()
            cf = x[n_dof:].copy()
            dcf = cf - contact_force_all

            err = dcf.reshape(-1, 1) - Ks_jaco @ dq_a.reshape(-1, 1)
            constraint = err.reshape(-1, 3)
            return constraint.reshape(-1)  # == 0

        def contact_model_constraint_grad(x):
            grad_cf = np.eye(3 * n_con)
            grad_dq_a = -Ks_jaco
            jacobian = np.hstack([grad_dq_a, grad_cf])
            return jacobian  # shape: (n_contacts, x_dim)

        def q_limits_constraint(x):
            dq_a = x[:n_dof].copy()
            q_a = curr_q_a + dq_a
            q_f = doa2dof_matrix @ q_a.reshape(-1, 1)

            In = np.eye(n_dof)
            A = np.concatenate([-In, In], axis=0)
            b = np.concatenate([joint_limits_f[0, :], -joint_limits_f[1, :]], axis=0).reshape(-1, 1)
            constraint = A @ q_f + b
            return -constraint.reshape(-1)  # >= 0

        def q_limits_constraint_grad(x):
            grad = np.zeros((2 * n_dof, len(x)))  # shape: (2*n_dof, len(x))
            A = np.concatenate([-np.eye(n_dof), np.eye(n_dof)], axis=0)
            grad_wrt_dq_a = -A @ doa2dof_matrix
            grad[:, :n_dof] = grad_wrt_dq_a
            return grad

        def force_magnitude_constraint(x):
            """
            Input normal forces must be positive.
            """
            cf = x[n_dof:].reshape(-1, 3)
            constraint = desired_sum_force - np.sum(cf[:, 0])  # == 0 / >= 0
            return constraint.reshape(-1)

        def force_magnitude_constraint_grad(x):
            n_vars = x.shape[0]
            grad = np.zeros((1, n_vars))  # shape: (1, len(x))
            idx = np.arange(n_con) * 3 + 0  # index of normal force in each contact
            grad[0, n_dof + idx] = -1.0
            return grad  # shape: (1, len(x))

        def arm_doa_constraint(x):
            dq_a_arm = x[:n_arm_dof].copy()
            constraint = dq_a_arm - 0  # == 0
            return constraint.reshape(-1)

        def arm_doa_constraint_grad(x):
            grad = np.zeros((n_arm_dof, x.shape[0]))
            grad[:, :n_arm_dof] = np.eye(n_arm_dof)
            return grad

        if stage == 1:
            if n_con == 0:
                constraints_list = [dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad)]
            else:
                constraints_list = [
                    dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad),
                    dict(type="eq", fun=contact_model_constraint, jac=contact_model_constraint_grad),
                    dict(type="ineq", fun=force_magnitude_constraint, jac=force_magnitude_constraint_grad),
                ]
        elif stage == 2:
            if n_con == 0:
                constraints_list = [
                    dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad),
                    dict(type="eq", fun=arm_doa_constraint, jac=arm_doa_constraint_grad),
                ]
            else:
                constraints_list = [
                    dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad),
                    dict(type="eq", fun=arm_doa_constraint, jac=arm_doa_constraint_grad),
                    dict(type="eq", fun=contact_model_constraint, jac=contact_model_constraint_grad),
                ]

        bounds_dq = [(-q_step_max[i], q_step_max[i]) for i in range(q_step_max.shape[0])]
        bounds_cf = [(0, 100), (-50, 50), (-50, 50)] * n_con
        bounds = bounds_dq + bounds_cf
        x0 = np.concatenate([np.zeros((n_dof)), contact_force_all], axis=0)

        res = minimize(
            fun=objective,
            jac=jacobian,
            constraints=constraints_list,
            x0=x0,
            bounds=bounds,
            method="SLSQP",
            options={"ftol": 1e-6, "disp": b_print_opt_details, "maxiter": 200},
        )

        res_var = res.x.reshape(-1)
        dq_a = res_var[:n_dof]
        cf = res_var[n_dof:]

        res = {}
        res["q_a"] = curr_q_a + dq_a
        res["dq_a"] = dq_a
        res["cf"] = cf
        return res