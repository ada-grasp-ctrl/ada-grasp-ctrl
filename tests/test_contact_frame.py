"""Tests for geom-order-independent contact frame and wrench semantics."""

import unittest

import mujoco
import numpy as np

from ada_grasp_ctrl.utils.hand_util import canonicalize_contact_frame_wrench


class ContactCanonicalizationTest(unittest.TestCase):
    """Verify pure and native MuJoCo canonical contact invariants."""

    def test_pure_transform_preserves_arbitrary_world_wrench(self):
        """Canonicalize equivalent orderings with all six wrench entries nonzero."""
        canonical_frame = np.array(
            [
                [0.36, -0.48, 0.8],
                [0.8, 0.60, 0.0],
                [-0.48, 0.64, 0.60],
            ]
        )
        canonical_wrench = np.array([4.2, -0.7, 1.1, 0.3, -0.5, 0.9])
        direct_frame, direct_wrench = canonicalize_contact_frame_wrench(
            canonical_frame.T,
            canonical_wrench,
            hand_is_geom1=True,
        )

        swap = np.diag([-1.0, 1.0, -1.0])
        reversed_raw_frame = (canonical_frame @ swap).T
        reversed_raw_wrench = np.concatenate([-swap @ canonical_wrench[:3], -swap @ canonical_wrench[3:]])
        reversed_frame, reversed_wrench = canonicalize_contact_frame_wrench(
            reversed_raw_frame,
            reversed_raw_wrench,
            hand_is_geom1=False,
        )

        np.testing.assert_allclose(direct_frame, canonical_frame, atol=1e-12)
        np.testing.assert_allclose(reversed_frame, canonical_frame, atol=1e-12)
        np.testing.assert_allclose(direct_wrench, canonical_wrench, atol=1e-12)
        np.testing.assert_allclose(reversed_wrench, canonical_wrench, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(reversed_frame), 1.0)
        self.assertGreaterEqual(reversed_wrench[0], 0.0)
        np.testing.assert_allclose(
            reversed_frame @ reversed_wrench[:3],
            canonical_frame @ canonical_wrench[:3],
        )
        np.testing.assert_allclose(
            reversed_frame @ reversed_wrench[3:],
            canonical_frame @ canonical_wrench[3:],
        )

    @staticmethod
    def _native_contact(object_first: bool):
        """Create one sliding/twisting sphere contact with a selected geom order.

        Args:
            object_first: Whether the object body/geom is declared first.

        Returns:
            Canonical frame, local wrench, world wrench, and generalized wrench.
        """
        hand = '<body name="hand"><geom name="hand_geom" type="sphere" size=".05"/></body>'
        obj = '<body name="object" pos=".09 0 0"><freejoint/><geom name="object_geom" type="sphere" size=".05"/></body>'
        bodies = obj + hand if object_first else hand + obj
        xml = (
            '<mujoco><option gravity="0 0 0"/>'
            '<default><geom solref="0.005 1" friction="1 .1 .05" condim="6"/></default>'
            f"<worldbody>{bodies}</worldbody></mujoco>"
        )
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        # Translation produces two tangential forces; rotation around the
        # contact normal and tangents produces nonzero contact torque.
        data.qvel[:] = [0.0, 0.5, 0.2, 3.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        if data.ncon != 1:
            raise AssertionError(f"Expected one native contact, got {data.ncon}.")
        contact = data.contact[0]
        raw_wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, 0, raw_wrench)
        hand_is_geom1 = model.geom(contact.geom1).name == "hand_geom"
        frame, local_wrench = canonicalize_contact_frame_wrench(
            contact.frame,
            raw_wrench,
            hand_is_geom1=hand_is_geom1,
        )
        world_force = frame @ local_wrench[:3]
        world_torque_at_contact = frame @ local_wrench[3:]
        object_position = data.body("object").xpos.copy()
        lever = contact.pos.copy() - object_position
        world_wrench_at_com = np.concatenate([world_force, world_torque_at_contact + np.cross(lever, world_force)])
        return frame, local_wrench, world_wrench_at_com, data.qfrc_constraint.copy()

    def test_native_mujoco_geom_order_matches_object_generalized_wrench(self):
        """Swap geom declarations and retain normal, sliding, and torsional wrench."""
        hand_first = self._native_contact(object_first=False)
        object_first = self._native_contact(object_first=True)
        for frame, wrench, world_wrench, generalized_wrench in (hand_first, object_first):
            np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(frame), 1.0)
            self.assertGreaterEqual(wrench[0], 0.0)
            self.assertGreater(abs(wrench[1]), 1e-6)
            self.assertGreater(abs(wrench[2]), 1e-6)
            self.assertGreater(np.linalg.norm(wrench[3:]), 1e-6)
            np.testing.assert_allclose(world_wrench, generalized_wrench, rtol=1e-10, atol=1e-10)

        np.testing.assert_allclose(hand_first[0], object_first[0], atol=1e-12)
        np.testing.assert_allclose(hand_first[1], object_first[1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(hand_first[2], object_first[2], rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
