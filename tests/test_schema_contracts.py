"""Focused tests for exact raw, grasp, scene, asset, and control schemas."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ada_grasp_ctrl.schema import (
    SchemaError,
    load_npy_record,
    validate_control_record,
    validate_grasp_record,
    validate_raw_record,
    validate_scene_record,
)
from ada_grasp_ctrl.utils.robots.base import RobotFactory


class ExactSchemaContractTest(unittest.TestCase):
    """Reject malformed data before MuJoCo, IK, or statistics consume it."""

    def setUp(self):
        """Create an isolated asset root.

        Returns:
            None.
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        """Release temporary schema fixtures.

        Returns:
            None.
        """
        self.temporary.cleanup()

    def _pose_qpos(self, shape: tuple[int, ...]) -> np.ndarray:
        """Create finite pose-plus-joint arrays with valid WXYZ quaternions.

        Args:
            shape: Full array shape whose final dimension includes a pose.

        Returns:
            Zero array with unit scalar quaternion entries.
        """
        values = np.zeros(shape)
        values[..., 3] = 1.0
        return values

    def _grasp_record(self, qpos_dim: int) -> dict:
        """Create a minimal structurally valid grasp record.

        Args:
            qpos_dim: Length of each qpos vector.

        Returns:
            Mutable grasp record.
        """
        qpos = np.zeros(qpos_dim)
        return {
            "obj_path": str(self.root / "object"),
            "obj_pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "obj_scale": 1.0,
            "pregrasp_qpos": qpos.copy(),
            "grasp_qpos": qpos.copy(),
            "squeeze_qpos": qpos.copy(),
        }

    def test_raw_converters_require_exact_rank_and_final_dimension(self):
        """Distinguish BODex trajectories, Learning singles, and Batched arrays."""
        joint_names = [f"joint_{index}" for index in range(16)]
        bodex = {
            "scene_path": "scene.npy",
            "robot_pose": self._pose_qpos((1, 2, 3, 23)),
            "joint_names": joint_names,
        }
        validate_raw_record(bodex, "BODex", "bodex.npy", expected_qpos_dim=23)
        malformed_bodex = {**bodex, "robot_pose": self._pose_qpos((2, 3, 23))}
        with self.assertRaisesRegex(SchemaError, "robot_pose"):
            validate_raw_record(malformed_bodex, "BODex", "bodex.npy", expected_qpos_dim=23)

        single = self._pose_qpos((23,))
        learning = {
            "scene_path": "scene.npy",
            "pregrasp_qpos": single,
            "grasp_qpos": single,
            "squeeze_qpos": single,
        }
        validate_raw_record(learning, "Learning", "learning.npy", expected_qpos_dim=23)
        with self.assertRaisesRegex(SchemaError, "ndim=1"):
            validate_raw_record(
                {key: np.stack([value, value]) if key.endswith("qpos") else value for key, value in learning.items()},
                "Learning",
                "learning.npy",
                expected_qpos_dim=23,
            )

        batched = {
            "scene_path": "scene.npy",
            "pregrasp_qpos": np.stack([single, single]),
            "grasp_qpos": np.stack([single, single]),
            "squeeze_qpos": np.stack([single, single]),
            "scene_scale": np.ones(2),
        }
        validate_raw_record(batched, "Batched", "batch.npy", expected_qpos_dim=23)
        with self.assertRaisesRegex(SchemaError, "ndim=2"):
            validate_raw_record(learning, "Batched", "batch.npy", expected_qpos_dim=23)

    def test_grasp_layout_validates_exact_joint_count_and_order(self):
        """Support pose-prefixed hand data and exact full-joint control data."""
        hand_names = ["joint_a", "joint_b"]
        pose_prefixed = self._grasp_record(9)
        pose_prefixed["joint_names"] = hand_names
        validate_grasp_record(
            pose_prefixed,
            "formatted.npy",
            expected_joint_dim=9,
            expected_joint_names=hand_names,
            qpos_prefix_dim=7,
        )

        full_joint = self._grasp_record(2)
        full_joint["joint_names"] = hand_names
        validate_grasp_record(
            full_joint,
            "control-input.npy",
            expected_joint_dim=2,
            expected_joint_names=hand_names,
            require_joint_names=True,
        )
        full_joint["joint_names"] = list(reversed(hand_names))
        with self.assertRaisesRegex(SchemaError, "configured joint order"):
            validate_grasp_record(
                full_joint,
                "control-input.npy",
                expected_joint_dim=2,
                expected_joint_names=hand_names,
                require_joint_names=True,
            )

    def test_scene_and_object_assets_are_validated_with_precise_paths(self):
        """Name missing scene meshes and malformed metadata fields exactly."""
        mesh_path = self.root / "scene_object" / "mesh" / "object.obj"
        scene = {
            "task": {"obj_name": "target"},
            "scene": {
                "target": {
                    "file_path": str(mesh_path),
                    "pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                    "scale": [1.0],
                }
            },
        }
        with self.assertRaisesRegex(SchemaError, "scene.target.file_path"):
            validate_scene_record(scene, "scene.npy")
        mesh_path.parent.mkdir(parents=True)
        mesh_path.write_text("o object\n", encoding="utf-8")
        validate_scene_record(scene, "scene.npy")

        object_root = self.root / "object"
        (object_root / "info").mkdir(parents=True)
        collision_root = object_root / "urdf" / "meshes"
        collision_root.mkdir(parents=True)
        (collision_root / "convex_piece_000.obj").write_text("o collision\n", encoding="utf-8")
        (object_root / "info" / "simplified.json").write_text(json.dumps({"mass": 1.0, "scale": 1.0}))
        grasp = self._grasp_record(2)
        with self.assertRaisesRegex(SchemaError, "simplified.json.density"):
            validate_grasp_record(grasp, "grasp.npy", require_assets=True)

    def test_control_contacts_require_shapes_frames_and_step_alignment(self):
        """Validate six-value wrenches, right-handed frames, and control lengths."""
        pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        contact = {
            "contact_pos": np.zeros(3),
            "contact_force": np.zeros(6),
            "contact_frame": np.eye(3),
        }
        valid = {
            "obj_pose": np.stack([pose, pose]),
            "contacts": [[contact]],
            "dof": np.zeros((1, 2)),
            "doa": np.zeros((1, 2)),
            "planned_dof": np.zeros((1, 2)),
            "episode_status": "completed",
        }
        validate_control_record(valid, "control.npy")

        wrong_force = deepcopy(valid)
        wrong_force["contacts"][0][0]["contact_force"] = np.zeros(3)
        with self.assertRaisesRegex(SchemaError, "contact_force"):
            validate_control_record(wrong_force, "control.npy")

        left_handed = deepcopy(valid)
        left_handed["contacts"][0][0]["contact_frame"] = np.diag([-1.0, 1.0, 1.0])
        with self.assertRaisesRegex(SchemaError, "right-handed"):
            validate_control_record(left_handed, "control.npy")

        misaligned = deepcopy(valid)
        misaligned["dof"] = np.zeros((2, 2))
        with self.assertRaisesRegex(SchemaError, "aligned with contacts"):
            validate_control_record(misaligned, "control.npy")

    def test_bundled_quick_records_satisfy_exact_grasp_schema(self):
        """Validate all 300 checked-in quick records and configured joint orders."""
        project_root = Path(__file__).resolve().parents[1]
        for hand in ("shadow", "allegro", "leap_tac3d"):
            prefix = "rh_" if hand != "allegro" else ""
            robot = RobotFactory.create_robot(robot_type=f"dummy_arm_{hand}", prefix=prefix)
            paths = sorted((project_root / f"examples/data/{hand}/dummy_arm").rglob("*.npy"))
            self.assertEqual(len(paths), 100)
            for path in paths:
                record = load_npy_record(path)
                self.assertEqual(record.get("schema_version"), 1)
                validate_grasp_record(
                    record,
                    path,
                    expected_joint_dim=robot.n_dof,
                    expected_joint_names=robot.dof_names,
                    require_joint_names=True,
                )


if __name__ == "__main__":
    unittest.main()
