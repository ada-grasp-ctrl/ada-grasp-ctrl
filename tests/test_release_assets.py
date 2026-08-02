"""Tests for release asset reachability and recorded checksums."""

from __future__ import annotations

import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from script.audit_example_fixtures import audit_manifest
from script.build_example_fixtures import sha256_file
from script.validate_quick_results import classification_digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECT_ROOT = PROJECT_ROOT / "examples/assets/object/DGN_2k"
QUICK_MANIFEST = PROJECT_ROOT / "examples/quick_manifest.json"
EXPECTED_STATUS = PROJECT_ROOT / "examples/quick_expected_status.json"
ACTIVE_HAND_XML = (
    PROJECT_ROOT / "assets/hand/allegro/right_hand.xml",
    PROJECT_ROOT / "assets/hand/dummy_arm_allegro/right.xml",
    PROJECT_ROOT / "assets/hand/shadow/right_hand.xml",
    PROJECT_ROOT / "assets/hand/dummy_arm_shadow/right_no_tendon.xml",
    PROJECT_ROOT / "assets/hand/leap_tac3d/leap_tac3d.xml",
    PROJECT_ROOT / "assets/hand/dummy_arm_leap_tac3d/leap_tac3d.xml",
)
ACTIVE_LEAP_MESHES = {
    "leap_hand/meshes/dip.stl",
    "leap_hand/meshes/fingertip_base_tac3d.stl",
    "leap_hand/meshes/fingertip_tac3d.stl",
    "leap_hand/meshes/mcp_joint.stl",
    "leap_hand/meshes/palm_lower.stl",
    "leap_hand/meshes/pip.stl",
    "leap_hand/meshes/thumb_dip.stl",
    "leap_hand/meshes/thumb_fingertip_base_tac3d.stl",
    "leap_hand/meshes/thumb_pip.stl",
}
REMOVED_HAND_ASSETS = (
    PROJECT_ROOT / "assets/hand/leap",
    PROJECT_ROOT / "assets/hand/ur5_leap_tac3d",
    PROJECT_ROOT / "assets/hand/ur10e_shadow",
    PROJECT_ROOT / "assets/hand/attach_hand_to_arm.py",
    PROJECT_ROOT / "assets/hand/dummy_arm_shadow/right.xml",
    PROJECT_ROOT / "assets/hand/shadow/right_with_forearm.xml",
    PROJECT_ROOT / "assets/hand/leap_tac3d/leap_hand/meshes/fingertip_base.stl",
    PROJECT_ROOT / "assets/hand/leap_tac3d/leap_hand/meshes/fingertip_custom.stl",
    PROJECT_ROOT / "assets/hand/leap_tac3d/leap_hand/meshes/palm_lower_left.stl",
    PROJECT_ROOT / "assets/hand/leap_tac3d/leap_hand/meshes/thumb_fingertip_base.stl",
    PROJECT_ROOT / "assets/hand/leap_tac3d/leap_hand/meshes/thumb_left_temp_base.stl",
)


class ReleaseAssetTest(unittest.TestCase):
    """Protect release assets from stale records and broken mesh paths."""

    def test_quick_manifest_audits_all_bundled_dgn_files(self) -> None:
        """Verify the exact 3x100 fixture and 89-object checksum inventory."""
        counts = audit_manifest(QUICK_MANIFEST, PROJECT_ROOT)

        self.assertEqual(counts["grasp_records"], 300)
        self.assertEqual(counts["object_ids"], 89)
        attribution = (OBJECT_ROOT / "ATTRIBUTION.md").read_text(encoding="utf-8")
        self.assertIn("DGN 2k", attribution)
        manifest = json.loads(QUICK_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["counts"]["scene_configs"], 100)

    def test_expected_status_inventory_is_tied_to_fixture_manifest(self) -> None:
        """Verify all three 100-record classification inventories and their digests."""
        inventory = json.loads(EXPECTED_STATUS.read_text(encoding="utf-8"))

        self.assertEqual(inventory["schema_version"], 1)
        self.assertEqual(inventory["fixture_manifest_sha256"], sha256_file(QUICK_MANIFEST))
        self.assertEqual(set(inventory["hands"]), {"shadow", "allegro", "leap_tac3d"})
        for hand, hand_inventory in inventory["hands"].items():
            with self.subTest(hand=hand):
                records = hand_inventory["records"]
                self.assertEqual(len(records), 100)
                self.assertEqual(sum(hand_inventory["counts"].values()), 100)
                self.assertEqual(hand_inventory["aggregate_sha256"], classification_digest(records))

    def test_active_hand_mjcf_meshes_exist(self) -> None:
        """Resolve every mesh referenced by each maintained hand MJCF."""
        for xml_path in ACTIVE_HAND_XML:
            root = ET.parse(xml_path).getroot()
            compiler = root.find("compiler")
            mesh_root = xml_path.parent
            if compiler is not None and compiler.get("meshdir"):
                mesh_root = (mesh_root / compiler.get("meshdir", "")).resolve()
            for mesh in root.findall(".//mesh"):
                relative_path = mesh.get("file")
                if relative_path is None:
                    continue
                with self.subTest(xml=xml_path, mesh=relative_path):
                    self.assertTrue((mesh_root / relative_path).is_file())

    def test_active_leap_mjcf_uses_only_retained_meshes(self) -> None:
        """Keep both LEAP Tac3D models on the exact retained nine-mesh set."""
        for xml_path in ACTIVE_HAND_XML[-2:]:
            root = ET.parse(xml_path).getroot()
            referenced = {mesh.get("file") for mesh in root.findall(".//mesh") if mesh.get("file") is not None}
            with self.subTest(xml=xml_path):
                self.assertEqual(referenced, ACTIVE_LEAP_MESHES)

    def test_removed_legacy_hand_assets_stay_absent(self) -> None:
        """Prevent statically unreachable legacy assets from returning."""
        for asset_path in REMOVED_HAND_ASSETS:
            with self.subTest(asset=asset_path):
                self.assertFalse(asset_path.exists(), asset_path)


if __name__ == "__main__":
    unittest.main()
