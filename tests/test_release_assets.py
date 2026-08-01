"""Tests for release asset reachability and recorded checksums."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECT_ROOT = PROJECT_ROOT / "examples/assets/object/core_bottle_15787789482f045d8add95bf56d3d2fa"
ACTIVE_HAND_XML = (
    PROJECT_ROOT / "assets/hand/allegro/right_hand.xml",
    PROJECT_ROOT / "assets/hand/dummy_arm_allegro/right.xml",
    PROJECT_ROOT / "assets/hand/shadow/right_hand.xml",
    PROJECT_ROOT / "assets/hand/dummy_arm_shadow/right_no_tendon.xml",
    PROJECT_ROOT / "assets/hand/leap_tac3d/leap_tac3d.xml",
    PROJECT_ROOT / "assets/hand/dummy_arm_leap_tac3d/leap_tac3d.xml",
)


class ReleaseAssetTest(unittest.TestCase):
    """Protect release assets from stale records and broken mesh paths."""

    def test_object_attribution_checksums_match_files(self) -> None:
        """Verify every object checksum recorded in the attribution file."""
        attribution = (OBJECT_ROOT / "ATTRIBUTION.md").read_text(encoding="utf-8")
        records = re.findall(r"^([0-9a-f]{64})  (.+)$", attribution, flags=re.MULTILINE)
        self.assertTrue(records, "ATTRIBUTION.md contains no SHA-256 records")
        for expected, relative_path in records:
            asset_path = OBJECT_ROOT / relative_path
            self.assertTrue(asset_path.is_file(), asset_path)
            actual = hashlib.sha256(asset_path.read_bytes()).hexdigest()
            self.assertEqual(actual, expected, asset_path)

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


if __name__ == "__main__":
    unittest.main()
