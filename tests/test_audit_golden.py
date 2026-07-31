"""Tests for compact golden artifact hashing and verification."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from script.audit_golden import difference_report, inventory_tree, scientific_sha256, verify_inventory


class GoldenAuditTest(unittest.TestCase):
    """Verify compact evidence remains strict for scientific data."""

    def setUp(self):
        """Create an isolated directory for NPY evidence.

        Returns:
            None.
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        """Release temporary evidence files.

        Returns:
            None.
        """
        self.temporary.cleanup()

    def test_scientific_digest_ignores_only_approved_metadata(self):
        """Ignore timings and additive status metadata but retain trajectories."""
        baseline = {"value": np.array([1.0]), "t_total": 1.0}
        metadata_changed = {
            "value": np.array([1.0]),
            "t_total": 9.0,
            "schema_version": 1,
            "episode_status": "completed",
            "solver_diagnostics": [],
        }
        science_changed = {**metadata_changed, "value": np.array([2.0])}

        self.assertEqual(scientific_sha256(baseline), scientific_sha256(metadata_changed))
        self.assertNotEqual(scientific_sha256(baseline), scientific_sha256(science_changed))

    def test_inventory_verification_detects_scientific_changes(self):
        """Verify portable relative paths and scientific hashes."""
        expected_root = self.root / "expected"
        current_root = self.root / "current"
        expected_root.mkdir()
        current_root.mkdir()
        np.save(expected_root / "sample.npy", {"value": np.array([1.0])})
        np.save(current_root / "sample.npy", {"value": np.array([1.0]), "t_step": 2.0})
        expected = inventory_tree(expected_root, include_scientific=True)

        self.assertEqual(verify_inventory(expected, current_root, "scientific_sha256"), [])
        np.save(current_root / "sample.npy", {"value": np.array([1.0]), "episode_status": "execution_error"})
        self.assertEqual(
            verify_inventory(expected, current_root, "scientific_sha256"),
            ["sample.npy: classification differs"],
        )
        np.save(current_root / "sample.npy", {"value": np.array([3.0])})
        self.assertEqual(
            verify_inventory(expected, current_root, "scientific_sha256"),
            ["sample.npy: scientific_sha256 differs"],
        )

    def test_difference_report_records_classification_flips(self):
        """Retain the exact file and category for an approved behavior change."""
        historical_path = self.root / "historical.npy"
        current_path = self.root / "current.npy"
        np.save(
            historical_path,
            {
                "obj_pose": np.array(
                    [
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
                    ]
                )
            },
        )
        np.save(
            current_path,
            {
                "obj_pose": np.array(
                    [
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    ]
                )
            },
        )

        report = difference_report([("sample.npy", historical_path, current_path)])

        self.assertEqual(report["changed_file_count"], 1)
        self.assertEqual(report["classification_change_count"], 1)
        self.assertEqual(report["mismatch_categories"], {"numeric": 1})


if __name__ == "__main__":
    unittest.main()
