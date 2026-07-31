"""Tests for strict and machine-readable golden trajectory comparison."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from script.compare_golden import (
    build_report,
    classify,
    compare_directories,
    write_json_report,
)


class GoldenComparisonTest(unittest.TestCase):
    """Protect release comparison semantics from silent weakening."""

    def setUp(self):
        """Create isolated baseline and current directories.

        Returns:
            None.
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.baseline_root = self.root / "baseline"
        self.current_root = self.root / "current"
        self.baseline_root.mkdir()
        self.current_root.mkdir()

    def tearDown(self):
        """Release temporary comparison files.

        Returns:
            None.
        """
        self.temporary.cleanup()

    def _write_pair(self, expected: dict, actual: dict) -> None:
        """Write one corresponding baseline/current record pair.

        Args:
            expected: Baseline record.
            actual: Current record.

        Returns:
            None.
        """
        np.save(self.baseline_root / "sample.npy", expected)
        np.save(self.current_root / "sample.npy", actual)

    def test_unexpected_scientific_keys_are_not_silently_accepted(self):
        """Allow declared metadata additions but reject new scientific fields."""
        expected = {"obj_pose": np.zeros((1, 7)), "contacts": [[]]}
        actual = {
            **expected,
            "schema_version": 1,
            "episode_status": "completed",
            "solver_diagnostics": [],
            "unreviewed_signal": np.ones(1),
        }
        self._write_pair(expected, actual)

        failures = compare_directories(self.baseline_root, self.current_root)

        self.assertEqual(len(failures), 1)
        self.assertIn("unexpected keys ['unreviewed_signal']", failures[0])

    def test_declared_execution_error_controls_classification(self):
        """Never reinterpret a declared execution error as a scientific result."""
        record = {
            "episode_status": "execution_error",
            "obj_pose": np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
                ]
            ),
        }

        self.assertEqual(classify(record), "execution_error")

    def test_json_report_records_tolerances_counts_and_failures(self):
        """Persist enough structured evidence for release automation."""
        self._write_pair({"value": np.zeros(1)}, {"value": np.ones(1)})
        failures = compare_directories(self.baseline_root, self.current_root)
        report = build_report(self.baseline_root, self.current_root, failures)
        report_path = self.root / "reports" / "comparison.json"

        write_json_report(report_path, report)
        persisted = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertFalse(persisted["passed"])
        self.assertEqual(persisted["expected_file_count"], 1)
        self.assertEqual(persisted["actual_file_count"], 1)
        self.assertEqual(persisted["failure_count"], 1)
        self.assertEqual(persisted["rtol"], 1e-5)
        self.assertEqual(persisted["atol"], 1e-6)


if __name__ == "__main__":
    unittest.main()
