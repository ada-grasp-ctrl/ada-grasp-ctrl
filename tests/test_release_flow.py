"""Tests for current-run-only statistics and example reporting."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
import yaml

from ada_grasp_ctrl.errors import PreflightError
from ada_grasp_ctrl.tasks.control_stat import _discover_control_paths, _reported_control_paths
from script.report_example import ReportError, summarize


class ReleaseFlowTest(unittest.TestCase):
    """Protect release scripts from scanning outputs from older runs."""

    def setUp(self) -> None:
        """Create an isolated current-run directory.

        Returns:
            None.
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        """Release the isolated directory.

        Returns:
            None.
        """
        self.temporary.cleanup()

    def _write_json(self, path: Path, value: dict) -> None:
        """Write one JSON fixture below the temporary root.

        Args:
            path: Destination path.
            value: JSON mapping to serialize.

        Returns:
            None.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def _control_config(self, control_root: Path, report_path: Path) -> OmegaConf:
        """Build the control-stat fields needed by discovery helpers.

        Args:
            control_root: Configured current-run control directory.
            report_path: Explicit control-eval report.

        Returns:
            Minimal OmegaConf configuration.
        """
        return OmegaConf.create(
            {
                "control_dir": str(control_root),
                "task": {
                    "method": "ours",
                    "ablation_name": "default",
                    "setting_name": "dist_0",
                    "input_report": str(report_path),
                },
            }
        )

    def _run_example_with_recording_python(
        self, *arguments: str
    ) -> tuple[subprocess.CompletedProcess, list[list[str]]]:
        """Run the example script with a Python stub that records stage arguments.

        Args:
            *arguments: Arguments passed after ``script/run_example.sh``.

        Returns:
            The completed shell process and one argument list per Python invocation.
        """
        invocation_log = self.root / "python-invocations.jsonl"
        invocation_log.unlink(missing_ok=True)
        python_stub = self.root / "record-python"
        python_stub.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "import os\n"
            "import sys\n"
            "with open(os.environ['ADA_GRASP_CTRL_TEST_INVOCATIONS'], 'a', encoding='utf-8') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n",
            encoding="utf-8",
        )
        python_stub.chmod(0o755)
        environment = os.environ.copy()
        environment.update(
            {
                "ADA_GRASP_CTRL_EXAMPLE_BASE": str(self.root / "examples"),
                "ADA_GRASP_CTRL_RUN_ID": "viewer-" + "-".join(argument.replace("--", "") for argument in arguments),
                "ADA_GRASP_CTRL_TEST_INVOCATIONS": str(invocation_log),
                "PYTHON_BIN": str(python_stub),
            }
        )

        completed = subprocess.run(
            ["bash", "script/run_example.sh", *arguments],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        invocations = []
        if invocation_log.exists():
            invocations = [json.loads(line) for line in invocation_log.read_text(encoding="utf-8").splitlines()]
        return completed, invocations

    @staticmethod
    def _control_eval_invocation(invocations: list[list[str]]) -> list[str]:
        """Select the recorded control-eval invocation.

        Args:
            invocations: Recorded Python argument lists.

        Returns:
            The single control-eval argument list.
        """
        matches = [arguments for arguments in invocations if "task=control_eval" in arguments]
        if len(matches) != 1:
            raise AssertionError(f"Expected one control_eval invocation, found {len(matches)}: {invocations}")
        return matches[0]

    def test_control_stat_uses_only_reported_current_outputs(self) -> None:
        """Ignore an old file in control_dir when the current report is explicit."""
        control_root = self.root / "control"
        method_root = control_root / "ours_default"
        method_root.mkdir(parents=True)
        current = method_root / "current_dist_0.npy"
        stale = method_root / "stale_dist_0.npy"
        np.save(current, {"value": 1})
        np.save(stale, {"value": 2})
        report_path = self.root / "log" / "control_eval" / "run_report.json"
        self._write_json(
            report_path,
            {
                "task": "control_eval",
                "results": [{"output_paths": [str(current)], "status": "completed"}],
            },
        )

        discovered, source = _discover_control_paths(self._control_config(control_root, report_path))

        self.assertEqual(discovered, [str(current)])
        self.assertIn(str(report_path), source)

    def test_control_report_cannot_escape_configured_control_root(self) -> None:
        """Reject a stale or unrelated report that points outside this run."""
        control_root = self.root / "current" / "control"
        outside = self.root / "old" / "control" / "ours_default" / "old_dist_0.npy"
        outside.parent.mkdir(parents=True)
        np.save(outside, {"value": 1})
        report_path = self.root / "current" / "log" / "control_eval" / "run_report.json"
        self._write_json(report_path, {"task": "control_eval", "results": [{"output_paths": [str(outside)]}]})

        with self.assertRaisesRegex(PreflightError, "outside control_dir"):
            _reported_control_paths(report_path, control_root)

    def test_control_stat_rejects_report_for_another_method(self) -> None:
        """Fail when explicit outputs cannot satisfy the requested method."""
        control_root = self.root / "control"
        current = control_root / "op" / "current_dist_0.npy"
        current.parent.mkdir(parents=True)
        np.save(current, {"value": 1})
        report_path = self.root / "log" / "control_eval" / "run_report.json"
        self._write_json(report_path, {"task": "control_eval", "results": [{"output_paths": [str(current)]}]})

        with self.assertRaisesRegex(PreflightError, "none match"):
            _discover_control_paths(self._control_config(control_root, report_path))

    def test_example_report_uses_current_manifest_and_requires_statistics(self) -> None:
        """Summarize one declared output while ignoring an unreported stale file."""
        control_root = self.root / "control" / "ours_default"
        control_root.mkdir(parents=True)
        current = control_root / "current_dist_0.npy"
        stale = control_root / "stale_dist_0.npy"
        pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        lifted = pose.copy()
        lifted[2] = 0.2
        np.save(current, {"episode_status": "completed", "obj_pose": np.stack([pose, lifted])})
        np.save(stale, {"episode_status": "execution_error", "obj_pose": []})
        self._write_json(
            self.root / "log" / "control_eval" / "run_report.json",
            {"task": "control_eval", "results": [{"output_paths": [str(current)]}]},
        )
        self._write_json(
            self.root / "log" / "control_stat" / "run_report.json",
            {
                "task": "control_stat",
                "num_discovered": 1,
                "num_processed": 1,
                "results": [{"input_path": str(current), "output_paths": [str(current)]}],
            },
        )
        statistics_path = self.root / "control_stat_res" / "dist_0_ours_default.yaml"
        statistics_path.parent.mkdir()
        statistics_path.write_text(
            yaml.safe_dump(
                {
                    "success_rate": 1.0,
                    "num_valid_cases": 1,
                    "num_total": 1,
                    "success": 1,
                    "failure": 0,
                    "invalid_initialization": 0,
                    "solver_degraded": 0,
                    "execution_error": 0,
                    "success_rate_denominator": 1,
                }
            ),
            encoding="utf-8",
        )

        output = io.StringIO()
        with redirect_stdout(output):
            summarize(self.root)

        rendered = output.getvalue()
        self.assertIn("current_dist_0.npy", rendered)
        self.assertNotIn("stale_dist_0.npy", rendered)
        self.assertIn("success_rate=1.0", rendered)

    def test_example_report_fails_instead_of_claiming_empty_success(self) -> None:
        """Treat missing current-run reports as a report-stage failure."""
        with self.assertRaisesRegex(ReportError, "required report"):
            summarize(self.root)

    def test_example_report_rejects_stale_statistics_counts(self) -> None:
        """Reject a statistics file whose totals describe another run."""
        control = self.root / "control" / "ours_default" / "current_dist_0.npy"
        control.parent.mkdir(parents=True)
        np.save(control, {"episode_status": "completed", "obj_pose": []})
        self._write_json(
            self.root / "log" / "control_eval" / "run_report.json",
            {"task": "control_eval", "results": [{"output_paths": [str(control)]}]},
        )
        self._write_json(
            self.root / "log" / "control_stat" / "run_report.json",
            {
                "task": "control_stat",
                "num_discovered": 1,
                "num_processed": 1,
                "results": [{"input_path": str(control)}],
            },
        )
        statistics_path = self.root / "control_stat_res" / "dist_0_ours_default.yaml"
        statistics_path.parent.mkdir()
        statistics_path.write_text(
            yaml.safe_dump(
                {
                    "success_rate": 1.0,
                    "num_valid_cases": 2,
                    "num_total": 2,
                    "success": 2,
                    "failure": 0,
                    "invalid_initialization": 0,
                    "solver_degraded": 0,
                    "execution_error": 0,
                    "success_rate_denominator": 2,
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ReportError, "counts do not match"):
            summarize(self.root)

    def test_example_script_refuses_an_existing_run_directory(self) -> None:
        """Prevent an explicit run ID from reusing reports from an older run."""
        existing = self.root / "shadow" / "fixed-run"
        existing.mkdir(parents=True)
        environment = os.environ.copy()
        environment.update(
            {
                "ADA_GRASP_CTRL_EXAMPLE_BASE": str(self.root),
                "ADA_GRASP_CTRL_RUN_ID": "fixed-run",
                "PYTHON_BIN": sys.executable,
            }
        )

        completed = subprocess.run(
            ["bash", "script/run_example.sh", "shadow", "quick"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("already exists", completed.stderr)

    def test_example_script_defaults_to_headless_viewing(self) -> None:
        """Keep the existing two-argument quick command non-interactive."""
        commands = (
            ("shadow", "quick"),
            ("shadow", "quick", "--viewer", "none"),
        )
        for arguments in commands:
            with self.subTest(arguments=arguments):
                completed, invocations = self._run_example_with_recording_python(*arguments)

                self.assertEqual(completed.returncode, 0, completed.stderr)
                control_eval = self._control_eval_invocation(invocations)
                self.assertIn("task.debug_viewer=false", control_eval)
                self.assertNotIn("task.debug_viewer_backend=mjviser", control_eval)

    def test_example_script_passes_exact_mjviser_overrides_for_all_hands(self) -> None:
        """Expose the browser viewer consistently for every quick fixture."""
        for hand in ("shadow", "allegro", "leap_tac3d"):
            with self.subTest(hand=hand):
                completed, invocations = self._run_example_with_recording_python(
                    hand,
                    "quick",
                    "--viewer",
                    "mjviser",
                )

                self.assertEqual(completed.returncode, 0, completed.stderr)
                control_eval = self._control_eval_invocation(invocations)
                self.assertIn("task.debug_viewer=true", control_eval)
                self.assertIn("task.debug_viewer_backend=mjviser", control_eval)
                self.assertNotIn("task.debug_viewer=false", control_eval)

    def test_example_script_rejects_malformed_viewer_arguments_before_pipeline(self) -> None:
        """Return usage exit code 2 without invoking any task stage."""
        invalid_arguments = (
            ("shadow", "quick", "--viewer"),
            ("shadow", "quick", "--view", "mjviser"),
            ("shadow", "quick", "--viewer", "mujoco"),
            ("shadow", "quick", "--viewer", "mjviser", "extra"),
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                completed, invocations = self._run_example_with_recording_python(*arguments)

                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertIn("Usage: bash script/run_example.sh", completed.stderr)
                self.assertEqual(invocations, [])

    def test_release_gate_refuses_a_nonempty_artifact_root(self) -> None:
        """Prevent a release comparison from consuming artifacts from an older run."""
        release_root = self.root / "release"
        release_root.mkdir()
        (release_root / "stale.txt").write_text("old run", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "ADA_GRASP_CTRL_RELEASE_GATE_ROOT": str(release_root),
                "PYTHON_BIN": sys.executable,
            }
        )

        completed = subprocess.run(
            ["bash", "script/run_release_gate.sh", "quick"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("not empty", completed.stderr)


if __name__ == "__main__":
    unittest.main()
