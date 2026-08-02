"""Tests for repository-maintenance scripts and ignore contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from script.build_example_fixtures import (
    HAND_NAMES,
    OBJECT_RELATIVE,
    SAMPLE_RELATIVE,
    FixtureBuildError,
    build_fixtures,
)
from script.run_ablation_baselines import ExperimentConfig, ExperimentRunnerError, run_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FixtureBuilderTest(unittest.TestCase):
    """Protect the fixture builder from hidden repository-output dependencies."""

    def setUp(self) -> None:
        """Create isolated source and destination roots.

        Returns:
            None.
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        """Release temporary files.

        Returns:
            None.
        """
        self.temporary.cleanup()

    @staticmethod
    def _record(dimension: int, *, dummy_arm: bool) -> dict[str, object]:
        """Build one minimally valid archived source record.

        Args:
            dimension: Joint-position dimension.
            dummy_arm: Whether to add dummy-arm-only fields.

        Returns:
            NumPy record mapping.
        """
        qpos = np.linspace(0.0, 1.0, dimension)
        record: dict[str, object] = {
            "obj_path": "/archive/original-object",
            "obj_pose": np.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]),
            "obj_scale": 0.06,
            "pregrasp_qpos": qpos,
            "grasp_qpos": qpos + 1.0,
            "squeeze_qpos": qpos + 2.0,
        }
        if dummy_arm:
            record.update(
                {
                    "joint_names": [f"joint_{index}" for index in range(dimension)],
                    "approach_qpos": np.stack((qpos, qpos + 0.5)),
                }
            )
        return record

    def _source_roots(self) -> tuple[dict[str, Path], dict[str, Path]]:
        """Write complete external source trees for all maintained hands.

        Returns:
            Formatted and dummy-arm root mappings.
        """
        formatted_roots = {}
        dummy_arm_roots = {}
        dimensions = {"shadow": 29, "allegro": 23, "leap_tac3d": 23}
        for hand in HAND_NAMES:
            formatted_root = self.root / "archive/formatted" / hand
            dummy_arm_root = self.root / "archive/dummy-arm" / hand
            for root, record in (
                (formatted_root, self._record(dimensions[hand], dummy_arm=False)),
                (dummy_arm_root, self._record(dimensions[hand] - 1, dummy_arm=True)),
            ):
                source = root / SAMPLE_RELATIVE
                source.parent.mkdir(parents=True, exist_ok=True)
                np.save(source, record)
            formatted_roots[hand] = formatted_root.resolve()
            dummy_arm_roots[hand] = dummy_arm_root.resolve()
        return formatted_roots, dummy_arm_roots

    def test_builder_uses_explicit_sources_outside_repository_output(self) -> None:
        """Generate every fixture without consulting the checkout's output tree."""
        formatted_roots, dummy_arm_roots = self._source_roots()
        destination = self.root / "generated/examples/data"

        generated = build_fixtures(formatted_roots, dummy_arm_roots, destination)

        self.assertEqual(len(generated), 10)
        self.assertTrue(all(path.is_file() for path in generated))
        for hand in HAND_NAMES:
            formatted = np.load(destination / hand / "formatted/grasp.npy", allow_pickle=True).item()
            dummy_arm = np.load(destination / hand / "dummy_arm/grasp.npy", allow_pickle=True).item()
            raw = np.load(destination / hand / "raw/learning.npy", allow_pickle=True).item()
            self.assertEqual(formatted["obj_path"], str(OBJECT_RELATIVE))
            self.assertEqual(dummy_arm["obj_path"], str(OBJECT_RELATIVE))
            self.assertEqual(formatted["schema_version"], 1)
            self.assertEqual(dummy_arm["schema_version"], 1)
            self.assertEqual(raw["schema_version"], 1)

    def test_builder_preflights_all_sources_before_writing(self) -> None:
        """Leave the destination untouched when one archived sample is missing."""
        formatted_roots, dummy_arm_roots = self._source_roots()
        (dummy_arm_roots["allegro"] / SAMPLE_RELATIVE).unlink()
        destination = self.root / "generated/examples/data"

        with self.assertRaisesRegex(FixtureBuildError, "Fixture source is missing"):
            build_fixtures(formatted_roots, dummy_arm_roots, destination)

        self.assertFalse(destination.exists())

    def test_builder_rejects_relative_source_roots(self) -> None:
        """Do not make fixture regeneration depend on the caller's CWD."""
        formatted_roots, dummy_arm_roots = self._source_roots()
        formatted_roots["shadow"] = Path("relative/shadow")

        with self.assertRaisesRegex(FixtureBuildError, "must be absolute"):
            build_fixtures(formatted_roots, dummy_arm_roots, self.root / "destination")

    def test_builder_reports_nonnumeric_archived_qpos(self) -> None:
        """Convert object-typed finite-check failures into an actionable error."""
        formatted_roots, dummy_arm_roots = self._source_roots()
        malformed_path = formatted_roots["shadow"] / SAMPLE_RELATIVE
        malformed = np.load(malformed_path, allow_pickle=True).item()
        malformed["pregrasp_qpos"] = np.array(["not-a-number"], dtype=object)
        np.save(malformed_path, malformed)

        with self.assertRaisesRegex(FixtureBuildError, "invalid finite one-dimensional pregrasp_qpos"):
            build_fixtures(formatted_roots, dummy_arm_roots, self.root / "destination")


class AblationRunnerTest(unittest.TestCase):
    """Protect experiment runs from stale outputs and implicit reports."""

    def setUp(self) -> None:
        """Create an isolated input archive and runtime roots.

        Returns:
            None.
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_root = self.root / "inputs"
        (self.input_root / "learn_dummy_arm_shadow/graspdata").mkdir(parents=True)
        self.asset_root = self.root / "assets"
        self.data_root = self.root / "data"
        self.asset_root.mkdir()
        self.data_root.mkdir()

    def tearDown(self) -> None:
        """Release temporary files.

        Returns:
            None.
        """
        self.temporary.cleanup()

    def _config(self, output_root: Path) -> ExperimentConfig:
        """Build one single-case runner configuration.

        Args:
            output_root: Fresh experiment path.

        Returns:
            Experiment configuration.
        """
        return ExperimentConfig(
            python_executable=sys.executable,
            input_root=self.input_root.resolve(),
            output_root=output_root.resolve(),
            asset_root=self.asset_root.resolve(),
            data_root=self.data_root.resolve(),
            hands=("shadow",),
            methods=("ours",),
            settings=("dist_2",),
            ours_ablations=("default",),
            seed=37,
            workers=3,
            max_num=5,
        )

    def test_runner_uses_unique_case_root_seed_and_current_eval_report(self) -> None:
        """Pass the exact eval report to statistics and record child exit one."""
        commands: list[list[str]] = []

        def run_command(command: list[str], **_: object) -> SimpleNamespace:
            """Record one command and create its declared structured report."""
            commands.append(command)
            log_directory = Path(next(value.split("=", 1)[1] for value in command if value.startswith("log_dir=")))
            task = next(value.split("=", 1)[1] for value in command if value.startswith("task="))
            log_directory.mkdir(parents=True, exist_ok=True)
            (log_directory / "run_report.json").write_text(json.dumps({"task": task}), encoding="utf-8")
            return SimpleNamespace(returncode=1 if task == "control_eval" else 0)

        output_root = self.root / "fresh-experiment"
        status = run_experiment(self._config(output_root), command_runner=run_command)

        self.assertEqual(status, 1)
        self.assertEqual(len(commands), 2)
        eval_command, stat_command = commands
        case_root = output_root / "dist_2/shadow/ours_default"
        eval_report = case_root / "log/control_eval/run_report.json"
        self.assertIn(f"output_root={case_root}", eval_command)
        self.assertIn(f"grasp_dir={self.input_root / 'learn_dummy_arm_shadow/graspdata'}", eval_command)
        self.assertIn("seed=37", eval_command)
        self.assertIn("n_worker=3", eval_command)
        self.assertIn("task.offsets=[0.02]", eval_command)
        self.assertIn("task.max_num=5", eval_command)
        self.assertIn(f"task.input_report={eval_report}", stat_command)

        report = json.loads((output_root / "experiment_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["exit_code"], 1)
        self.assertEqual(len(report["results"]), 1)
        self.assertEqual(report["results"][0]["control_eval"]["exit_code"], 1)
        self.assertTrue(report["results"][0]["control_stat"]["report_exists"])

    def test_runner_refuses_an_existing_output_root(self) -> None:
        """Prevent a new matrix from consuming reports from an older run."""
        output_root = self.root / "existing"
        output_root.mkdir()

        with self.assertRaisesRegex(ExperimentRunnerError, "already exists"):
            run_experiment(self._config(output_root), command_runner=lambda *_args, **_kwargs: None)

    def test_runner_rejects_ablation_path_traversal(self) -> None:
        """Keep every case below the fresh experiment root."""
        config = self._config(self.root / "fresh-experiment")
        config = ExperimentConfig(**{**config.__dict__, "ours_ablations": ("../escape",)})

        with self.assertRaisesRegex(ExperimentRunnerError, "safe path/config identifiers"):
            run_experiment(config, command_runner=lambda *_args, **_kwargs: None)


class RepositoryHygieneTest(unittest.TestCase):
    """Verify ignored generated paths without blanket data-format rules."""

    def setUp(self) -> None:
        """Skip Git-index checks outside a source checkout.

        Returns:
            None.
        """
        if not (PROJECT_ROOT / ".git").exists():
            self.skipTest("repository metadata is unavailable")

    def _git(self, *arguments: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        """Run one read-only Git query.

        Args:
            *arguments: Git arguments.
            input_text: Optional standard input.

        Returns:
            Completed Git process.
        """
        return subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_generated_and_external_paths_are_ignored(self) -> None:
        """Cover runtime, build, cache, editor, and restored-evidence paths."""
        paths = (
            "output/run/control.npy",
            "output/run/log/.hydra/config.yaml",
            "release/golden/artifact.json",
            "assets/object",
            "src/ada_grasp_ctrl/__pycache__/runtime.cpython-310.pyc",
            "build/lib/module.py",
            "dist/ada_grasp_ctrl.whl",
            "src/ada_grasp_ctrl.egg-info/PKG-INFO",
            ".pytest_cache/CACHEDIR.TAG",
            ".ruff_cache/CACHEDIR.TAG",
            ".coverage",
            ".venv/bin/python",
            "MUJOCO_LOG.TXT",
            "debug.xml",
            ".vscode/settings.json",
            ".DS_Store",
        )
        completed = self._git("check-ignore", "--no-index", "--stdin", input_text="\n".join(paths) + "\n")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.splitlines(), list(paths))

    def test_project_data_formats_are_not_blanket_ignored(self) -> None:
        """Keep maintained NumPy, MJCF/XML, and mesh/STL files eligible for Git."""
        for path in ("fixtures/sample.npy", "assets/hand/new_hand.xml", "assets/hand/mesh/finger.stl"):
            with self.subTest(path=path):
                completed = self._git("check-ignore", "--no-index", "--quiet", path)
                self.assertEqual(completed.returncode, 1, f"Unexpectedly ignored: {path}")


if __name__ == "__main__":
    unittest.main()
