"""Tests for repository-maintenance scripts and ignore contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from script.build_example_fixtures import (
    HAND_NAMES,
    LEGACY_OBJECT_ID,
    FixtureBuildError,
    build_fixtures,
)
from script.audit_example_fixtures import audit_manifest
from script.run_ablation_baselines import ExperimentConfig, ExperimentRunnerError, run_experiment
from ada_grasp_ctrl.utils.robots.base import RobotFactory


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
    def _record(hand: str, object_path: Path, pose: np.ndarray, scale: float) -> dict[str, object]:
        """Build one valid dummy-arm source record.

        Args:
            hand: Maintained short hand name.
            object_path: Absolute processed-object directory.
            pose: Seven-value object pose.
            scale: Positive object scale.

        Returns:
            NumPy record mapping.
        """
        prefix = "rh_" if hand != "allegro" else ""
        robot = RobotFactory.create_robot(robot_type=f"dummy_arm_{hand}", prefix=prefix)
        dimension = robot.n_dof
        qpos = np.linspace(0.0, 1.0, dimension)
        return {
            "obj_path": str(object_path),
            "obj_pose": pose,
            "obj_scale": scale,
            "pregrasp_qpos": qpos,
            "grasp_qpos": qpos + 1.0,
            "squeeze_qpos": qpos + 2.0,
            "joint_names": list(robot.dof_names),
            "approach_qpos": np.stack((qpos, qpos + 0.5)),
        }

    def _sources(self) -> tuple[dict[str, Path], Path, Path]:
        """Write two-record source trees, DGN assets, and maintained full fixtures.

        Returns:
            Dummy-arm roots, DGN source root, and destination checkout root.
        """
        object_ids = (LEGACY_OBJECT_ID, "core_test_second")
        relative_paths = (
            Path(object_ids[0]) / "tabletop_ur10e/scale006_pose004_0/partial_pc_00_6.npy",
            Path(object_ids[1]) / "tabletop_ur10e/scale008_pose001_0/partial_pc_01_0.npy",
        )
        dgn_root = self.root / "external/DGN_2k"
        poses = (
            np.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]),
        )
        scales = (0.06, 0.08)
        for object_id, pose, scale, relative in zip(object_ids, poses, scales, relative_paths):
            processed = dgn_root / "processed_data" / object_id
            required = (
                processed / "info/simplified.json",
                processed / "mesh/simplified.obj",
                processed / "urdf/coacd.xml",
                processed / "urdf/coacd.urdf",
                processed / "urdf/meshes/convex_piece_000.obj",
            )
            for path in required:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            scene_path = dgn_root / "scene_cfg" / object_id / relative.parts[1] / f"{relative.parts[2]}.npy"
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(
                scene_path,
                {
                    "scene": {
                        object_id: {
                            "file_path": f"../../../processed_data/{object_id}/mesh/simplified.obj",
                            "xml_path": f"../../../processed_data/{object_id}/urdf/coacd.xml",
                            "urdf_path": f"../../../processed_data/{object_id}/urdf/coacd.urdf",
                            "info_path": f"../../../processed_data/{object_id}/info/simplified.json",
                            "pose": pose,
                            "scale": np.full(3, scale),
                        }
                    },
                    "task": {"obj_name": object_id},
                },
            )

        dummy_arm_roots = {}
        for hand in HAND_NAMES:
            dummy_arm_root = self.root / "archive/dummy-arm" / hand
            for relative, object_id, pose, scale in zip(relative_paths, object_ids, poses, scales):
                source = dummy_arm_root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                np.save(source, self._record(hand, dgn_root / "processed_data" / object_id, pose, scale))
            dummy_arm_roots[hand] = dummy_arm_root.resolve()

        project_root = self.root / "checkout"
        data_root = project_root / "examples/data"
        data_root.mkdir(parents=True)
        np.save(
            data_root / "scene.npy",
            {"scene": {"target": {"file_path": f"../assets/object/{LEGACY_OBJECT_ID}/mesh/simplified.obj"}}},
        )
        for hand in HAND_NAMES:
            formatted = data_root / hand / "formatted/grasp.npy"
            formatted.parent.mkdir(parents=True, exist_ok=True)
            np.save(
                formatted,
                self._record(hand, Path(f"examples/assets/object/{LEGACY_OBJECT_ID}"), poses[0], scales[0]),
            )
        legacy = project_root / "examples/assets/object" / LEGACY_OBJECT_ID / "mesh"
        legacy.mkdir(parents=True)
        (legacy / "simplified.obj").write_text("legacy\n", encoding="utf-8")
        return dummy_arm_roots, dgn_root.resolve(), project_root

    def test_builder_creates_exact_manifest_and_auditable_subset(self) -> None:
        """Generate and audit the selected fixtures without consulting repository output."""
        dummy_arm_roots, dgn_root, project_root = self._sources()

        with patch("script.build_example_fixtures.EXPECTED_RECORDS_PER_HAND", 2):
            with patch("script.build_example_fixtures.EXPECTED_OBJECT_COUNT", 2):
                manifest_path = build_fixtures(dummy_arm_roots, dgn_root, project_root)
        attribution = project_root / "examples/assets/object/DGN_2k/ATTRIBUTION.md"
        attribution.write_text("DGN 2k\n", encoding="utf-8")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["counts"]["grasp_records"], 6)
        self.assertEqual(manifest["counts"]["object_ids"], 2)
        self.assertFalse((project_root / "examples/assets/object" / LEGACY_OBJECT_ID).exists())
        for hand in HAND_NAMES:
            records = sorted((project_root / f"examples/data/{hand}/dummy_arm").rglob("*.npy"))
            self.assertEqual(len(records), 2)
            self.assertTrue(all(np.load(path, allow_pickle=True).item()["schema_version"] == 1 for path in records))
        with patch("script.audit_example_fixtures.EXPECTED_RECORDS_PER_HAND", 2):
            with patch("script.audit_example_fixtures.EXPECTED_OBJECT_COUNT", 2):
                counts = audit_manifest(manifest_path, project_root)
        self.assertEqual(counts["grasp_records"], 6)

    def test_builder_preflights_all_sources_before_writing(self) -> None:
        """Leave the destination untouched when one archived sample is missing."""
        dummy_arm_roots, dgn_root, project_root = self._sources()
        next(dummy_arm_roots["allegro"].rglob("*.npy")).unlink()

        with patch("script.build_example_fixtures.EXPECTED_RECORDS_PER_HAND", 2):
            with patch("script.build_example_fixtures.EXPECTED_OBJECT_COUNT", 2):
                with self.assertRaisesRegex(FixtureBuildError, "contains 1 .npy records"):
                    build_fixtures(dummy_arm_roots, dgn_root, project_root)

        self.assertFalse((project_root / "examples/quick_manifest.json").exists())

    def test_builder_rejects_relative_source_roots(self) -> None:
        """Do not make fixture regeneration depend on the caller's CWD."""
        dummy_arm_roots, dgn_root, project_root = self._sources()
        dummy_arm_roots["shadow"] = Path("relative/shadow")

        with self.assertRaisesRegex(FixtureBuildError, "must be absolute"):
            build_fixtures(dummy_arm_roots, dgn_root, project_root)

    def test_builder_reports_nonnumeric_archived_qpos(self) -> None:
        """Convert object-typed finite-check failures into an actionable error."""
        dummy_arm_roots, dgn_root, project_root = self._sources()
        malformed_path = next(dummy_arm_roots["shadow"].rglob("*.npy"))
        malformed = np.load(malformed_path, allow_pickle=True).item()
        malformed["pregrasp_qpos"] = np.array(["not-a-number"], dtype=object)
        np.save(malformed_path, malformed)

        with patch("script.build_example_fixtures.EXPECTED_RECORDS_PER_HAND", 2):
            with patch("script.build_example_fixtures.EXPECTED_OBJECT_COUNT", 2):
                with self.assertRaisesRegex(FixtureBuildError, "Invalid dummy-arm fixture source"):
                    build_fixtures(dummy_arm_roots, dgn_root, project_root)


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
        for path in (
            "fixtures/sample.npy",
            "assets/hand/new_hand.xml",
            "assets/hand/mesh/finger.stl",
            "examples/assets/object/DGN_2k/processed_data/object/info/simplified.json",
        ):
            with self.subTest(path=path):
                completed = self._git("check-ignore", "--no-index", "--quiet", path)
                self.assertEqual(completed.returncode, 1, f"Unexpectedly ignored: {path}")

    def test_maintained_release_automation_is_quick_only(self) -> None:
        """Keep scripts, workflows, README, and normative release rules on the quick gate."""
        paths = (
            PROJECT_ROOT / "script/run_release_gate.sh",
            PROJECT_ROOT / ".github/workflows/ci.yml",
            PROJECT_ROOT / ".github/workflows/release.yml",
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "agents/rules/testing-release.md",
        )
        removed_tokens = (
            "release300",
            "release_input_root",
            "ADA_GRASP_CTRL_RELEASE_INPUT_ROOT",
            "run_release_gate.sh fixed",
            "run_release_gate.sh wheel",
            "run_release_gate.sh portable",
            "run_release_gate.sh all",
        )
        for path in paths:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertTrue(all(token not in text for token in removed_tokens))
        for workflow in (PROJECT_ROOT / ".github/workflows/ci.yml", PROJECT_ROOT / ".github/workflows/release.yml"):
            text = workflow.read_text(encoding="utf-8")
            self.assertIn("matrix:", text)
            self.assertIn("hand: [shadow, allegro, leap_tac3d]", text)


if __name__ == "__main__":
    unittest.main()
