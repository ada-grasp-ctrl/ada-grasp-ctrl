"""Unit tests for schemas, converters, paths, reports, and empty statistics."""

import json
import multiprocessing
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch
import yaml

from ada_grasp_ctrl.batch import (
    SampleResult,
    SampleStatus,
    choose_inputs,
    raise_for_batch_failures,
    write_batch_report,
)
from ada_grasp_ctrl.errors import BatchExecutionError
from ada_grasp_ctrl.runtime import resolve_worker_count, sample_seed, seed_everything, seed_sample
from ada_grasp_ctrl.schema import SchemaError, load_npy_record, validate_grasp_record
from ada_grasp_ctrl.tasks import TASK_REGISTRY
from ada_grasp_ctrl.tasks.control_eval import METHOD_REGISTRY
from ada_grasp_ctrl.tasks.control_stat import get_control_results
from ada_grasp_ctrl.tasks.convert_format import BODex, Batched, Learning
from ada_grasp_ctrl.utils.robots.base import RobotFactory


def _seeded_worker(params: tuple[int, int]) -> tuple[int, float, float, float]:
    """Return random draws after applying one deterministic sample seed.

    Args:
        params: Global seed and stable sample index.

    Returns:
        Sample index followed by Python, NumPy, and Torch random draws.
    """
    global_seed, sample_index = params
    seed_sample(global_seed, sample_index)
    return sample_index, random.random(), float(np.random.rand()), float(torch.rand(1).item())


class PipelineContractTest(unittest.TestCase):
    """Exercise the public data and task orchestration contracts."""

    def setUp(self):
        """Create an isolated filesystem root for each test.

        Returns:
            None.
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        """Release the isolated filesystem root.

        Returns:
            None.
        """
        self.temporary.cleanup()

    def _scene_path(self):
        """Write a minimal valid object scene record.

        Returns:
            Scene ``.npy`` path.
        """
        scene_path = self.root / "scene.npy"
        object_mesh = self.root / "object" / "meshes" / "object.obj"
        object_mesh.parent.mkdir(parents=True)
        object_mesh.write_text("o object\n", encoding="utf-8")
        scene = {
            "task": {"obj_name": "target"},
            "scene": {
                "target": {
                    "file_path": str(object_mesh),
                    "pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                    "scale": [1.0],
                }
            },
        }
        np.save(scene_path, scene)
        return scene_path

    def _converter_config(self):
        """Build the converter configuration fields used by direct unit calls.

        Returns:
            OmegaConf configuration.
        """
        return OmegaConf.create(
            {
                "task": {"data_path": str(self.root / "raw")},
                "grasp_dir": str(self.root / "formatted"),
                "hand_name": "shadow",
                "hand": {"mocap": True},
            }
        )

    def _hand_converter_config(self, hand_name: str, *, mocap: bool = True):
        """Build a converter configuration for one concrete hand.

        Args:
            hand_name: Registered hand name.
            mocap: Whether BODex records contain exactly three named poses.

        Returns:
            OmegaConf configuration rooted in the current fixture.
        """
        return OmegaConf.create(
            {
                "task": {"data_path": str(self.root / "raw")},
                "grasp_dir": str(self.root / "formatted" / hand_name),
                "hand_name": hand_name,
                "hand": {"mocap": mocap},
            }
        )

    def _run_cli(self, *overrides: str) -> subprocess.CompletedProcess[str]:
        """Run the source-checkout CLI and capture its public process contract.

        Args:
            overrides: Hydra overrides appended to the command line.

        Returns:
            Completed subprocess with captured text output.
        """
        project_root = Path(__file__).resolve().parents[1]
        return subprocess.run(
            [sys.executable, str(project_root / "src" / "main.py"), *overrides],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_learning_and_batched_write_v1_without_nested_batch_paths(self):
        """Convert fixtures and prove Batched outputs remain siblings."""
        scene_path = self._scene_path()
        raw_root = self.root / "raw"
        raw_root.mkdir()
        qpos = np.zeros(29)
        qpos[3] = 1.0
        qpos[7:] = 0.2
        learning_path = raw_root / "learning.npy"
        np.save(
            learning_path,
            {
                "scene_path": str(scene_path),
                "pregrasp_qpos": qpos,
                "grasp_qpos": qpos,
                "squeeze_qpos": qpos,
            },
        )
        learning_outputs = Learning((str(learning_path), self._converter_config()))
        learning = load_npy_record(learning_outputs[0])
        self.assertEqual(learning["schema_version"], 1)
        validate_grasp_record(learning, learning_outputs[0])

        batched_path = raw_root / "batch.npy"
        np.save(
            batched_path,
            {
                "scene_path": str(scene_path),
                "pregrasp_qpos": np.stack([qpos, qpos]),
                "grasp_qpos": np.stack([qpos, qpos]),
                "squeeze_qpos": np.stack([qpos, qpos]),
                "scene_scale": np.array([0.5, 1.5]),
            },
        )
        batched_outputs = [Path(path) for path in Batched((str(batched_path), self._converter_config()))]
        self.assertEqual(
            [path.relative_to(self.root / "formatted") for path in batched_outputs],
            [
                Path("batch/0.npy"),
                Path("batch/1.npy"),
            ],
        )
        self.assertEqual(load_npy_record(batched_outputs[0])["obj_scale"], 0.5)
        self.assertEqual(load_npy_record(batched_outputs[1])["obj_scale"], 1.5)

    def test_bodex_converts_all_hands_with_exact_joint_order(self):
        """Exercise hand-specific pose transforms and preserve verified names."""
        scene_path = self._scene_path()
        raw_root = self.root / "raw"
        raw_root.mkdir()

        for hand_name in ("shadow", "allegro", "leap_tac3d"):
            prefix = "" if hand_name == "allegro" else "rh_"
            robot = RobotFactory.create_robot(robot_type=hand_name, prefix=prefix)
            expected_names = list(robot.dof_names)
            raw_names = expected_names[-5:] + expected_names[:-5] if hand_name == "shadow" else expected_names
            qpos_dim = 7 + robot.n_dof
            robot_pose = np.zeros((1, 2, 3, qpos_dim))
            robot_pose[..., 3] = 1.0
            robot_pose[..., 7:] = np.arange(robot.n_dof)
            raw_path = raw_root / f"{hand_name}_mogen.npy"
            np.save(
                raw_path,
                {
                    "scene_path": str(scene_path),
                    "robot_pose": robot_pose,
                    "joint_names": raw_names,
                },
            )

            outputs = BODex((str(raw_path), self._hand_converter_config(hand_name)))

            self.assertEqual(len(outputs), 2)
            for output in outputs:
                record = load_npy_record(output)
                self.assertEqual(record["joint_names"], expected_names)
                validate_grasp_record(
                    record,
                    output,
                    expected_joint_dim=qpos_dim,
                    expected_joint_names=expected_names,
                    qpos_prefix_dim=7,
                    require_joint_names=True,
                )
            if hand_name == "shadow":
                expected_joint_qpos = np.concatenate([np.arange(robot.n_dof)[5:], np.arange(robot.n_dof)[:5]])
                np.testing.assert_array_equal(load_npy_record(outputs[0])["grasp_qpos"][7:], expected_joint_qpos)

    def test_bodex_non_mocap_layout_preserves_approach_and_lift(self):
        """Map the final four trajectory poses without dropping approach data."""
        scene_path = self._scene_path()
        raw_root = self.root / "raw"
        raw_root.mkdir()
        robot = RobotFactory.create_robot(robot_type="leap_tac3d", prefix="rh_")
        qpos_dim = 7 + robot.n_dof
        robot_pose = np.zeros((1, 1, 5, qpos_dim))
        robot_pose[..., 3] = 1.0
        for step in range(robot_pose.shape[2]):
            robot_pose[:, :, step, 7:] = step
        raw_path = raw_root / "leap_full_mogen.npy"
        np.save(
            raw_path,
            {
                "scene_path": str(scene_path),
                "robot_pose": robot_pose,
                "joint_names": robot.dof_names,
            },
        )

        output = BODex((str(raw_path), self._hand_converter_config("leap_tac3d", mocap=False)))[0]
        record = load_npy_record(output)

        self.assertEqual(record["approach_qpos"].shape, (1, qpos_dim))
        np.testing.assert_array_equal(record["pregrasp_qpos"][7:], np.ones(robot.n_dof))
        np.testing.assert_array_equal(record["grasp_qpos"][7:], np.full(robot.n_dof, 2.0))
        np.testing.assert_array_equal(record["squeeze_qpos"][7:], np.full(robot.n_dof, 3.0))
        np.testing.assert_array_equal(record["lift_qpos"][7:], np.full(robot.n_dof, 4.0))

    def test_schema_error_names_sample_field_expected_and_actual(self):
        """Return actionable context for a malformed grasp field."""
        invalid = {
            "obj_path": "missing",
            "obj_pose": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "obj_scale": 1.0,
            "pregrasp_qpos": np.zeros(2),
            "grasp_qpos": np.zeros(2),
            "squeeze_qpos": np.zeros(2),
        }
        with self.assertRaises(SchemaError) as context:
            validate_grasp_record(invalid, "broken.npy")
        message = str(context.exception)
        self.assertIn("broken.npy", message)
        self.assertIn("obj_pose", message)
        self.assertIn("expected", message)
        self.assertIn("actual", message)

    def test_deterministic_skip_report_and_failure_exit(self):
        """Select deterministically, write both reports, and raise only after writing."""
        existing = self.root / "existing.npy"
        existing.touch()
        selected, skipped = choose_inputs(
            ["b.npy", "a.npy", "c.npy"],
            skip=True,
            output_paths=[[str(existing)], [str(self.root / "a.out")], [str(self.root / "c.out")]],
            max_num=1,
            seed=12,
        )
        self.assertEqual(skipped, 1)
        self.assertEqual(len(selected), 1)
        results = [
            SampleResult("a.npy", SampleStatus.COMPLETED, output_paths=["a.out"]),
            SampleResult("c.npy", SampleStatus.SOLVER_DEGRADED, message="held"),
        ]
        summary = write_batch_report(
            self.root / "report",
            "control_eval",
            results,
            num_discovered=3,
            num_skipped=1,
        )
        self.assertEqual(summary["num_solver_degraded"], 1)
        self.assertTrue((self.root / "report" / "failures.jsonl").is_file())
        with self.assertRaises(BatchExecutionError):
            raise_for_batch_failures(summary)
        parsed = json.loads((self.root / "report" / "run_report.json").read_text())
        self.assertEqual(parsed["num_processed"], 2)

    def test_empty_statistics_are_null_and_have_zero_denominator(self):
        """Save defined integer counts and YAML null instead of division warnings/NaN."""
        configs = OmegaConf.create(
            {
                "control_dir": str(self.root / "control"),
                "hand_name": "dummy_arm_shadow",
                "task": {
                    "method": "ours",
                    "ablation_name": "default",
                    "setting_name": "dist_0",
                    "lift_height": 0.2,
                    "n_terminal_steps": 5,
                },
            }
        )
        statistics, results, output_path = get_control_results([], configs)
        self.assertEqual(results, [])
        self.assertIsNone(statistics["success_rate"])
        self.assertIsNone(statistics["ave_obj_pos_err"]["mean"])
        self.assertEqual(statistics["success_rate_denominator"], 0)
        saved = yaml.safe_load(output_path.read_text())
        self.assertIsNone(saved["success_rate"])
        self.assertNotIn("nan", output_path.read_text().lower())

    def test_declared_execution_errors_remain_errors_in_statistics(self):
        """Exclude declared execution errors from every scientific outcome bucket."""
        configs = OmegaConf.create(
            {
                "control_dir": str(self.root / "control"),
                "hand_name": "dummy_arm_shadow",
                "task": {
                    "method": "ours",
                    "ablation_name": "default",
                    "setting_name": "dist_0",
                    "lift_height": 0.2,
                    "n_terminal_steps": 5,
                },
            }
        )
        pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        indexed_data = [
            (
                0,
                "empty_error.npy",
                {"episode_status": "execution_error", "obj_pose": [], "contacts": []},
                None,
            ),
            (
                1,
                "trajectory_error.npy",
                {
                    "episode_status": "execution_error",
                    "obj_pose": np.stack([pose, pose]),
                    "contacts": [[], []],
                    "error_message": "worker failed",
                },
                None,
            ),
        ]

        statistics, results, _ = get_control_results(indexed_data, configs)

        self.assertEqual([result.status for result in results], [SampleStatus.EXECUTION_ERROR] * 2)
        self.assertEqual(statistics["execution_error"], 2)
        self.assertEqual(statistics["invalid_initialization"], 0)
        self.assertEqual(statistics["success"], 0)
        self.assertEqual(statistics["failure"], 0)
        self.assertEqual(statistics["success_rate_denominator"], 0)
        self.assertIsNone(statistics["success_rate"])

    def test_registries_workers_and_seed_are_explicit_and_reproducible(self):
        """Expose only public tasks/methods and reproduce NumPy samples."""
        self.assertEqual(
            set(TASK_REGISTRY),
            {"format", "dummy_arm_qpos", "control_eval", "control_stat"},
        )
        self.assertEqual(
            {method for setting, method in METHOD_REGISTRY if setting == "tabletop"},
            {"ours", "op", "bs1", "bs2", "bs3"},
        )
        self.assertGreaterEqual(resolve_worker_count("auto"), 1)
        with self.assertRaisesRegex(Exception, "greater than zero"):
            resolve_worker_count(0)
        seed_everything(42)
        first = np.random.rand(4)
        seed_everything(42)
        np.testing.assert_array_equal(first, np.random.rand(4))

        self.assertEqual(sample_seed(12, 3), sample_seed(12, 3))
        self.assertNotEqual(sample_seed(12, 3), sample_seed(12, 4))
        params = [(12, index) for index in range(8)]
        with multiprocessing.Pool(processes=1) as pool:
            serial = sorted(pool.imap_unordered(_seeded_worker, params))
        with multiprocessing.Pool(processes=3) as pool:
            parallel = sorted(pool.imap_unordered(_seeded_worker, reversed(params)))
        self.assertEqual(serial, parallel)

    def test_format_outputs_are_invariant_to_serial_parallel_and_auto_workers(self):
        """Compare real converter outputs and sample seeds across worker policies."""
        scene_path = self._scene_path()
        raw_root = self.root / "deterministic_raw"
        raw_root.mkdir()
        for index in range(4):
            qpos = np.zeros(29)
            qpos[3] = 1.0
            qpos[7:] = index + np.arange(22) / 100.0
            np.save(
                raw_root / f"sample_{index}.npy",
                {
                    "scene_path": str(scene_path),
                    "pregrasp_qpos": qpos - 0.1,
                    "grasp_qpos": qpos,
                    "squeeze_qpos": qpos + 0.1,
                },
            )

        records_by_policy = {}
        seeds_by_policy = {}
        for policy in ("1", "2", "auto"):
            run_root = self.root / f"workers_{policy}"
            process = self._run_cli(
                "task=format",
                "hand=shadow",
                "task.data_name=Learning",
                f"task.data_path={raw_root}",
                f"n_worker={policy}",
                "seed=77",
                f"save_dir={run_root}",
                f"grasp_dir={run_root / 'formatted'}",
                f"log_dir={run_root / 'log'}",
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            records_by_policy[policy] = {
                path.name: load_npy_record(path) for path in sorted((run_root / "formatted").glob("*.npy"))
            }
            report = json.loads((run_root / "log" / "run_report.json").read_text())
            seeds_by_policy[policy] = [result["details"]["sample_seed"] for result in report["results"]]

        self.assertEqual(seeds_by_policy["1"], seeds_by_policy["2"])
        self.assertEqual(seeds_by_policy["1"], seeds_by_policy["auto"])
        self.assertEqual(set(records_by_policy["1"]), {f"sample_{index}.npy" for index in range(4)})
        for policy in ("2", "auto"):
            self.assertEqual(records_by_policy["1"].keys(), records_by_policy[policy].keys())
            for filename, expected in records_by_policy["1"].items():
                actual = records_by_policy[policy][filename]
                self.assertEqual(expected.keys(), actual.keys())
                for field in expected:
                    np.testing.assert_equal(actual[field], expected[field])

    def test_cli_preserves_exit_codes_without_hydra_tracebacks(self):
        """Prove successful, batch-failure, and preflight exit semantics in subprocesses."""
        unknown_task = self._run_cli("task=unsupported")
        unknown_hand = self._run_cli("task=control_stat", "hand=unsupported")

        empty_root = self.root / "empty"
        successful = self._run_cli(
            "task=control_stat",
            "hand=dummy_arm_shadow",
            "n_worker=1",
            f"save_dir={empty_root}",
            f"control_dir={empty_root / 'control'}",
            f"log_dir={empty_root / 'log'}",
        )

        damaged_root = self.root / "damaged"
        raw_root = damaged_root / "raw"
        raw_root.mkdir(parents=True)
        (raw_root / "broken.npy").write_bytes(b"not a NumPy file")
        qpos = np.zeros(29)
        qpos[3] = 1.0
        np.save(
            raw_root / "valid.npy",
            {
                "scene_path": str(self._scene_path()),
                "pregrasp_qpos": qpos,
                "grasp_qpos": qpos,
                "squeeze_qpos": qpos,
            },
        )
        batch_failure = self._run_cli(
            "task=format",
            "hand=shadow",
            "task.data_name=Learning",
            f"task.data_path={raw_root}",
            "n_worker=1",
            f"save_dir={damaged_root}",
            f"grasp_dir={damaged_root / 'grasp'}",
            f"log_dir={damaged_root / 'log'}",
        )

        invalid_root = self.root / "invalid"
        preflight_failure = self._run_cli(
            "task=control_stat",
            "hand=dummy_arm_shadow",
            "task_name=unsupported",
            "n_worker=1",
            f"save_dir={invalid_root}",
            f"control_dir={invalid_root / 'control'}",
            f"log_dir={invalid_root / 'log'}",
        )

        missing_asset_root = self.root / "missing_asset"
        project_root = Path(__file__).resolve().parents[1]
        missing_asset = self._run_cli(
            "task=control_eval",
            "hand=dummy_arm_shadow",
            "n_worker=1",
            f"hand.xml_path={missing_asset_root / 'missing.xml'}",
            f"grasp_dir={project_root / 'examples' / 'data' / 'shadow' / 'dummy_arm'}",
            f"control_dir={missing_asset_root / 'control'}",
            f"save_dir={missing_asset_root}",
            f"log_dir={missing_asset_root / 'log'}",
        )

        project_grasp = np.load(
            project_root / "examples" / "data" / "shadow" / "dummy_arm" / "grasp.npy",
            allow_pickle=True,
        ).item()
        missing_object_root = self.root / "missing_object"
        missing_object_grasp_dir = missing_object_root / "grasp"
        missing_object_grasp_dir.mkdir(parents=True)
        missing_object_record = dict(project_grasp)
        missing_object_record["obj_path"] = str(missing_object_root / "absent_object")
        np.save(missing_object_grasp_dir / "grasp.npy", missing_object_record)
        missing_object = self._run_cli(
            "task=control_eval",
            "hand=dummy_arm_shadow",
            "n_worker=1",
            f"grasp_dir={missing_object_grasp_dir}",
            f"control_dir={missing_object_root / 'control'}",
            f"save_dir={missing_object_root}",
            f"log_dir={missing_object_root / 'log'}",
        )

        missing_mesh_root = self.root / "missing_mesh"
        object_root = missing_mesh_root / "object"
        (object_root / "info").mkdir(parents=True)
        (object_root / "urdf" / "meshes").mkdir(parents=True)
        (object_root / "info" / "simplified.json").write_text(
            json.dumps({"mass": 1.0, "density": 1.0, "scale": 1.0}),
            encoding="utf-8",
        )
        missing_mesh_grasp_dir = missing_mesh_root / "grasp"
        missing_mesh_grasp_dir.mkdir()
        missing_mesh_record = dict(project_grasp)
        missing_mesh_record["obj_path"] = str(object_root)
        np.save(missing_mesh_grasp_dir / "grasp.npy", missing_mesh_record)
        missing_mesh = self._run_cli(
            "task=control_eval",
            "hand=dummy_arm_shadow",
            "n_worker=1",
            f"grasp_dir={missing_mesh_grasp_dir}",
            f"control_dir={missing_mesh_root / 'control'}",
            f"save_dir={missing_mesh_root}",
            f"log_dir={missing_mesh_root / 'log'}",
        )

        damaged_control_root = self.root / "damaged_control"
        damaged_control_grasp_dir = damaged_control_root / "grasp"
        damaged_control_grasp_dir.mkdir(parents=True)
        (damaged_control_grasp_dir / "broken.npy").write_bytes(b"not a NumPy file")
        damaged_control = self._run_cli(
            "task=control_eval",
            "hand=dummy_arm_shadow",
            "n_worker=1",
            f"grasp_dir={damaged_control_grasp_dir}",
            f"control_dir={damaged_control_root / 'control'}",
            f"save_dir={damaged_control_root}",
            f"log_dir={damaged_control_root / 'log'}",
        )

        declared_error_root = self.root / "declared_error"
        declared_error_method_dir = declared_error_root / "control" / "ours_default"
        declared_error_method_dir.mkdir(parents=True)
        np.save(
            declared_error_method_dir / "sample_dist_0.npy",
            {
                "schema_version": 1,
                "episode_status": "execution_error",
                "obj_pose": np.stack([np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])]),
                "contacts": [[]],
                "error_message": "recorded worker failure",
            },
        )
        declared_error_stat = self._run_cli(
            "task=control_stat",
            "hand=dummy_arm_shadow",
            "n_worker=1",
            f"control_dir={declared_error_root / 'control'}",
            f"save_dir={declared_error_root}",
            f"log_dir={declared_error_root / 'log'}",
        )

        self.assertEqual(unknown_task.returncode, 2, unknown_task.stderr)
        self.assertEqual(unknown_hand.returncode, 2, unknown_hand.stderr)
        self.assertEqual(successful.returncode, 0, successful.stderr)
        self.assertEqual(batch_failure.returncode, 1, batch_failure.stderr)
        self.assertTrue((damaged_root / "grasp" / "valid.npy").is_file())
        damaged_report = json.loads((damaged_root / "log" / "run_report.json").read_text())
        self.assertEqual(damaged_report["num_completed"], 1)
        self.assertEqual(damaged_report["num_execution_error"], 1)
        self.assertEqual(preflight_failure.returncode, 2, preflight_failure.stderr)
        self.assertEqual(missing_asset.returncode, 2, missing_asset.stderr)
        self.assertEqual(missing_object.returncode, 2, missing_object.stderr)
        self.assertEqual(missing_mesh.returncode, 2, missing_mesh.stderr)
        self.assertEqual(damaged_control.returncode, 1, damaged_control.stderr)
        self.assertEqual(declared_error_stat.returncode, 1, declared_error_stat.stderr)
        declared_error_statistics = yaml.safe_load(
            (declared_error_root / "control_stat_res" / "dist_0_ours_default.yaml").read_text()
        )
        self.assertEqual(declared_error_statistics["execution_error"], 1)
        self.assertEqual(declared_error_statistics["success_rate_denominator"], 0)
        for process in (
            unknown_task,
            unknown_hand,
            successful,
            batch_failure,
            preflight_failure,
            missing_asset,
            missing_object,
            missing_mesh,
            damaged_control,
            declared_error_stat,
        ):
            self.assertNotIn("Traceback", process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
