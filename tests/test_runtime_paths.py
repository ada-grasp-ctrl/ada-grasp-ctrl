"""Tests for installation-independent roots and reproducibility manifests."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from ada_grasp_ctrl.errors import PreflightError
from ada_grasp_ctrl.paths import (
    ROOT_ENVIRONMENT,
    configure_runtime_roots,
    reset_runtime_roots,
    resolve_external_root,
    resolve_from_root,
    source_checkout_root,
)
from ada_grasp_ctrl.runtime import _dependency_versions, _git_metadata, configure_runtime
from ada_grasp_ctrl.tasks.convert_format import _resolve_scene_path
from ada_grasp_ctrl.utils.robots.base import RobotFactory


class RuntimePathTest(unittest.TestCase):
    """Exercise root precedence without depending on the process CWD."""

    def setUp(self) -> None:
        """Create one isolated root and clear process-local path state.

        Returns:
            None.
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        reset_runtime_roots()

    def tearDown(self) -> None:
        """Restore default root discovery and release temporary files.

        Returns:
            None.
        """
        reset_runtime_roots()
        self.temporary.cleanup()

    def test_source_checkout_uses_stable_markers(self) -> None:
        """Detect a checkout from a nested directory without fixed parent counts."""
        checkout = self.root / "checkout"
        nested = checkout / "src" / "ada_grasp_ctrl" / "nested"
        nested.mkdir(parents=True)
        (checkout / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")

        self.assertEqual(source_checkout_root(nested), checkout)

    def test_root_precedence_is_explicit_then_environment_then_checkout(self) -> None:
        """Apply the documented root precedence with absolute normalization."""
        checkout = self.root / "checkout"
        explicit = self.root / "explicit"
        environment = self.root / "environment"
        values = {name: "" for name in ROOT_ENVIRONMENT.values()}
        values[ROOT_ENVIRONMENT["asset"]] = str(environment)
        with patch("ada_grasp_ctrl.paths.source_checkout_root", return_value=checkout):
            with patch.dict(os.environ, values, clear=False):
                self.assertEqual(resolve_external_root("asset", explicit), explicit)
                self.assertEqual(resolve_external_root("asset"), environment)
                self.assertEqual(resolve_external_root("data"), checkout)

    def test_default_root_resolution_is_independent_of_cwd(self) -> None:
        """Resolve source defaults identically before and after changing CWD."""
        expected = resolve_external_root("data")
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            actual = resolve_external_root("data")
        finally:
            os.chdir(previous)
        self.assertEqual(actual, expected)

    def test_relative_scene_path_never_prefers_caller_cwd(self) -> None:
        """Resolve relative scene records from input/data roots, not the shell CWD."""
        data_root = self.root / "data"
        raw_root = self.root / "raw"
        caller_root = self.root / "caller"
        expected = data_root / "fixtures" / "scene.npy"
        decoy = caller_root / "fixtures" / "scene.npy"
        expected.parent.mkdir(parents=True)
        decoy.parent.mkdir(parents=True)
        raw_root.mkdir()
        expected.write_bytes(b"expected")
        decoy.write_bytes(b"decoy")
        config = OmegaConf.create({"data_root": str(data_root)})

        previous = Path.cwd()
        try:
            os.chdir(caller_root)
            actual = _resolve_scene_path("fixtures/scene.npy", str(raw_root / "sample.npy"), config)
        finally:
            os.chdir(previous)

        self.assertEqual(actual, expected)

    def test_wheel_mode_requires_explicit_external_roots(self) -> None:
        """Fail clearly when neither checkout, config, nor environment supplies roots."""
        values = {name: "" for name in ROOT_ENVIRONMENT.values()}
        with patch("ada_grasp_ctrl.paths.source_checkout_root", return_value=None):
            with patch.dict(os.environ, values, clear=False):
                with self.assertRaisesRegex(ValueError, "Missing asset_root for wheel execution"):
                    resolve_external_root("asset")

    def test_configure_runtime_activates_each_root_kind_in_wheel_mode(self) -> None:
        """Normalize config paths and make legacy robot paths use asset_root."""
        asset_root = self.root / "external_assets"
        data_root = self.root / "external_data"
        output_root = self.root / "external_output"
        config = OmegaConf.create(
            {
                "asset_root": str(asset_root),
                "data_root": str(data_root),
                "output_root": str(output_root),
                "save_root": None,
                "save_dir": "run",
                "grasp_dir": "run/grasp",
                "dummy_arm_grasp_dir": "run/dummy",
                "control_dir": "run/control",
                "log_dir": "run/log",
                "n_worker": 1,
                "seed": 12,
                "hand": {"xml_path": "assets/hand/dummy_arm_shadow/right_no_tendon.xml"},
                "task": {"data_path": "raw", "debug_dir": "debug"},
            }
        )
        with patch("ada_grasp_ctrl.paths.source_checkout_root", return_value=None):
            with patch("ada_grasp_ctrl.runtime.source_checkout_root", return_value=None):
                configure_runtime(config)

        self.assertIsNone(config.project_root)
        self.assertEqual(Path(config.hand.xml_path), asset_root / "hand/dummy_arm_shadow/right_no_tendon.xml")
        self.assertEqual(Path(config.task.data_path), data_root / "raw")
        self.assertEqual(Path(config.save_dir), output_root / "run")
        self.assertEqual(Path(config.task.debug_dir), output_root / "debug")
        robot = RobotFactory.create_robot(robot_type="dummy_arm_shadow", prefix="rh_")
        self.assertEqual(Path(robot.get_file_path("mjcf")), asset_root / "hand/dummy_arm_shadow/right_no_tendon.xml")

    def test_activated_wheel_roots_do_not_reenter_missing_root_discovery(self) -> None:
        """Use process-local roots without eagerly evaluating checkout fallback."""
        asset_root = self.root / "asset"
        configure_runtime_roots(asset_root=asset_root, data_root=self.root / "data", output_root=self.root / "out")
        with patch("ada_grasp_ctrl.paths.source_checkout_root", return_value=None):
            self.assertEqual(resolve_from_root("assets/hand/test.xml", root_kind="asset"), asset_root / "hand/test.xml")

    def test_legacy_save_root_selects_canonical_output_root(self) -> None:
        """Keep old save_root overrides working while recording output_root."""
        legacy_output = self.root / "legacy"
        config = OmegaConf.create(
            {
                "asset_root": str(self.root / "assets"),
                "data_root": str(self.root / "data"),
                "output_root": None,
                "save_root": str(legacy_output),
                "save_dir": "${save_root}/run",
                "n_worker": 1,
                "seed": 12,
            }
        )
        configure_runtime(config)

        self.assertEqual(Path(config.output_root), legacy_output)
        self.assertEqual(Path(config.save_root), legacy_output)
        self.assertEqual(Path(config.save_dir), legacy_output / "run")

    def test_pinocchio_version_falls_back_to_imported_module(self) -> None:
        """Record Pinocchio even when conda exposes no Python distribution metadata."""

        def metadata_version(distribution: str) -> str:
            """Return fixture versions except for the missing Pin distribution.

            Args:
                distribution: Requested package distribution name.

            Returns:
                Fixture version string.

            Raises:
                PackageNotFoundError: When the requested distribution is ``pin``.
            """
            if distribution == "pin":
                raise importlib.metadata.PackageNotFoundError(distribution)
            return "1.2.3"

        with patch("ada_grasp_ctrl.runtime.importlib.metadata.version", side_effect=metadata_version):
            with patch(
                "ada_grasp_ctrl.runtime.importlib.import_module",
                return_value=SimpleNamespace(__version__="3.0.0"),
            ):
                versions = _dependency_versions()

        self.assertEqual(versions["pinocchio"], "3.0.0")

    def test_git_metadata_is_unavailable_outside_a_checkout(self) -> None:
        """Avoid invoking Git against an installed wheel or caller CWD."""
        with patch("ada_grasp_ctrl.runtime.source_checkout_root", return_value=None):
            self.assertEqual(_git_metadata(), {"commit": None, "dirty": None})

    def test_configure_runtime_maps_missing_wheel_roots_to_preflight(self) -> None:
        """Expose missing wheel roots through the public exit-code-two error type."""
        config = OmegaConf.create(
            {
                "asset_root": None,
                "data_root": None,
                "output_root": None,
                "save_root": None,
                "n_worker": 1,
                "seed": 12,
            }
        )
        values = {name: "" for name in ROOT_ENVIRONMENT.values()}
        with patch("ada_grasp_ctrl.paths.source_checkout_root", return_value=None):
            with patch.dict(os.environ, values, clear=False):
                with self.assertRaises(PreflightError):
                    configure_runtime(config)


if __name__ == "__main__":
    unittest.main()
