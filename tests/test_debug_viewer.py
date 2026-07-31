"""Tests for the task-level native and mjviser debug-viewer session."""

import importlib.util
import io
import math
import os
import socket
import unittest
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from ada_grasp_ctrl.utils.viewer_session import (
    DebugViewerError,
    DebugViewerSession,
    create_debug_viewer_session,
)


def find_free_port():
    """Reserve and return a currently unused localhost TCP port.

    Args:
        None.

    Returns:
        An integer TCP port available at the time of the check.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_model(geom_type):
    """Create a minimal dynamic model for viewer lifecycle tests.

    Args:
        geom_type: MuJoCo primitive geom type used by the model.

    Returns:
        A compiled MuJoCo model and initialized data pair.
    """
    geom_size = "0.1 0.1 0.1" if geom_type == "box" else "0.1"
    model = mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody><body><freejoint/><geom type='{geom_type}' "
        f"size='{geom_size}'/></body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class DebugViewerSessionTest(unittest.TestCase):
    """Verify configuration, failure handling, camera alignment, and reuse."""

    def test_invalid_backend_is_rejected(self):
        """Reject unknown backend names before any viewer resources start."""
        with self.assertRaisesRegex(DebugViewerError, "Unsupported debug viewer backend"):
            DebugViewerSession("unknown")

    def test_disabled_factory_does_not_start_a_viewer(self):
        """Return None without reading backend-specific configuration when disabled."""
        task_config = SimpleNamespace(debug_viewer=False)
        self.assertIsNone(create_debug_viewer_session(task_config))

    def test_native_viewer_reports_headless_environment(self):
        """Fail clearly before GLFW starts when Linux has no display session."""
        model, data = make_model("sphere")
        session = DebugViewerSession("mujoco")
        with mock.patch.dict(os.environ, {"DISPLAY": "", "WAYLAND_DISPLAY": ""}, clear=False):
            with self.assertRaisesRegex(DebugViewerError, "requires a graphical session"):
                session.attach(model, data)

    def test_web_camera_matches_native_free_camera(self):
        """Use MuJoCo camera conventions for the initial browser viewpoint."""
        model, _ = make_model("sphere")
        session = DebugViewerSession("mujoco")
        session._model = model
        client = SimpleNamespace(camera=SimpleNamespace())

        session._apply_camera(client)

        np.testing.assert_allclose(client.camera.look_at, [0.7, 0.0, 0.3])
        np.testing.assert_allclose(client.camera.position, [0.18316905, 0.0, 0.48811108], atol=1e-7)
        np.testing.assert_allclose(client.camera.up_direction, [0.0, 0.0, 1.0])
        self.assertAlmostEqual(client.camera.fov, math.radians(45.0))

    @unittest.skipUnless(
        importlib.util.find_spec("mjviser") is not None and importlib.util.find_spec("viser") is not None,
        "mjviser and viser are not installed",
    )
    def test_mjviser_falls_back_from_occupied_port_and_reports_actual_url(self):
        """Use viser's next free port and expose it in session output.

        Args:
            None.

        Returns:
            None.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied_socket:
            occupied_socket.bind(("127.0.0.1", 0))
            occupied_socket.listen()
            requested_port = occupied_socket.getsockname()[1]

            output = io.StringIO()
            session = None
            try:
                with mock.patch("sys.stdout", output):
                    session = DebugViewerSession(
                        "mjviser",
                        host="127.0.0.1",
                        port=requested_port,
                        wait_for_client=False,
                    )

                self.assertGreater(session.port, requested_port)
                self.assertEqual(session.port, session._server.get_port())
                self.assertIn(
                    f"mjviser port {requested_port} is occupied; using the next available port {session.port}.",
                    output.getvalue(),
                )
                self.assertIn(
                    f"mjviser debug viewer: http://127.0.0.1:{session.port}",
                    output.getvalue(),
                )
            finally:
                if session is not None:
                    session.close()

    @unittest.skipUnless(
        importlib.util.find_spec("mjviser") is not None and importlib.util.find_spec("viser") is not None,
        "mjviser and viser are not installed",
    )
    def test_mjviser_close_releases_actual_port(self):
        """Release the selected listener port synchronously during close.

        Args:
            None.

        Returns:
            None.
        """
        session = DebugViewerSession(
            "mjviser",
            host="127.0.0.1",
            port=find_free_port(),
            wait_for_client=False,
        )
        actual_port = session.port

        session.close()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as replacement_socket:
            replacement_socket.bind(("127.0.0.1", actual_port))
            self.assertEqual(replacement_socket.getsockname()[1], actual_port)

    @unittest.skipUnless(
        importlib.util.find_spec("mjviser") is not None and importlib.util.find_spec("viser") is not None,
        "mjviser and viser are not installed",
    )
    def test_mjviser_reuses_server_and_waits_only_once(self):
        """Replace two model scenes without restarting or waiting a second time."""
        session = DebugViewerSession(
            "mjviser",
            host="127.0.0.1",
            port=find_free_port(),
            wait_for_client=True,
        )
        try:
            model1, data1 = make_model("sphere")
            model2, data2 = make_model("box")
            visual_before = model1.vis.rgba.contactpoint.copy()

            # Pretend the browser connected so the test exercises the real
            # wait path without requiring an interactive WebSocket client.
            session._client_connected.set()
            session.attach(model1, data1)
            server_id = id(session._server)
            first_scene = session._scene
            np.testing.assert_allclose(model1.vis.rgba.contactpoint, visual_before)

            session._client_connected.clear()
            session.attach(model2, data2)
            session.sync()

            self.assertEqual(id(session._server), server_id)
            self.assertIsNot(session._scene, first_scene)
            self.assertIs(session._scene.mj_model, model2)
            self.assertTrue(session._first_client_wait_completed)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
