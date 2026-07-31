"""Shared debug-viewer lifecycle for native MuJoCo and mjviser."""

import math
import os
import sys
import threading

import numpy as np


_CAMERA_LOOKAT = np.array([0.7, 0.0, 0.3], dtype=float)
_CAMERA_DISTANCE = 0.55
_CAMERA_AZIMUTH = 0.0
_CAMERA_ELEVATION = -20.0


class DebugViewerError(RuntimeError):
    """Raised when a requested debug viewer cannot be initialized or updated."""


class DebugViewerSession:
    """Own one task-level debug viewer session across multiple grasp samples."""

    def __init__(self, backend, host="127.0.0.1", port=8080, wait_for_client=True):
        """Initialize a debug-viewer session.

        Args:
            backend: Viewer backend name, either ``mujoco`` or ``mjviser``.
            host: Host interface used by the mjviser HTTP server.
            port: Preferred starting port for the mjviser HTTP server.
            wait_for_client: Whether the first mjviser scene waits for a browser.

        Returns:
            None.
        """
        backend = str(backend).lower()
        if backend not in {"mujoco", "mjviser"}:
            raise DebugViewerError(f"Unsupported debug viewer backend '{backend}'. Expected one of: mujoco, mjviser.")

        self.backend = backend
        self.host = str(host)
        self.port = int(port)
        self.wait_for_client = bool(wait_for_client)

        self._model = None
        self._data = None
        self._native_viewer = None
        self._server = None
        self._scene = None
        self._scene_type = None
        self._client_connected = threading.Event()
        self._first_client_wait_completed = False

        if self.backend == "mjviser":
            self._start_mjviser_server()

    def attach(self, model, data):
        """Attach the session to a newly created MuJoCo model and data pair.

        Args:
            model: Compiled MuJoCo model to visualize.
            data: Live MuJoCo data updated by the simulation.

        Returns:
            None.
        """
        self.detach()
        self._model = model
        self._data = data

        try:
            if self.backend == "mujoco":
                self._attach_mujoco()
            else:
                self._attach_mjviser()
        except DebugViewerError:
            raise
        except Exception as exc:
            raise DebugViewerError(f"Failed to attach the {self.backend} debug viewer: {exc}") from exc

    def sync(self):
        """Push the current MuJoCo state to the active viewer.

        Args:
            None.

        Returns:
            None.
        """
        try:
            if self.backend == "mujoco" and self._native_viewer is not None:
                if not self._native_viewer.is_running():
                    raise DebugViewerError("The native MuJoCo viewer was closed.")
                self._native_viewer.sync()
            elif self.backend == "mjviser" and self._scene is not None and self._data is not None:
                self._scene.update_from_mjdata(self._data)
        except DebugViewerError:
            raise
        except Exception as exc:
            raise DebugViewerError(f"Failed to update the {self.backend} debug viewer: {exc}") from exc

    def detach(self):
        """Detach the current sample while preserving a task-level web server.

        Args:
            None.

        Returns:
            None.
        """
        if self._native_viewer is not None:
            self._native_viewer.close()
            self._native_viewer = None

        # The mjviser scene intentionally remains visible until the next model
        # replaces it, so a connected browser can inspect the terminal state.
        if self.backend == "mujoco":
            self._model = None
            self._data = None

    def close(self):
        """Release all native or web viewer resources owned by this session.

        Args:
            None.

        Returns:
            None.
        """
        self.detach()
        if self._server is not None:
            self._server.stop()
            self._server = None
            self._scene = None
            self._model = None
            self._data = None

    def _start_mjviser_server(self):
        """Start one persistent Viser server and register one client callback.

        Args:
            None.

        Returns:
            None.
        """
        requested_port = self.port
        try:
            import viser
            from mjviser import ViserMujocoScene

            self._scene_type = ViserMujocoScene
            self._server = viser.ViserServer(host=self.host, port=self.port, label="Ada Grasp Debug Viewer")
            # Viser searches upward from the requested port when that port is
            # occupied. Store its selected port so every user-facing URL and
            # caller-visible session field reflects the actual listener.
            self.port = self._server.get_port()

            @self._server.on_client_connect
            def on_client_connect(client):
                """Record the first browser and align its camera with MuJoCo.

                Args:
                    client: Newly connected Viser client handle.

                Returns:
                    None.
                """
                self._client_connected.set()
                self._apply_camera(client)

            if self.port != requested_port:
                print(f"mjviser port {requested_port} is occupied; using the next available port {self.port}.")
            print(f"mjviser debug viewer: http://{self.host}:{self.port}")
            if self.host in {"127.0.0.1", "localhost"}:
                print(f"For remote access, forward the port with: ssh -L {self.port}:127.0.0.1:{self.port} <server>")
        except ImportError as exc:
            raise DebugViewerError(
                "The mjviser backend requires mjviser==0.0.14 and viser==1.0.27 to be installed."
            ) from exc
        except Exception as exc:
            if self._server is not None:
                self._server.stop()
                self._server = None
            raise DebugViewerError(
                f"Failed to start the mjviser server on {self.host}:{requested_port}: {exc}"
            ) from exc

    def _attach_mujoco(self):
        """Launch the native passive viewer for the current model.

        Args:
            None.

        Returns:
            None.
        """
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise DebugViewerError(
                "The native MuJoCo viewer requires a graphical session, but DISPLAY and "
                "WAYLAND_DISPLAY are both unset. Use task.debug_viewer_backend=mjviser on a headless server."
            )

        import mujoco.viewer

        self._native_viewer = mujoco.viewer.launch_passive(self._model, self._data)
        self._native_viewer.cam.lookat[:] = _CAMERA_LOOKAT
        self._native_viewer.cam.distance = _CAMERA_DISTANCE
        self._native_viewer.cam.azimuth = _CAMERA_AZIMUTH
        self._native_viewer.cam.elevation = _CAMERA_ELEVATION
        self._native_viewer.sync()

    def _attach_mjviser(self):
        """Replace the web scene while keeping the Viser server alive.

        Args:
            None.

        Returns:
            None.
        """
        if self._server is None or self._scene_type is None:
            raise DebugViewerError("The mjviser server is not running.")

        self._server.scene.reset()
        self._server.gui.reset()

        # mjviser adjusts several model.vis defaults during construction. Save
        # and restore them so visualization never mutates the shared sim model.
        visual_state = self._capture_visual_state()
        try:
            self._scene = self._scene_type(self._server, self._model, num_envs=1)
        finally:
            self._restore_visual_state(visual_state)

        self._scene.camera_tracking_enabled = False
        self._scene.create_overlay_gui()
        self._scene.create_groups_gui()
        self._scene.update_from_mjdata(self._data)
        self._apply_camera_to_connected_clients()

        if self.wait_for_client and not self._first_client_wait_completed:
            print("Waiting for the first mjviser browser client...")
            self._client_connected.wait()
            self._first_client_wait_completed = True
            self._apply_camera_to_connected_clients()
            print("mjviser browser connected; starting simulation.")

    def _capture_visual_state(self):
        """Capture model visualization fields changed by mjviser construction.

        Args:
            None.

        Returns:
            Dictionary containing copies of affected MuJoCo visualization fields.
        """
        rgba = self._model.vis.rgba
        scale = self._model.vis.scale
        return {
            "contactpoint": rgba.contactpoint.copy(),
            "contactforce": rgba.contactforce.copy(),
            "inertia": rgba.inertia.copy(),
            "framewidth": scale.framewidth,
            "jointwidth": scale.jointwidth,
            "actuatorlength": scale.actuatorlength,
            "actuatorwidth": scale.actuatorwidth,
            "contactwidth": scale.contactwidth,
            "contactheight": scale.contactheight,
            "forcewidth": scale.forcewidth,
        }

    def _restore_visual_state(self, visual_state):
        """Restore visualization fields after mjviser creates its scene handles.

        Args:
            visual_state: Dictionary returned by ``_capture_visual_state``.

        Returns:
            None.
        """
        rgba = self._model.vis.rgba
        scale = self._model.vis.scale
        rgba.contactpoint[:] = visual_state["contactpoint"]
        rgba.contactforce[:] = visual_state["contactforce"]
        rgba.inertia[:] = visual_state["inertia"]
        scale.framewidth = visual_state["framewidth"]
        scale.jointwidth = visual_state["jointwidth"]
        scale.actuatorlength = visual_state["actuatorlength"]
        scale.actuatorwidth = visual_state["actuatorwidth"]
        scale.contactwidth = visual_state["contactwidth"]
        scale.contactheight = visual_state["contactheight"]
        scale.forcewidth = visual_state["forcewidth"]

    def _apply_camera_to_connected_clients(self):
        """Apply the shared initial camera to all currently connected clients.

        Args:
            None.

        Returns:
            None.
        """
        if self._server is None:
            return
        for client in self._server.get_clients().values():
            self._apply_camera(client)

    def _apply_camera(self, client):
        """Align one Viser client camera with the native MuJoCo free camera.

        Args:
            client: Viser client handle whose camera should be updated.

        Returns:
            None.
        """
        if self._model is None:
            return

        azimuth = math.radians(_CAMERA_AZIMUTH)
        elevation = math.radians(_CAMERA_ELEVATION)
        offset = np.array(
            [
                -math.cos(elevation) * math.cos(azimuth),
                -math.cos(elevation) * math.sin(azimuth),
                -math.sin(elevation),
            ]
        )
        client.camera.position = _CAMERA_LOOKAT + _CAMERA_DISTANCE * offset
        client.camera.look_at = _CAMERA_LOOKAT.copy()
        client.camera.up_direction = np.array([0.0, 0.0, 1.0])
        client.camera.fov = math.radians(float(self._model.vis.global_.fovy))


def create_debug_viewer_session(task_config):
    """Create a task-level viewer session from a Hydra task configuration.

    Args:
        task_config: Hydra task configuration containing viewer settings.

    Returns:
        A ``DebugViewerSession`` when debug viewing is enabled, otherwise None.
    """
    if not bool(task_config.debug_viewer):
        return None
    return DebugViewerSession(
        backend=task_config.debug_viewer_backend,
        host=task_config.mjviser.host,
        port=task_config.mjviser.port,
        wait_for_client=True,
    )
