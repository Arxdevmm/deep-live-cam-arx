"""True 3D scene rendering for live webcam backgrounds."""
from __future__ import annotations

import os
import sys
from typing import Optional

import cv2
import numpy as np


class BackgroundScene3DEngine:
    """Render a GLB/glTF scene behind an AI-segmented webcam foreground."""

    def __init__(self) -> None:
        self._remove = None
        self._session = None
        self._scene_path: Optional[str] = None
        self._scene = None
        self._renderer = None
        self._renderer_size: Optional[tuple[int, int]] = None
        self._camera = None
        self._camera_node = None
        self._center: Optional[np.ndarray] = None
        self._distance = 1.0
        self._radius = 1.0
        self._cached_mask: Optional[np.ndarray] = None
        self._frame_index = 0
        self._smoothed_offset = np.zeros(2, dtype=np.float32)
        self._last_error: Optional[str] = None
        self._load_dependencies()

    def _load_dependencies(self) -> None:
        try:
            if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
                os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            import OpenGL
            pyopengl_version = tuple(
                int(part) for part in OpenGL.__version__.split(".")[:3]
            )
            if pyopengl_version < (3, 1, 10):
                raise RuntimeError(
                    "PyOpenGL 3.1.10 or newer is required for textured GLB "
                    "scenes. After installing optional background dependencies, "
                    "run `python -m pip install --no-deps --force-reinstall "
                    "PyOpenGL==3.1.10`, then restart the app/runtime."
                )
            import pyrender
            import trimesh
            from rembg import new_session, remove

            try:
                import onnxruntime as ort

                available = set(ort.get_available_providers())
                providers = (
                    ["CUDAExecutionProvider"]
                    if "CUDAExecutionProvider" in available
                    else []
                )
                providers.append("CPUExecutionProvider")
                self._session = new_session("u2netp", providers=providers)
            except Exception:
                self._session = new_session("u2netp")
            self._remove = remove
            self._pyrender = pyrender
            self._trimesh = trimesh
        except Exception as exc:
            self._last_error = (
                "GLB backgrounds need the optional packages in "
                "requirements-background-3d.txt. "
                f"Details: {exc}"
            )
            print("[3D Background] " + self._last_error, flush=True)

    @property
    def available(self) -> bool:
        return (
            self._session is not None
            and self._remove is not None
            and hasattr(self, "_pyrender")
            and hasattr(self, "_trimesh")
        )

    @property
    def error(self) -> Optional[str]:
        return self._last_error

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None
            self._renderer_size = None
        self._session = None
        self._remove = None

    @staticmethod
    def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        forward = target - eye
        forward /= max(float(np.linalg.norm(forward)), 1e-8)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up)
        right /= max(float(np.linalg.norm(right)), 1e-8)
        up = np.cross(right, forward)

        pose = np.eye(4, dtype=np.float32)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = eye
        return pose

    def _load_scene(self, path: str) -> bool:
        try:
            if self._renderer is not None:
                self._renderer.delete()
                self._renderer = None
                self._renderer_size = None

            loaded = self._trimesh.load(path, force="scene")
            if not isinstance(loaded, self._trimesh.Scene) or not loaded.geometry:
                raise ValueError("The selected GLB/glTF file contains no renderable geometry.")

            bounds = np.asarray(loaded.bounds, dtype=np.float32)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
                raise ValueError("The selected scene has invalid or empty bounds.")
            self._center = (bounds[0] + bounds[1]) * 0.5
            extents = np.maximum(bounds[1] - bounds[0], 1e-3)
            self._radius = max(float(np.linalg.norm(extents)) * 0.5, 1e-2)

            scene = self._pyrender.Scene.from_trimesh_scene(
                loaded,
                bg_color=np.array([0.02, 0.02, 0.02, 1.0]),
                ambient_light=np.array([0.55, 0.55, 0.55]),
            )
            for camera_node in list(scene.camera_nodes):
                scene.remove_node(camera_node)
            light = self._pyrender.DirectionalLight(
                color=np.ones(3), intensity=2.0
            )
            light_pose = self._look_at(
                self._center + np.array(
                    [self._radius, self._radius * 1.5, self._radius * 2.0],
                    dtype=np.float32,
                ),
                self._center,
            )
            scene.add(light, pose=light_pose)

            yfov = np.deg2rad(50.0)
            camera = self._pyrender.PerspectiveCamera(yfov=yfov)
            self._distance = self._radius / max(np.sin(yfov * 0.5), 1e-3) * 1.15
            self._camera = camera
            eye = self._center + np.array(
                [0.0, 0.0, self._distance], dtype=np.float32
            )
            self._camera_node = scene.add(
                camera, pose=self._look_at(eye, self._center)
            )
            self._scene = scene
            self._scene_path = path
            self._last_error = None
            return True
        except Exception as exc:
            self._scene = None
            self._scene_path = path
            self._last_error = f"Could not load 3D background scene: {exc}"
            print("[3D Background] " + self._last_error, flush=True)
            return False

    def _ensure_renderer(self, width: int, height: int) -> None:
        size = (width, height)
        if self._renderer is not None and self._renderer_size == size:
            return
        if self._renderer is not None:
            self._renderer.delete()

        self._renderer = self._pyrender.OffscreenRenderer(
            viewport_width=width,
            viewport_height=height,
        )
        self._renderer_size = size

    def _track_head(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        upper = mask[:max(1, int(h * 0.42)), :]
        ys, xs = np.where(upper > 0.45)
        if len(xs) > 20:
            cx = float(np.mean(xs)) / max(1, w - 1)
            cy = float(np.mean(ys)) / max(1, h - 1)
            target = np.array(
                [(cx - 0.5) * 2.0, (cy - 0.21) * 2.0],
                dtype=np.float32,
            )
            self._smoothed_offset = (
                0.82 * self._smoothed_offset + 0.18 * target
            )
        else:
            self._smoothed_offset *= 0.96
        return self._smoothed_offset

    def process(
        self,
        frame: np.ndarray,
        scene_path: str,
        parallax_strength: float = 0.35,
    ) -> np.ndarray:
        if not self.available or frame is None:
            return frame

        if scene_path != self._scene_path and not self._load_scene(scene_path):
            return frame
        if self._scene is None:
            return frame

        try:
            from PIL import Image

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._frame_index += 1
            if self._cached_mask is None or self._frame_index % 4 == 1:
                mask_image = self._remove(
                    Image.fromarray(rgb),
                    session=self._session,
                    only_mask=True,
                )
                self._cached_mask = (
                    np.asarray(mask_image.convert("L"), dtype=np.float32) / 255.0
                )
            mask = self._cached_mask
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                self._cached_mask = mask

            offset = self._track_head(mask)
            strength = float(np.clip(parallax_strength, 0.0, 1.0))
            eye = self._center + np.array(
                [
                    offset[0] * self._radius * strength * 0.22,
                    -offset[1] * self._radius * strength * 0.16,
                    self._distance,
                ],
                dtype=np.float32,
            )
            camera_pose = self._look_at(eye, self._center)

            self._ensure_renderer(w, h)
            self._scene.set_pose(self._camera_node, pose=camera_pose)
            rendered, _depth = self._renderer.render(self._scene)

            rendered = rendered[:, :, :3]
            if rendered.dtype != np.uint8:
                multiplier = 255.0 if float(np.max(rendered)) <= 1.0 else 1.0
                rendered = np.clip(rendered * multiplier, 0, 255).astype(np.uint8)
            background = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            mask = cv2.GaussianBlur(
                mask.astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=2.0
            )
            mask = np.clip((mask - 0.12) / 0.68, 0.0, 1.0)[:, :, None]
            return np.clip(
                frame.astype(np.float32) * mask
                + background.astype(np.float32) * (1.0 - mask),
                0,
                255,
            ).astype(np.uint8)
        except Exception as exc:
            self._last_error = f"3D background rendering failed: {exc}"
            if self._last_error != getattr(self, "_reported_error", None):
                print("[3D Background] " + self._last_error, flush=True)
                self._reported_error = self._last_error
            return frame
