"""Optional 2.5D virtual background for the live webcam preview.

Requires the optional ``mediapipe`` package.  Person segmentation is AI-based;
the room effect uses a lightweight projective warp and a synthetic depth field
so it runs without a heavyweight monocular-depth model.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np


class Background3DEngine:
    """AI foreground cutout + depth-inspired perspective/parallax background."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session = None
        self._remove = None
        self._background_path: Optional[str] = None
        self._background: Optional[np.ndarray] = None
        self._smoothed_offset = np.zeros(2, dtype=np.float32)
        self._cached_mask: Optional[np.ndarray] = None
        self._frame_index = 0
        self._last_error: Optional[str] = None
        self._active_providers = []
        self._load_models()

    def _load_models(self) -> None:
        try:
            from rembg import new_session, remove
            # Prefer CUDA for the RTX 4060, but retain a CPU fallback so the
            # feature still starts on systems without a working CUDA runtime.
            try:
                import onnxruntime as ort
                available = set(ort.get_available_providers())
                providers = (["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else [])
                providers.append("CPUExecutionProvider")
                self._session = new_session("u2netp", providers=providers)
                self._active_providers = providers
            except Exception:
                self._session = new_session("u2netp")
                self._active_providers = ["default"]
            self._remove = remove
        except Exception as exc:
            self._last_error = (
                "3D Background needs the optional dependency 'rembg'. "
                "Install it with: pip install -r requirements-background-3d.txt "
                f"(details: {exc})"
            )
            print("[3D Background] " + self._last_error, flush=True)

    @property
    def available(self) -> bool:
        return self._session is not None and self._remove is not None

    @property
    def error(self) -> Optional[str]:
        return self._last_error

    @property
    def active_providers(self) -> list[str]:
        """ONNX Runtime providers selected for segmentation, when available."""
        return list(self._active_providers)

    def close(self) -> None:
        self._session = None
        self._remove = None

    def _get_background(self, path: str, width: int, height: int) -> Optional[np.ndarray]:
        if not path or not os.path.isfile(path):
            return None
        if path != self._background_path or self._background is None:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                self._last_error = f"Could not read background image: {path}"
                return None
            self._background = image
            self._background_path = path
        bg = self._background
        h, w = bg.shape[:2]
        scale = max(width / w, height / h)
        nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
        bg = cv2.resize(bg, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x0, y0 = (nw - width) // 2, (nh - height) // 2
        return bg[y0:y0 + height, x0:x0 + width].copy()

    def _track_head(self, mask: np.ndarray) -> np.ndarray:
        """Estimate head position from the upper portion of the person mask."""
        h, w = mask.shape[:2]
        upper = mask[:max(1, int(h * 0.42)), :]
        ys, xs = np.where(upper > 0.45)
        if len(xs) > 20:
            cx = float(np.mean(xs)) / max(1, w - 1)
            cy = float(np.mean(ys)) / max(1, h - 1)
            target = np.array([(cx - 0.5) * 2.0, (cy - 0.21) * 2.0],
                              dtype=np.float32)
            self._smoothed_offset = 0.82 * self._smoothed_offset + 0.18 * target
        else:
            self._smoothed_offset *= 0.96
        return self._smoothed_offset

    @staticmethod
    def _warp_perspective(background: np.ndarray, offset: np.ndarray,
                          parallax: float, perspective: float) -> np.ndarray:
        h, w = background.shape[:2]
        dx = float(np.clip(offset[0], -1, 1)) * parallax * w
        dy = float(np.clip(offset[1], -1, 1)) * parallax * h
        # Four corners move in opposite directions, giving a subtle 3D-room
        # perspective cue. Extra translation supplies the head-tracked parallax.
        px, py = perspective * w, perspective * h
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32([
            [dx - px, dy - py],
            [w - 1 + dx + px, dy - py],
            [w - 1 + dx - px, h - 1 + dy + py],
            [dx - px, h - 1 + dy + py],
        ])
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(
            background, matrix, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

    def process(self, frame: np.ndarray, background_path: str,
                parallax_strength: float = 0.35,
                perspective_strength: float = 0.25,
                blur_strength: float = 0.0) -> np.ndarray:
        if not self.available or frame is None:
            return frame
        h, w = frame.shape[:2]
        background = self._get_background(background_path, w, h)
        if background is None:
            return frame

        try:
            from PIL import Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._frame_index += 1
            # U2Netp inference is comparatively expensive on CPU. Refresh the
            # segmentation mask every fourth frame and reuse it in between.
            if self._cached_mask is None or self._frame_index % 4 == 1:
                mask_image = self._remove(
                    Image.fromarray(rgb), session=self._session, only_mask=True
                )
                new_mask = np.asarray(mask_image.convert("L"), dtype=np.float32) / 255.0
                if new_mask.size == 0:
                    return frame
                self._cached_mask = new_mask
            mask = self._cached_mask
            if mask.shape[:2] != (h, w):
                self._cached_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = self._cached_mask
            offset = self._track_head(mask)
            moved_bg = self._warp_perspective(
                background, offset,
                float(np.clip(parallax_strength, 0.0, 1.0)) * 0.10,
                float(np.clip(perspective_strength, 0.0, 1.0)) * 0.035,
            )
            if blur_strength > 0:
                radius = max(1, int(round(float(blur_strength) * 15)))
                ksize = radius * 2 + 1
                moved_bg = cv2.GaussianBlur(moved_bg, (ksize, ksize), 0)

            # Feather the AI mask to reduce hair/shoulder halos.
            mask = cv2.resize(mask.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_LINEAR)
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0, sigmaY=2.0)
            mask = np.clip((mask - 0.12) / 0.68, 0.0, 1.0)
            alpha = mask[:, :, None]
            return np.clip(frame.astype(np.float32) * alpha +
                           moved_bg.astype(np.float32) * (1.0 - alpha),
                           0, 255).astype(np.uint8)
        except Exception as exc:
            self._last_error = f"Background processing failed: {exc}"
            print("[3D Background] " + self._last_error, flush=True)
            return frame
