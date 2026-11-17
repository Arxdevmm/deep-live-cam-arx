# --- START OF FILE globals.py ---

import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

# Canonical media extensions, defined once so the file dialogs and
# has_image_extension never drift. GIF is intentionally excluded: OpenCV's
# cv2.imread/imwrite (the only image I/O this app uses) cannot decode or
# encode GIF on 4.10 or 4.11, so offering it would silently fail. WEBP works
# via the libwebp bundled with opencv-python.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mkv")


@dataclass
class RuntimeConfig:
    """Typed runtime state for app startup and live-processing configuration."""

    source_target_map: List[Dict[str, Any]] = field(default_factory=list)
    simple_map: Dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None
    target_path: str | None = None
    output_path: str | None = None
    frame_processors: List[str] = field(default_factory=list)
    keep_fps: bool = True
    keep_audio: bool = True
    keep_frames: bool = False
    many_faces: bool = False
    map_faces: bool = False
    poisson_blend: bool = False
    color_correction: bool = False
    nsfw_filter: bool = False
    video_encoder: str | None = None
    video_quality: int | None = None
    live_mirror: bool = False
    live_resizable: bool = True
    camera_input_combobox: Any | None = None
    webcam_preview_running: bool = False
    show_fps: bool = False
    live_async_swap: bool = True
    max_memory: int | None = None
    execution_providers: List[str] = field(default_factory=list)
    execution_threads: int | None = None
    headless: bool | None = None
    log_level: str = "error"
    lang: str = "en"
    fp_ui: Dict[str, bool] = field(
        default_factory=lambda: {
            "face_enhancer": False,
            "face_enhancer_gpen256": False,
            "face_enhancer_gpen512": False,
        }
    )
    face_swapper_enabled: bool = True
    opacity: float = 1.0
    sharpness: float = 0.0
    mouth_mask: bool = False
    show_mouth_mask_box: bool = False
    mask_feather_ratio: int = 12
    mask_down_size: float = 0.1
    mask_size: float = 1.0
    mouth_mask_size: float = 0.0
    enable_interpolation: bool = True
    interpolation_weight: float = 0
    DEFAULT_DET_SIZE: int = 640
    det_size: int = 320
    capture_resolution: tuple = (1280, 720)
    dml_lock: threading.Lock = field(default_factory=threading.Lock)
    enable_3d_background: bool = True
    background_image_path: str | None = None
    background_scene_path: str | None = None
    background_parallax_strength: float = 0.35
    background_perspective_strength: float = 0.25
    background_blur: float = 0.05


runtime_config = RuntimeConfig()


def apply_runtime_config(config: RuntimeConfig | None = None) -> RuntimeConfig:
    """Synchronize the typed runtime config with legacy module-level globals."""
    if config is None:
        config = runtime_config

    for key, value in config.__dict__.items():
        setattr(runtime_config, key, value)
        setattr(sys.modules[__name__], key, value)
    return config


# Mirror legacy globals for compatibility while the app is being migrated.
# These remain the same names existing modules use today.
source_target_map: List[Dict[str, Any]] = runtime_config.source_target_map
simple_map: Dict[str, Any] = runtime_config.simple_map
source_path: str | None = runtime_config.source_path
target_path: str | None = runtime_config.target_path
output_path: str | None = runtime_config.output_path
frame_processors: List[str] = runtime_config.frame_processors
keep_fps: bool = runtime_config.keep_fps
keep_audio: bool = runtime_config.keep_audio
keep_frames: bool = runtime_config.keep_frames
many_faces: bool = runtime_config.many_faces
map_faces: bool = runtime_config.map_faces
poisson_blend: bool = runtime_config.poisson_blend
color_correction: bool = runtime_config.color_correction
nsfw_filter: bool = runtime_config.nsfw_filter
video_encoder: str | None = runtime_config.video_encoder
video_quality: int | None = runtime_config.video_quality
live_mirror: bool = runtime_config.live_mirror
live_resizable: bool = runtime_config.live_resizable
camera_input_combobox: Any | None = runtime_config.camera_input_combobox
webcam_preview_running: bool = runtime_config.webcam_preview_running
show_fps: bool = runtime_config.show_fps
live_async_swap: bool = runtime_config.live_async_swap
max_memory: int | None = runtime_config.max_memory
execution_providers: List[str] = runtime_config.execution_providers
execution_threads: int | None = runtime_config.execution_threads
headless: bool | None = runtime_config.headless
log_level: str = runtime_config.log_level
lang: str = runtime_config.lang
fp_ui: Dict[str, bool] = runtime_config.fp_ui
face_swapper_enabled: bool = runtime_config.face_swapper_enabled
opacity: float = runtime_config.opacity
sharpness: float = runtime_config.sharpness
mouth_mask: bool = runtime_config.mouth_mask
show_mouth_mask_box: bool = runtime_config.show_mouth_mask_box
mask_feather_ratio: int = runtime_config.mask_feather_ratio
mask_down_size: float = runtime_config.mask_down_size
mask_size: float = runtime_config.mask_size
mouth_mask_size: float = runtime_config.mouth_mask_size
enable_interpolation: bool = runtime_config.enable_interpolation
interpolation_weight: float = runtime_config.interpolation_weight
DEFAULT_DET_SIZE: int = runtime_config.DEFAULT_DET_SIZE
det_size: int = runtime_config.det_size
capture_resolution: tuple = runtime_config.capture_resolution
dml_lock = runtime_config.dml_lock
enable_3d_background: bool = runtime_config.enable_3d_background
background_image_path: str | None = runtime_config.background_image_path
background_scene_path: str | None = runtime_config.background_scene_path
background_parallax_strength: float = runtime_config.background_parallax_strength
background_perspective_strength: float = runtime_config.background_perspective_strength
background_blur: float = runtime_config.background_blur

# Keep the old module namespace values in sync with the config object after any
# direct assignment. This preserves current call sites while giving us a strong
# typed config source to move towards.
apply_runtime_config(runtime_config)

# --- END OF FILE globals.py ---
