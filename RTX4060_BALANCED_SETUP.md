# RTX 4060 Balanced preset (Windows 11)

Target hardware: NVIDIA GeForce RTX 4060 8 GB, Intel Core i9-14900K, 32 GB RAM.

## Apply the preset

1. Install the project's normal CUDA dependencies using its existing Windows CUDA setup.
2. Start with `run-cuda.bat` (the project command is `python run.py --execution-provider cuda`).
3. In the Camera panel, click **Apply RTX 4060 balanced preset**.
4. Choose a background image for 2.5D or a `.glb`/`.gltf` scene for true 3D,
   enable **3D background**, then start **Live**.
5. Select 1280x720 capture if your webcam supports it. If the camera drops frames, use 960x540 or 640x480.

## Preset values

- Capture: 1280x720
- Face detection: 320x320
- Async face swap: enabled
- 3D background: enabled (requires a selected background image)
- Parallax slider: 35%
- Room perspective slider: 25%
- Background blur slider: 5%
- Segmentation: CUDAExecutionProvider first when ONNX Runtime reports it, otherwise CPU fallback

## Important limitations

The photo mode is depth-aware 2.5D. GLB/glTF mode renders supplied geometry as a real 3D scene, but does not reconstruct 3D from a photograph. The webcam segmentation mask refreshes periodically and is reused between inference frames to reduce load. Actual FPS depends on webcam, drivers, CUDA/cuDNN compatibility, OpenGL support, scene complexity, and which other face processors are enabled. The UI preset cannot guarantee a particular frame rate.

Install optional support with `pip install -r requirements-background-3d.txt`, then run `python -m pip install --no-deps --force-reinstall PyOpenGL==3.1.10` and restart the app. This overrides pyrender's older pinned version to avoid a Python 3.12 ctypes pointer issue when loading GLB textures. The project uses `rembg` with the lightweight `u2netp` model for person segmentation. The depth model downloads automatically for photo mode and is cached under `models/background/`. If CUDA is unavailable to ONNX Runtime, segmentation and depth inference fall back to CPU.
