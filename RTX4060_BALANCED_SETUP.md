# RTX 4060 Balanced preset (Windows 11)

Target hardware: NVIDIA GeForce RTX 4060 8 GB, Intel Core i9-14900K, 32 GB RAM.

## Apply the preset

1. Install the project's normal CUDA dependencies using its existing Windows CUDA setup.
2. Start with `run-cuda.bat` (the project command is `python run.py --execution-provider cuda`).
3. In the Camera panel, click **Apply RTX 4060 balanced preset**.
4. Choose a background image, enable **3D background**, then start **Live**.
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

The effect is depth-aware 2.5D, not a true reconstructed 3D room mesh. Depth Anything V2 estimates relative depth from the selected background image once per image; the webcam segmentation mask refreshes periodically and is reused between inference frames to reduce load. Actual FPS depends on webcam, drivers, CUDA/cuDNN compatibility, and which other face processors are enabled. The UI preset cannot guarantee a particular frame rate.

The project uses `rembg` with the lightweight `u2netp` model for person segmentation. Install the optional package with `pip install -r requirements-background-3d.txt`. The depth model downloads automatically on first use and is cached under `models/background/`. If CUDA is unavailable to ONNX Runtime, segmentation and depth inference fall back to CPU.
