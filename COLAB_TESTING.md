# Colab testing

Colab can validate the Python application and test suite. It cannot run the Windows
`.bat` installer or create Windows shortcuts.

Use the ready-made notebook
[Deep-Live-Cam-Colab-Test.ipynb](Deep-Live-Cam-Colab-Test.ipynb).

## 1. Clone and enter the repository

```python
!git clone --depth 1 https://github.com/Arxdevmm/deep-live-cam-arx.git
%cd deep-live-cam-arx
```

To test a specific commit:

```python
!git fetch --depth 1 origin 911e888
!git checkout 911e888
```

## 2. Install dependencies

```python
!pip install -r requirements.txt
!pip install pytest
```

Restart the Colab runtime if pip reports that imported packages require a restart.

## 3. Check code and run tests

```python
!python -m compileall -q run.py modules tests
!pytest -q
```

## 4. Optional dependency and model smoke check

```python
import modules.model_downloader
print(modules.model_downloader.ensure_model("gfpgan-1024.onnx"))
print(modules.model_downloader.ensure_model("inswapper_128_fp16.onnx"))
```

Model downloads can consume several hundred megabytes and should only be run when
you are ready to test model loading.

## 5. Optional combined face-swap and background test

The notebook also includes an offline single-image test for the 2.5D background
and face-swap pipeline. Install `rembg` when prompted, then upload a source face,
a target portrait, and a replacement background in the following cells. The
background is replaced before the face swap, matching the live preview order.
The first run downloads segmentation, depth-estimation, and face-swap models.
This tests image processing only; Colab does not run the webcam UI or GLB/glTF
scene renderer.
