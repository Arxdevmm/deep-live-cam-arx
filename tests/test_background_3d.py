import unittest

import numpy as np

from modules.background_3d import Background3DEngine


class FakeDepthSession:
    def __init__(self):
        self.input_shape = None

    def run(self, _outputs, inputs):
        tensor = next(iter(inputs.values()))
        self.input_shape = tensor.shape
        h, w = tensor.shape[-2:]
        return [np.linspace(0.0, 1.0, h * w, dtype=np.float32).reshape(1, h, w)]


class Background3DWarpTests(unittest.TestCase):
    def test_zero_strength_preserves_dimensions_and_content(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[20:100, 30:130] = (40, 120, 200)
        result = Background3DEngine._warp_perspective(
            frame, np.zeros(2, dtype=np.float32), 0.0, 0.0
        )
        self.assertEqual(result.shape, frame.shape)
        self.assertTrue(np.array_equal(result, frame))

    def test_perspective_warp_preserves_output_dimensions(self):
        frame = np.full((120, 160, 3), 100, dtype=np.uint8)
        offset = np.array([0.4, -0.2], dtype=np.float32)
        result = Background3DEngine._warp_perspective(frame, offset, 0.05, 0.02)
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_depth_warp_zero_strength_preserves_content(self):
        frame = np.arange(120 * 160 * 3, dtype=np.uint8).reshape((120, 160, 3))
        depth = np.linspace(0.0, 1.0, 160, dtype=np.float32)[None, :]
        depth = np.repeat(depth, 120, axis=0)
        result = Background3DEngine._warp_with_depth(
            frame, depth, np.array([0.5, 0.0], dtype=np.float32), 0.0
        )
        self.assertTrue(np.array_equal(result, frame))

    def test_depth_warp_uses_per_pixel_depth(self):
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
        depth = np.linspace(0.0, 1.0, 160, dtype=np.float32)[None, :]
        depth = np.repeat(depth, 120, axis=0)
        result = Background3DEngine._warp_with_depth(
            frame, depth, np.array([1.0, 0.0], dtype=np.float32), 1.0
        )
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertFalse(np.array_equal(result, frame))

    def test_depth_normalization_handles_constant_and_non_finite_values(self):
        constant = Background3DEngine._normalize_depth(np.full((4, 4), 2.0))
        self.assertTrue(np.all(constant == 0.5))

        mixed = Background3DEngine._normalize_depth(
            np.array([[0.0, 1.0], [np.nan, 2.0]], dtype=np.float32)
        )
        self.assertTrue(np.isfinite(mixed).all())
        self.assertGreaterEqual(float(mixed.min()), 0.0)
        self.assertLessEqual(float(mixed.max()), 1.0)

    def test_depth_estimation_prepares_model_input_and_resizes_output(self):
        engine = Background3DEngine.__new__(Background3DEngine)
        engine._depth_session = FakeDepthSession()
        engine._depth_input_name = "pixel_values"
        background = np.zeros((64, 96, 3), dtype=np.uint8)

        depth = engine._estimate_background_depth(background)

        self.assertEqual(engine._depth_session.input_shape[0:2], (1, 3))
        self.assertEqual(engine._depth_session.input_shape[2] % 14, 0)
        self.assertEqual(engine._depth_session.input_shape[3] % 14, 0)
        self.assertEqual(depth.shape, background.shape[:2])
        self.assertTrue(np.isfinite(depth).all())


if __name__ == "__main__":
    unittest.main()
