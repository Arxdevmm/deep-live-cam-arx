import unittest

import numpy as np

from modules.background_3d import Background3DEngine


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


if __name__ == "__main__":
    unittest.main()
