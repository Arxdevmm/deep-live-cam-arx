import unittest

import numpy as np

from modules.background_scene_3d import BackgroundScene3DEngine


class BackgroundScene3DTests(unittest.TestCase):
    def test_camera_pose_looks_at_scene_center(self):
        eye = np.array([2.0, 3.0, 8.0], dtype=np.float32)
        target = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        pose = BackgroundScene3DEngine._look_at(eye, target)
        camera_forward = -pose[:3, 2]
        expected_forward = target - eye
        expected_forward /= np.linalg.norm(expected_forward)

        np.testing.assert_allclose(camera_forward, expected_forward, atol=1e-6)
        np.testing.assert_allclose(pose[:3, 3], eye, atol=1e-6)
        np.testing.assert_allclose(
            pose[:3, :3].T @ pose[:3, :3],
            np.eye(3),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
