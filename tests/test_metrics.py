import math
import unittest

import numpy as np

from bone_iqa.metrics import (
    average_gradient,
    cnr_background,
    cnr_local,
    pooled_background_noise,
    relative_change,
    roi_statistics,
    saturation_ratio,
    structural_similarity,
)


class MetricsTest(unittest.TestCase):
    def test_roi_statistics_uses_sample_std(self):
        values = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        stats = roi_statistics(values)
        self.assertEqual(stats["pixel_count"], 4)
        self.assertEqual(stats["unique_pixel_count"], 4)
        self.assertAlmostEqual(stats["mean_intensity"], 2.5)
        self.assertAlmostEqual(stats["std_intensity"], np.std(values, ddof=1))

    def test_pooled_background_noise(self):
        first = roi_statistics(np.array([0, 2, 4], dtype=np.uint8))
        second = roi_statistics(np.array([10, 12, 14], dtype=np.uint8))
        self.assertAlmostEqual(pooled_background_noise([first, second]), 2.0)

    def test_background_and_local_cnr(self):
        self.assertAlmostEqual(cnr_background(20, 10, 2), 5.0)
        expected = 10 / math.sqrt((2**2 + 4**2) / 2)
        self.assertAlmostEqual(cnr_local(20, 10, 2, 4), expected)
        self.assertIsNone(cnr_background(20, 10, 0))
        self.assertIsNone(cnr_local(20, 10, 0, 0))

    def test_average_gradient(self):
        self.assertEqual(average_gradient(np.ones((3, 3))), 0.0)
        horizontal = np.tile(np.arange(4, dtype=np.float64), (4, 1))
        self.assertAlmostEqual(average_gradient(horizontal), 1 / math.sqrt(2))
        self.assertIsNone(average_gradient(np.ones((1, 4))))

    def test_saturation_ratio_uint8_default(self):
        values = np.array([249, 250, 255, 0], dtype=np.uint8)
        ratio, count, threshold = saturation_ratio(values, 0, 255, 0.98)
        self.assertAlmostEqual(threshold, 249.9)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(ratio, 0.5)

    def test_relative_change(self):
        self.assertAlmostEqual(relative_change(15, 10), 50.0)
        self.assertIsNone(relative_change(1, 0))

    def test_ssim_identity_and_change(self):
        image = np.arange(32 * 32, dtype=np.float64).reshape(32, 32) % 256
        self.assertAlmostEqual(structural_similarity(image, image, 255), 1.0, places=12)
        changed = image.copy()
        changed[12:20, 12:20] += 20
        score = structural_similarity(image, changed, 255)
        self.assertIsNotNone(score)
        self.assertLess(score, 1.0)


if __name__ == "__main__":
    unittest.main()
