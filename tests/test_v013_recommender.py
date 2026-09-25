import unittest

import numpy as np

from bone_iqa.app import BoneIQAApp, ROI_LABELS
from bone_iqa.models import ROI
from bone_iqa.roi_recommender import (
    _gradient_map,
    _structure_continuity,
    build_effective_signal_mask,
    recommend_rois,
    recommend_surrounding_for_roi,
)
from test_v011_regressions import synthetic_body_image
from test_v012_recommender import dual_projection_image, roi_iou


class ChineseUiAndCompatibilityTest(unittest.TestCase):
    def test_visible_labels_are_chinese_and_internal_types_are_unchanged(self):
        self.assertEqual(
            ROI_LABELS,
            {"weak_bone": "弱骨骼", "strong_bone": "强骨骼", "surrounding": "邻域", "background": "背景"},
        )
        rois = [
            ROI.create("a", "weak_bone", 0, 0, 8, 8),
            ROI.create("b", "strong_bone", 8, 0, 8, 8),
            ROI.create("c", "surrounding", 16, 0, 8, 8),
            ROI.create("d", "background", 24, 0, 8, 8),
        ]
        app = object.__new__(BoneIQAApp)
        self.assertEqual(list(app._short_roi_labels(rois).values()), ["弱骨骼1", "强骨骼1", "邻域1", "背景1"])
        self.assertEqual([roi.type for roi in rois], ["weak_bone", "strong_bone", "surrounding", "background"])

    def test_recommended_names_do_not_contain_recommendation_marker(self):
        recommendations = recommend_rois(synthetic_body_image())
        self.assertTrue(recommendations)
        self.assertTrue(all("推荐" not in item.roi.name for item in recommendations))
        self.assertTrue(any(item.roi.name.startswith("弱骨骼") for item in recommendations))
        self.assertTrue(any(item.roi.name.startswith("强骨骼") for item in recommendations))
        self.assertTrue(any(item.roi.name.endswith("-邻域") for item in recommendations if item.roi.type == "surrounding"))


class WeakStructureRecommendationTest(unittest.TestCase):
    def test_continuous_edge_scores_above_comparable_random_texture(self):
        rng = np.random.default_rng(13)
        continuous = np.zeros((24, 24), dtype=np.float64)
        continuous[:, 12:] = 50.0
        random_texture = rng.normal(25.0, 25.0, (24, 24))
        continuous_gradient = _gradient_map(continuous)
        random_gradient = _gradient_map(random_texture)
        continuous_edges = continuous_gradient >= np.percentile(continuous_gradient, 90)
        random_edges = random_gradient >= np.percentile(random_gradient, 90)
        self.assertGreater(
            _structure_continuity(continuous, continuous_edges),
            _structure_continuity(random_texture, random_edges),
        )

    def test_weak_candidates_cover_distinct_spatial_grid_cells(self):
        image = dual_projection_image()
        weak = [item.roi for item in recommend_rois(image) if item.roi.type == "weak_bone"]
        cells = {(min(2, int(3 * (roi.x + roi.width / 2) / image.shape[1])), min(2, int(3 * (roi.y + roi.height / 2) / image.shape[0]))) for roi in weak}
        self.assertGreaterEqual(len(cells), 2)


class SignalAwareSurroundingTest(unittest.TestCase):
    def test_effective_signal_mask_keeps_both_subjects(self):
        image = dual_projection_image()
        mask = build_effective_signal_mask(image)
        self.assertGreater(float(np.mean(mask[:, 15:65])), 0.5)
        self.assertGreater(float(np.mean(mask[:, 95:145])), 0.5)

    def test_low_signal_overlap_is_not_marked_recommended(self):
        image = np.zeros((80, 80), dtype=np.uint8)
        image[32:44, 32:44] = 100
        image[32:44, 44:47] = 35
        image[32:44, 47:56] = 2
        bone = ROI.create("弱骨1", "weak_bone", 32, 32, 12, 12)
        result = recommend_surrounding_for_roi(image, bone)
        self.assertIsNotNone(result)
        self.assertNotEqual(result.quality, "推荐")

    def test_body_neighbor_beats_pure_background_and_does_not_overlap(self):
        rng = np.random.default_rng(3)
        image = np.zeros((80, 80), dtype=np.uint8)
        image[32:44, 32:44] = rng.integers(85, 110, (12, 12), dtype=np.uint8)
        image[32:44, 20:32] = rng.integers(2, 8, (12, 12), dtype=np.uint8)
        image[32:44, 44:56] = rng.integers(35, 50, (12, 12), dtype=np.uint8)
        bone = ROI.create("弱骨1", "weak_bone", 32, 32, 12, 12)
        result = recommend_surrounding_for_roi(image, bone)
        self.assertIsNotNone(result)
        self.assertEqual(result.roi.geometry(), (44, 32, 12, 12))
        self.assertEqual(roi_iou(bone, result.roi), 0.0)


if __name__ == "__main__":
    unittest.main()
