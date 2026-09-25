import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from bone_iqa.evaluator import evaluate_project
from bone_iqa.metrics import cnr_background, cnr_local, pooled_background_noise, roi_statistics
from bone_iqa.models import ROI
from bone_iqa.project_store import ProjectStore
from bone_iqa.roi_recommender import recommend_rois
from bone_iqa.validation import validate_project


def synthetic_body_image() -> np.ndarray:
    rng = np.random.default_rng(20260925)
    image = np.zeros((128, 96), dtype=np.uint8)
    image[4:28, 4:24] = rng.integers(2, 8, size=(24, 20), dtype=np.uint8)
    image[4:28, 72:92] = rng.integers(3, 9, size=(24, 20), dtype=np.uint8)
    image[20:112, 28:68] = rng.integers(18, 35, size=(92, 40), dtype=np.uint8)
    image[28:105, 45:52] = 220
    image[72:92, 34:64] = 185
    image[38:52, 30:44] += np.tile(np.array([0, 20], dtype=np.uint8), (14, 7))
    return image


class BackgroundRegressionTest(unittest.TestCase):
    def test_zero_background_keeps_local_cnr(self):
        stats = roi_statistics(np.zeros((8, 8), dtype=np.uint8))
        noise = pooled_background_noise([stats])
        self.assertEqual(noise, 0.0)
        self.assertIsNone(cnr_background(50, 20, noise))
        self.assertIsNotNone(cnr_local(50, 20, 5, 4))

    def test_constant_twenty_background_is_invalid_for_background_cnr(self):
        stats = roi_statistics(np.full((8, 8), 20, dtype=np.uint8))
        noise = pooled_background_noise([stats])
        self.assertEqual(noise, 0.0)
        self.assertEqual(stats["unique_pixel_count"], 1)
        self.assertIsNone(cnr_background(50, 20, noise))

    def test_varying_background_supports_background_cnr(self):
        values = np.array([18, 19, 20, 21] * 16, dtype=np.uint8).reshape(8, 8)
        noise = pooled_background_noise([roi_statistics(values)])
        self.assertGreater(noise, 0)
        self.assertIsNotNone(cnr_background(50, 20, noise))


class PairingAndStatusTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image_path = self.root / "original.png"
        Image.fromarray(synthetic_body_image()).save(self.image_path)
        self.store = ProjectStore.create(self.root / "project", "v011")
        self.store.set_original(self.image_path)

    def tearDown(self):
        self.temp.cleanup()

    def test_only_surrounding_is_selected_automatically(self):
        surrounding = ROI.create("周围", "surrounding", 24, 36, 12, 12)
        self.store.upsert_roi(surrounding)
        bone = ROI.create("弱骨", "weak_bone", 36, 36, 12, 12)
        self.store.upsert_roi(bone)
        self.assertEqual(bone.paired_surrounding_roi_id, surrounding.id)

    def test_nearest_surrounding_is_selected(self):
        far = ROI.create("远", "surrounding", 2, 2, 8, 8)
        near = ROI.create("近", "surrounding", 30, 30, 8, 8)
        self.store.upsert_roi(far)
        self.store.upsert_roi(near)
        bone = ROI.create("弱骨", "weak_bone", 40, 32, 8, 8)
        self.store.upsert_roi(bone)
        self.assertEqual(bone.paired_surrounding_roi_id, near.id)

    def test_delete_surrounding_clears_pair_and_warns(self):
        surrounding = ROI.create("周围", "surrounding", 24, 36, 12, 12)
        self.store.upsert_roi(surrounding)
        bone = ROI.create("弱骨", "weak_bone", 36, 36, 12, 12, surrounding.id)
        self.store.upsert_roi(bone)
        self.store.remove_roi(surrounding.id)
        self.assertIsNone(bone.paired_surrounding_roi_id)
        codes = {issue.code for issue in validate_project(self.store)}
        self.assertIn("missing_surrounding_pair", codes)

    def test_missing_strong_bone_is_not_partial(self):
        surrounding = ROI.create("周围", "surrounding", 28, 30, 12, 12)
        weak = ROI.create("弱骨", "weak_bone", 40, 30, 12, 12, surrounding.id)
        background = ROI.create("背景", "background", 4, 4, 18, 18)
        for roi in (surrounding, weak, background):
            self.store.upsert_roi(roi)
        result = evaluate_project(self.store)
        self.assertIn(result["metrics"][0]["status"], {"valid", "valid_with_warnings"})
        self.assertNotEqual(result["metrics"][0]["status"], "partial")

    def test_zero_background_reports_reason_while_local_cnr_remains_valid(self):
        surrounding = ROI.create("周围", "surrounding", 28, 30, 12, 12)
        weak = ROI.create("弱骨", "weak_bone", 40, 30, 12, 12, surrounding.id)
        background = ROI.create("纯黑背景", "background", 0, 112, 16, 16)
        for roi in (surrounding, weak, background):
            self.store.upsert_roi(roi)
        result = evaluate_project(self.store)
        summary = result["metrics"][0]
        self.assertIsNone(summary["weak_bone_mean_cnr_background"])
        self.assertIsNotNone(summary["weak_bone_mean_cnr_local"])
        self.assertEqual(summary["status"], "valid_with_warnings")
        self.assertIn("背景 ROI 方差为 0", summary["message"])

    def test_saved_background_primary_is_respected_for_existing_project(self):
        self.store.project.evaluation_config.primary_cnr = "background"
        self.store.save()
        reopened = ProjectStore.open(self.store.root)
        self.assertEqual(reopened.project.evaluation_config.primary_cnr, "background")


class RecommendationTest(unittest.TestCase):
    def test_recommendations_are_in_bounds_and_background_not_constant(self):
        image = synthetic_body_image()
        recommendations = recommend_rois(image)
        self.assertTrue(recommendations)
        height, width = image.shape
        backgrounds = [item.roi for item in recommendations if item.roi.type == "background"]
        self.assertTrue(backgrounds)
        for item in recommendations:
            roi = item.roi
            self.assertGreater(roi.width, 0)
            self.assertGreater(roi.height, 0)
            self.assertGreaterEqual(roi.x, 0)
            self.assertGreaterEqual(roi.y, 0)
            self.assertLessEqual(roi.x + roi.width, width)
            self.assertLessEqual(roi.y + roi.height, height)
        for roi in backgrounds:
            region = image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
            self.assertGreater(np.std(region, ddof=1), 0)
            self.assertGreater(np.unique(region).size, 1)
            self.assertGreater(np.mean(region), 0)

    def test_accept_save_reopen_preserves_pairs_and_same_rois_for_all_schemes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = synthetic_body_image()
            original_path = root / "original.png"
            method_path = root / "method.png"
            Image.fromarray(image).save(original_path)
            Image.fromarray(np.clip(image.astype(np.int16) + 2, 0, 255).astype(np.uint8)).save(method_path)
            store = ProjectStore.create(root / "project", "recommendation")
            store.set_original(original_path)
            store.add_algorithm(method_path, "方法 A")
            recommendations = recommend_rois(image)
            for item in sorted(recommendations, key=lambda value: value.roi.type != "surrounding"):
                store.upsert_roi(item.roi)
            reopened = ProjectStore.open(store.root)
            self.assertEqual(len(reopened.project.rois), len(recommendations))
            roi_ids = {roi.id for roi in reopened.project.rois}
            for roi in reopened.project.rois:
                if roi.type in {"weak_bone", "strong_bone"} and roi.paired_surrounding_roi_id:
                    self.assertIn(roi.paired_surrounding_roi_id, roi_ids)
            result = evaluate_project(reopened)
            by_scheme = {}
            for row in result["roi_metrics"]:
                by_scheme.setdefault(row["scheme_id"], set()).add(row["roi_id"])
            self.assertEqual(len(by_scheme), 2)
            self.assertEqual(*by_scheme.values())


if __name__ == "__main__":
    unittest.main()
