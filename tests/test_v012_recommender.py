import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from bone_iqa.app import BoneIQAApp
from bone_iqa.evaluator import evaluate_project
from bone_iqa.models import ROI
from bone_iqa.project_store import ProjectStore
from bone_iqa.roi_recommender import (
    build_strong_response_masks,
    recommend_rois,
    recommend_surrounding_for_roi,
    remove_recommendation,
)
from test_v011_regressions import synthetic_body_image


def roi_region(image: np.ndarray, roi: ROI) -> np.ndarray:
    return image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]


def roi_iou(first: ROI, second: ROI) -> float:
    left, top = max(first.x, second.x), max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


def dual_projection_image(dtype=np.uint8) -> np.ndarray:
    rng = np.random.default_rng(20261002)
    image = np.zeros((160, 160), dtype=np.uint8)
    image[15:150, 15:65] = rng.integers(12, 32, (135, 50), dtype=np.uint8)
    image[15:150, 95:145] = rng.integers(12, 32, (135, 50), dtype=np.uint8)
    image[25:140, 36:42] = 220
    image[25:140, 117:123] = 215
    image[55:80, 18:34] += np.tile(np.array([0, 20], dtype=np.uint8), (25, 8))
    image[90:115, 126:142] += np.tile(np.array([0, 18], dtype=np.uint8), (25, 8))
    if dtype == np.uint16:
        return image.astype(np.uint16) * 257
    return image


class WeakAndStrongRecommendationTest(unittest.TestCase):
    def setUp(self):
        self.image = synthetic_body_image()
        self.recommendations = recommend_rois(self.image)
        self.weak = [item.roi for item in self.recommendations if item.roi.type == "weak_bone"]
        self.strong = [item.roi for item in self.recommendations if item.roi.type == "strong_bone"]

    def test_weak_candidates_avoid_strong_response(self):
        _, exclusion = build_strong_response_masks(self.image)
        self.assertTrue(self.weak)
        for roi in self.weak:
            overlap = float(np.mean(roi_region(exclusion, roi)))
            self.assertLessEqual(overlap, 0.18)
            self.assertTrue(all(roi_iou(roi, strong) <= 0.12 for strong in self.strong))

    def test_weak_candidates_are_spatially_separated(self):
        diagonal = float(np.hypot(*self.image.shape))
        centers = [(roi.x + roi.width / 2, roi.y + roi.height / 2) for roi in self.weak]
        for index, first in enumerate(centers):
            for second in centers[index + 1 :]:
                self.assertGreaterEqual(float(np.hypot(first[0] - second[0], first[1] - second[1])) / diagonal, 0.12)

    def test_weak_does_not_prefer_highest_response(self):
        self.assertTrue(self.strong)
        strongest_mean = max(float(np.mean(roi_region(self.image, roi))) for roi in self.strong)
        self.assertTrue(all(float(np.mean(roi_region(self.image, roi))) < strongest_mean for roi in self.weak))


class SurroundingRecommendationTest(unittest.TestCase):
    def test_surrounding_is_lower_than_bone_nonconstant_and_nonoverlapping(self):
        image = synthetic_body_image()
        recommendations = recommend_rois(image)
        by_id = {item.roi.id: item.roi for item in recommendations}
        bones = [item.roi for item in recommendations if item.roi.type in {"weak_bone", "strong_bone"} and item.roi.paired_surrounding_roi_id]
        self.assertTrue(bones)
        for bone in bones:
            surrounding = by_id[bone.paired_surrounding_roi_id]
            self.assertEqual(roi_iou(bone, surrounding), 0.0)
            bone_mean = float(np.mean(roi_region(image, bone)))
            surrounding_region = roi_region(image, surrounding)
            self.assertLess(float(np.mean(surrounding_region)), bone_mean)
            self.assertGreater(float(np.std(surrounding_region, ddof=1)), 0.0)
            self.assertGreater(np.unique(surrounding_region).size, 1)

    def test_surrounding_avoids_adjacent_strong_structure(self):
        rng = np.random.default_rng(4)
        image = np.zeros((80, 80), dtype=np.uint8)
        image[32:44, 32:44] = rng.integers(85, 116, (12, 12), dtype=np.uint8)
        image[32:44, 20:32] = rng.integers(210, 241, (12, 12), dtype=np.uint8)
        image[32:44, 44:56] = rng.integers(30, 50, (12, 12), dtype=np.uint8)
        bone = ROI.create("测试 Bone", "weak_bone", 32, 32, 12, 12)
        recommendation = recommend_surrounding_for_roi(image, bone)
        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.roi.geometry(), (44, 32, 12, 12))
        self.assertEqual(roi_iou(bone, recommendation.roi), 0.0)

    def test_deleting_recommended_surrounding_clears_pair(self):
        recommendations = recommend_rois(synthetic_body_image())
        bone = next(item for item in recommendations if item.roi.type == "weak_bone" and item.roi.paired_surrounding_roi_id)
        pair_id = bone.roi.paired_surrounding_roi_id
        remaining = remove_recommendation(recommendations, pair_id)
        updated_bone = next(item for item in remaining if item.roi.id == bone.roi.id)
        self.assertIsNone(updated_bone.roi.paired_surrounding_roi_id)
        self.assertEqual(updated_bone.quality, "需检查")


class CompatibilityAndPersistenceTest(unittest.TestCase):
    def test_canvas_short_labels_and_pair_highlight(self):
        neighbor = ROI.create("完整周围名称", "surrounding", 2, 2, 8, 8)
        weak = ROI.create("完整弱骨名称", "weak_bone", 12, 2, 8, 8, neighbor.id)
        background = ROI.create("完整背景名称", "background", 2, 20, 8, 8)
        app = object.__new__(BoneIQAApp)
        app._selected_roi_id = weak.id
        app._selected_recommendation_id = None
        labels = app._short_roi_labels([weak, neighbor, background])
        self.assertEqual(labels[weak.id], "W1")
        self.assertEqual(labels[neighbor.id], "N1")
        self.assertEqual(labels[background.id], "BG1")
        self.assertEqual(app._linked_highlight_ids([weak, neighbor, background]), {weak.id, neighbor.id})

    def test_uint16_recommendations_are_dynamic_and_in_bounds(self):
        image = dual_projection_image(np.uint16)
        recommendations = recommend_rois(image)
        self.assertTrue(any(item.roi.type == "weak_bone" for item in recommendations))
        height, width = image.shape
        for item in recommendations:
            roi = item.roi
            self.assertGreaterEqual(roi.x, 0)
            self.assertGreaterEqual(roi.y, 0)
            self.assertLessEqual(roi.x + roi.width, width)
            self.assertLessEqual(roi.y + roi.height, height)

    def test_dual_projection_keeps_candidates_on_both_sides(self):
        image = dual_projection_image()
        weak = [item.roi for item in recommend_rois(image) if item.roi.type == "weak_bone"]
        centers = [roi.x + roi.width / 2 for roi in weak]
        self.assertTrue(any(center < image.shape[1] / 2 for center in centers))
        self.assertTrue(any(center > image.shape[1] / 2 for center in centers))

    def test_accept_bone_saves_neighbor_and_all_algorithms_share_rois(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = synthetic_body_image()
            paths = []
            for name, offset in (("original", 0), ("method_a", 2), ("method_b", 4)):
                path = root / f"{name}.png"
                Image.fromarray(np.clip(image.astype(np.int16) + offset, 0, 255).astype(np.uint8)).save(path)
                paths.append(path)
            store = ProjectStore.create(root / "project", "v012")
            store.set_original(paths[0])
            store.add_algorithm(paths[1], "方法 A")
            store.add_algorithm(paths[2], "方法 B")
            recommendations = recommend_rois(image)
            bone = next(item.roi for item in recommendations if item.roi.type == "weak_bone" and item.roi.paired_surrounding_roi_id)
            neighbor = next(item.roi for item in recommendations if item.roi.id == bone.paired_surrounding_roi_id)
            store.upsert_roi(neighbor)
            store.upsert_roi(bone)
            background = next((item.roi for item in recommendations if item.roi.type == "background"), None)
            if background:
                store.upsert_roi(background)
            reopened = ProjectStore.open(store.root)
            reopened_bone = next(roi for roi in reopened.project.rois if roi.id == bone.id)
            self.assertEqual(reopened_bone.paired_surrounding_roi_id, neighbor.id)
            result = evaluate_project(reopened)
            ids_by_scheme = {}
            for row in result["roi_metrics"]:
                ids_by_scheme.setdefault(row["scheme_id"], set()).add(row["roi_id"])
            self.assertEqual(len(ids_by_scheme), 3)
            first = next(iter(ids_by_scheme.values()))
            self.assertTrue(all(ids == first for ids in ids_by_scheme.values()))


if __name__ == "__main__":
    unittest.main()
