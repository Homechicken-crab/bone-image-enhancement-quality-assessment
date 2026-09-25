import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from bone_iqa.app import BoneIQAApp
from bone_iqa.models import ROI
from bone_iqa.project_store import ProjectStore


class ClearSavedRoisTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        source = root / "original.png"
        algorithm = root / "algorithm.png"
        Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16)).save(source)
        Image.fromarray(np.flipud(np.arange(256, dtype=np.uint8).reshape(16, 16))).save(algorithm)
        self.store = ProjectStore.create(root / "project", "清空测试")
        self.store.set_original(source)
        self.store.add_algorithm(algorithm, "方案 A")
        neighbor = ROI.create("邻域1", "surrounding", 0, 0, 4, 4)
        for roi in (
            neighbor,
            ROI.create("弱骨骼1", "weak_bone", 4, 0, 4, 4, neighbor.id),
            ROI.create("强骨骼1", "strong_bone", 8, 0, 4, 4, neighbor.id),
            ROI.create("背景1", "background", 12, 0, 4, 4),
        ):
            self.store.upsert_roi(roi)
        self.store.project.latest_evaluation_id = "evaluation-before-clear"
        self.store.save()

    def tearDown(self):
        self.temporary.cleanup()

    def test_clear_rois_removes_all_types_at_once(self):
        self.assertEqual({roi.type for roi in self.store.project.rois}, {"weak_bone", "strong_bone", "surrounding", "background"})
        self.store.clear_rois()
        self.assertEqual(len(self.store.project.rois), 0)

    def test_clear_rois_leaves_empty_project_list(self):
        self.store.clear_rois()
        self.assertEqual(self.store.project.rois, [])

    def test_clear_rois_invalidates_latest_evaluation(self):
        self.store.clear_rois()
        self.assertIsNone(self.store.project.latest_evaluation_id)

    def test_clear_rois_persists_after_reopen(self):
        self.store.clear_rois()
        reopened = ProjectStore.open(self.store.root)
        self.assertEqual(reopened.project.rois, [])
        self.assertIsNone(reopened.project.latest_evaluation_id)

    def test_clear_rois_preserves_images_algorithms_and_settings(self):
        original_id = self.store.project.original.id
        original_path = self.store.project.original.relative_path
        algorithm_ids = [item.id for item in self.store.project.algorithms]
        config = self.store.project.evaluation_config
        primary_cnr = config.primary_cnr
        gray_range = (config.gray_min, config.gray_max)
        self.store.clear_rois()
        self.assertIsNotNone(self.store.project.original)
        self.assertEqual(self.store.project.original.id, original_id)
        self.assertEqual(self.store.project.original.relative_path, original_path)
        self.assertEqual([item.id for item in self.store.project.algorithms], algorithm_ids)
        self.assertEqual(self.store.project.evaluation_config.primary_cnr, primary_cnr)
        self.assertEqual((self.store.project.evaluation_config.gray_min, self.store.project.evaluation_config.gray_max), gray_range)


class FullChineseCanvasLabelsTest(unittest.TestCase):
    @staticmethod
    def label_for(roi_type: str, width: int = 8) -> str:
        roi = ROI.create("任意名称", roi_type, 0, 0, width, 8)
        app = object.__new__(BoneIQAApp)
        return app._short_roi_labels([roi])[roi.id]

    def test_background_label_is_complete(self):
        self.assertEqual(self.label_for("background"), "背景1")

    def test_weak_bone_label_is_complete(self):
        self.assertEqual(self.label_for("weak_bone"), "弱骨骼1")

    def test_strong_bone_label_is_complete(self):
        self.assertEqual(self.label_for("strong_bone"), "强骨骼1")

    def test_surrounding_label_is_complete(self):
        self.assertEqual(self.label_for("surrounding"), "邻域1")

    def test_tiny_rois_are_not_abbreviated(self):
        self.assertEqual(self.label_for("weak_bone", width=1), "弱骨骼1")
        self.assertEqual(self.label_for("strong_bone", width=1), "强骨骼1")
        self.assertEqual(self.label_for("surrounding", width=1), "邻域1")
        self.assertEqual(self.label_for("background", width=1), "背景1")


if __name__ == "__main__":
    unittest.main()
