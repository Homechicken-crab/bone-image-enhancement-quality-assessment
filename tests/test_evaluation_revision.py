import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from bone_iqa.evaluator import evaluate_project
from bone_iqa.project_store import ProjectStore
from bone_iqa.roi_templates import ROI_TEMPLATE_VERSION, build_roi_template, is_template_v1


class FixedTemplateTest(unittest.TestCase):
    def test_template_has_all_twelve_fixed_rois_and_expected_pairs(self):
        rois = build_roi_template()
        self.assertEqual(len(rois), 12)
        self.assertEqual(rois[0].geometry(), (43, 190, 9, 60))
        self.assertEqual(rois[0].type, "weak_bone")
        self.assertEqual(rois[0].paired_surrounding_roi_id, rois[1].id)
        self.assertEqual(rois[1].paired_surrounding_roi_id, rois[0].id)
        self.assertEqual([roi.name for roi in rois[-2:]], ["背景1", "背景2"])
        self.assertEqual([roi.paired_surrounding_roi_id for roi in rois[8:10]], [None, None])

    def test_template_is_recognized_only_at_500_by_800(self):
        rois = build_roi_template()
        self.assertTrue(is_template_v1(rois, 500, 800))
        self.assertFalse(is_template_v1(rois, 800, 500))


class EvaluationFreshnessTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        source = root / "original.png"
        array = np.zeros((800, 500), dtype=np.uint8)
        array[190:250, 43:52] = np.tile(np.arange(60, dtype=np.uint8)[:, None], (1, 9)) + 30
        array[180:320, 10:30] = np.tile(np.arange(20, dtype=np.uint8), (140, 1)) + 10
        Image.fromarray(array).save(source)
        self.store = ProjectStore.create(root / "project", "freshness")
        self.store.set_original(source)
        self.store.replace_rois(build_roi_template())

    def tearDown(self):
        self.temporary.cleanup()

    def test_evaluation_records_both_cnr_fields_and_template_metadata(self):
        result = evaluate_project(self.store)
        self.assertIn("weak_bone_mean_cnr_local", result["metrics"][0])
        self.assertIn("weak_bone_mean_cnr_background", result["metrics"][0])
        self.assertEqual(result["roi_template_version"], ROI_TEMPLATE_VERSION)
        self.assertEqual(len(result["roi_snapshot"]), 12)
        self.assertTrue(result["roi_config_hash"])
        self.assertTrue(result["input_hashes"]["original"])

    def test_external_project_image_change_expires_cached_result(self):
        evaluate_project(self.store)
        internal = self.store.resolve(self.store.project.original.relative_path)
        with internal.open("ab") as stream:
            stream.write(b"changed")
        self.assertEqual(self.store.evaluation_status(), "expired")
        self.assertIsNone(self.store.load_latest_evaluation())

    def test_roi_change_invalidates_cached_result(self):
        evaluate_project(self.store)
        roi = self.store.project.rois[0]
        roi.x += 1
        self.store.upsert_roi(roi)
        self.assertEqual(self.store.evaluation_status(), "none")
        self.assertIsNone(self.store.load_latest_evaluation())

    def test_old_evaluation_without_new_metadata_is_loadable_as_legacy(self):
        result = evaluate_project(self.store)
        result.pop("roi_config_hash")
        result.pop("roi_template_version")
        result.pop("roi_snapshot")
        result.pop("metric_algorithm_version")
        self.store.save_evaluation(result)
        self.assertEqual(self.store.evaluation_status(), "legacy")
        self.assertIsNotNone(self.store.load_latest_evaluation())


if __name__ == "__main__":
    unittest.main()
