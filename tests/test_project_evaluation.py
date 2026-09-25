import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from bone_iqa.evaluator import evaluate_project, export_csv_bundle
from bone_iqa.models import ROI
from bone_iqa.project_store import ProjectStore


class ProjectEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        base = np.full((32, 32), 20, dtype=np.uint8)
        base[4:12, 4:12] = np.arange(64, dtype=np.uint8).reshape(8, 8) + 40
        base[20:28, 20:28] = 245
        base[0:4, 0:4] = np.array([18, 19, 20, 21] * 4, dtype=np.uint8).reshape(4, 4)
        enhanced = base.copy()
        enhanced[4:12, 4:12] = np.clip((enhanced[4:12, 4:12].astype(int) - 20) * 2 + 20, 0, 255)
        enhanced[20:28, 20:28] = 255
        self.original_path = self.root / "source.png"
        self.enhanced_path = self.root / "enhanced.png"
        Image.fromarray(base).save(self.original_path)
        Image.fromarray(enhanced).save(self.enhanced_path)

    def tearDown(self):
        self.temp.cleanup()

    def test_end_to_end_project(self):
        store = ProjectStore.create(self.root / "project", "test")
        store.set_original(self.original_path)
        store.add_algorithm(self.enhanced_path, "增强方案")
        surrounding = ROI.create("弱骨周围", "surrounding", 12, 4, 4, 8)
        weak = ROI.create("弱骨", "weak_bone", 4, 4, 8, 8, surrounding.id)
        strong = ROI.create("强骨", "strong_bone", 20, 20, 8, 8, surrounding.id)
        background = ROI.create("背景", "background", 0, 0, 4, 4)
        for roi in (surrounding, weak, strong, background):
            store.upsert_roi(roi)

        result = evaluate_project(store)
        self.assertEqual(len(result["metrics"]), 2)
        original, enhanced = result["metrics"]
        self.assertEqual(original["ssim"], 1.0)
        self.assertIsNotNone(enhanced["weak_bone_mean_cnr_background"])
        self.assertIsNotNone(enhanced["weak_bone_mean_cnr_local"])
        self.assertEqual(enhanced["primary_cnr"], enhanced["weak_bone_mean_cnr_local"])
        self.assertIsNotNone(enhanced["weak_bone_mean_average_gradient"])
        self.assertGreater(enhanced["strong_bone_saturation_mean"], original["strong_bone_saturation_mean"])

        algorithm_id = store.project.algorithms[0].id
        store.rename_algorithm(algorithm_id, "重命名方案")
        synchronized = store.load_latest_evaluation()
        self.assertEqual(synchronized["metrics"][1]["scheme_name"], "重命名方案")

        reopened = ProjectStore.open(store.root)
        self.assertEqual(len(reopened.project.rois), 4)
        export_dir = self.root / "export"
        paths = export_csv_bundle(reopened, export_dir)
        self.assertEqual(len(paths), 4)
        with (export_dir / "metrics.csv").open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        with (export_dir / "evaluation.json").open("r", encoding="utf-8") as stream:
            saved = json.load(stream)
        self.assertEqual(saved["metric_spec_version"], "1.1.1")

        reopened.project.rois[0].x += 1
        reopened.upsert_roi(reopened.project.rois[0])
        self.assertIsNone(reopened.load_latest_evaluation())


if __name__ == "__main__":
    unittest.main()
