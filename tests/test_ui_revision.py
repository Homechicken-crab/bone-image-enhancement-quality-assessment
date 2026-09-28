import inspect
import unittest

from bone_iqa.app import (
    MAX_COMPARISON_SCHEMES,
    SUMMARY_THUMBNAIL_SIZE,
    BoneIQAApp,
    MetricSummaryTable,
    comparison_grid_columns,
)


class UiRevisionTest(unittest.TestCase):
    def test_comparison_grid_keeps_readable_layout(self):
        self.assertEqual([comparison_grid_columns(count) for count in (2, 3, 4)], [2, 3, 4])
        self.assertEqual([comparison_grid_columns(count) for count in (5, 6, 7, 10)], [3, 3, 3, 3])
        self.assertEqual(MAX_COMPARISON_SCHEMES, 10)

    def test_summary_table_has_scrollable_metric_columns_and_status(self):
        headings = [heading for heading, _width in MetricSummaryTable.COLUMNS]
        for heading in (
            "Local CNR", "Local Δ%", "Background CNR", "Background Δ%", "Weak AG", "AG Δ%",
            "Noise", "Noise Δ%", "SSIM", "Strong Sat.%", "Sat. Δpp", "状态",
        ):
            self.assertIn(heading, headings)
        self.assertEqual(headings[-1], "状态")
        self.assertGreater(sum(width for _heading, width in MetricSummaryTable.COLUMNS), 1200)

    def test_thumbnail_cell_uses_portrait_fit_area(self):
        self.assertEqual(SUMMARY_THUMBNAIL_SIZE, (60, 96))

    def test_summary_rows_and_canvas_windows_keep_fixed_total_width(self):
        add_row_source = inspect.getsource(MetricSummaryTable.add_row)
        init_source = inspect.getsource(MetricSummaryTable.__init__)
        self.assertIn("width=self.total_width", add_row_source)
        self.assertIn("self.body_inner.columnconfigure(0, minsize=self.total_width", init_source)
        self.assertIn("self.header_canvas.itemconfigure(self.header_window, width=self.total_width)", init_source)
        self.assertIn("self.body_canvas.itemconfigure(self.body_window, width=self.total_width)", init_source)

    def test_diagnostic_is_specific_to_selected_scheme(self):
        app = object.__new__(BoneIQAApp)
        text = app._diagnostic_for_row({
            "role": "algorithm",
            "scheme_name": "方案 A",
            "weak_bone_mean_cnr_local": 3.2,
            "weak_bone_mean_cnr_background": 2.4,
            "weak_bone_mean_average_gradient": 8.1,
            "background_noise_pooled": 6.0,
            "ssim": 0.96,
            "strong_bone_saturation_mean": 0.12,
            "local_cnr_change_percent": -15.0,
            "background_cnr_change_percent": -12.0,
            "weak_ag_change_percent": 20.0,
            "background_noise_change_percent": 18.0,
            "saturation_change_pp": 7.0,
        })
        self.assertIn("方案 A", text)
        self.assertIn("Local CNR 下降", text)
        self.assertIn("AG 明显上升且 Noise 也上升", text)
        self.assertIn("Strong Sat.% 相比原图上升", text)


if __name__ == "__main__":
    unittest.main()
