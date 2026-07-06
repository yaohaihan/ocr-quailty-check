import unittest

from PIL import Image

from ocr_quality.config import QualityConfig
from ocr_quality.text_detection import TextDetectionResult
from ocr_quality.text_regions import TextRegionAnalyzer


class TextRegionAnalyzerTests(unittest.TestCase):
    def test_analyzes_small_text_and_border_crop_risk(self):
        image = Image.new("RGB", (200, 100), "white")
        text = TextDetectionResult(
            boxes=[(0, 10, 40, 18), (80, 40, 160, 52)],
            coverage_ratio=0.0,
            median_height=0.0,
            low_sharpness_ratio=0.0,
            border_touch_ratio=0.0,
        )

        summary = TextRegionAnalyzer(QualityConfig()).analyze(image, text)

        self.assertEqual(summary.box_count, 2)
        self.assertEqual(summary.small_text_ratio, 0.5)
        self.assertEqual(summary.border_touch_ratio, 0.5)

    def test_text_region_summary_updates_detection_result(self):
        image = Image.new("RGB", (200, 100), "white")
        text = TextDetectionResult(
            boxes=[(20, 20, 120, 50)],
            coverage_ratio=0.0,
            median_height=0.0,
            low_sharpness_ratio=0.0,
            border_touch_ratio=0.0,
        )

        summary = TextRegionAnalyzer(QualityConfig()).analyze(image, text)

        self.assertGreater(summary.coverage_ratio, 0)
        self.assertEqual(summary.median_height, 30)


if __name__ == "__main__":
    unittest.main()

