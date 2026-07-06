import unittest

from PIL import Image, ImageDraw

from ocr_quality.config import QualityConfig
from ocr_quality.models import ReasonCode
from ocr_quality.occlusion import OcclusionAnalyzer
from ocr_quality.text_detection import TextDetectionResult


def text_result():
    return TextDetectionResult(
        boxes=[(100, 90, 300, 120), (100, 150, 320, 180)],
        coverage_ratio=0.08,
        median_height=30,
        low_sharpness_ratio=0.0,
        border_touch_ratio=0.0,
    )


def base_page():
    image = Image.new("RGB", (500, 350), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 90, 300, 120), fill="black")
    draw.rectangle((100, 150, 320, 180), fill="black")
    return image


class OcclusionConfigTests(unittest.TestCase):
    def test_occlusion_config_defaults_are_available(self):
        config = QualityConfig()

        self.assertTrue(config.enable_occlusion_detection)
        self.assertGreater(config.occlusion_min_area_ratio, 0)
        self.assertGreater(config.occlusion_min_text_overlap_ratio, 0)
        self.assertGreater(config.occlusion_min_affected_box_ratio, 0)
        self.assertGreater(config.occlusion_min_box_overlap_ratio, 0)
        self.assertGreater(config.occlusion_text_box_padding_ratio, 0)

    def test_occlusion_reason_codes_are_stable(self):
        self.assertEqual(ReasonCode.HAND_OCCLUSION_RISK.value, "HAND_OCCLUSION_RISK")
        self.assertEqual(ReasonCode.TEXT_OCCLUSION_RISK.value, "TEXT_OCCLUSION_RISK")


class OcclusionAnalyzerTests(unittest.TestCase):
    def test_normal_text_page_has_no_occlusion(self):
        result = OcclusionAnalyzer(QualityConfig()).analyze(base_page(), text_result())

        self.assertFalse(result.has_hand_occlusion)
        self.assertFalse(result.has_text_occlusion)
        self.assertEqual(result.candidate_count, 0)

    def test_skin_colored_region_over_text_reports_hand_occlusion(self):
        image = base_page()
        draw = ImageDraw.Draw(image)
        draw.ellipse((130, 75, 260, 140), fill=(210, 145, 110))

        result = OcclusionAnalyzer(QualityConfig()).analyze(image, text_result())

        self.assertTrue(result.has_hand_occlusion)
        self.assertGreater(result.text_overlap_ratio, 0)
        self.assertGreater(result.max_box_overlap_ratio, 0)

    def test_skin_colored_region_away_from_text_is_ignored(self):
        image = base_page()
        draw = ImageDraw.Draw(image)
        draw.ellipse((360, 240, 470, 330), fill=(210, 145, 110))

        result = OcclusionAnalyzer(QualityConfig()).analyze(image, text_result())

        self.assertFalse(result.has_hand_occlusion)
        self.assertFalse(result.has_text_occlusion)

    def test_dark_non_text_region_over_text_reports_generic_occlusion(self):
        image = base_page()
        draw = ImageDraw.Draw(image)
        draw.rectangle((130, 80, 280, 135), fill=(80, 80, 80))

        result = OcclusionAnalyzer(QualityConfig()).analyze(image, text_result())

        self.assertFalse(result.has_hand_occlusion)
        self.assertTrue(result.has_text_occlusion)


if __name__ == "__main__":
    unittest.main()
