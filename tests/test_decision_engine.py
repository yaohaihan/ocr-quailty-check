import unittest

from ocr_quality.config import QualityConfig
from ocr_quality.decision import DecisionEngine
from ocr_quality.image_metrics import ImageMetrics
from ocr_quality.models import Decision, ReasonCode
from ocr_quality.occlusion import OcclusionResult
from ocr_quality.ocr_probe import OcrProbeResult
from ocr_quality.text_detection import TextDetectionResult


def metrics(**overrides):
    data = dict(
        width=900,
        height=700,
        pixels=630000,
        mean=220.0,
        std=45.0,
        edge_density=0.02,
        sharpness=120.0,
        low_quality_block_ratio=0.1,
        noise_estimate=3.0,
        illumination_unevenness=5.0,
        laplacian_variance=100.0,
        tenengrad=120.0,
        low_contrast=False,
        content_block_count=8,
        total_block_count=16,
        skew_angle_degrees=None,
        compression_risk=0.1,
    )
    data.update(overrides)
    return ImageMetrics(**data)


def occlusion(**overrides):
    data = dict(
        candidate_count=1,
        hand_like_candidate_count=1,
        occlusion_area_ratio=0.08,
        text_overlap_ratio=0.2,
        affected_text_box_ratio=0.5,
        max_box_overlap_ratio=0.5,
        has_hand_occlusion=True,
        has_text_occlusion=False,
        source="skin-color-rule",
    )
    data.update(overrides)
    return OcclusionResult(**data)


class DecisionEngineTests(unittest.TestCase):
    def test_single_risk_does_not_reject(self):
        text = TextDetectionResult([(10, 10, 100, 25)], 0.01, 15, 0.0, 0.0)

        result = DecisionEngine(QualityConfig()).page_decision(0, metrics(low_quality_block_ratio=0.8), text, None, {})

        self.assertTrue(result.accepted)
        self.assertEqual(result.decision, Decision.ACCEPT_WITH_WARNINGS)
        self.assertIn(ReasonCode.LOCAL_BLUR_RISK, result.reasonCodes)

    def test_real_ocr_probe_low_confidence_can_reject(self):
        text = TextDetectionResult([(10, 10, 100, 25)], 0.01, 15, 0.0, 0.0)
        probe = OcrProbeResult(0.2, 0.8, 0.2, 0.2, 1.0, 0.1, sample_count=1, source="paddleocr")

        result = DecisionEngine(QualityConfig()).page_decision(0, metrics(), text, probe, {})

        self.assertFalse(result.accepted)
        self.assertIn(ReasonCode.OCR_PROBE_LOW_CONFIDENCE, result.reasonCodes)

    def test_heuristic_probe_low_confidence_does_not_hard_reject(self):
        text = TextDetectionResult([(10, 10, 100, 25)], 0.01, 15, 0.0, 0.0)
        probe = OcrProbeResult(0.2, 0.8, 0.2, 0.2, 1.0, 0.1, source="heuristic")

        result = DecisionEngine(QualityConfig()).page_decision(0, metrics(), text, probe, {})

        self.assertTrue(result.accepted)
        self.assertNotIn(ReasonCode.OCR_PROBE_LOW_CONFIDENCE, result.reasonCodes)

    def test_empty_real_ocr_probe_sample_does_not_add_low_confidence_reason(self):
        text = TextDetectionResult([], 0.0, 0.0, 0.0, 0.0)
        probe = OcrProbeResult(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, sample_count=0, source="paddleocr")

        result = DecisionEngine(QualityConfig()).page_decision(0, metrics(), text, probe, {})

        self.assertFalse(result.accepted)
        self.assertIn(ReasonCode.NO_EFFECTIVE_TEXT_DETECTED, result.reasonCodes)
        self.assertNotIn(ReasonCode.OCR_PROBE_LOW_CONFIDENCE, result.reasonCodes)

    def test_occlusion_risk_warns_without_rejecting(self):
        text = TextDetectionResult([(10, 10, 100, 25)], 0.01, 15, 0.0, 0.0)

        result = DecisionEngine(QualityConfig()).page_decision(0, metrics(), text, None, {}, occlusion())

        self.assertTrue(result.accepted)
        self.assertEqual(result.decision, Decision.ACCEPT_WITH_WARNINGS)
        self.assertIn(ReasonCode.HAND_OCCLUSION_RISK, result.reasonCodes)
        self.assertIsNotNone(result.occlusionSummary)

    def test_occlusion_plus_real_low_confidence_probe_rejects(self):
        text = TextDetectionResult([(10, 10, 100, 25)], 0.01, 15, 0.0, 0.0)
        probe = OcrProbeResult(0.2, 0.8, 0.2, 0.2, 1.0, 0.1, sample_count=1, source="paddleocr")

        result = DecisionEngine(QualityConfig()).page_decision(0, metrics(), text, probe, {}, occlusion())

        self.assertFalse(result.accepted)
        self.assertIn(ReasonCode.OCR_PROBE_LOW_CONFIDENCE, result.reasonCodes)
        self.assertIn(ReasonCode.HAND_OCCLUSION_RISK, result.reasonCodes)


if __name__ == "__main__":
    unittest.main()
