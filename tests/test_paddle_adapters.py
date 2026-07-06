import unittest

import numpy as np
from PIL import Image

from ocr_quality.adapters.paddle_ocr import (
    PaddleOcrBundle,
    PaddleOcrProbe,
    PaddleOcrUnavailable,
    PaddleTextDetector,
)
from ocr_quality.config import QualityConfig
from ocr_quality.text_detection import TextDetectionResult


class FakePaddleOCR:
    def __init__(self):
        self.ocr_calls = 0

    def ocr(self, image, cls=False, det=True, rec=True):
        self.ocr_calls += 1
        return [[[[10, 10], [110, 10], [110, 30], [10, 30]], ("hello", 0.91)]]


class FakeDetectionOnlyPaddle:
    def ocr(self, image, cls=False, det=True, rec=False):
        return [[[[10, 10], [110, 10], [110, 30], [10, 30]], None]]


class FakeRecognitionPaddle:
    def ocr(self, image, cls=True, det=False, rec=True):
        return [[("hello", 0.92)]]


class FakePaddleModuleV3:
    def __init__(self):
        self.kwargs = None

    def PaddleOCR(self, **kwargs):
        self.kwargs = kwargs
        return FakePaddlePredictV3()


class FakePaddlePredictV3:
    def __init__(self):
        self.predict_calls = 0
        self.last_input = None

    def predict(self, image):
        self.predict_calls += 1
        self.last_input = image
        return [
            {
                "dt_polys": [
                    [[10, 10], [110, 10], [110, 30], [10, 30]],
                    [[20, 50], [160, 50], [160, 75], [20, 75]],
                ],
                "rec_texts": ["hello", "world"],
                "rec_scores": [0.93, 0.81],
            }
        ]


class FakePaddlePredictV3RecBoxes:
    def predict(self, image):
        return [
            {
                "rec_boxes": [
                    [10, 10, 110, 30],
                    [20, 50, 160, 75],
                ],
                "rec_texts": ["hello", "world"],
                "rec_scores": [0.93, 0.81],
            }
        ]


class PaddleAdapterTests(unittest.TestCase):
    def test_bundle_warmup_calls_fake_model_once(self):
        fake = FakePaddleOCR()
        bundle = PaddleOcrBundle(config=QualityConfig(), paddle=fake)

        bundle.warmup()

        self.assertEqual(fake.ocr_calls, 1)
        self.assertIn("paddleocr", bundle.model_versions())

    def test_bundle_warmup_passes_numpy_array_to_paddleocr_3_predict(self):
        fake = FakePaddlePredictV3()
        bundle = PaddleOcrBundle(config=QualityConfig(), paddle=fake)

        bundle.warmup()

        self.assertIsInstance(fake.last_input, np.ndarray)

    def test_missing_paddle_dependency_raises_adapter_error(self):
        with self.assertRaises(PaddleOcrUnavailable):
            PaddleOcrBundle.create(QualityConfig(), import_name="module_that_does_not_exist")

    def test_bundle_uses_paddleocr_3_constructor_arguments(self):
        fake_module = FakePaddleModuleV3()

        bundle = PaddleOcrBundle.from_module(QualityConfig(), fake_module)

        self.assertIsInstance(bundle.paddle, FakePaddlePredictV3)
        self.assertEqual(fake_module.kwargs["lang"], "ch")
        self.assertFalse(fake_module.kwargs["use_doc_orientation_classify"])
        self.assertFalse(fake_module.kwargs["use_doc_unwarping"])
        self.assertFalse(fake_module.kwargs["use_textline_orientation"])
        self.assertNotIn("use_angle_cls", fake_module.kwargs)
        self.assertNotIn("use_gpu", fake_module.kwargs)
        self.assertNotIn("det_model_dir", fake_module.kwargs)

    def test_text_detector_converts_paddle_boxes_to_summary_metrics(self):
        detector = PaddleTextDetector(FakeDetectionOnlyPaddle())
        image = Image.new("RGB", (200, 100), "white")

        result = detector.detect(image)

        self.assertEqual(len(result.boxes), 1)
        self.assertEqual(result.boxes[0], (10, 10, 110, 30))
        self.assertGreater(result.coverage_ratio, 0)
        self.assertEqual(result.median_height, 20)

    def test_text_detector_supports_paddleocr_3_predict_results(self):
        fake = FakePaddlePredictV3()
        detector = PaddleTextDetector(fake)
        image = Image.new("RGB", (200, 100), "white")

        result = detector.detect(image)

        self.assertIsInstance(fake.last_input, np.ndarray)
        self.assertEqual(len(result.boxes), 2)
        self.assertEqual(result.boxes[0], (10, 10, 110, 30))
        self.assertEqual(result.median_height, 22.5)

    def test_text_detector_supports_paddleocr_3_rec_boxes(self):
        detector = PaddleTextDetector(FakePaddlePredictV3RecBoxes())
        image = Image.new("RGB", (200, 100), "white")

        result = detector.detect(image)

        self.assertEqual(len(result.boxes), 2)
        self.assertEqual(result.boxes[0], (10, 10, 110, 30))
        self.assertEqual(result.median_height, 22.5)

    def test_probe_samples_limited_boxes_and_reports_confidence(self):
        probe = PaddleOcrProbe(FakeRecognitionPaddle(), sample_size=2)
        text = TextDetectionResult(
            boxes=[(0, 0, 100, 20), (0, 40, 100, 60), (0, 80, 100, 100)],
            coverage_ratio=0.1,
            median_height=20,
            low_sharpness_ratio=0.0,
            border_touch_ratio=0.0,
        )

        result = probe.probe(Image.new("RGB", (200, 120), "white"), text)

        self.assertEqual(result.sample_count, 2)
        self.assertEqual(result.source, "paddleocr")
        self.assertGreater(result.average_confidence, 0.9)
        self.assertEqual(result.empty_ratio, 0.0)

    def test_probe_supports_paddleocr_3_predict_results(self):
        probe = PaddleOcrProbe(FakePaddlePredictV3(), sample_size=2)
        text = TextDetectionResult(
            boxes=[(0, 0, 100, 20), (0, 40, 100, 60)],
            coverage_ratio=0.1,
            median_height=20,
            low_sharpness_ratio=0.0,
            border_touch_ratio=0.0,
        )

        result = probe.probe(Image.new("RGB", (200, 120), "white"), text)

        self.assertEqual(result.sample_count, 2)
        self.assertEqual(result.source, "paddleocr")
        self.assertGreater(result.average_confidence, 0.8)
        self.assertEqual(result.empty_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
