import os
import unittest

from PIL import Image

from ocr_quality.adapters.paddle_ocr import PaddleOcrBundle, PaddleOcrUnavailable, PaddleTextDetector
from ocr_quality.config import QualityConfig


@unittest.skipUnless(os.environ.get("OCR_QUALITY_RUN_PADDLE_INTEGRATION") == "1", "set OCR_QUALITY_RUN_PADDLE_INTEGRATION=1 to run")
class OptionalPaddleOcrIntegrationTests(unittest.TestCase):
    def test_paddleocr_bundle_loads_warms_and_detects_text(self):
        try:
            bundle = PaddleOcrBundle.create(QualityConfig())
        except PaddleOcrUnavailable as exc:
            self.skipTest(str(exc))

        bundle.warmup()
        detector = PaddleTextDetector(bundle.paddle)
        image = Image.new("RGB", (360, 120), "white")

        result = detector.detect(image)

        self.assertIsNotNone(result.boxes)
        self.assertIn("paddleocr", bundle.model_versions())


if __name__ == "__main__":
    unittest.main()

