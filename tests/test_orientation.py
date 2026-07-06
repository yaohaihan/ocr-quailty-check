import unittest

from PIL import Image

from ocr_quality.orientation import FixedOrientationDetector, OrientationResult, apply_orientation_correction


class OrientationTests(unittest.TestCase):
    def test_180_degree_orientation_is_auto_corrected(self):
        image = Image.new("RGB", (100, 50), "white")
        detector = FixedOrientationDetector(OrientationResult(angle=180, confidence=0.95, model_version="fake"))

        corrected, fix = apply_orientation_correction(image, detector.detect(image))

        self.assertEqual(corrected.size, image.size)
        self.assertEqual(fix.fixType, "ORIENTATION_CORRECTED")
        self.assertEqual(fix.before["angle"], 180)
        self.assertTrue(fix.recomputed)

    def test_low_confidence_orientation_is_not_corrected(self):
        image = Image.new("RGB", (100, 50), "white")
        detector = FixedOrientationDetector(OrientationResult(angle=90, confidence=0.10, model_version="fake"))

        corrected, fix = apply_orientation_correction(image, detector.detect(image), min_confidence=0.8)

        self.assertEqual(corrected.size, image.size)
        self.assertIsNone(fix)


if __name__ == "__main__":
    unittest.main()

