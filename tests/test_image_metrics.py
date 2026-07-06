import unittest

from PIL import Image, ImageEnhance, ImageFilter

from ocr_quality.config import QualityConfig
from ocr_quality.image_metrics import FastImageQualityAnalyzer


def text_like_image(size=(900, 700)):
    image = Image.new("RGB", size, "white")
    pixels = image.load()
    for y in range(100, 600, 42):
        for x in range(80, 820, 130):
            for yy in range(y, y + 10):
                for xx in range(x, x + 80):
                    pixels[xx, yy] = (20, 20, 20)
    return image


class ImageMetricsTests(unittest.TestCase):
    def test_blur_lowers_laplacian_and_tenengrad(self):
        analyzer = FastImageQualityAnalyzer(QualityConfig())
        sharp = analyzer.analyze(text_like_image())
        blurred = analyzer.analyze(text_like_image().filter(ImageFilter.GaussianBlur(radius=3)))

        self.assertGreater(sharp.laplacian_variance, blurred.laplacian_variance)
        self.assertGreater(sharp.tenengrad, blurred.tenengrad)

    def test_low_contrast_is_detected_by_std_and_flag(self):
        analyzer = FastImageQualityAnalyzer(QualityConfig())
        low_contrast = ImageEnhance.Contrast(text_like_image()).enhance(0.15)

        metrics = analyzer.analyze(low_contrast)

        self.assertLess(metrics.std, 18.0)
        self.assertTrue(metrics.low_contrast)

    def test_blank_blocks_are_excluded_from_local_blur_ratio(self):
        analyzer = FastImageQualityAnalyzer(QualityConfig())
        image = Image.new("RGB", (900, 700), "white")
        pixels = image.load()
        for y in range(80, 180, 24):
            for x in range(80, 260, 90):
                for yy in range(y, y + 10):
                    for xx in range(x, x + 70):
                        pixels[xx, yy] = (20, 20, 20)

        metrics = analyzer.analyze(image)

        self.assertGreater(metrics.content_block_count, 0)
        self.assertLess(metrics.content_block_count, metrics.total_block_count)


if __name__ == "__main__":
    unittest.main()
