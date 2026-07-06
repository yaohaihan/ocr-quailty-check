import tempfile
import unittest
from pathlib import Path

from PIL import Image
try:
    from reportlab.pdfgen import canvas
except ImportError:
    canvas = None

from ocr_quality.config import QualityConfig
from ocr_quality.file_decoder import FileDecoder
from ocr_quality.models import ReasonCode


class FileDecoderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_decodes_image_to_single_page_with_metadata(self):
        path = self.root / "page.png"
        Image.new("RGB", (640, 480), "white").save(path)

        result = FileDecoder(QualityConfig()).decode(path)

        self.assertEqual(result.reason_codes, [])
        self.assertEqual(len(result.pages), 1)
        self.assertEqual(result.pages[0].page_index, 0)
        self.assertEqual(result.pages[0].original_size, (640, 480))
        self.assertEqual(result.file_info["extension"], ".png")

    def test_decodes_pdf_to_page_images(self):
        if canvas is None:
            self.skipTest("reportlab is not installed")
        path = self.root / "sample.pdf"
        pdf = canvas.Canvas(str(path))
        pdf.drawString(72, 720, "OCR quality sample")
        pdf.showPage()
        pdf.save()

        result = FileDecoder(QualityConfig()).decode(path)

        self.assertEqual(result.reason_codes, [])
        self.assertEqual(len(result.pages), 1)
        self.assertGreater(result.pages[0].image.size[0], 0)

    def test_rejects_corrupted_image_as_decode_failed(self):
        path = self.root / "bad.png"
        path.write_bytes(b"bad")

        result = FileDecoder(QualityConfig()).decode(path)

        self.assertEqual(result.pages, [])
        self.assertEqual(result.reason_codes, [ReasonCode.IMAGE_DECODE_FAILED])


if __name__ == "__main__":
    unittest.main()
