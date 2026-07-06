import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageFilter
from pypdf import PdfReader, PdfWriter
try:
    from reportlab.pdfgen import canvas
except ImportError:
    canvas = None

from ocr_quality import QualityPipeline, load_default_config
from ocr_quality.file_inspector import FileInspector
from ocr_quality.models import Decision, ReasonCode
from ocr_quality.text_detection import TextDetectionResult, TextDetector


class FixedTextDetector(TextDetector):
    def __init__(self, result):
        self.result = result

    def detect(self, image):
        return self.result


class FakeOcclusionAnalyzer:
    def analyze(self, image, text):
        from ocr_quality.occlusion import OcclusionResult

        return OcclusionResult(
            candidate_count=1,
            hand_like_candidate_count=1,
            occlusion_area_ratio=0.08,
            text_overlap_ratio=0.2,
            affected_text_box_ratio=0.5,
            max_box_overlap_ratio=0.5,
            has_hand_occlusion=True,
            has_text_occlusion=False,
            source="test",
        )


class QualityPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def image_path(self, name, image):
        path = self.root / name
        image.save(path)
        return path

    def pdf_path(self, name, pages=1):
        if canvas is None:
            self.skipTest("reportlab is not installed")
        path = self.root / name
        pdf = canvas.Canvas(str(path))
        for page in range(pages):
            for row in range(10):
                pdf.drawString(72, 720 - row * 28, f"page {page + 1} OCR quality sample line {row + 1}")
            pdf.showPage()
        pdf.save()
        return path

    def text_like_image(self, size=(900, 700)):
        image = Image.new("RGB", size, "white")
        pixels = image.load()
        for y in range(100, size[1] - 100, 42):
            for x in range(80, size[0] - 80, 130):
                for yy in range(y, min(y + 10, size[1])):
                    for xx in range(x, min(x + 80, size[0])):
                        pixels[xx, yy] = (20, 20, 20)
        return image

    def test_accepts_readable_text_like_image_with_stable_result_shape(self):
        path = self.image_path("page.png", self.text_like_image())

        result = QualityPipeline().evaluate(path)

        self.assertTrue(result["accepted"])
        self.assertEqual(result["decision"], Decision.ACCEPT.value)
        self.assertEqual(result["reasonCodes"], [])
        self.assertIn("ocrReadinessScore", result)
        self.assertIn("metrics", result)
        self.assertIn("qualityMetrics", result)
        self.assertIn("detectedRisks", result)
        self.assertIn("appliedCorrections", result)
        self.assertIn("modelVersions", result)
        self.assertIn("stageDurations", result)
        self.assertIn("pageResults", result)
        self.assertIn("failedPageIndexes", result)
        self.assertEqual(len(result["pages"]), 1)
        self.assertEqual(result["thresholdConfigVersion"], "ocr-quality-v1")
        self.assertFalse(result["ocrProbeExecuted"])

    def test_rejects_unsupported_file_with_reason_code(self):
        path = self.root / "note.txt"
        path.write_text("not an image", encoding="utf-8")

        result = QualityPipeline().evaluate(path)

        self.assertFalse(result["accepted"])
        self.assertEqual(result["decision"], Decision.REJECT.value)
        self.assertEqual(result["reasonCodes"], [ReasonCode.UNSUPPORTED_FILE_TYPE.value])
        self.assertEqual(result["failedPages"], [])

    def test_rejects_corrupted_image_with_decode_reason(self):
        path = self.root / "bad.png"
        path.write_bytes(b"not a real png")

        result = QualityPipeline().evaluate(path)

        self.assertFalse(result["accepted"])
        self.assertIn(ReasonCode.IMAGE_DECODE_FAILED.value, result["reasonCodes"])

    def test_file_inspector_accepts_plain_pdf_and_reports_page_count(self):
        path = self.pdf_path("sample.pdf", pages=2)

        inspection = FileInspector(load_default_config()).inspect(path)

        self.assertTrue(inspection.accepted)
        self.assertEqual(inspection.page_count, 2)
        self.assertEqual(inspection.reason_codes, [])

    def test_file_inspector_rejects_encrypted_pdf_with_stable_reason(self):
        plain = self.pdf_path("plain.pdf")
        encrypted = self.root / "encrypted.pdf"
        reader = PdfReader(str(plain))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.encrypt("secret")
        with encrypted.open("wb") as handle:
            writer.write(handle)

        inspection = FileInspector(load_default_config()).inspect(encrypted)

        self.assertFalse(inspection.accepted)
        self.assertEqual(inspection.reason_codes, [ReasonCode.PDF_ENCRYPTED])

    def test_pipeline_renders_pdf_pages_and_returns_page_level_results(self):
        path = self.pdf_path("two-pages.pdf", pages=2)

        result = QualityPipeline().evaluate(path)

        self.assertTrue(result["accepted"])
        self.assertEqual(result["metrics"]["pageCount"], 2)
        self.assertEqual(len(result["pages"]), 2)
        self.assertEqual(result["failedPages"], [])

    def test_rejects_blank_page_as_hard_failure(self):
        path = self.image_path("blank.png", Image.new("RGB", (900, 700), "white"))

        result = QualityPipeline().evaluate(path)

        self.assertFalse(result["accepted"])
        self.assertIn(ReasonCode.PAGE_BLANK.value, result["reasonCodes"])

    def test_rejects_extreme_low_resolution(self):
        path = self.image_path("tiny.png", self.text_like_image(size=(120, 80)))

        result = QualityPipeline().evaluate(path)

        self.assertFalse(result["accepted"])
        self.assertIn(ReasonCode.PAGE_TOO_LOW_RESOLUTION.value, result["reasonCodes"])

    def test_long_screenshot_uses_original_resolution_for_low_resolution_decision(self):
        path = self.image_path("long.png", self.text_like_image(size=(1200, 8000)))
        detector = FixedTextDetector(
            TextDetectionResult(
                boxes=[(20, 30, 160, 45), (20, 90, 160, 105)],
                coverage_ratio=0.02,
                median_height=15,
                low_sharpness_ratio=0.0,
                border_touch_ratio=0.0,
            )
        )

        result = QualityPipeline(text_detector=detector).evaluate(path)

        self.assertNotIn(ReasonCode.PAGE_TOO_LOW_RESOLUTION.value, result["reasonCodes"])
        self.assertEqual(result["pages"][0]["metrics"]["width"], 1200)
        self.assertEqual(result["pages"][0]["metrics"]["height"], 8000)

    def test_risk_item_alone_does_not_reject_when_ocr_probe_passes(self):
        config = load_default_config()
        image = self.text_like_image()
        blurred = image.filter(ImageFilter.GaussianBlur(radius=1.2))
        path = self.image_path("slightly_blurred.png", blurred)
        detector = FixedTextDetector(
            TextDetectionResult(
                boxes=[(80, 100, 500, 130), (80, 180, 520, 210)],
                coverage_ratio=0.04,
                median_height=30,
                low_sharpness_ratio=0.25,
                border_touch_ratio=0.0,
            )
        )

        result = QualityPipeline(config=config, text_detector=detector).evaluate(path)

        self.assertTrue(result["accepted"])
        self.assertIn(result["decision"], [Decision.ACCEPT.value, Decision.ACCEPT_WITH_WARNINGS.value])

    def test_heuristic_ocr_probe_runs_for_boundary_text_detection_without_hard_reject(self):
        path = self.image_path("page.png", self.text_like_image())
        detector = FixedTextDetector(
            TextDetectionResult(
                boxes=[(10, 10, 30, 22)],
                coverage_ratio=0.001,
                median_height=12,
                low_sharpness_ratio=0.1,
                border_touch_ratio=1.0,
            )
        )

        result = QualityPipeline(text_detector=detector).evaluate(path)

        self.assertTrue(result["ocrProbeExecuted"])
        self.assertTrue(result["accepted"])
        self.assertNotIn(ReasonCode.OCR_PROBE_LOW_CONFIDENCE.value, result["reasonCodes"])

    def test_occlusion_risk_forces_ocr_probe_and_returns_warning(self):
        path = self.image_path("page.png", self.text_like_image())
        detector = FixedTextDetector(
            TextDetectionResult(
                boxes=[(80, 100, 500, 130), (80, 180, 520, 210)],
                coverage_ratio=0.04,
                median_height=30,
                low_sharpness_ratio=0.0,
                border_touch_ratio=0.0,
            )
        )
        pipeline = QualityPipeline(text_detector=detector)
        pipeline.occlusion_analyzer = FakeOcclusionAnalyzer()

        result = pipeline.evaluate(path)

        self.assertTrue(result["ocrProbeExecuted"])
        self.assertTrue(result["accepted"])
        self.assertIn(ReasonCode.HAND_OCCLUSION_RISK.value, result["reasonCodes"])
        self.assertIsNotNone(result["pages"][0]["occlusionSummary"])

    def test_allow_partial_aggregation_accepts_when_failure_ratio_is_within_limit(self):
        config = load_default_config()
        config.aggregation_policy = "ALLOW_PARTIAL"
        config.max_failed_page_ratio = 0.5
        good = self.image_path("good.png", self.text_like_image())
        blank = self.image_path("blank.png", Image.new("RGB", (900, 700), "white"))

        result = QualityPipeline(config=config).evaluate_many([good, blank])

        self.assertTrue(result["accepted"])
        self.assertEqual(result["decision"], Decision.ACCEPT_WITH_WARNINGS.value)
        self.assertEqual(result["failedPages"], [1])

    def test_first_n_required_aggregation_rejects_when_first_page_fails(self):
        config = load_default_config()
        config.aggregation_policy = "FIRST_N_REQUIRED"
        config.first_n_required = 1
        blank = self.image_path("blank-first.png", Image.new("RGB", (900, 700), "white"))
        good = self.image_path("good-second.png", self.text_like_image())

        result = QualityPipeline(config=config).evaluate_many([blank, good])

        self.assertFalse(result["accepted"])
        self.assertEqual(result["failedPageIndexes"], [0])

    def test_allow_partial_aggregation_reports_failed_ratio_metrics(self):
        config = load_default_config()
        config.aggregation_policy = "ALLOW_PARTIAL"
        config.max_failed_page_ratio = 0.5
        good = self.image_path("good.png", self.text_like_image())
        blank = self.image_path("blank.png", Image.new("RGB", (900, 700), "white"))

        result = QualityPipeline(config=config).evaluate_many([good, blank])

        self.assertTrue(result["accepted"])
        self.assertEqual(result["qualityMetrics"]["failedPageCount"], 1)
        self.assertEqual(result["qualityMetrics"]["pageCount"], 2)

    def test_cli_outputs_json_result(self):
        path = self.image_path("page.png", self.text_like_image())
        project_root = Path(__file__).resolve().parents[1]

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "ocr_quality.cli",
                str(path),
            ],
            cwd=project_root,
            env={**os.environ, "PYTHONPATH": str(project_root / "src")},
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["accepted"])
        self.assertEqual(payload["thresholdConfigVersion"], "ocr-quality-v1")


if __name__ == "__main__":
    unittest.main()
