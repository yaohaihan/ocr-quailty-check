import tempfile
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
    category=Warning,
)

from fastapi.testclient import TestClient
from PIL import Image

from ocr_quality.config import QualityConfig
from ocr_quality.models import ReasonCode
from ocr_quality.config_loader import load_quality_config
from ocr_quality.web.app import create_app


class StubPipeline:
    config = load_quality_config("config/quality.default.json")
    model_versions = {"textDetector": "paddleocr", "ocrProbe": "paddleocr"}

    def evaluate(self, path):
        if Path(path).suffix == ".txt":
            return {
                "accepted": False,
                "decision": "REJECT",
                "ocrReadinessScore": 0.0,
                "reasonCodes": [ReasonCode.UNSUPPORTED_FILE_TYPE.value],
                "userMessages": [],
                "metrics": {},
                "qualityMetrics": {},
                "detectedRisks": [],
                "autoFixes": [],
                "appliedCorrections": [],
                "pages": [],
                "pageResults": [],
                "failedPages": [],
                "failedPageIndexes": [],
                "timingsMs": {},
                "stageDurations": {},
                "thresholdConfigVersion": self.config.version,
                "modelVersions": self.model_versions,
                "ocrProbeExecuted": False,
                "aggregationPolicy": self.config.aggregation_policy,
            }
        return {
            "accepted": True,
            "decision": "ACCEPT",
            "ocrReadinessScore": 0.9,
            "reasonCodes": [],
            "userMessages": [],
            "metrics": {"pageCount": 1},
            "qualityMetrics": {"pageCount": 1},
            "detectedRisks": [],
            "autoFixes": [],
            "appliedCorrections": [],
            "pages": [{"pageIndex": 0, "accepted": True, "decision": "ACCEPT"}],
            "pageResults": [{"pageIndex": 0, "accepted": True, "decision": "ACCEPT"}],
            "failedPages": [],
            "failedPageIndexes": [],
            "timingsMs": {"total": 1.0},
            "stageDurations": {"total": 1.0},
            "thresholdConfigVersion": self.config.version,
            "modelVersions": self.model_versions,
            "ocrProbeExecuted": False,
            "aggregationPolicy": self.config.aggregation_policy,
        }


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(pipeline=StubPipeline()))
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def text_like_image(self):
        image = Image.new("RGB", (900, 700), "white")
        pixels = image.load()
        for y in range(100, 600, 42):
            for x in range(80, 820, 130):
                for yy in range(y, y + 10):
                    for xx in range(x, x + 80):
                        pixels[xx, yy] = (20, 20, 20)
        path = self.root / "page.png"
        image.save(path)
        return path

    def test_health_reports_ready_config_version(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok", "thresholdConfigVersion": "ocr-quality-v2-paddle"},
        )

    def test_index_serves_upload_page(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("OCR", response.text)
        self.assertIn("/api/quality/check", response.text)

    def test_config_endpoint_exposes_supported_formats_and_limits(self):
        response = self.client.get("/api/quality/config")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["version"], "ocr-quality-v2-paddle")
        self.assertIn(".png", payload["supportedExtensions"])
        self.assertIn("maxPages", payload)
        self.assertIn("model", payload)
        self.assertEqual(payload["model"]["textDetectorProvider"], "paddleocr")
        self.assertIn("features", payload)
        self.assertTrue(payload["features"]["textDetection"])

    def test_upload_image_returns_quality_result(self):
        path = self.text_like_image()

        with path.open("rb") as handle:
            response = self.client.post(
                "/api/quality/check",
                files={"file": ("page.png", handle, "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["accepted"])
        self.assertIn("ocrReadinessScore", payload)
        self.assertEqual(payload["thresholdConfigVersion"], "ocr-quality-v2-paddle")
        self.assertIn("modelVersions", payload)
        self.assertIn("stageDurations", payload)
        self.assertIn("pageResults", payload)
        self.assertEqual(len(payload["pages"]), 1)

    def test_upload_unsupported_file_returns_stable_reason_code(self):
        response = self.client.post(
            "/api/quality/check",
            files={"file": ("note.txt", b"not an image", "text/plain")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["accepted"])
        self.assertEqual(payload["reasonCodes"], [ReasonCode.UNSUPPORTED_FILE_TYPE.value])

    def test_upload_pipeline_exception_returns_json_error(self):
        class ExplodingPipeline:
            config = QualityConfig(version="ocr-quality-test")

            def evaluate(self, path):
                raise RuntimeError("boom")

        client = TestClient(create_app(pipeline=ExplodingPipeline()), raise_server_exceptions=False)

        response = client.post(
            "/api/quality/check",
            files={"file": ("page.png", b"not important", "image/png")},
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn("application/json", response.headers["content-type"])
        payload = response.json()
        self.assertFalse(payload["accepted"])
        self.assertEqual(payload["decision"], "REJECT")
        self.assertEqual(payload["reasonCodes"], ["INTERNAL_ERROR"])
        self.assertIn("boom", payload["error"])

    def test_frontend_renders_occlusion_summary_fields(self):
        html = (Path(__file__).resolve().parents[1] / "src" / "ocr_quality" / "web" / "static" / "index.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("occlusionSummary", html)
        self.assertIn("textOverlapRatio", html)
        self.assertIn("affectedTextBoxRatio", html)


if __name__ == "__main__":
    unittest.main()
