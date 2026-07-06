# OCR Quality PaddleOCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the current OCR quality prototype into a configurable FastAPI image-quality service that uses traditional CV rules plus optional PaddleOCR adapters for orientation, text detection, and boundary-sample OCR probing.

**Architecture:** Keep the current independent FastAPI service and Python core package. Split model-dependent behavior behind adapters so `DecisionEngine` consumes normalized metrics only, never PaddleOCR SDK objects. Move thresholds and model strategy to external versioned configuration, add model warmup, richer page/file results, instrumentation, regression tests, integration tests, and benchmark commands.

**Tech Stack:** Python 3.12, `unittest`, FastAPI, Uvicorn, Pillow, NumPy, pypdf, pdfplumber, optional PaddleOCR/PaddlePaddle adapters, optional psutil for benchmark memory metrics.

---

## Design Inputs

- Base design document: `docs/superpowers/specs/2026-06-26-ocr-quality-design.md`
- Confirmed design supplement from 2026-06-30:
  - PaddleOCR is allowed as the preferred pretrained OCR component.
  - PaddleOCR must be isolated behind `OrientationDetector`, `TextDetector`, and `OcrProbe` adapters.
  - No self-trained models.
  - No multimodal model calls.
  - No complete OCR business parsing inside quality detection.
  - `OCR_PROBE_LOW_CONFIDENCE` may only come from a real OCR probe adapter result, not from the current heuristic implementation.

## File Structure

### Create

- `config/quality.default.json`: default versioned quality, risk, aggregation, model, and benchmark configuration.
- `src/ocr_quality/config_loader.py`: load and validate external config into `QualityConfig`.
- `src/ocr_quality/file_decoder.py`: unified image/PDF decoding and page rendering contract.
- `src/ocr_quality/page.py`: page image, file metadata, and coordinate mapping dataclasses.
- `src/ocr_quality/orientation.py`: orientation detector protocol, disabled detector, PaddleOCR adapter shell.
- `src/ocr_quality/text_regions.py`: text-box geometry and text-region quality metrics.
- `src/ocr_quality/adapters/__init__.py`: adapter package marker.
- `src/ocr_quality/adapters/paddle_ocr.py`: PaddleOCR factory and adapter implementations with lazy optional imports.
- `src/ocr_quality/instrumentation.py`: structured stage timing and model version helpers.
- `src/ocr_quality/benchmark.py`: local benchmark runner for latency percentiles and memory.
- `tests/test_config_loader.py`: external configuration tests.
- `tests/test_encoding_contract.py`: UTF-8/readability regression tests.
- `tests/test_file_decoder.py`: file/PDF decode contract tests.
- `tests/test_image_metrics.py`: traditional CV metric trend tests.
- `tests/test_orientation.py`: orientation adapter and correction tests.
- `tests/test_text_regions.py`: text-region metric tests.
- `tests/test_decision_engine.py`: decision/risk/OCR-probe rule tests.
- `tests/test_paddle_adapters.py`: PaddleOCR adapter tests using fake PaddleOCR objects.
- `tests/test_instrumentation.py`: timing/model-version result tests.
- `tests/test_benchmark.py`: benchmark output-shape tests.
- `tests/integration/test_optional_paddleocr.py`: skipped-by-default real PaddleOCR smoke tests.

### Modify

- `pyproject.toml`: add optional dependency groups for `paddle`, `benchmark`, and `test`.
- `README.md`: document config file, one-command startup, optional PaddleOCR install, tests, integration tests, and benchmark commands.
- `src/ocr_quality/config.py`: extend `QualityConfig` with externalized thresholds, feature flags, model paths/devices/threads, OCR-probe sampling, and risk-combination settings.
- `src/ocr_quality/models.py`: repair UTF-8 messages, add stable reason codes and richer result fields.
- `src/ocr_quality/image_metrics.py`: add Laplacian, Tenengrad/Sobel, content-aware block stats, low contrast, skew proxy, compression/noise risk metrics.
- `src/ocr_quality/normalizer.py`: preserve coordinate mapping and use `DecodedPage` inputs.
- `src/ocr_quality/text_detection.py`: keep protocol, remove production reliance on heuristic detector, add disabled detector.
- `src/ocr_quality/ocr_probe.py`: keep protocol, make heuristic probe test-only or disabled by default.
- `src/ocr_quality/decision.py`: consume richer metrics; stop hard-rejecting on heuristic probe; separate hard failures, risks, and corrections.
- `src/ocr_quality/pipeline.py`: orchestrate decode, normalize, CV metrics, orientation correction, text detection, text-region analysis, OCR probe, decision, and timing.
- `src/ocr_quality/web/app.py`: load config once, expose model/config state, preserve single pipeline instance.
- `src/ocr_quality/web/static/index.html`: repair UTF-8 text and show new metrics/model versions.
- `start.py`: support `OCR_QUALITY_CONFIG`, optional worker count remains out of scope for this task.
- `tests/test_quality_pipeline.py`: update expected result contract and remove assumptions tied to heuristic OCR confidence.
- `tests/test_web_app.py`: assert config/model fields and readable UTF-8 HTML.

### Forbidden

- Do not modify `.git/`.
- Do not add model binaries to the repository.
- Do not add generated cache folders such as `__pycache__/`, `.pytest_cache/`, `.venv/`, Paddle model download caches, or benchmark output artifacts.
- Do not replace FastAPI with another web framework.
- Do not add business document classification, fixed-template matching, field extraction, fraud detection, super-resolution, GAN/diffusion, online learning, or custom model training.

## Commands Used Throughout

Use the bundled Python for all commands in this plan:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Compile check:

```powershell
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m compileall -q src tests
```

Optional PaddleOCR dependency install for integration testing:

```powershell
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pip install "paddleocr" "paddlepaddle"
```

Run optional PaddleOCR integration tests only when dependencies and models are available:

```powershell
$env:PYTHONPATH='src'
$env:OCR_QUALITY_RUN_PADDLE_INTEGRATION='1'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests/integration -v
```

---

## Task 1: UTF-8 Contract and User Message Repair

**Files:**
- Modify: `src/ocr_quality/models.py`
- Modify: `src/ocr_quality/web/static/index.html`
- Test: `tests/test_encoding_contract.py`

- [ ] **Step 1: Write failing UTF-8 regression tests**

Create `tests/test_encoding_contract.py` with:

```python
import unittest
from pathlib import Path

from ocr_quality.models import ReasonCode, USER_MESSAGES


ROOT = Path(__file__).resolve().parents[1]


class EncodingContractTests(unittest.TestCase):
    def test_user_messages_are_readable_chinese(self):
        self.assertEqual(USER_MESSAGES[ReasonCode.OCR_PROBE_LOW_CONFIDENCE], "OCR 抽样探测置信度较低，请重新上传更清晰文件。")
        self.assertIn("文件格式暂不支持", USER_MESSAGES[ReasonCode.UNSUPPORTED_FILE_TYPE])

    def test_frontend_contains_readable_chinese_labels(self):
        html = (ROOT / "src" / "ocr_quality" / "web" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("OCR 影像质量校验", html)
        self.assertIn("开始质检", html)
        self.assertIn("原因码", html)
        self.assertNotIn("璐", html)
        self.assertNotIn("�", html)
```

- [ ] **Step 2: Run the encoding tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_encoding_contract -v
```

Expected: FAIL because `models.py` and `index.html` currently contain mojibake Chinese text.

- [ ] **Step 3: Repair only user-facing Chinese strings**

Modify `src/ocr_quality/models.py` `USER_MESSAGES` to exactly:

```python
USER_MESSAGES = {
    ReasonCode.UNSUPPORTED_FILE_TYPE: "文件格式暂不支持，请上传图片或 PDF。",
    ReasonCode.FILE_CORRUPTED: "文件可能已损坏，请重新上传清晰完整的文件。",
    ReasonCode.IMAGE_DECODE_FAILED: "图片无法解码，请重新上传有效图片。",
    ReasonCode.PDF_ENCRYPTED: "PDF 无法处理，请上传未加密文件。",
    ReasonCode.PAGE_BLANK: "页面未检测到有效内容，请重新上传。",
    ReasonCode.PAGE_TOO_LOW_RESOLUTION: "页面分辨率过低，可能无法支持 OCR 识别。",
    ReasonCode.SEVERE_BLUR: "页面严重模糊，请重新拍摄或扫描。",
    ReasonCode.SEVERE_EXPOSURE_ABNORMAL: "页面曝光异常，请重新拍摄或扫描。",
    ReasonCode.NO_EFFECTIVE_TEXT_DETECTED: "页面未检测到足够文字内容。",
    ReasonCode.TEXT_TOO_SMALL_RISK: "页面文字偏小，可能影响 OCR 识别。",
    ReasonCode.LOCAL_BLUR_RISK: "页面存在局部模糊风险。",
    ReasonCode.BORDER_CROP_RISK: "页面文字疑似贴近边界，可能存在裁切风险。",
    ReasonCode.OCR_PROBE_LOW_CONFIDENCE: "OCR 抽样探测置信度较低，请重新上传更清晰文件。",
}
```

Modify `src/ocr_quality/web/static/index.html` to contain readable Chinese labels: `OCR 影像质量校验`, `上传图片或 PDF`, `开始质检`, `质检中...`, `完成`, `结论`, `耗时`, `原因码`, `页面级结果`, `页码`, `分数`, `关键指标`, `文字检测`, `原始 JSON`, `通过`, `拒绝`, `无`.

- [ ] **Step 4: Run the encoding tests and verify they pass**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_encoding_contract -v
```

Expected: PASS.

- [ ] **Step 5: Run current full regression**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Expected: PASS.

---

## Task 2: External Versioned Configuration

**Files:**
- Create: `config/quality.default.json`
- Create: `src/ocr_quality/config_loader.py`
- Modify: `src/ocr_quality/config.py`
- Modify: `src/ocr_quality/web/app.py`
- Modify: `start.py`
- Test: `tests/test_config_loader.py`

- [ ] **Step 1: Write failing configuration tests**

Create `tests/test_config_loader.py` with:

```python
import json
import tempfile
import unittest
from pathlib import Path

from ocr_quality.config_loader import load_quality_config


class ConfigLoaderTests(unittest.TestCase):
    def test_default_config_file_loads_version_and_model_flags(self):
        config = load_quality_config(Path("config/quality.default.json"))

        self.assertEqual(config.version, "ocr-quality-v2-paddle")
        self.assertTrue(config.enable_text_detection)
        self.assertTrue(config.enable_orientation_detection)
        self.assertTrue(config.enable_ocr_probe)
        self.assertEqual(config.ocr_probe_sample_size, 6)
        self.assertEqual(config.text_detector_provider, "paddleocr")

    def test_external_config_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality.json"
            path.write_text(
                json.dumps(
                    {
                        "version": "local-test",
                        "max_pages": 3,
                        "enable_ocr_probe": False,
                        "text_detector_provider": "disabled",
                        "risk_count_for_probe": 4,
                    }
                ),
                encoding="utf-8",
            )

            config = load_quality_config(path)

        self.assertEqual(config.version, "local-test")
        self.assertEqual(config.max_pages, 3)
        self.assertFalse(config.enable_ocr_probe)
        self.assertEqual(config.text_detector_provider, "disabled")
        self.assertEqual(config.risk_count_for_probe, 4)
```

- [ ] **Step 2: Run tests and verify missing loader failure**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_config_loader -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ocr_quality.config_loader'`.

- [ ] **Step 3: Extend `QualityConfig` fields**

Modify `src/ocr_quality/config.py` by adding these dataclass fields with defaults:

```python
enable_orientation_detection: bool = True
enable_text_detection: bool = True
enable_ocr_probe: bool = True
orientation_detector_provider: str = "paddleocr"
text_detector_provider: str = "paddleocr"
ocr_probe_provider: str = "paddleocr"
paddle_use_gpu: bool = False
paddle_lang: str = "ch"
paddle_det_model_dir: str | None = None
paddle_rec_model_dir: str | None = None
paddle_cls_model_dir: str | None = None
paddle_cpu_threads: int = 4
ocr_probe_sample_size: int = 6
model_load_timeout_ms: int = 30000
```

Keep existing fields intact so current tests still compile.

- [ ] **Step 4: Add default external config**

Create `config/quality.default.json` with:

```json
{
  "version": "ocr-quality-v2-paddle",
  "supported_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pdf"],
  "max_file_bytes": 26214400,
  "max_pages": 20,
  "pdf_render_dpi": 120,
  "min_width": 240,
  "min_height": 240,
  "min_pixels": 80000,
  "max_analysis_side": 1200,
  "blank_std_threshold": 2.0,
  "blank_edge_density_threshold": 0.002,
  "black_mean_threshold": 8.0,
  "overexposed_mean_threshold": 246.0,
  "underexposed_mean_threshold": 18.0,
  "min_contrast_std": 18.0,
  "severe_blur_threshold": 22.0,
  "blur_risk_threshold": 55.0,
  "max_low_quality_block_ratio": 0.45,
  "min_text_boxes": 1,
  "min_text_coverage_ratio": 0.003,
  "min_text_median_height": 10.0,
  "max_border_touch_ratio": 0.45,
  "max_text_low_sharpness_ratio": 0.60,
  "risk_count_for_probe": 2,
  "ocr_probe_min_confidence": 0.60,
  "ocr_probe_max_empty_ratio": 0.50,
  "aggregation_policy": "ALL_PAGES_REQUIRED",
  "max_failed_page_ratio": 0.0,
  "first_n_required": 1,
  "require_text": true,
  "enable_orientation_detection": true,
  "enable_text_detection": true,
  "enable_ocr_probe": true,
  "orientation_detector_provider": "paddleocr",
  "text_detector_provider": "paddleocr",
  "ocr_probe_provider": "paddleocr",
  "paddle_use_gpu": false,
  "paddle_lang": "ch",
  "paddle_det_model_dir": null,
  "paddle_rec_model_dir": null,
  "paddle_cls_model_dir": null,
  "paddle_cpu_threads": 4,
  "ocr_probe_sample_size": 6,
  "model_load_timeout_ms": 30000
}
```

- [ ] **Step 5: Implement config loader**

Create `src/ocr_quality/config_loader.py`:

```python
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from .config import QualityConfig


def load_quality_config(path: str | Path | None = None) -> QualityConfig:
    if path is None:
        path = Path("config/quality.default.json")
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    field_names = {field.name for field in fields(QualityConfig)}
    filtered = {key: value for key, value in data.items() if key in field_names}
    if "supported_extensions" in filtered:
        filtered["supported_extensions"] = tuple(filtered["supported_extensions"])
    return QualityConfig(**filtered)
```

- [ ] **Step 6: Wire service startup to external config**

Modify `src/ocr_quality/web/app.py`:

```python
import os
from ocr_quality.config_loader import load_quality_config

def create_app(pipeline: QualityPipeline | None = None) -> FastAPI:
    config_path = os.environ.get("OCR_QUALITY_CONFIG")
    quality_pipeline = pipeline or QualityPipeline(config=load_quality_config(config_path))
```

Modify `start.py` to print config path when set:

```powershell
if ($env:OCR_QUALITY_CONFIG) {
    Write-Host "Config: $env:OCR_QUALITY_CONFIG"
}
```

- [ ] **Step 7: Run config tests and full regression**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_config_loader -v
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Expected: both PASS.

---

## Task 3: Unified File Decoder and PDF Page Contract

**Files:**
- Create: `src/ocr_quality/page.py`
- Create: `src/ocr_quality/file_decoder.py`
- Modify: `src/ocr_quality/pipeline.py`
- Modify: `src/ocr_quality/normalizer.py`
- Test: `tests/test_file_decoder.py`

- [ ] **Step 1: Write failing decoder tests**

Create `tests/test_file_decoder.py` with:

```python
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from reportlab.pdfgen import canvas

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
```

- [ ] **Step 2: Run tests and verify module missing**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_file_decoder -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ocr_quality.file_decoder'`.

- [ ] **Step 3: Implement page dataclasses**

Create `src/ocr_quality/page.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .models import AutoFixRecord, ReasonCode


@dataclass
class CoordinateMapping:
    scale_x: float = 1.0
    scale_y: float = 1.0


@dataclass
class DecodedPage:
    page_index: int
    image: Image.Image
    original_size: tuple[int, int]
    mapping: CoordinateMapping = field(default_factory=CoordinateMapping)
    auto_fixes: list[AutoFixRecord] = field(default_factory=list)
    source: str = "image"


@dataclass
class DecodeResult:
    pages: list[DecodedPage]
    reason_codes: list[ReasonCode]
    file_info: dict[str, Any]
```

- [ ] **Step 4: Implement `FileDecoder` using existing behavior**

Create `src/ocr_quality/file_decoder.py`:

```python
from __future__ import annotations

from pathlib import Path

from .config import QualityConfig
from .file_inspector import FileInspector
from .models import ReasonCode
from .normalizer import PageNormalizer
from .page import DecodeResult, DecodedPage


class FileDecoder:
    def __init__(self, config: QualityConfig):
        self.config = config
        self.inspector = FileInspector(config)
        self.normalizer = PageNormalizer(config)

    def decode(self, path: str | Path) -> DecodeResult:
        file_path = Path(path)
        inspection = self.inspector.inspect(file_path)
        file_info = {
            "extension": inspection.extension,
            "fileSize": inspection.file_size,
            "pageCount": inspection.page_count,
        }
        if not inspection.accepted:
            return DecodeResult([], inspection.reason_codes, file_info)
        if inspection.extension == ".pdf":
            images, reasons = self.normalizer.render_pdf_pages(file_path)
            pages = [
                DecodedPage(index, image, image.size, source="pdf")
                for index, image in enumerate(images)
            ]
            return DecodeResult(pages, reasons, file_info)
        image, reasons, fixes = self.normalizer.load_image(file_path)
        if image is None:
            return DecodeResult([], reasons, file_info)
        page = DecodedPage(0, image, image.size, auto_fixes=fixes, source="image")
        return DecodeResult([page], [], file_info)
```

- [ ] **Step 5: Run decoder tests**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_file_decoder -v
```

Expected: PASS.

- [ ] **Step 6: Update pipeline to use `FileDecoder`**

Modify `src/ocr_quality/pipeline.py` so `QualityPipeline.__init__` creates `self.decoder = FileDecoder(self.config)` and `evaluate(path)` decodes once, then evaluates each `DecodedPage`. Preserve API output shape.

Expected internal call direction:

```python
decode_result = self.decoder.decode(path)
if decode_result.reason_codes:
    return self._file_failure(decode_result.reason_codes, decode_result.file_info)
pages = [self._evaluate_decoded_page(page) for page in decode_result.pages]
return self.decision_engine.aggregate(pages, timings)
```

- [ ] **Step 7: Run full regression**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Expected: PASS.

---

## Task 4: Pure CV Quality Metrics

**Files:**
- Modify: `src/ocr_quality/image_metrics.py`
- Modify: `src/ocr_quality/config.py`
- Test: `tests/test_image_metrics.py`

- [ ] **Step 1: Write failing CV metric trend tests**

Create `tests/test_image_metrics.py` with:

```python
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

        metrics = analyzer.analyze(text_like_image())

        self.assertGreater(metrics.content_block_count, 0)
        self.assertLess(metrics.content_block_count, metrics.total_block_count)
```

- [ ] **Step 2: Run tests and verify missing fields**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_image_metrics -v
```

Expected: FAIL because `ImageMetrics` lacks `laplacian_variance`, `tenengrad`, `low_contrast`, `content_block_count`, and `total_block_count`.

- [ ] **Step 3: Extend `ImageMetrics` dataclass**

Modify `src/ocr_quality/image_metrics.py` `ImageMetrics` to add:

```python
laplacian_variance: float
tenengrad: float
low_contrast: bool
content_block_count: int
total_block_count: int
skew_angle_degrees: float | None
compression_risk: float
```

Add these to `to_dict()` as `laplacianVariance`, `tenengrad`, `lowContrast`, `contentBlockCount`, `totalBlockCount`, `skewAngleDegrees`, `compressionRisk`.

- [ ] **Step 4: Implement metrics**

Implement:

```python
def _laplacian(self, gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="edge")
    return (
        -4 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
```

Set:

```python
lap = self._laplacian(gray)
laplacian_variance = float(np.var(lap))
tenengrad = sharpness
low_contrast = float(np.std(gray)) < self.config.min_contrast_std
```

Change `_block_sharpness` to return scores only for blocks whose `std >= blank_std_threshold * 2` or `edge_density >= blank_edge_density_threshold`; track total block count separately.

Use `skew_angle_degrees=None` for this task; small-angle skew is planned in Task 6.

Set `compression_risk` initially to `min(1.0, noise_estimate / 40.0)`.

- [ ] **Step 5: Run CV metric tests and full regression**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_image_metrics -v
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Expected: PASS.

---

## Task 5: Model Loading, Warmup, and Adapter Factory

**Files:**
- Create: `src/ocr_quality/adapters/__init__.py`
- Create: `src/ocr_quality/adapters/paddle_ocr.py`
- Modify: `src/ocr_quality/pipeline.py`
- Modify: `src/ocr_quality/web/app.py`
- Test: `tests/test_paddle_adapters.py`

- [ ] **Step 1: Write failing adapter factory tests using fakes**

Create `tests/test_paddle_adapters.py` with:

```python
import unittest

from PIL import Image

from ocr_quality.adapters.paddle_ocr import PaddleOcrBundle, PaddleOcrUnavailable
from ocr_quality.config import QualityConfig


class FakePaddleOCR:
    def __init__(self):
        self.ocr_calls = 0

    def ocr(self, image, cls=False, det=True, rec=True):
        self.ocr_calls += 1
        return [[[[10, 10], [110, 10], [110, 30], [10, 30]], ("hello", 0.91)]]


class PaddleAdapterTests(unittest.TestCase):
    def test_bundle_warmup_calls_fake_model_once(self):
        fake = FakePaddleOCR()
        bundle = PaddleOcrBundle(config=QualityConfig(), paddle=fake)

        bundle.warmup()

        self.assertEqual(fake.ocr_calls, 1)
        self.assertIn("paddleocr", bundle.model_versions())

    def test_missing_paddle_dependency_raises_adapter_error(self):
        with self.assertRaises(PaddleOcrUnavailable):
            PaddleOcrBundle.create(QualityConfig(), import_name="module_that_does_not_exist")
```

- [ ] **Step 2: Run tests and verify module missing**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_paddle_adapters -v
```

Expected: FAIL because `ocr_quality.adapters.paddle_ocr` does not exist.

- [ ] **Step 3: Implement adapter bundle with lazy optional import**

Create `src/ocr_quality/adapters/paddle_ocr.py`:

```python
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from PIL import Image

from ocr_quality.config import QualityConfig


class PaddleOcrUnavailable(RuntimeError):
    pass


@dataclass
class PaddleOcrBundle:
    config: QualityConfig
    paddle: Any
    warmed: bool = False

    @classmethod
    def create(cls, config: QualityConfig, import_name: str = "paddleocr") -> "PaddleOcrBundle":
        try:
            module = importlib.import_module(import_name)
        except ImportError as exc:
            raise PaddleOcrUnavailable("PaddleOCR is not installed") from exc
        paddle = module.PaddleOCR(
            use_angle_cls=config.enable_orientation_detection,
            lang=config.paddle_lang,
            use_gpu=config.paddle_use_gpu,
            det_model_dir=config.paddle_det_model_dir,
            rec_model_dir=config.paddle_rec_model_dir,
            cls_model_dir=config.paddle_cls_model_dir,
            cpu_threads=config.paddle_cpu_threads,
        )
        return cls(config=config, paddle=paddle)

    def warmup(self) -> None:
        image = Image.new("RGB", (64, 64), "white")
        self.paddle.ocr(image, cls=False, det=True, rec=True)
        self.warmed = True

    def model_versions(self) -> dict[str, str]:
        return {"paddleocr": type(self.paddle).__name__}
```

- [ ] **Step 4: Keep app initialization unchanged in this task**

Do not modify `src/ocr_quality/web/app.py` in this task. Model bundle wiring is scheduled in Task 11 after orientation, text detection, text-region analysis, and OCR probe adapters all exist.

- [ ] **Step 5: Run adapter tests**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_paddle_adapters -v
```

Expected: PASS without installing PaddleOCR because tests use fakes and missing-module behavior.

---

## Task 6: Document Orientation Detection and Auto-Correction

**Files:**
- Create: `src/ocr_quality/orientation.py`
- Modify: `src/ocr_quality/pipeline.py`
- Modify: `src/ocr_quality/models.py`
- Test: `tests/test_orientation.py`

- [ ] **Step 1: Write failing orientation tests**

Create `tests/test_orientation.py`:

```python
import unittest

from PIL import Image

from ocr_quality.models import ReasonCode
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
```

- [ ] **Step 2: Run tests and verify module missing**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_orientation -v
```

Expected: FAIL because `ocr_quality.orientation` does not exist.

- [ ] **Step 3: Add reason code**

Modify `src/ocr_quality/models.py` `ReasonCode`:

```python
ORIENTATION_CORRECTED = "ORIENTATION_CORRECTED"
ORIENTATION_DETECTION_UNAVAILABLE = "ORIENTATION_DETECTION_UNAVAILABLE"
```

Add readable `USER_MESSAGES` for both.

- [ ] **Step 4: Implement orientation protocol and correction**

Create `src/ocr_quality/orientation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from .models import AutoFixRecord


@dataclass
class OrientationResult:
    angle: int
    confidence: float
    model_version: str


class OrientationDetector(Protocol):
    def detect(self, image: Image.Image) -> OrientationResult:
        ...


class DisabledOrientationDetector:
    def detect(self, image: Image.Image) -> OrientationResult:
        return OrientationResult(angle=0, confidence=1.0, model_version="disabled")


class FixedOrientationDetector:
    def __init__(self, result: OrientationResult):
        self.result = result

    def detect(self, image: Image.Image) -> OrientationResult:
        return self.result


def apply_orientation_correction(
    image: Image.Image,
    result: OrientationResult,
    min_confidence: float = 0.8,
) -> tuple[Image.Image, AutoFixRecord | None]:
    if result.confidence < min_confidence or result.angle % 360 == 0:
        return image, None
    corrected = image.rotate(-result.angle, expand=True)
    fix = AutoFixRecord(
        fixType="ORIENTATION_CORRECTED",
        before={"angle": result.angle, "confidence": result.confidence, "size": image.size},
        after={"angle": 0, "size": corrected.size},
        recomputed=True,
    )
    return corrected, fix
```

- [ ] **Step 5: Wire pipeline after Task 3 decode**

Modify `QualityPipeline` constructor to accept `orientation_detector: OrientationDetector | None = None` and default to `DisabledOrientationDetector()` when disabled or unavailable. In page evaluation, call detector before `prepare_for_analysis`, apply correction, append fix, then compute metrics.

- [ ] **Step 6: Run orientation tests and full regression**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_orientation -v
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Expected: PASS.

---

## Task 7: PaddleOCR Text Detection Adapter

**Files:**
- Modify: `src/ocr_quality/text_detection.py`
- Modify: `src/ocr_quality/adapters/paddle_ocr.py`
- Modify: `src/ocr_quality/pipeline.py`
- Test: `tests/test_paddle_adapters.py`

- [ ] **Step 1: Add failing fake Paddle text-detection tests**

Append to `tests/test_paddle_adapters.py`:

```python
from ocr_quality.adapters.paddle_ocr import PaddleTextDetector


class FakeDetectionOnlyPaddle:
    def ocr(self, image, cls=False, det=True, rec=False):
        return [[[[10, 10], [110, 10], [110, 30], [10, 30]], None]]


class PaddleTextDetectorTests(unittest.TestCase):
    def test_text_detector_converts_paddle_boxes_to_summary_metrics(self):
        detector = PaddleTextDetector(FakeDetectionOnlyPaddle())
        image = Image.new("RGB", (200, 100), "white")

        result = detector.detect(image)

        self.assertEqual(len(result.boxes), 1)
        self.assertEqual(result.boxes[0], (10, 10, 110, 30))
        self.assertGreater(result.coverage_ratio, 0)
        self.assertEqual(result.median_height, 20)
```

- [ ] **Step 2: Run adapter tests and verify missing class**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_paddle_adapters -v
```

Expected: FAIL because `PaddleTextDetector` is missing.

- [ ] **Step 3: Implement Paddle text detector**

Add to `src/ocr_quality/adapters/paddle_ocr.py`:

```python
import numpy as np
from ocr_quality.text_detection import TextDetectionResult


class PaddleTextDetector:
    def __init__(self, paddle):
        self.paddle = paddle

    def detect(self, image: Image.Image) -> TextDetectionResult:
        raw = self.paddle.ocr(image, cls=False, det=True, rec=False)
        boxes = []
        for item in raw or []:
            points = item[0]
            xs = [int(point[0]) for point in points]
            ys = [int(point[1]) for point in points]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        if not boxes:
            return TextDetectionResult([], 0.0, 0.0, 0.0, 0.0)
        width, height = image.size
        areas = [(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes]
        heights = [y1 - y0 for _, y0, _, y1 in boxes]
        border = [box for box in boxes if box[0] <= 3 or box[1] <= 3 or box[2] >= width - 3 or box[3] >= height - 3]
        return TextDetectionResult(
            boxes=boxes,
            coverage_ratio=sum(areas) / float(width * height),
            median_height=float(np.median(heights)),
            low_sharpness_ratio=0.0,
            border_touch_ratio=len(border) / float(len(boxes)),
        )
```

- [ ] **Step 4: Add disabled text detector**

Modify `src/ocr_quality/text_detection.py`:

```python
class DisabledTextDetector:
    def detect(self, image: Image.Image) -> TextDetectionResult:
        return TextDetectionResult([], 0.0, 0.0, 0.0, 0.0)
```

- [ ] **Step 5: Run adapter tests and full regression**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_paddle_adapters -v
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Expected: PASS.

---

## Task 8: Text Region Quality Analysis

**Files:**
- Create: `src/ocr_quality/text_regions.py`
- Modify: `src/ocr_quality/text_detection.py`
- Modify: `src/ocr_quality/pipeline.py`
- Test: `tests/test_text_regions.py`

- [ ] **Step 1: Write failing text-region tests**

Create `tests/test_text_regions.py`:

```python
import unittest

from PIL import Image

from ocr_quality.image_metrics import FastImageQualityAnalyzer
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
```

- [ ] **Step 2: Run tests and verify module missing**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_text_regions -v
```

Expected: FAIL because `ocr_quality.text_regions` is missing.

- [ ] **Step 3: Implement text-region summary**

Create `src/ocr_quality/text_regions.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import QualityConfig
from .text_detection import TextDetectionResult


@dataclass
class TextRegionSummary:
    box_count: int
    coverage_ratio: float
    average_height: float
    median_height: float
    small_text_ratio: float
    low_sharpness_ratio: float
    border_touch_ratio: float
    top_box_ratio: float
    middle_box_ratio: float
    bottom_box_ratio: float

    def to_dict(self) -> dict:
        return {
            "boxCount": self.box_count,
            "coverageRatio": round(self.coverage_ratio, 6),
            "averageHeight": round(self.average_height, 4),
            "medianHeight": round(self.median_height, 4),
            "smallTextRatio": round(self.small_text_ratio, 4),
            "lowSharpnessRatio": round(self.low_sharpness_ratio, 4),
            "borderTouchRatio": round(self.border_touch_ratio, 4),
            "topBoxRatio": round(self.top_box_ratio, 4),
            "middleBoxRatio": round(self.middle_box_ratio, 4),
            "bottomBoxRatio": round(self.bottom_box_ratio, 4),
        }


class TextRegionAnalyzer:
    def __init__(self, config: QualityConfig):
        self.config = config

    def analyze(self, image: Image.Image, text: TextDetectionResult) -> TextRegionSummary:
        width, height = image.size
        boxes = text.boxes
        if not boxes:
            return TextRegionSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        box_heights = [y1 - y0 for _, y0, _, y1 in boxes]
        areas = [(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes]
        small = [h for h in box_heights if h < self.config.min_text_median_height]
        border = [box for box in boxes if box[0] <= 3 or box[1] <= 3 or box[2] >= width - 3 or box[3] >= height - 3]
        centers = [((y0 + y1) / 2.0) / height for _, y0, _, y1 in boxes]
        top = [c for c in centers if c < 0.33]
        middle = [c for c in centers if 0.33 <= c < 0.66]
        bottom = [c for c in centers if c >= 0.66]
        count = len(boxes)
        return TextRegionSummary(
            box_count=count,
            coverage_ratio=sum(areas) / float(width * height),
            average_height=float(np.mean(box_heights)),
            median_height=float(np.median(box_heights)),
            small_text_ratio=len(small) / count,
            low_sharpness_ratio=text.low_sharpness_ratio,
            border_touch_ratio=len(border) / count,
            top_box_ratio=len(top) / count,
            middle_box_ratio=len(middle) / count,
            bottom_box_ratio=len(bottom) / count,
        )
```

- [ ] **Step 4: Wire summary into pipeline**

In `QualityPipeline`, after text detection:

```python
text_summary = self.text_region_analyzer.analyze(analysis_image, text)
text = text.with_region_summary(text_summary)
```

If adding `with_region_summary` is too broad, update `DecisionEngine.page_decision` to accept `text_region_summary` as a separate parameter and use it for `textDetectionSummary`.

- [ ] **Step 5: Run tests**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_text_regions -v
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Expected: PASS.

---

## Task 9: Decision Engine Hard Failures, Risks, and Probe Rules

**Files:**
- Modify: `src/ocr_quality/models.py`
- Modify: `src/ocr_quality/decision.py`
- Modify: `src/ocr_quality/ocr_probe.py`
- Test: `tests/test_decision_engine.py`

- [ ] **Step 1: Write failing decision tests**

Create `tests/test_decision_engine.py`:

```python
import unittest

from ocr_quality.config import QualityConfig
from ocr_quality.decision import DecisionEngine
from ocr_quality.image_metrics import ImageMetrics
from ocr_quality.models import Decision, ReasonCode
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


class DecisionEngineTests(unittest.TestCase):
    def test_single_risk_does_not_reject(self):
        text = TextDetectionResult([(10, 10, 100, 25)], 0.01, 15, 0.0, 0.0)

        result = DecisionEngine(QualityConfig()).page_decision(0, metrics(low_quality_block_ratio=0.8), text, None, {})

        self.assertTrue(result.accepted)
        self.assertEqual(result.decision, Decision.ACCEPT_WITH_WARNINGS)
        self.assertIn(ReasonCode.LOCAL_BLUR_RISK, result.reasonCodes)

    def test_real_ocr_probe_low_confidence_can_reject(self):
        text = TextDetectionResult([(10, 10, 100, 25)], 0.01, 15, 0.0, 0.0)
        probe = OcrProbeResult(0.2, 0.8, 0.2, 0.2, 1.0, 0.1, source="paddleocr")

        result = DecisionEngine(QualityConfig()).page_decision(0, metrics(), text, probe, {})

        self.assertFalse(result.accepted)
        self.assertIn(ReasonCode.OCR_PROBE_LOW_CONFIDENCE, result.reasonCodes)

    def test_heuristic_probe_low_confidence_does_not_hard_reject(self):
        text = TextDetectionResult([(10, 10, 100, 25)], 0.01, 15, 0.0, 0.0)
        probe = OcrProbeResult(0.2, 0.8, 0.2, 0.2, 1.0, 0.1, source="heuristic")

        result = DecisionEngine(QualityConfig()).page_decision(0, metrics(), text, probe, {})

        self.assertTrue(result.accepted)
        self.assertNotIn(ReasonCode.OCR_PROBE_LOW_CONFIDENCE, result.reasonCodes)
```

- [ ] **Step 2: Run tests and verify signature/field failures**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_decision_engine -v
```

Expected: FAIL with `OcrProbeResult.__init__() got an unexpected keyword argument 'source'`.

- [ ] **Step 3: Add source to OCR probe result**

Modify `src/ocr_quality/ocr_probe.py`:

```python
source: str = "unknown"
```

Add `source` to `to_summary()`.

- [ ] **Step 4: Update decision rules**

Modify `src/ocr_quality/decision.py` so:

```python
if ocr_probe is not None and ocr_probe.source != "heuristic":
    if (
        ocr_probe.average_confidence < self.config.ocr_probe_min_confidence
        or ocr_probe.empty_ratio > self.config.ocr_probe_max_empty_ratio
    ):
        hard.append(ReasonCode.OCR_PROBE_LOW_CONFIDENCE)
```

Add low contrast and compression risk as risks:

```python
if metrics.low_contrast:
    risks.append(ReasonCode.LOW_CONTRAST_RISK)
if metrics.compression_risk >= 0.8:
    risks.append(ReasonCode.COMPRESSION_ARTIFACT_RISK)
```

Add `LOW_CONTRAST_RISK` and `COMPRESSION_ARTIFACT_RISK` to `ReasonCode` and `USER_MESSAGES`.

- [ ] **Step 5: Run decision tests and full regression**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_decision_engine -v
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Expected: PASS.

---

## Task 10: OCR Sampling Probe

**Files:**
- Modify: `src/ocr_quality/ocr_probe.py`
- Modify: `src/ocr_quality/adapters/paddle_ocr.py`
- Modify: `src/ocr_quality/decision.py`
- Test: `tests/test_paddle_adapters.py`

- [ ] **Step 1: Add failing Paddle OCR probe tests**

Append to `tests/test_paddle_adapters.py`:

```python
from ocr_quality.adapters.paddle_ocr import PaddleOcrProbe
from ocr_quality.text_detection import TextDetectionResult


class FakeRecognitionPaddle:
    def ocr(self, image, cls=True, det=False, rec=True):
        return [[("hello", 0.92)]]


class PaddleOcrProbeTests(unittest.TestCase):
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
```

- [ ] **Step 2: Run tests and verify missing class/field**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_paddle_adapters -v
```

Expected: FAIL because `PaddleOcrProbe` and `sample_count` do not exist.

- [ ] **Step 3: Extend `OcrProbeResult`**

Modify `src/ocr_quality/ocr_probe.py`:

```python
sample_count: int = 0
source: str = "unknown"
```

Add both to `to_summary()`.

- [ ] **Step 4: Implement Paddle OCR probe**

Add to `src/ocr_quality/adapters/paddle_ocr.py`:

```python
class PaddleOcrProbe:
    def __init__(self, paddle, sample_size: int):
        self.paddle = paddle
        self.sample_size = sample_size

    def probe(self, image: Image.Image, text_detection: TextDetectionResult) -> OcrProbeResult:
        boxes = text_detection.boxes[: self.sample_size]
        confidences = []
        empty = 0
        for x0, y0, x1, y1 in boxes:
            crop = image.crop((x0, y0, x1, y1))
            raw = self.paddle.ocr(crop, cls=True, det=False, rec=True)
            if not raw:
                empty += 1
                continue
            item = raw[0][0] if isinstance(raw[0], list) else raw[0]
            text, confidence = item
            if not text:
                empty += 1
            confidences.append(float(confidence))
        sample_count = len(boxes)
        if sample_count == 0:
            return OcrProbeResult(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, sample_count=0, source="paddleocr")
        import numpy as np
        avg = float(np.mean(confidences)) if confidences else 0.0
        med = float(np.median(confidences)) if confidences else 0.0
        low = len([c for c in confidences if c < 0.6]) / sample_count
        return OcrProbeResult(
            success_ratio=(sample_count - empty) / sample_count,
            empty_ratio=empty / sample_count,
            average_confidence=avg,
            median_confidence=med,
            low_confidence_ratio=low,
            effective_character_ratio=(sample_count - empty) / sample_count,
            sample_count=sample_count,
            source="paddleocr",
        )
```

- [ ] **Step 5: Run adapter and decision tests**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_paddle_adapters tests.test_decision_engine -v
```

Expected: PASS.

---

## Task 11: Pipeline Orchestration and Result Object

**Files:**
- Modify: `src/ocr_quality/models.py`
- Modify: `src/ocr_quality/pipeline.py`
- Modify: `src/ocr_quality/web/app.py`
- Modify: `src/ocr_quality/web/static/index.html`
- Test: `tests/test_quality_pipeline.py`
- Test: `tests/test_web_app.py`

- [ ] **Step 1: Add failing result-contract tests**

Modify `tests/test_quality_pipeline.py` `test_accepts_readable_text_like_image_with_stable_result_shape` to assert:

```python
self.assertIn("qualityMetrics", result)
self.assertIn("detectedRisks", result)
self.assertIn("appliedCorrections", result)
self.assertIn("modelVersions", result)
self.assertIn("stageDurations", result)
self.assertIn("pageResults", result)
self.assertIn("failedPageIndexes", result)
```

Modify `tests/test_web_app.py` `test_upload_image_returns_quality_result` to assert:

```python
self.assertIn("modelVersions", payload)
self.assertIn("stageDurations", payload)
self.assertIn("pageResults", payload)
```

- [ ] **Step 2: Run tests and verify missing fields**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_quality_pipeline tests.test_web_app -v
```

Expected: FAIL due missing new result fields.

- [ ] **Step 3: Add backward-compatible aliases**

Modify `DecisionEngine.aggregate` to keep old fields and add new fields:

```python
"qualityMetrics": metrics,
"detectedRisks": [code.value for code in reason_codes if code.name.endswith("_RISK")],
"appliedCorrections": auto_fixes,
"modelVersions": model_versions,
"stageDurations": timings_ms,
"pageResults": [page.to_dict() for page in pages],
"failedPageIndexes": failed_pages,
```

Keep existing `metrics`, `autoFixes`, `timingsMs`, `pages`, and `failedPages` for compatibility.

- [ ] **Step 4: Add `modelVersions` path**

Modify `QualityPipeline` to hold:

```python
self.model_versions = {}
```

When adapter bundles are used, merge `bundle.model_versions()`.

Pass `model_versions` into `aggregate`. If this requires signature change, update all call sites.

- [ ] **Step 5: Update frontend display**

Modify `index.html` to display `payload.modelVersions` and prefer `payload.pageResults || payload.pages`.

- [ ] **Step 6: Run full regression**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Expected: PASS.

---

## Task 12: Logs and Stage Timing Instrumentation

**Files:**
- Create: `src/ocr_quality/instrumentation.py`
- Modify: `src/ocr_quality/pipeline.py`
- Modify: `src/ocr_quality/web/app.py`
- Test: `tests/test_instrumentation.py`

- [ ] **Step 1: Write failing timing tests**

Create `tests/test_instrumentation.py`:

```python
import time
import unittest

from ocr_quality.instrumentation import StageTimer


class InstrumentationTests(unittest.TestCase):
    def test_stage_timer_records_named_duration(self):
        timer = StageTimer()

        with timer.measure("decode"):
            time.sleep(0.001)

        payload = timer.to_dict()
        self.assertIn("decode", payload)
        self.assertGreaterEqual(payload["decode"], 0.0)

    def test_nested_stage_names_are_preserved(self):
        timer = StageTimer()

        with timer.measure("page.0.textDetection"):
            pass

        self.assertIn("page.0.textDetection", timer.to_dict())
```

- [ ] **Step 2: Run tests and verify missing module**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_instrumentation -v
```

Expected: FAIL because `ocr_quality.instrumentation` does not exist.

- [ ] **Step 3: Implement timer**

Create `src/ocr_quality/instrumentation.py`:

```python
from __future__ import annotations

import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self):
        self._durations: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self._durations[name] = round((time.perf_counter() - started) * 1000.0, 3)

    def to_dict(self) -> dict[str, float]:
        return dict(self._durations)
```

- [ ] **Step 4: Use timer in pipeline**

Replace scattered `time.perf_counter()` stage measurements in `QualityPipeline` with `StageTimer`. Keep old output keys where tests require them.

- [ ] **Step 5: Run instrumentation tests and full regression**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_instrumentation -v
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Expected: PASS.

---

## Task 13: PDF Multi-Page Aggregation

**Files:**
- Modify: `src/ocr_quality/file_decoder.py`
- Modify: `src/ocr_quality/decision.py`
- Modify: `src/ocr_quality/pipeline.py`
- Test: `tests/test_quality_pipeline.py`

- [ ] **Step 1: Add failing aggregation tests**

Append to `tests/test_quality_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run tests and verify aggregation behavior**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_quality_pipeline -v
```

Expected: PASS because Task 11 has already added `failedPageIndexes` and `qualityMetrics`.

- [ ] **Step 3: Ensure aliases and metrics are complete**

In `DecisionEngine.aggregate`, ensure:

```python
failed_ratio = len(failed_pages) / max(len(pages), 1)
metrics = {"pageCount": len(pages), "failedPageCount": len(failed_pages), "failedPageRatio": failed_ratio}
```

Use this same `metrics` object for both `metrics` and `qualityMetrics`.

- [ ] **Step 4: Run full regression**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Expected: PASS.

---

## Task 14: Optional Real PaddleOCR Integration Tests

**Files:**
- Create: `tests/integration/test_optional_paddleocr.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] **Step 1: Write skipped-by-default integration tests**

Create `tests/integration/test_optional_paddleocr.py`:

```python
import os
import unittest

from PIL import Image

from ocr_quality.config import QualityConfig
from ocr_quality.adapters.paddle_ocr import PaddleOcrBundle, PaddleOcrUnavailable, PaddleTextDetector


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
```

- [ ] **Step 2: Run integration tests without env var**

Run:

```powershell
$env:PYTHONPATH='src'
Remove-Item Env:\OCR_QUALITY_RUN_PADDLE_INTEGRATION -ErrorAction SilentlyContinue
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests/integration -v
```

Expected: SKIPPED, not failed.

- [ ] **Step 3: Add optional dependencies**

Modify `pyproject.toml`:

```toml
[project.optional-dependencies]
test = [
  "httpx",
]
paddle = [
  "paddleocr",
  "paddlepaddle",
]
benchmark = [
  "psutil",
]
```

If `test` already exists, merge without duplication.

- [ ] **Step 4: Document optional install**

Add to `README.md`:

```markdown
Optional PaddleOCR integration:

```powershell
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pip install ".[paddle]"
$env:OCR_QUALITY_RUN_PADDLE_INTEGRATION='1'
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests/integration -v
```
```

- [ ] **Step 5: Run default full regression**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Expected: PASS with integration test skipped unless env var is set.

---

## Task 15: Performance Benchmark Runner

**Files:**
- Create: `src/ocr_quality/benchmark.py`
- Test: `tests/test_benchmark.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing benchmark output tests**

Create `tests/test_benchmark.py`:

```python
import unittest

from ocr_quality.benchmark import percentile, summarize_durations


class BenchmarkTests(unittest.TestCase):
    def test_percentile_calculation(self):
        values = [1.0, 2.0, 3.0, 4.0]

        self.assertEqual(percentile(values, 50), 2.5)
        self.assertEqual(percentile(values, 95), 4.0)

    def test_summary_contains_required_percentiles(self):
        summary = summarize_durations([10.0, 20.0, 30.0])

        self.assertEqual(summary["count"], 3)
        self.assertIn("p50Ms", summary)
        self.assertIn("p95Ms", summary)
        self.assertIn("p99Ms", summary)
```

- [ ] **Step 2: Run tests and verify missing module**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_benchmark -v
```

Expected: FAIL because `ocr_quality.benchmark` does not exist.

- [ ] **Step 3: Implement benchmark helpers and CLI**

Create `src/ocr_quality/benchmark.py`:

```python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .pipeline import QualityPipeline


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 4)


def summarize_durations(values: list[float]) -> dict:
    return {
        "count": len(values),
        "p50Ms": percentile(values, 50),
        "p95Ms": percentile(values, 95),
        "p99Ms": percentile(values, 99),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark OCR quality pipeline latency.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args(argv)
    pipeline = QualityPipeline()
    durations = []
    for _ in range(args.repeat):
        for path in args.paths:
            started = time.perf_counter()
            pipeline.evaluate(Path(path))
            durations.append((time.perf_counter() - started) * 1000.0)
    print(json.dumps(summarize_durations(durations), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run benchmark tests**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_benchmark -v
```

Expected: PASS.

- [ ] **Step 5: Document benchmark command**

Add to `README.md`:

```markdown
Benchmark:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ocr_quality.benchmark path\to\sample.png --repeat 10
```
```

- [ ] **Step 6: Run full regression**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Expected: PASS.

---

## Task 16: API and Frontend Configuration Visibility

**Files:**
- Modify: `src/ocr_quality/web/app.py`
- Modify: `src/ocr_quality/web/static/index.html`
- Test: `tests/test_web_app.py`

- [ ] **Step 1: Add failing Web API tests**

Modify `tests/test_web_app.py` `test_config_endpoint_exposes_supported_formats_and_limits`:

```python
self.assertIn("model", payload)
self.assertEqual(payload["model"]["textDetectorProvider"], "paddleocr")
self.assertIn("features", payload)
self.assertTrue(payload["features"]["textDetection"])
```

- [ ] **Step 2: Run tests and verify missing fields**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_web_app -v
```

Expected: FAIL because config endpoint does not expose `model` and `features`.

- [ ] **Step 3: Update config endpoint**

Modify `/api/quality/config` response in `src/ocr_quality/web/app.py`:

```python
"features": {
    "orientationDetection": quality_pipeline.config.enable_orientation_detection,
    "textDetection": quality_pipeline.config.enable_text_detection,
    "ocrProbe": quality_pipeline.config.enable_ocr_probe,
},
"model": {
    "orientationDetectorProvider": quality_pipeline.config.orientation_detector_provider,
    "textDetectorProvider": quality_pipeline.config.text_detector_provider,
    "ocrProbeProvider": quality_pipeline.config.ocr_probe_provider,
    "useGpu": quality_pipeline.config.paddle_use_gpu,
    "lang": quality_pipeline.config.paddle_lang,
},
```

- [ ] **Step 4: Update frontend**

Modify `index.html` to keep raw JSON visible and add display for `modelVersions`; do not add a build system.

- [ ] **Step 5: Run Web tests and full regression**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_web_app -v
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Expected: PASS.

---

## Task 17: README and Operational Runbook

**Files:**
- Modify: `README.md`
- Test: `tests/test_start_scripts.py`

- [ ] **Step 1: Add failing README assertions**

Modify `tests/test_start_scripts.py`:

```python
def test_readme_documents_config_paddle_and_benchmark(self):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    self.assertIn("OCR_QUALITY_CONFIG", readme)
    self.assertIn("PaddleOCR", readme)
    self.assertIn("ocr_quality.benchmark", readme)
    self.assertIn("OCR_QUALITY_RUN_PADDLE_INTEGRATION", readme)
```

- [ ] **Step 2: Run README tests and verify failure**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_start_scripts -v
```

Expected: FAIL until README is updated.

- [ ] **Step 3: Update README**

Add sections:

```markdown
## Configuration

Default config is `config/quality.default.json`.

```powershell
$env:OCR_QUALITY_CONFIG='config\quality.default.json'
python start.py
```

## PaddleOCR

PaddleOCR is optional at install time. When unavailable, the service starts with disabled model adapters and does not emit `OCR_PROBE_LOW_CONFIDENCE` from heuristic probe output.

## Benchmark

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ocr_quality.benchmark path\to\sample.png --repeat 10
```
```

- [ ] **Step 4: Run README tests**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_start_scripts -v
```

Expected: PASS.

---

## Task 18: Final Verification and Acceptance

**Files:**
- No production code changes.
- Read-only verification over all changed files.

- [ ] **Step 1: Run full unit and API regression**

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Expected: PASS. Integration tests may be skipped unless `OCR_QUALITY_RUN_PADDLE_INTEGRATION=1`.

- [ ] **Step 2: Run compile check**

Run:

```powershell
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m compileall -q src tests
```

Expected: exit code 0.

- [ ] **Step 3: Run optional PaddleOCR integration check**

Only if PaddleOCR is installed:

```powershell
$env:PYTHONPATH='src'
$env:OCR_QUALITY_RUN_PADDLE_INTEGRATION='1'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests/integration -v
```

Expected: PASS or SKIP with explicit dependency/model reason. It must not fail because default unit tests depend on remote model downloads.

- [ ] **Step 4: Run service smoke test**

Start:

```powershell
python start.py --no-browser --port 8010
```

Expected: foreground process prints `Uvicorn running on http://127.0.0.1:8010`.

In a second terminal:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8010/health'
Invoke-RestMethod -Uri 'http://127.0.0.1:8010/api/quality/config'
```

Expected: health returns `status=ok`; config returns `version=ocr-quality-v2-paddle`.

- [ ] **Step 5: Run benchmark smoke**

Use a local sample image:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ocr_quality.benchmark path\to\sample.png --repeat 3
```

Expected: JSON includes `count`, `p50Ms`, `p95Ms`, `p99Ms`.

- [ ] **Step 6: Clean generated caches**

Run:

```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

Expected: no `__pycache__` directories remain.

## Acceptance Checklist

- [ ] Config file exists at `config/quality.default.json` and `thresholdConfigVersion` returns `ocr-quality-v2-paddle`.
- [ ] `QualityConfig` includes model provider, model path, device, thread, feature-flag, OCR-probe, and aggregation settings.
- [ ] PaddleOCR is optional and isolated behind adapters.
- [ ] `DecisionEngine` does not import or reference PaddleOCR.
- [ ] No self-trained model or training pipeline is introduced.
- [ ] No complete OCR business parsing is introduced.
- [ ] No model binaries are committed.
- [ ] `OCR_PROBE_LOW_CONFIDENCE` is emitted only for non-heuristic OCR probe results.
- [ ] Missing PaddleOCR dependency degrades without crashing default unit tests.
- [ ] Orientation detection can auto-correct 90/180/270 degree pages when adapter confidence is high.
- [ ] Text detection returns normalized boxes and summary metrics.
- [ ] Text-region analyzer reports small text, coverage, border risk, distribution, and region quality metrics.
- [ ] Pure CV metrics include Laplacian, Tenengrad/Sobel, content-aware block sharpness, low contrast, exposure, noise/compression risk, and illumination unevenness.
- [ ] PDF multi-page handling preserves page-level results and aggregation policy.
- [ ] Result object includes both backward-compatible fields and new fields: `qualityMetrics`, `detectedRisks`, `appliedCorrections`, `modelVersions`, `stageDurations`, `pageResults`, `failedPageIndexes`.
- [ ] API `/api/quality/config` exposes feature flags and model providers.
- [ ] Frontend displays readable UTF-8 Chinese and raw JSON.
- [ ] Tests cover unit, adapter fake integration, optional PaddleOCR integration, Web API, startup script, and benchmark helpers.
- [ ] README documents startup, config, PaddleOCR optional install, integration tests, and benchmark.

## Regression Commands

Default regression:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m compileall -q src tests
```

Optional PaddleOCR integration:

```powershell
$env:PYTHONPATH='src'
$env:OCR_QUALITY_RUN_PADDLE_INTEGRATION='1'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests/integration -v
```

Service smoke:

```powershell
python start.py --no-browser --port 8010
```

Benchmark smoke:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ocr_quality.benchmark path\to\sample.png --repeat 3
```
