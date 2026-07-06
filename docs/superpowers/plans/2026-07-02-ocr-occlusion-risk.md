# OCR Text Occlusion Risk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add conservative hand/object-over-text occlusion risk detection to the OCR quality pipeline.

**Architecture:** Introduce a focused `occlusion.py` analyzer that consumes the analysis image and text boxes, returns normalized overlap metrics, and never makes reject decisions itself. Wire the analyzer into `QualityPipeline`, pass its result to `DecisionEngine`, and use it to force OCR probing when suspicious regions overlap text.

**Tech Stack:** Python standard library, Pillow, NumPy, existing unittest suite, existing FastAPI response model.

---

## File Structure

- Create: `src/ocr_quality/occlusion.py` for occlusion candidates, result summary, and NumPy/Pillow analyzer.
- Create: `tests/test_occlusion.py` for synthetic image analyzer tests.
- Modify: `src/ocr_quality/config.py` to add externalized occlusion thresholds.
- Modify: `config/quality.default.json` to expose default occlusion thresholds.
- Modify: `src/ocr_quality/models.py` to add reason codes and page-level `occlusionSummary`.
- Modify: `src/ocr_quality/decision.py` to consume `OcclusionResult`, add warning reason codes, and allow occlusion to force OCR probe.
- Modify: `src/ocr_quality/pipeline.py` to instantiate and call `OcclusionAnalyzer`.
- Modify: `src/ocr_quality/web/static/index.html` to show `occlusionSummary` in page metrics.
- Modify: `tests/test_decision_engine.py` and `tests/test_quality_pipeline.py` for integration behavior.

Do not modify PaddleOCR adapter code, PDF rendering code, file inspection code, or startup scripts for this feature.

## Task 1: Configuration and Reason Codes

**Files:**
- Modify: `src/ocr_quality/config.py`
- Modify: `config/quality.default.json`
- Modify: `src/ocr_quality/models.py`
- Test: `tests/test_config.py` if present, otherwise create `tests/test_occlusion.py`

- [ ] **Step 1: Write failing config and reason-code tests**

Create `tests/test_occlusion.py` with:

```python
import unittest

from ocr_quality.config import QualityConfig
from ocr_quality.models import ReasonCode


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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_occlusion -v`

Expected: FAIL with missing `QualityConfig.enable_occlusion_detection` or missing `ReasonCode.HAND_OCCLUSION_RISK`.

- [ ] **Step 3: Add minimal config and reason codes**

Add to `QualityConfig`:

```python
enable_occlusion_detection: bool = True
occlusion_min_area_ratio: float = 0.01
occlusion_min_text_overlap_ratio: float = 0.08
occlusion_min_affected_box_ratio: float = 0.10
occlusion_min_box_overlap_ratio: float = 0.35
occlusion_text_box_padding_ratio: float = 0.25
occlusion_skin_min_red: int = 95
occlusion_skin_min_red_green_delta: int = 12
occlusion_skin_min_red_blue_delta: int = 20
```

Add matching JSON keys to `config/quality.default.json`.

Add to `ReasonCode`:

```python
HAND_OCCLUSION_RISK = "HAND_OCCLUSION_RISK"
TEXT_OCCLUSION_RISK = "TEXT_OCCLUSION_RISK"
```

Add clear Chinese `USER_MESSAGES` entries for both reason codes.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_occlusion -v`

Expected: PASS.

## Task 2: Occlusion Analyzer

**Files:**
- Create: `src/ocr_quality/occlusion.py`
- Modify: `tests/test_occlusion.py`

- [ ] **Step 1: Add failing analyzer tests**

Append to `tests/test_occlusion.py`:

```python
from PIL import Image, ImageDraw

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_occlusion -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'ocr_quality.occlusion'`.

- [ ] **Step 3: Implement minimal analyzer**

Create `src/ocr_quality/occlusion.py` with dataclasses `OcclusionCandidate`, `OcclusionResult`, and class `OcclusionAnalyzer`. Implement:

```python
def analyze(self, image: Image.Image, text: TextDetectionResult) -> OcclusionResult:
    if not self.config.enable_occlusion_detection:
        return OcclusionResult.empty("disabled")
    if not text.boxes:
        return OcclusionResult.empty("no-text")
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    skin_mask = self._skin_mask(rgb)
    generic_mask = self._generic_obstruction_mask(rgb)
    text_mask = self._text_mask(image.size, text.boxes)
    return self._summarize(skin_mask, generic_mask, text_mask, text.boxes, image.size)
```

Use a small stack-based or BFS connected component helper over boolean masks. Filter components with:

```python
area_ratio >= config.occlusion_min_area_ratio
width >= 8
height >= 8
```

Use expanded text boxes where padding is `box_height * occlusion_text_box_padding_ratio`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_occlusion -v`

Expected: PASS.

## Task 3: Result Contract and Decision Engine

**Files:**
- Modify: `src/ocr_quality/models.py`
- Modify: `src/ocr_quality/decision.py`
- Modify: `tests/test_decision_engine.py`

- [ ] **Step 1: Add failing decision tests**

Add to `tests/test_decision_engine.py`:

```python
from ocr_quality.occlusion import OcclusionResult


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
```

Add tests:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_decision_engine -v`

Expected: FAIL because `page_decision()` does not accept `occlusion`.

- [ ] **Step 3: Add result and decision support**

Add `occlusionSummary: dict[str, Any] | None = None` to `PageResult`, include it in `to_dict()`.

Change `DecisionEngine.page_decision()` signature to:

```python
def page_decision(..., timings_ms: dict[str, float], occlusion: OcclusionResult | None = None) -> PageResult:
```

Add risks:

```python
if occlusion is not None:
    if occlusion.has_hand_occlusion:
        risks.append(ReasonCode.HAND_OCCLUSION_RISK)
    if occlusion.has_text_occlusion:
        risks.append(ReasonCode.TEXT_OCCLUSION_RISK)
```

Pass `occlusionSummary=occlusion.to_summary() if occlusion else None` to all `PageResult` constructors.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_decision_engine -v`

Expected: PASS.

## Task 4: Pipeline Wiring and Probe Trigger

**Files:**
- Modify: `src/ocr_quality/pipeline.py`
- Modify: `src/ocr_quality/decision.py`
- Modify: `tests/test_quality_pipeline.py`

- [ ] **Step 1: Add failing pipeline test**

In `tests/test_quality_pipeline.py`, add a fake occlusion analyzer and fake probe test that asserts OCR probe executes when occlusion risk exists even if ordinary risk count is below threshold.

Use:

```python
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
```

Inject it into `pipeline.occlusion_analyzer` after construction, run a clear text-like image, and assert:

```python
self.assertTrue(result["ocrProbeExecuted"])
self.assertIn("HAND_OCCLUSION_RISK", result["reasonCodes"])
self.assertIsNotNone(result["pages"][0]["occlusionSummary"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_quality_pipeline -v`

Expected: FAIL because `QualityPipeline` has no `occlusion_analyzer`.

- [ ] **Step 3: Wire analyzer and probe trigger**

In `QualityPipeline.__init__`:

```python
from .occlusion import OcclusionAnalyzer
self.occlusion_analyzer = OcclusionAnalyzer(self.config)
```

In `_evaluate_image()` after text region metrics:

```python
started = time.perf_counter()
occlusion = self.occlusion_analyzer.analyze(analysis_image, text)
page_timings["occlusionDetection"] = self._elapsed(started)
```

Probe condition:

```python
should_probe = self.decision_engine.should_probe(metrics, text, occlusion)
```

Update `should_probe()` signature to accept `occlusion` and return true when `occlusion.has_hand_occlusion or occlusion.has_text_occlusion`.

Pass `occlusion` into `page_decision()`.

- [ ] **Step 4: Run pipeline tests**

Run: `python -m unittest tests.test_quality_pipeline tests.test_decision_engine tests.test_occlusion -v`

Expected: PASS.

## Task 5: Frontend and Full Regression

**Files:**
- Modify: `src/ocr_quality/web/static/index.html`
- Test: `tests/test_web_app.py`

- [ ] **Step 1: Add failing web/static test**

Add to `tests/test_web_app.py`:

```python
    def test_frontend_renders_occlusion_summary_fields(self):
        html = (ROOT / "src" / "ocr_quality" / "web" / "static" / "index.html").read_text(encoding="utf-8")

        self.assertIn("occlusionSummary", html)
        self.assertIn("textOverlapRatio", html)
        self.assertIn("affectedTextBoxRatio", html)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_web_app -v`

Expected: FAIL because frontend does not reference `occlusionSummary`.

- [ ] **Step 3: Render occlusion summary in page table**

In `index.html`, read:

```javascript
const occlusion = page.occlusionSummary || {};
```

Add to the page table cell:

```javascript
遮挡=${occlusion.textOverlapRatio ?? "-"}<br>影响框=${occlusion.affectedTextBoxRatio ?? "-"}
```

- [ ] **Step 4: Run full regression**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -v
python -m compileall -q src tests start.py
```

Expected: all tests PASS, compileall has no output and exit code 0.

## Acceptance Checklist

- [ ] Normal text pages do not report occlusion.
- [ ] Skin-colored regions overlapping text report `HAND_OCCLUSION_RISK`.
- [ ] Skin-colored regions away from text do not report occlusion.
- [ ] Non-skin obstruction overlapping text reports `TEXT_OCCLUSION_RISK`.
- [ ] Occlusion risk forces OCR probe when OCR probe is enabled.
- [ ] Occlusion alone returns `ACCEPT_WITH_WARNINGS`, not `REJECT`.
- [ ] Occlusion plus real low-confidence OCR probe returns `REJECT`.
- [ ] Page result includes `occlusionSummary`.
- [ ] File-level `detectedRisks` includes occlusion risk codes.
- [ ] Thresholds are configurable and not hard-coded in decision logic.
