# Architecture

## Current Shape

The project is a Python package with a FastAPI service and a small frontend page.

Core modules:

- `src/ocr_quality/file_inspector.py`: file type, file size, and PDF processability checks.
- `src/ocr_quality/normalizer.py`: image loading, EXIF transpose, downscaling, and PDF rendering.
- `src/ocr_quality/image_metrics.py`: Pillow/NumPy quality metrics.
- `src/ocr_quality/text_detection.py`: text-detection protocol plus the current heuristic detector.
- `src/ocr_quality/ocr_probe.py`: OCR probe protocol plus the current heuristic probe.
- `src/ocr_quality/decision.py`: hard failure, warning, and aggregation decisions.
- `src/ocr_quality/pipeline.py`: end-to-end orchestration.
- `src/ocr_quality/web/app.py`: FastAPI app and upload API.

## Important Current Limitation

The current `HeuristicTextDetector` and `HeuristicOcrProbe` are not pretrained OCR components. They are lightweight placeholders around image thresholds and text-box counts. Because of that, `OCR_PROBE_LOW_CONFIDENCE` can be triggered by heuristic conditions rather than real OCR recognition confidence.

The current implementation now makes this safer:

- PaddleOCR is allowed as the preferred pretrained OCR component.
- PaddleOCR must be isolated behind adapters.
- `DecisionEngine` must not import or depend on PaddleOCR directly.
- `OCR_PROBE_LOW_CONFIDENCE` should only be emitted from a real OCR probe adapter result.
 - When PaddleOCR is not installed, the service records an unavailable model version and uses fallback adapters.

## Planned Direction

The confirmed implementation plan is:

[OCR Quality PaddleOCR Implementation Plan](superpowers/plans/2026-06-30-ocr-quality-paddle.md)
