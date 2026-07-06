# OCR Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable Python OCR image-material quality gate with a CLI, centralized thresholds, stable result objects, layered checks, and tests.

**Architecture:** The implementation is a small Python package under `src/ocr_quality`. A `QualityPipeline` coordinates file inspection, page normalization, fast image analysis, pluggable text detection, optional OCR probing, decisioning, and page aggregation. Dependencies are limited to the standard library, Pillow, and NumPy because the repository has no existing stack and the bundled runtime does not include OpenCV or pytest.

**Tech Stack:** Python 3, `unittest`, Pillow, NumPy.

---

### Task 1: Contract and Decision Tests

**Files:**
- Create: `tests/test_quality_pipeline.py`
- Create: `src/ocr_quality/models.py`
- Create: `src/ocr_quality/config.py`
- Create: `src/ocr_quality/decision.py`

- [x] **Step 1: Write failing tests for result shape, hard failures, risk handling, and aggregation.**
- [x] **Step 2: Run tests and confirm failure because `ocr_quality` does not exist.**
- [x] **Step 3: Implement dataclasses, config defaults, decision rules, and aggregation.**
- [x] **Step 4: Run tests and confirm they pass.**

### Task 2: Image Analysis and File Inspection

**Files:**
- Create: `src/ocr_quality/file_inspector.py`
- Create: `src/ocr_quality/image_metrics.py`
- Create: `src/ocr_quality/normalizer.py`

- [x] **Step 1: Write failing tests for unsupported files, corrupted images, blank pages, low resolution, exposure, blur, and auto-fix records.**
- [x] **Step 2: Run tests and confirm failure due missing behavior.**
- [x] **Step 3: Implement file inspection, Pillow decoding, EXIF transpose, grayscale conversion, downscaling, metrics, and light auto-fix detection.**
- [x] **Step 4: Run tests and confirm they pass.**

### Task 3: Pipeline, CLI, Docs, and Verification

**Files:**
- Create: `src/ocr_quality/pipeline.py`
- Create: `src/ocr_quality/text_detection.py`
- Create: `src/ocr_quality/ocr_probe.py`
- Create: `src/ocr_quality/cli.py`
- Create: `src/ocr_quality/__init__.py`
- Create: `pyproject.toml`
- Create: `README.md`

- [x] **Step 1: Write failing tests for full pipeline output, OCR probe triggering, CLI JSON, and config version.**
- [x] **Step 2: Run tests and confirm failure due missing behavior.**
- [x] **Step 3: Implement the pipeline, injectable adapters, CLI, package metadata, and README.**
- [x] **Step 4: Run full test suite and compile check.**

