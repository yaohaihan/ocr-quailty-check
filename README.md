# OCR Quality

Generic, low-latency image-material quality checks for OCR readiness.

## Overview

OCR Quality is a Python/FastAPI service for checking whether uploaded images or PDF pages are suitable for downstream OCR text detection, recognition, and structured parsing.

The current implementation is intentionally small and explainable:

- central threshold configuration with version `ocr-quality-v1`
- stable file-level and page-level result objects
- hard failures for reliable conditions
- risk codes that do not reject by themselves
- injectable text detection and OCR probe adapters
- CLI JSON output
- FastAPI service with a simple upload page

## Start

Start the API, frontend page, and quality pipeline together:

```powershell
python start.py
```

The startup script launches the algorithm API and the frontend page together at <http://127.0.0.1:8000/>. Set `OCR_QUALITY_PORT` before running the script if port `8000` is already in use.

## Documentation

- [Usage](docs/usage.md): startup, frontend, CLI, and manual service commands.
- [API](docs/api.md): HTTP endpoints and request formats.
- [Development](docs/development.md): test, compile, optional dependency, and benchmark commands.
- [Architecture](docs/architecture.md): current module boundaries and planned PaddleOCR adapter direction.
- [Design](docs/superpowers/specs/2026-06-26-ocr-quality-design.md): original quality-gate design.
- [Implementation Plan](docs/superpowers/plans/2026-06-30-ocr-quality-paddle.md): confirmed PaddleOCR implementation plan.

## Quick Commands

Use an explicit config file:

```powershell
$env:OCR_QUALITY_CONFIG='config\quality.default.json'
python start.py
```

Run tests with the bundled Python used by this workspace:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Run the CLI quality check:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ocr_quality.cli path\to\page.png
```

Optional PaddleOCR integration and benchmark commands are documented in [Development](docs/development.md).
