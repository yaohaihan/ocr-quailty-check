# Development

## Test

Run the full regression suite:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

Run a compile check:

```powershell
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m compileall -q src tests
```

## Optional PaddleOCR Integration

PaddleOCR is optional at install time. When unavailable, the default service starts with fallback adapters and does not emit `OCR_PROBE_LOW_CONFIDENCE` from heuristic probe output.

Install optional PaddleOCR dependencies:

```powershell
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pip install ".[paddle]"
```

Run optional integration tests:

```powershell
$env:PYTHONPATH='src'
$env:OCR_QUALITY_RUN_PADDLE_INTEGRATION='1'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests/integration -v
```

## Benchmark

Run the benchmark helper:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ocr_quality.benchmark path\to\sample.png --repeat 10
```
