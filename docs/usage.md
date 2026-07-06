# Usage

## One-Command Startup

Start the FastAPI service, algorithm pipeline, and frontend page together:

```powershell
python start.py
```

The frontend is available at:

```text
http://127.0.0.1:8000/
```

Use a different port when `8000` is already occupied:

```powershell
$env:OCR_QUALITY_PORT=8010
python start.py
```

Use an explicit quality configuration:

```powershell
$env:OCR_QUALITY_CONFIG='config\quality.default.json'
python start.py
```

The startup script sets `PADDLE_PDX_CACHE_HOME` to `.paddlex-cache` under the project root unless you have already set it. PaddleOCR/PaddleX model files are downloaded there so the service does not depend on a user-profile cache with incompatible permissions.

The startup script also sets `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False` by default. This keeps PaddleOCR 3.x on the standard CPU Paddle backend instead of the PaddleX oneDNN/MKLDNN path, which can fail on some PaddlePaddle 3.x CPU builds with `ConvertPirAttribute2RuntimeAttribute` errors. Set the environment variable yourself before startup if you explicitly want to re-enable MKLDNN.

The default PaddleX CPU thread count is `PADDLE_PDX_CPU_NUM_THREADS=4`. Override it before startup when benchmarking on a larger or smaller machine.

Run without opening a browser:

```powershell
python start.py --no-browser
```

## Manual FastAPI Command

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m uvicorn ocr_quality.web.app:app --host 127.0.0.1 --port 8000
```

## CLI Quality Check

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\20399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ocr_quality.cli path\to\page.png
```
