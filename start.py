from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DEFAULT_CONFIG = ROOT / "config" / "quality.default.json"
DEFAULT_PADDLEX_CACHE = ROOT / ".paddlex-cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the OCR Quality FastAPI service and frontend.")
    parser.add_argument("--host", default=os.environ.get("OCR_QUALITY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("OCR_QUALITY_PORT", "8000")))
    parser.add_argument("--config", default=os.environ.get("OCR_QUALITY_CONFIG"))
    parser.add_argument("--no-browser", action="store_true", help="Do not open the frontend page.")
    return parser.parse_args()


def configure_environment(config: str | None) -> None:
    os.environ["PYTHONPATH"] = str(SRC)
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(DEFAULT_PADDLEX_CACHE))
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
    os.environ.setdefault("PADDLE_PDX_CPU_NUM_THREADS", "4")

    if config:
        os.environ["OCR_QUALITY_CONFIG"] = config
    elif DEFAULT_CONFIG.exists():
        os.environ["OCR_QUALITY_CONFIG"] = str(DEFAULT_CONFIG)


def main() -> None:
    args = parse_args()
    configure_environment(args.config)
    url = f"http://{args.host}:{args.port}/"

    print("Starting OCR Quality service...")
    print(f"URL: {url}")
    print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")
    print(f"PADDLE_PDX_CACHE_HOME: {os.environ['PADDLE_PDX_CACHE_HOME']}")
    print(f"PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT: {os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT']}")
    print(f"PADDLE_PDX_CPU_NUM_THREADS: {os.environ['PADDLE_PDX_CPU_NUM_THREADS']}")
    if os.environ.get("OCR_QUALITY_CONFIG"):
        print(f"OCR_QUALITY_CONFIG: {os.environ['OCR_QUALITY_CONFIG']}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        webbrowser.open(url)

    uvicorn.run("ocr_quality.web.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
