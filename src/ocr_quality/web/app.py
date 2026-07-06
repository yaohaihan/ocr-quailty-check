from __future__ import annotations

import tempfile
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from ocr_quality import QualityPipeline
from ocr_quality.config_loader import load_quality_config


STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(pipeline: QualityPipeline | None = None) -> FastAPI:
    config_path = os.environ.get("OCR_QUALITY_CONFIG")
    config = pipeline.config if pipeline is not None else load_quality_config(config_path)
    quality_pipeline = pipeline
    app = FastAPI(
        title="OCR Quality Service",
        description="Generic OCR image-material quality checks.",
        version="0.1.0",
    )

    def get_pipeline() -> QualityPipeline:
        nonlocal quality_pipeline
        if quality_pipeline is None:
            quality_pipeline = QualityPipeline(config=config)
        return quality_pipeline

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "thresholdConfigVersion": config.version,
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/quality/config")
    def quality_config() -> dict:
        config_dict = asdict(config)
        return {
            "version": config_dict.pop("version"),
            "supportedExtensions": list(config_dict.pop("supported_extensions")),
            "maxPages": config_dict.pop("max_pages"),
            "maxFileBytes": config_dict.pop("max_file_bytes"),
            "aggregationPolicy": config_dict.pop("aggregation_policy"),
            "features": {
                "orientationDetection": config.enable_orientation_detection,
                "textDetection": config.enable_text_detection,
                "ocrProbe": config.enable_ocr_probe,
            },
            "model": {
                "orientationDetectorProvider": config.orientation_detector_provider,
                "textDetectorProvider": config.text_detector_provider,
                "ocrProbeProvider": config.ocr_probe_provider,
                "useGpu": config.paddle_use_gpu,
                "lang": config.paddle_lang,
            },
            "thresholds": config_dict,
        }

    @app.post("/api/quality/check")
    async def check_quality(file: UploadFile = File(...)) -> Any:
        suffix = Path(file.filename or "").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            temp_path = Path(handle.name)
            while chunk := await file.read(1024 * 1024):
                handle.write(chunk)
        try:
            active_pipeline = get_pipeline()
            return active_pipeline.evaluate(temp_path)
        except Exception as exc:
            active_pipeline = quality_pipeline
            return JSONResponse(
                status_code=500,
                content={
                    "accepted": False,
                    "decision": "REJECT",
                    "ocrReadinessScore": 0.0,
                    "reasonCodes": ["INTERNAL_ERROR"],
                    "userMessages": ["质检服务内部错误，请查看服务端日志"],
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
                    "thresholdConfigVersion": config.version,
                    "modelVersions": getattr(active_pipeline, "model_versions", {}),
                    "ocrProbeExecuted": False,
                    "aggregationPolicy": config.aggregation_policy,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        finally:
            temp_path.unlink(missing_ok=True)

    return app


app = create_app()
