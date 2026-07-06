from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ocr_quality.config import QualityConfig
from ocr_quality.ocr_probe import OcrProbeResult
from ocr_quality.text_detection import TextDetectionResult


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
        return cls.from_module(config, module)

    @classmethod
    def from_module(cls, config: QualityConfig, module: Any) -> "PaddleOcrBundle":
        kwargs = {
            "lang": config.paddle_lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": bool(config.enable_orientation_detection),
        }
        if config.paddle_det_model_dir:
            kwargs["text_detection_model_dir"] = config.paddle_det_model_dir
        if config.paddle_rec_model_dir:
            kwargs["text_recognition_model_dir"] = config.paddle_rec_model_dir
        paddle = module.PaddleOCR(
            **kwargs,
        )
        return cls(config=config, paddle=paddle)

    def warmup(self) -> None:
        image = Image.new("RGB", (64, 64), "white")
        _predict(self.paddle, image, det=True, rec=True)
        self.warmed = True

    def model_versions(self) -> dict[str, str]:
        return {"paddleocr": type(self.paddle).__name__}


class PaddleTextDetector:
    def __init__(self, paddle):
        self.paddle = paddle

    def detect(self, image: Image.Image) -> TextDetectionResult:
        raw = _predict(self.paddle, image, det=True, rec=False)
        boxes = _extract_boxes(raw)
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
            raw = _predict(self.paddle, crop, det=False, rec=True)
            texts = _extract_recognition(raw)
            if not texts:
                empty += 1
                continue
            text, confidence = texts[0]
            if not text:
                empty += 1
            confidences.append(float(confidence))
        sample_count = len(boxes)
        if sample_count == 0:
            return OcrProbeResult(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, sample_count=0, source="paddleocr")
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


def _predict(paddle: Any, image: Image.Image, det: bool, rec: bool) -> Any:
    if hasattr(paddle, "predict"):
        return paddle.predict(np.asarray(image.convert("RGB")))
    return paddle.ocr(image, cls=True, det=det, rec=rec)


def _extract_boxes(raw: Any) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for page in _as_pages(raw):
        if isinstance(page, dict):
            rec_boxes = page.get("rec_boxes")
            if rec_boxes is not None and len(rec_boxes) > 0:
                for box in rec_boxes:
                    boxes.append(_box_to_tuple(box))
                continue
            for points in page.get("dt_polys") or page.get("rec_polys") or []:
                boxes.append(_points_to_box(points))
            continue
        for item in page or []:
            if not item:
                continue
            points = item if _looks_like_points(item) else item[0]
            boxes.append(_points_to_box(points))
    return boxes


def _extract_recognition(raw: Any) -> list[tuple[str, float]]:
    texts: list[tuple[str, float]] = []
    for page in _as_pages(raw):
        if isinstance(page, dict):
            rec_texts = page.get("rec_texts") or []
            rec_scores = page.get("rec_scores") or []
            for text, score in zip(rec_texts, rec_scores):
                texts.append((str(text), float(score)))
            continue
        for item in page or []:
            candidate = item[0] if isinstance(item, list) and len(item) == 1 else item
            if isinstance(candidate, tuple) and len(candidate) == 2:
                texts.append((str(candidate[0]), float(candidate[1])))
            elif isinstance(candidate, list) and len(candidate) >= 2 and isinstance(candidate[1], (float, int)):
                texts.append((str(candidate[0]), float(candidate[1])))
            elif isinstance(item, list) and len(item) >= 2 and isinstance(item[1], tuple):
                texts.append((str(item[1][0]), float(item[1][1])))
    return texts


def _as_pages(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return raw
    return [raw]


def _points_to_box(points: Any) -> tuple[int, int, int, int]:
    xs = [int(point[0]) for point in points]
    ys = [int(point[1]) for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _box_to_tuple(box: Any) -> tuple[int, int, int, int]:
    values = np.asarray(box).reshape(-1).tolist()
    if len(values) < 4:
        raise ValueError("PaddleOCR box must contain at least four values")
    return (int(values[0]), int(values[1]), int(values[2]), int(values[3]))


def _looks_like_points(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return (
        isinstance(value, list)
        and len(value) >= 4
        and isinstance(value[0], (list, tuple))
        and len(value[0]) >= 2
        and isinstance(value[0][0], (int, float))
    )
