from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass
class TextDetectionResult:
    boxes: list[tuple[int, int, int, int]]
    coverage_ratio: float
    median_height: float
    low_sharpness_ratio: float
    border_touch_ratio: float

    def to_summary(self) -> dict[str, float | int]:
        return {
            "boxCount": len(self.boxes),
            "coverageRatio": round(self.coverage_ratio, 6),
            "medianHeight": round(self.median_height, 4),
            "lowSharpnessRatio": round(self.low_sharpness_ratio, 4),
            "borderTouchRatio": round(self.border_touch_ratio, 4),
        }


class TextDetector(Protocol):
    def detect(self, image: Image.Image) -> TextDetectionResult:
        ...


class DisabledTextDetector:
    def detect(self, image: Image.Image) -> TextDetectionResult:
        return TextDetectionResult([], 0.0, 0.0, 0.0, 0.0)


class HeuristicTextDetector:
    def detect(self, image: Image.Image) -> TextDetectionResult:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        mask = gray < 120.0
        height, width = mask.shape
        row_counts = np.sum(mask, axis=1)
        boxes: list[tuple[int, int, int, int]] = []
        in_run = False
        start = 0
        threshold = max(5, int(width * 0.01))
        for idx, count in enumerate(row_counts):
            if count >= threshold and not in_run:
                start = idx
                in_run = True
            elif count < threshold and in_run:
                self._append_box(mask, start, idx, boxes)
                in_run = False
        if in_run:
            self._append_box(mask, start, height, boxes)

        if not boxes:
            return TextDetectionResult([], 0.0, 0.0, 0.0, 0.0)
        area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes)
        heights = [y1 - y0 for _, y0, _, y1 in boxes]
        border = [b for b in boxes if b[0] <= 3 or b[1] <= 3 or b[2] >= width - 3 or b[3] >= height - 3]
        return TextDetectionResult(
            boxes=boxes,
            coverage_ratio=area / float(width * height),
            median_height=float(np.median(heights)),
            low_sharpness_ratio=0.0,
            border_touch_ratio=len(border) / float(len(boxes)),
        )

    def _append_box(self, mask: np.ndarray, y0: int, y1: int, boxes: list[tuple[int, int, int, int]]) -> None:
        segment = mask[y0:y1, :]
        cols = np.where(np.any(segment, axis=0))[0]
        if cols.size == 0:
            return
        x0 = int(cols[0])
        x1 = int(cols[-1]) + 1
        if y1 - y0 >= 3 and x1 - x0 >= 8:
            boxes.append((x0, y0, x1, y1))
