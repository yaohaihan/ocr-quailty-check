from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import QualityConfig
from .text_detection import TextDetectionResult


@dataclass
class TextRegionSummary:
    box_count: int
    coverage_ratio: float
    average_height: float
    median_height: float
    small_text_ratio: float
    low_sharpness_ratio: float
    border_touch_ratio: float
    top_box_ratio: float
    middle_box_ratio: float
    bottom_box_ratio: float

    def to_dict(self) -> dict:
        return {
            "boxCount": self.box_count,
            "coverageRatio": round(self.coverage_ratio, 6),
            "averageHeight": round(self.average_height, 4),
            "medianHeight": round(self.median_height, 4),
            "smallTextRatio": round(self.small_text_ratio, 4),
            "lowSharpnessRatio": round(self.low_sharpness_ratio, 4),
            "borderTouchRatio": round(self.border_touch_ratio, 4),
            "topBoxRatio": round(self.top_box_ratio, 4),
            "middleBoxRatio": round(self.middle_box_ratio, 4),
            "bottomBoxRatio": round(self.bottom_box_ratio, 4),
        }


class TextRegionAnalyzer:
    def __init__(self, config: QualityConfig):
        self.config = config

    def analyze(self, image: Image.Image, text: TextDetectionResult) -> TextRegionSummary:
        width, height = image.size
        boxes = text.boxes
        if not boxes:
            return TextRegionSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        box_heights = [y1 - y0 for _, y0, _, y1 in boxes]
        areas = [(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes]
        small = [h for h in box_heights if h < self.config.min_text_median_height]
        border = [box for box in boxes if box[0] <= 3 or box[1] <= 3 or box[2] >= width - 3 or box[3] >= height - 3]
        centers = [((y0 + y1) / 2.0) / height for _, y0, _, y1 in boxes]
        top = [c for c in centers if c < 0.33]
        middle = [c for c in centers if 0.33 <= c < 0.66]
        bottom = [c for c in centers if c >= 0.66]
        count = len(boxes)
        return TextRegionSummary(
            box_count=count,
            coverage_ratio=sum(areas) / float(width * height),
            average_height=float(np.mean(box_heights)),
            median_height=float(np.median(box_heights)),
            small_text_ratio=len(small) / count,
            low_sharpness_ratio=text.low_sharpness_ratio,
            border_touch_ratio=len(border) / count,
            top_box_ratio=len(top) / count,
            middle_box_ratio=len(middle) / count,
            bottom_box_ratio=len(bottom) / count,
        )

