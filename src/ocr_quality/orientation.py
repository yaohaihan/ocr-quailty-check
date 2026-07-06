from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from .models import AutoFixRecord


@dataclass
class OrientationResult:
    angle: int
    confidence: float
    model_version: str


class OrientationDetector(Protocol):
    def detect(self, image: Image.Image) -> OrientationResult:
        ...


class DisabledOrientationDetector:
    def detect(self, image: Image.Image) -> OrientationResult:
        return OrientationResult(angle=0, confidence=1.0, model_version="disabled")


class FixedOrientationDetector:
    def __init__(self, result: OrientationResult):
        self.result = result

    def detect(self, image: Image.Image) -> OrientationResult:
        return self.result


def apply_orientation_correction(
    image: Image.Image,
    result: OrientationResult,
    min_confidence: float = 0.8,
) -> tuple[Image.Image, AutoFixRecord | None]:
    if result.confidence < min_confidence or result.angle % 360 == 0:
        return image, None
    corrected = image.rotate(-result.angle, expand=True)
    fix = AutoFixRecord(
        fixType="ORIENTATION_CORRECTED",
        before={"angle": result.angle, "confidence": result.confidence, "size": image.size},
        after={"angle": 0, "size": corrected.size},
        recomputed=True,
    )
    return corrected, fix

