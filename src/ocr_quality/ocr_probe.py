from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from .text_detection import TextDetectionResult


@dataclass
class OcrProbeResult:
    success_ratio: float
    empty_ratio: float
    average_confidence: float
    median_confidence: float
    low_confidence_ratio: float
    effective_character_ratio: float
    sample_count: int = 0
    source: str = "unknown"

    def to_summary(self) -> dict[str, float]:
        return {
            "successRatio": round(self.success_ratio, 4),
            "emptyRatio": round(self.empty_ratio, 4),
            "averageConfidence": round(self.average_confidence, 4),
            "medianConfidence": round(self.median_confidence, 4),
            "lowConfidenceRatio": round(self.low_confidence_ratio, 4),
            "effectiveCharacterRatio": round(self.effective_character_ratio, 4),
            "sampleCount": self.sample_count,
            "source": self.source,
        }


class OcrProbe(Protocol):
    def probe(self, image: Image.Image, text_detection: TextDetectionResult) -> OcrProbeResult:
        ...


class DisabledOcrProbe:
    def probe(self, image: Image.Image, text_detection: TextDetectionResult) -> OcrProbeResult:
        return OcrProbeResult(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, sample_count=0, source="disabled")


class HeuristicOcrProbe:
    def probe(self, image: Image.Image, text_detection: TextDetectionResult) -> OcrProbeResult:
        box_count = len(text_detection.boxes)
        if box_count == 0:
            confidence = 0.0
            empty = 1.0
        elif box_count < 2 or text_detection.coverage_ratio < 0.003:
            confidence = 0.35
            empty = 0.75
        else:
            confidence = 0.75
            empty = 0.0
        return OcrProbeResult(
            success_ratio=1.0 - empty,
            empty_ratio=empty,
            average_confidence=confidence,
            median_confidence=confidence,
            low_confidence_ratio=1.0 if confidence < 0.6 else 0.0,
            effective_character_ratio=max(0.0, confidence - 0.1),
            sample_count=box_count,
            source="heuristic",
        )
