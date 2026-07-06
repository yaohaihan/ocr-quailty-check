from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import QualityConfig
from .text_detection import TextDetectionResult


@dataclass
class OcclusionCandidate:
    x0: int
    y0: int
    x1: int
    y1: int
    area: int
    hand_like: bool


@dataclass
class OcclusionResult:
    candidate_count: int
    hand_like_candidate_count: int
    occlusion_area_ratio: float
    text_overlap_ratio: float
    affected_text_box_ratio: float
    max_box_overlap_ratio: float
    has_hand_occlusion: bool
    has_text_occlusion: bool
    source: str

    @classmethod
    def empty(cls, source: str) -> "OcclusionResult":
        return cls(
            candidate_count=0,
            hand_like_candidate_count=0,
            occlusion_area_ratio=0.0,
            text_overlap_ratio=0.0,
            affected_text_box_ratio=0.0,
            max_box_overlap_ratio=0.0,
            has_hand_occlusion=False,
            has_text_occlusion=False,
            source=source,
        )

    def to_summary(self) -> dict[str, float | int | bool | str]:
        return {
            "candidateCount": self.candidate_count,
            "handLikeCandidateCount": self.hand_like_candidate_count,
            "occlusionAreaRatio": round(self.occlusion_area_ratio, 4),
            "textOverlapRatio": round(self.text_overlap_ratio, 4),
            "affectedTextBoxRatio": round(self.affected_text_box_ratio, 4),
            "maxBoxOverlapRatio": round(self.max_box_overlap_ratio, 4),
            "hasHandOcclusion": self.has_hand_occlusion,
            "hasTextOcclusion": self.has_text_occlusion,
            "source": self.source,
        }


class OcclusionAnalyzer:
    def __init__(self, config: QualityConfig):
        self.config = config

    def analyze(self, image: Image.Image, text: TextDetectionResult) -> OcclusionResult:
        if not self.config.enable_occlusion_detection:
            return OcclusionResult.empty("disabled")
        if not text.boxes:
            return OcclusionResult.empty("no-text")

        rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
        skin_mask = self._skin_mask(rgb)
        generic_mask = self._generic_obstruction_mask(rgb) & ~skin_mask
        text_mask = self._text_mask(image.size, text.boxes)
        return self._summarize(skin_mask, generic_mask, text_mask, text.boxes, image.size)

    def _skin_mask(self, rgb: np.ndarray) -> np.ndarray:
        red = rgb[:, :, 0]
        green = rgb[:, :, 1]
        blue = rgb[:, :, 2]
        return (
            (red >= self.config.occlusion_skin_min_red)
            & ((red - green) >= self.config.occlusion_skin_min_red_green_delta)
            & ((red - blue) >= self.config.occlusion_skin_min_red_blue_delta)
            & (green >= 40)
            & (blue >= 20)
        )

    def _generic_obstruction_mask(self, rgb: np.ndarray) -> np.ndarray:
        gray = np.mean(rgb, axis=2)
        spread = np.max(rgb, axis=2) - np.min(rgb, axis=2)
        return (gray >= 35) & (gray <= 210) & (spread <= 55)

    def _text_mask(self, size: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
        width, height = size
        mask = np.zeros((height, width), dtype=bool)
        for x0, y0, x1, y1 in boxes:
            ex0, ey0, ex1, ey1 = self._expand_box((x0, y0, x1, y1), width, height)
            mask[ey0:ey1, ex0:ex1] = True
        return mask

    def _expand_box(self, box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = box
        pad = max(1, int((y1 - y0) * self.config.occlusion_text_box_padding_ratio))
        return (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(width, x1 + pad),
            min(height, y1 + pad),
        )

    def _summarize(
        self,
        skin_mask: np.ndarray,
        generic_mask: np.ndarray,
        text_mask: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        size: tuple[int, int],
    ) -> OcclusionResult:
        width, height = size
        page_area = width * height
        candidates: list[tuple[OcclusionCandidate, np.ndarray]] = []
        candidates.extend(self._components(skin_mask, hand_like=True, page_area=page_area))
        candidates.extend(self._components(generic_mask, hand_like=False, page_area=page_area))

        risk_candidates: list[tuple[OcclusionCandidate, np.ndarray]] = []
        has_hand = False
        has_text = False
        for candidate, mask in candidates:
            text_overlap, affected_ratio, max_box_overlap = self._overlap_metrics(mask, text_mask, boxes, size)
            overlaps_text = (
                text_overlap >= self.config.occlusion_min_text_overlap_ratio
                or affected_ratio >= self.config.occlusion_min_affected_box_ratio
                or max_box_overlap >= self.config.occlusion_min_box_overlap_ratio
            )
            if not overlaps_text:
                continue
            risk_candidates.append((candidate, mask))
            has_hand = has_hand or candidate.hand_like
            has_text = has_text or not candidate.hand_like

        if not risk_candidates:
            return OcclusionResult.empty("skin-color-rule")

        combined = np.zeros_like(text_mask, dtype=bool)
        area = 0
        hand_count = 0
        for candidate, mask in risk_candidates:
            combined |= mask
            area += candidate.area
            if candidate.hand_like:
                hand_count += 1
        text_overlap, affected_ratio, max_box_overlap = self._overlap_metrics(combined, text_mask, boxes, size)
        return OcclusionResult(
            candidate_count=len(risk_candidates),
            hand_like_candidate_count=hand_count,
            occlusion_area_ratio=area / float(page_area),
            text_overlap_ratio=text_overlap,
            affected_text_box_ratio=affected_ratio,
            max_box_overlap_ratio=max_box_overlap,
            has_hand_occlusion=has_hand,
            has_text_occlusion=has_text,
            source="skin-color-rule" if has_hand and not has_text else "generic-obstruction-rule",
        )

    def _components(
        self,
        mask: np.ndarray,
        hand_like: bool,
        page_area: int,
    ) -> list[tuple[OcclusionCandidate, np.ndarray]]:
        visited = np.zeros(mask.shape, dtype=bool)
        height, width = mask.shape
        results: list[tuple[OcclusionCandidate, np.ndarray]] = []
        min_area = max(1, int(page_area * self.config.occlusion_min_area_ratio))

        for start_y, start_x in np.argwhere(mask & ~visited):
            if visited[start_y, start_x]:
                continue
            pixels = []
            stack = [(int(start_y), int(start_x))]
            visited[start_y, start_x] = True
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(pixels) < min_area:
                continue
            ys = [point[0] for point in pixels]
            xs = [point[1] for point in pixels]
            x0, x1 = min(xs), max(xs) + 1
            y0, y1 = min(ys), max(ys) + 1
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            component_mask = np.zeros_like(mask, dtype=bool)
            component_mask[ys, xs] = True
            results.append((OcclusionCandidate(x0, y0, x1, y1, len(pixels), hand_like), component_mask))
        return results

    def _overlap_metrics(
        self,
        mask: np.ndarray,
        text_mask: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        size: tuple[int, int],
    ) -> tuple[float, float, float]:
        text_pixels = int(np.sum(text_mask))
        if text_pixels == 0:
            return 0.0, 0.0, 0.0
        overlap_pixels = int(np.sum(mask & text_mask))
        text_overlap_ratio = overlap_pixels / float(text_pixels)

        width, height = size
        affected = 0
        max_overlap = 0.0
        for box in boxes:
            x0, y0, x1, y1 = self._expand_box(box, width, height)
            box_area = max((x1 - x0) * (y1 - y0), 1)
            box_overlap = int(np.sum(mask[y0:y1, x0:x1])) / float(box_area)
            if box_overlap >= self.config.occlusion_min_box_overlap_ratio:
                affected += 1
            max_overlap = max(max_overlap, box_overlap)
        affected_ratio = affected / float(len(boxes)) if boxes else 0.0
        return text_overlap_ratio, affected_ratio, max_overlap
