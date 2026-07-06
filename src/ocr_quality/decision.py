from __future__ import annotations

from .config import QualityConfig
from .image_metrics import ImageMetrics
from .models import Decision, PageResult, ReasonCode, messages_for
from .occlusion import OcclusionResult
from .ocr_probe import OcrProbeResult
from .text_detection import TextDetectionResult


class DecisionEngine:
    def __init__(self, config: QualityConfig):
        self.config = config

    def page_decision(
        self,
        page_index: int,
        metrics: ImageMetrics,
        text: TextDetectionResult | None,
        ocr_probe: OcrProbeResult | None,
        timings_ms: dict[str, float],
        occlusion: OcclusionResult | None = None,
    ) -> PageResult:
        hard: list[ReasonCode] = []
        risks: list[ReasonCode] = []

        if metrics.width < self.config.min_width or metrics.height < self.config.min_height or metrics.pixels < self.config.min_pixels:
            hard.append(ReasonCode.PAGE_TOO_LOW_RESOLUTION)
        if metrics.std <= self.config.blank_std_threshold and metrics.edge_density <= self.config.blank_edge_density_threshold:
            hard.append(ReasonCode.PAGE_BLANK)
        if metrics.mean <= self.config.black_mean_threshold:
            hard.append(ReasonCode.PAGE_BLANK)
        exposure_abnormal = metrics.mean >= self.config.overexposed_mean_threshold or metrics.mean <= self.config.underexposed_mean_threshold
        if metrics.sharpness < self.config.severe_blur_threshold:
            hard.append(ReasonCode.SEVERE_BLUR)
        elif metrics.sharpness < self.config.blur_risk_threshold or metrics.low_quality_block_ratio > self.config.max_low_quality_block_ratio:
            risks.append(ReasonCode.LOCAL_BLUR_RISK)
        if getattr(metrics, "low_contrast", False):
            risks.append(ReasonCode.LOW_CONTRAST_RISK)
        if getattr(metrics, "compression_risk", 0.0) >= 0.8:
            risks.append(ReasonCode.COMPRESSION_ARTIFACT_RISK)

        if text is not None:
            if self.config.require_text and len(text.boxes) < self.config.min_text_boxes and not hard:
                hard.append(ReasonCode.NO_EFFECTIVE_TEXT_DETECTED)
            if text.coverage_ratio < self.config.min_text_coverage_ratio and not hard:
                risks.append(ReasonCode.NO_EFFECTIVE_TEXT_DETECTED)
            if 0 < text.median_height < self.config.min_text_median_height:
                risks.append(ReasonCode.TEXT_TOO_SMALL_RISK)
            if text.border_touch_ratio > self.config.max_border_touch_ratio:
                risks.append(ReasonCode.BORDER_CROP_RISK)
            if text.low_sharpness_ratio > self.config.max_text_low_sharpness_ratio:
                risks.append(ReasonCode.LOCAL_BLUR_RISK)
            text_insufficient = len(text.boxes) < self.config.min_text_boxes or text.coverage_ratio < self.config.min_text_coverage_ratio
        else:
            text_insufficient = True

        if occlusion is not None:
            if occlusion.has_hand_occlusion:
                risks.append(ReasonCode.HAND_OCCLUSION_RISK)
            if occlusion.has_text_occlusion:
                risks.append(ReasonCode.TEXT_OCCLUSION_RISK)

        if exposure_abnormal and text_insufficient:
            hard.append(ReasonCode.SEVERE_EXPOSURE_ABNORMAL)

        if ocr_probe is not None and ocr_probe.source != "heuristic" and ocr_probe.sample_count > 0:
            if (
                ocr_probe.average_confidence < self.config.ocr_probe_min_confidence
                or ocr_probe.empty_ratio > self.config.ocr_probe_max_empty_ratio
            ):
                hard.append(ReasonCode.OCR_PROBE_LOW_CONFIDENCE)

        hard = self._dedupe(hard)
        risks = self._dedupe(risks)
        score = self._score(metrics, risks, hard, ocr_probe)

        if hard:
            reason_codes = self._dedupe(hard + risks)
            return PageResult(
                pageIndex=page_index,
                accepted=False,
                decision=Decision.REJECT,
                ocrReadinessScore=score,
                reasonCodes=reason_codes,
                userMessages=messages_for(reason_codes),
                metrics=metrics.to_dict(),
                timingsMs=timings_ms,
                textDetectionSummary=text.to_summary() if text else None,
                occlusionSummary=occlusion.to_summary() if occlusion else None,
                ocrProbeSummary=ocr_probe.to_summary() if ocr_probe else None,
                ocrProbeExecuted=ocr_probe is not None,
            )
        if risks:
            return PageResult(
                pageIndex=page_index,
                accepted=True,
                decision=Decision.ACCEPT_WITH_WARNINGS,
                ocrReadinessScore=score,
                reasonCodes=risks,
                userMessages=messages_for(risks),
                metrics=metrics.to_dict(),
                timingsMs=timings_ms,
                textDetectionSummary=text.to_summary() if text else None,
                occlusionSummary=occlusion.to_summary() if occlusion else None,
                ocrProbeSummary=ocr_probe.to_summary() if ocr_probe else None,
                ocrProbeExecuted=ocr_probe is not None,
            )
        return PageResult(
            pageIndex=page_index,
            accepted=True,
            decision=Decision.ACCEPT,
            ocrReadinessScore=score,
            metrics=metrics.to_dict(),
            timingsMs=timings_ms,
            textDetectionSummary=text.to_summary() if text else None,
            occlusionSummary=occlusion.to_summary() if occlusion else None,
            ocrProbeSummary=ocr_probe.to_summary() if ocr_probe else None,
            ocrProbeExecuted=ocr_probe is not None,
        )

    def should_probe(
        self,
        metrics: ImageMetrics,
        text: TextDetectionResult,
        occlusion: OcclusionResult | None = None,
    ) -> bool:
        if occlusion is not None and (occlusion.has_hand_occlusion or occlusion.has_text_occlusion):
            return True
        risks = 0
        if metrics.sharpness < self.config.blur_risk_threshold:
            risks += 1
        if text.coverage_ratio < self.config.min_text_coverage_ratio:
            risks += 1
        if 0 < text.median_height < self.config.min_text_median_height:
            risks += 1
        if text.border_touch_ratio > self.config.max_border_touch_ratio:
            risks += 1
        if len(text.boxes) <= self.config.min_text_boxes:
            risks += 1
        return risks >= self.config.risk_count_for_probe

    def aggregate(self, pages: list[PageResult], timings_ms: dict[str, float], model_versions: dict[str, str] | None = None) -> dict:
        failed_pages = [page.pageIndex for page in pages if not page.accepted]
        failed_ratio = len(failed_pages) / max(len(pages), 1)
        if self.config.aggregation_policy == "ALLOW_PARTIAL":
            failed_ratio = len(failed_pages) / max(len(pages), 1)
            accepted = failed_ratio <= self.config.max_failed_page_ratio
        elif self.config.aggregation_policy == "FIRST_N_REQUIRED":
            required = pages[: self.config.first_n_required]
            accepted = all(page.accepted for page in required)
        else:
            accepted = not failed_pages

        any_warning = any(page.decision == Decision.ACCEPT_WITH_WARNINGS for page in pages)
        any_probe = any(page.ocrProbeExecuted for page in pages)
        reason_codes = self._dedupe([code for page in pages for code in page.reasonCodes])
        score = min((page.ocrReadinessScore for page in pages), default=0.0)
        if accepted and (failed_pages or any_warning):
            decision = Decision.ACCEPT_WITH_WARNINGS
        elif accepted:
            decision = Decision.ACCEPT
        else:
            decision = Decision.REJECT
        metrics = {"pageCount": len(pages), "failedPageCount": len(failed_pages), "failedPageRatio": failed_ratio}
        auto_fixes = [fix.to_dict() for page in pages for fix in page.autoFixes]
        page_dicts = [page.to_dict() for page in pages]
        risk_codes = [code.value for code in reason_codes if code.name.endswith("_RISK")]
        return {
            "accepted": accepted,
            "decision": decision.value,
            "ocrReadinessScore": score,
            "reasonCodes": [code.value for code in reason_codes],
            "userMessages": messages_for(reason_codes),
            "metrics": metrics,
            "qualityMetrics": metrics,
            "detectedRisks": risk_codes,
            "autoFixes": auto_fixes,
            "appliedCorrections": auto_fixes,
            "pages": page_dicts,
            "pageResults": page_dicts,
            "failedPages": failed_pages,
            "failedPageIndexes": failed_pages,
            "timingsMs": timings_ms,
            "stageDurations": timings_ms,
            "thresholdConfigVersion": self.config.version,
            "modelVersions": model_versions or {},
            "ocrProbeExecuted": any_probe,
            "aggregationPolicy": self.config.aggregation_policy,
        }

    def _score(
        self,
        metrics: ImageMetrics,
        risks: list[ReasonCode],
        hard: list[ReasonCode],
        ocr_probe: OcrProbeResult | None,
    ) -> float:
        if hard:
            return 0.0
        contrast = min(1.0, metrics.std / max(self.config.min_contrast_std * 2.0, 1.0))
        sharpness = min(1.0, metrics.sharpness / max(self.config.blur_risk_threshold * 2.0, 1.0))
        exposure = 1.0 - min(1.0, abs(metrics.mean - 180.0) / 180.0)
        score = max(0.0, min(1.0, min(contrast, sharpness, exposure) - 0.08 * len(risks)))
        if ocr_probe is not None:
            score = min(score, ocr_probe.average_confidence)
        return round(score, 4)

    def _dedupe(self, codes: list[ReasonCode]) -> list[ReasonCode]:
        seen = set()
        result = []
        for code in codes:
            if code not in seen:
                seen.add(code)
                result.append(code)
        return result
