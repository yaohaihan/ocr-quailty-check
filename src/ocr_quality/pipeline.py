from __future__ import annotations

from dataclasses import replace
import time
from pathlib import Path

from .config import QualityConfig, load_default_config
from .decision import DecisionEngine
from .file_decoder import FileDecoder
from .file_inspector import FileInspector
from .image_metrics import FastImageQualityAnalyzer
from .models import Decision, PageResult, messages_for
from .normalizer import PageNormalizer
from .occlusion import OcclusionAnalyzer
from .ocr_probe import DisabledOcrProbe, HeuristicOcrProbe, OcrProbe
from .orientation import DisabledOrientationDetector, OrientationDetector, apply_orientation_correction
from .text_detection import DisabledTextDetector, HeuristicTextDetector, TextDetector
from .text_regions import TextRegionAnalyzer


class QualityPipeline:
    def __init__(
        self,
        config: QualityConfig | None = None,
        text_detector: TextDetector | None = None,
        ocr_probe: OcrProbe | None = None,
        orientation_detector: OrientationDetector | None = None,
    ):
        self.config = config or load_default_config()
        self.inspector = FileInspector(self.config)
        self.decoder = FileDecoder(self.config)
        self.normalizer = PageNormalizer(self.config)
        self.image_analyzer = FastImageQualityAnalyzer(self.config)
        self.model_versions: dict[str, str] = {}
        self.text_detector = text_detector or self._build_text_detector()
        self.ocr_probe = ocr_probe or self._build_ocr_probe()
        self.orientation_detector = orientation_detector or DisabledOrientationDetector()
        self.text_region_analyzer = TextRegionAnalyzer(self.config)
        self.occlusion_analyzer = OcclusionAnalyzer(self.config)
        self.decision_engine = DecisionEngine(self.config)

    def _build_text_detector(self) -> TextDetector:
        if not self.config.enable_text_detection:
            self.model_versions["textDetector"] = "disabled"
            return DisabledTextDetector()
        if self.config.text_detector_provider == "paddleocr":
            try:
                from .adapters.paddle_ocr import PaddleOcrBundle, PaddleTextDetector

                bundle = PaddleOcrBundle.create(self.config)
                bundle.warmup()
                self.model_versions.update(bundle.model_versions())
                self.model_versions["textDetector"] = "paddleocr"
                self._paddle_bundle = bundle
                return PaddleTextDetector(bundle.paddle)
            except Exception as exc:
                self.model_versions["paddleocr"] = f"unavailable:{type(exc).__name__}"
                self.model_versions["textDetector"] = "heuristic-fallback"
        else:
            self.model_versions["textDetector"] = self.config.text_detector_provider
        return HeuristicTextDetector()

    def _build_ocr_probe(self) -> OcrProbe:
        if not self.config.enable_ocr_probe:
            self.model_versions["ocrProbe"] = "disabled"
            return DisabledOcrProbe()
        bundle = getattr(self, "_paddle_bundle", None)
        if self.config.ocr_probe_provider == "paddleocr" and bundle is not None:
            from .adapters.paddle_ocr import PaddleOcrProbe

            self.model_versions["ocrProbe"] = "paddleocr"
            return PaddleOcrProbe(bundle.paddle, self.config.ocr_probe_sample_size)
        if self.config.ocr_probe_provider == "paddleocr":
            self.model_versions["ocrProbe"] = "heuristic-fallback"
        else:
            self.model_versions["ocrProbe"] = self.config.ocr_probe_provider
        return HeuristicOcrProbe()

    def evaluate(self, path: str | Path) -> dict:
        file_path = Path(path)
        inspection = self.inspector.inspect(file_path)
        if not inspection.accepted:
            reason_codes = [code.value for code in inspection.reason_codes]
            messages = messages_for(inspection.reason_codes)
            metrics = {"pageCount": 0, "failedPageCount": 0, "failedPageRatio": 0.0}
            return {
                "accepted": False,
                "decision": Decision.REJECT.value,
                "ocrReadinessScore": 0.0,
                "reasonCodes": reason_codes,
                "userMessages": messages,
                "metrics": metrics,
                "qualityMetrics": metrics,
                "detectedRisks": [],
                "autoFixes": [],
                "appliedCorrections": [],
                "pages": [],
                "pageResults": [],
                "failedPages": [],
                "failedPageIndexes": [],
                "timingsMs": {},
                "stageDurations": {},
                "thresholdConfigVersion": self.config.version,
                "modelVersions": self.model_versions,
                "ocrProbeExecuted": False,
                "aggregationPolicy": self.config.aggregation_policy,
            }
        if inspection.extension == ".pdf":
            return self._evaluate_pdf(file_path)
        return self.evaluate_many([path])

    def evaluate_many(self, paths: list[str | Path]) -> dict:
        start = time.perf_counter()
        pages: list[PageResult] = []
        for index, path in enumerate(paths):
            pages.append(self._evaluate_page(Path(path), index))
        timings = {"total": round((time.perf_counter() - start) * 1000.0, 3)}
        return self.decision_engine.aggregate(pages, timings, self.model_versions)

    def _evaluate_page(self, path: Path, page_index: int) -> PageResult:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        inspection = self.inspector.inspect(path)
        timings["fileInspection"] = self._elapsed(started)
        if not inspection.accepted:
            return PageResult(
                pageIndex=page_index,
                accepted=False,
                decision=Decision.REJECT,
                ocrReadinessScore=0.0,
                reasonCodes=inspection.reason_codes,
                userMessages=messages_for(inspection.reason_codes),
                timingsMs=timings,
            )

        started = time.perf_counter()
        image, decode_reasons, fixes = self.normalizer.load_image(path)
        timings["decode"] = self._elapsed(started)
        if image is None:
            return PageResult(
                pageIndex=page_index,
                accepted=False,
                decision=Decision.REJECT,
                ocrReadinessScore=0.0,
                reasonCodes=decode_reasons,
                userMessages=messages_for(decode_reasons),
                timingsMs=timings,
            )

        result = self._evaluate_image(image, page_index, timings)
        result.autoFixes = fixes + result.autoFixes
        if fixes and result.decision == Decision.ACCEPT:
            result.decision = Decision.AUTO_FIXED_ACCEPT
        return result

    def _evaluate_pdf(self, path: Path) -> dict:
        start = time.perf_counter()
        render_start = time.perf_counter()
        images, reasons = self.normalizer.render_pdf_pages(path)
        render_ms = self._elapsed(render_start)
        if reasons:
            return {
                "accepted": False,
                "decision": Decision.REJECT.value,
                "ocrReadinessScore": 0.0,
                "reasonCodes": [code.value for code in reasons],
                "userMessages": messages_for(reasons),
                "metrics": {"pageCount": 0, "failedPageCount": 0},
                "qualityMetrics": {"pageCount": 0, "failedPageCount": 0, "failedPageRatio": 0.0},
                "detectedRisks": [],
                "autoFixes": [],
                "appliedCorrections": [],
                "pages": [],
                "pageResults": [],
                "failedPages": [],
                "failedPageIndexes": [],
                "timingsMs": {"pdfRender": render_ms, "total": self._elapsed(start)},
                "stageDurations": {"pdfRender": render_ms, "total": self._elapsed(start)},
                "thresholdConfigVersion": self.config.version,
                "modelVersions": self.model_versions,
                "ocrProbeExecuted": False,
                "aggregationPolicy": self.config.aggregation_policy,
            }
        pages = [self._evaluate_image(image, page_index, {"pdfRender": render_ms if page_index == 0 else 0.0}) for page_index, image in enumerate(images)]
        timings = {"pdfRender": render_ms, "total": self._elapsed(start)}
        return self.decision_engine.aggregate(pages, timings, self.model_versions)

    def _evaluate_image(self, image, page_index: int, timings: dict[str, float] | None = None) -> PageResult:
        page_timings = dict(timings or {})
        fixes = []
        if self.config.enable_orientation_detection:
            started = time.perf_counter()
            orientation = self.orientation_detector.detect(image)
            image, fix = apply_orientation_correction(image, orientation)
            page_timings["orientationDetection"] = self._elapsed(started)
            if fix is not None:
                fixes.append(fix)
        started = time.perf_counter()
        analysis_image = self.normalizer.prepare_for_analysis(image)
        metrics = self.image_analyzer.analyze(analysis_image)
        if analysis_image.size != image.size:
            original_width, original_height = image.size
            metrics = replace(metrics, width=original_width, height=original_height, pixels=original_width * original_height)
        page_timings["fastImageQuality"] = self._elapsed(started)

        started = time.perf_counter()
        text = self.text_detector.detect(analysis_image)
        region_summary = self.text_region_analyzer.analyze(analysis_image, text)
        text.coverage_ratio = region_summary.coverage_ratio
        text.median_height = region_summary.median_height
        text.border_touch_ratio = region_summary.border_touch_ratio
        text.low_sharpness_ratio = region_summary.low_sharpness_ratio
        page_timings["textDetection"] = self._elapsed(started)

        started = time.perf_counter()
        occlusion = self.occlusion_analyzer.analyze(analysis_image, text)
        page_timings["occlusionDetection"] = self._elapsed(started)

        probe = None
        if self.config.enable_ocr_probe and self.decision_engine.should_probe(metrics, text, occlusion):
            started = time.perf_counter()
            probe = self.ocr_probe.probe(analysis_image, text)
            page_timings["ocrProbe"] = self._elapsed(started)

        result = self.decision_engine.page_decision(page_index, metrics, text, probe, page_timings, occlusion)
        result.autoFixes.extend(fixes)
        return result

    def _elapsed(self, started: float) -> float:
        return round((time.perf_counter() - started) * 1000.0, 3)
