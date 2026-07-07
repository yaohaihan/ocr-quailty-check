from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QualityConfig:
    version: str = "ocr-quality-v1"
    supported_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pdf")
    max_file_bytes: int = 25 * 1024 * 1024
    max_pages: int = 20
    pdf_render_dpi: int = 120
    min_width: int = 240
    min_height: int = 240
    min_pixels: int = 80_000
    max_analysis_side: int = 1200
    blank_std_threshold: float = 2.0
    blank_edge_density_threshold: float = 0.002
    black_mean_threshold: float = 8.0
    overexposed_mean_threshold: float = 246.0
    underexposed_mean_threshold: float = 18.0
    min_contrast_std: float = 18.0
    severe_blur_threshold: float = 22.0
    blur_risk_threshold: float = 55.0
    max_low_quality_block_ratio: float = 0.45
    min_text_boxes: int = 1
    min_text_coverage_ratio: float = 0.003
    min_text_median_height: float = 10.0
    max_border_touch_ratio: float = 0.45
    max_text_low_sharpness_ratio: float = 0.60
    risk_count_for_probe: int = 2
    ocr_probe_min_confidence: float = 0.60
    ocr_probe_max_empty_ratio: float = 0.50
    aggregation_policy: str = "ALL_PAGES_REQUIRED"
    max_failed_page_ratio: float = 0.0
    first_n_required: int = 1
    require_text: bool = True
    enable_orientation_detection: bool = False
    enable_text_detection: bool = True
    enable_ocr_probe: bool = True
    orientation_detector_provider: str = "disabled"
    text_detector_provider: str = "heuristic"
    ocr_probe_provider: str = "heuristic"
    paddle_use_gpu: bool = False
    paddle_lang: str = "ch"
    paddle_det_model_dir: str | None = None
    paddle_rec_model_dir: str | None = None
    paddle_cls_model_dir: str | None = None
    paddle_cpu_threads: int = 4
    ocr_probe_sample_size: int = 6
    model_load_timeout_ms: int = 30000
    enable_occlusion_detection: bool = True
    occlusion_min_area_ratio: float = 0.01
    occlusion_min_text_overlap_ratio: float = 0.08
    occlusion_min_affected_box_ratio: float = 0.10
    occlusion_min_box_overlap_ratio: float = 0.35
    occlusion_text_box_padding_ratio: float = 0.25
    occlusion_max_hand_area_ratio: float = 0.20
    occlusion_max_hand_rectangularity: float = 0.92
    occlusion_skin_min_red: int = 95
    occlusion_skin_min_red_green_delta: int = 12
    occlusion_skin_min_red_blue_delta: int = 20


def load_default_config() -> QualityConfig:
    return QualityConfig()
