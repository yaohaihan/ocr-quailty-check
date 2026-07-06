from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import QualityConfig


@dataclass
class ImageMetrics:
    width: int
    height: int
    pixels: int
    mean: float
    std: float
    edge_density: float
    sharpness: float
    low_quality_block_ratio: float
    noise_estimate: float
    illumination_unevenness: float
    laplacian_variance: float = 0.0
    tenengrad: float = 0.0
    low_contrast: bool = False
    content_block_count: int = 0
    total_block_count: int = 0
    skew_angle_degrees: float | None = None
    compression_risk: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "width": self.width,
            "height": self.height,
            "pixels": self.pixels,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "edgeDensity": round(self.edge_density, 6),
            "sharpness": round(self.sharpness, 4),
            "lowQualityBlockRatio": round(self.low_quality_block_ratio, 4),
            "noiseEstimate": round(self.noise_estimate, 4),
            "illuminationUnevenness": round(self.illumination_unevenness, 4),
            "laplacianVariance": round(self.laplacian_variance, 4),
            "tenengrad": round(self.tenengrad, 4),
            "lowContrast": self.low_contrast,
            "contentBlockCount": self.content_block_count,
            "totalBlockCount": self.total_block_count,
            "skewAngleDegrees": self.skew_angle_degrees,
            "compressionRisk": round(self.compression_risk, 4),
        }


class FastImageQualityAnalyzer:
    def __init__(self, config: QualityConfig):
        self.config = config

    def analyze(self, image: Image.Image) -> ImageMetrics:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        height, width = gray.shape
        gx = np.diff(gray, axis=1, append=gray[:, -1:])
        gy = np.diff(gray, axis=0, append=gray[-1:, :])
        gradient = np.hypot(gx, gy)
        edge_density = float(np.mean(gradient > 20.0))
        sharpness = float(np.mean(gradient * gradient))
        lap = self._laplacian(gray)
        laplacian_variance = float(np.var(lap))
        block_scores, total_blocks = self._block_sharpness(gray)
        low_quality = float(np.mean(block_scores < self.config.blur_risk_threshold)) if block_scores.size else 0.0
        noise = float(np.std(gray - self._box_blur(gray)))
        unevenness = self._illumination_unevenness(gray)
        low_contrast = float(np.std(gray)) < self.config.min_contrast_std
        compression_risk = min(1.0, noise / 40.0)
        return ImageMetrics(
            width=width,
            height=height,
            pixels=width * height,
            mean=float(np.mean(gray)),
            std=float(np.std(gray)),
            edge_density=edge_density,
            sharpness=sharpness,
            low_quality_block_ratio=low_quality,
            noise_estimate=noise,
            illumination_unevenness=unevenness,
            laplacian_variance=laplacian_variance,
            tenengrad=sharpness,
            low_contrast=low_contrast,
            content_block_count=int(block_scores.size),
            total_block_count=total_blocks,
            skew_angle_degrees=None,
            compression_risk=compression_risk,
        )

    def _block_sharpness(self, gray: np.ndarray) -> tuple[np.ndarray, int]:
        height, width = gray.shape
        rows = 4
        cols = 4
        scores = []
        total = rows * cols
        for row in range(rows):
            for col in range(cols):
                y0 = math.floor(row * height / rows)
                y1 = math.floor((row + 1) * height / rows)
                x0 = math.floor(col * width / cols)
                x1 = math.floor((col + 1) * width / cols)
                block = gray[y0:y1, x0:x1]
                if block.size == 0:
                    continue
                block_gx = np.diff(block, axis=1, append=block[:, -1:])
                block_gy = np.diff(block, axis=0, append=block[-1:, :])
                block_gradient = np.hypot(block_gx, block_gy)
                block_edge_density = float(np.mean(block_gradient > 20.0))
                if float(np.std(block)) < self.config.blank_std_threshold * 2 and block_edge_density < self.config.blank_edge_density_threshold:
                    continue
                gx = np.diff(block, axis=1, append=block[:, -1:])
                gy = np.diff(block, axis=0, append=block[-1:, :])
                scores.append(float(np.mean(gx * gx + gy * gy)))
        return np.asarray(scores, dtype=np.float32), total

    def _laplacian(self, gray: np.ndarray) -> np.ndarray:
        padded = np.pad(gray, 1, mode="edge")
        return (
            -4 * padded[1:-1, 1:-1]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )

    def _box_blur(self, gray: np.ndarray) -> np.ndarray:
        padded = np.pad(gray, 1, mode="edge")
        return (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0

    def _illumination_unevenness(self, gray: np.ndarray) -> float:
        height, width = gray.shape
        thirds = []
        for row in range(3):
            for col in range(3):
                y0 = math.floor(row * height / 3)
                y1 = math.floor((row + 1) * height / 3)
                x0 = math.floor(col * width / 3)
                x1 = math.floor((col + 1) * width / 3)
                thirds.append(float(np.mean(gray[y0:y1, x0:x1])))
        return float(max(thirds) - min(thirds))
