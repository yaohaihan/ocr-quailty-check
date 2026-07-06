from __future__ import annotations

from pathlib import Path

import pdfplumber
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import QualityConfig
from .models import AutoFixRecord, ReasonCode


class PageNormalizer:
    def __init__(self, config: QualityConfig):
        self.config = config

    def load_image(self, path: str | Path) -> tuple[Image.Image | None, list[ReasonCode], list[AutoFixRecord]]:
        try:
            image = Image.open(path)
            transposed = ImageOps.exif_transpose(image)
            fixes: list[AutoFixRecord] = []
            if transposed.size != image.size:
                fixes.append(
                    AutoFixRecord(
                        fixType="EXIF_ORIENTATION",
                        before={"size": image.size},
                        after={"size": transposed.size},
                        recomputed=True,
                    )
                )
            return transposed.convert("RGB"), [], fixes
        except (UnidentifiedImageError, OSError, ValueError):
            return None, [ReasonCode.IMAGE_DECODE_FAILED], []

    def prepare_for_analysis(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        longest = max(width, height)
        if longest <= self.config.max_analysis_side:
            return image
        ratio = self.config.max_analysis_side / longest
        return image.resize((max(1, int(width * ratio)), max(1, int(height * ratio))))

    def render_pdf_pages(self, path: str | Path) -> tuple[list[Image.Image], list[ReasonCode]]:
        try:
            pages: list[Image.Image] = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages[: self.config.max_pages]:
                    rendered = page.to_image(resolution=self.config.pdf_render_dpi).original
                    pages.append(rendered.convert("RGB"))
            if not pages:
                return [], [ReasonCode.FILE_CORRUPTED]
            return pages, []
        except Exception:
            return [], [ReasonCode.FILE_CORRUPTED]
