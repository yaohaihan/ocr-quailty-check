from __future__ import annotations

from pathlib import Path

from .config import QualityConfig
from .file_inspector import FileInspector
from .normalizer import PageNormalizer
from .page import DecodeResult, DecodedPage


class FileDecoder:
    def __init__(self, config: QualityConfig):
        self.config = config
        self.inspector = FileInspector(config)
        self.normalizer = PageNormalizer(config)

    def decode(self, path: str | Path) -> DecodeResult:
        file_path = Path(path)
        inspection = self.inspector.inspect(file_path)
        file_info = {
            "extension": inspection.extension,
            "fileSize": inspection.file_size,
            "pageCount": inspection.page_count,
        }
        if not inspection.accepted:
            return DecodeResult([], inspection.reason_codes, file_info)
        if inspection.extension == ".pdf":
            images, reasons = self.normalizer.render_pdf_pages(file_path)
            pages = [DecodedPage(index, image, image.size, source="pdf") for index, image in enumerate(images)]
            return DecodeResult(pages, reasons, file_info)
        image, reasons, fixes = self.normalizer.load_image(file_path)
        if image is None:
            return DecodeResult([], reasons, file_info)
        return DecodeResult([DecodedPage(0, image, image.size, auto_fixes=fixes, source="image")], [], file_info)

