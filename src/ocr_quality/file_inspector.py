from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .config import QualityConfig
from .models import ReasonCode


@dataclass
class FileInspection:
    accepted: bool
    reason_codes: list[ReasonCode]
    extension: str
    file_size: int
    page_count: int = 1


class FileInspector:
    def __init__(self, config: QualityConfig):
        self.config = config

    def inspect(self, path: str | Path) -> FileInspection:
        file_path = Path(path)
        extension = file_path.suffix.lower()
        if extension not in self.config.supported_extensions:
            return FileInspection(False, [ReasonCode.UNSUPPORTED_FILE_TYPE], extension, 0, 0)
        try:
            size = file_path.stat().st_size
        except OSError:
            return FileInspection(False, [ReasonCode.FILE_CORRUPTED], extension, 0, 0)
        if size <= 0 or size > self.config.max_file_bytes:
            return FileInspection(False, [ReasonCode.FILE_CORRUPTED], extension, size, 0)
        if extension == ".pdf":
            return self._inspect_pdf(file_path, extension, size)
        return FileInspection(True, [], extension, size, 1)

    def _inspect_pdf(self, path: Path, extension: str, size: int) -> FileInspection:
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                return FileInspection(False, [ReasonCode.PDF_ENCRYPTED], extension, size, 0)
            page_count = len(reader.pages)
        except Exception:
            return FileInspection(False, [ReasonCode.FILE_CORRUPTED], extension, size, 0)
        if page_count <= 0 or page_count > self.config.max_pages:
            return FileInspection(False, [ReasonCode.FILE_CORRUPTED], extension, size, page_count)
        return FileInspection(True, [], extension, size, page_count)
