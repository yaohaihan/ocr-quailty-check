from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .models import AutoFixRecord, ReasonCode


@dataclass
class CoordinateMapping:
    scale_x: float = 1.0
    scale_y: float = 1.0


@dataclass
class DecodedPage:
    page_index: int
    image: Image.Image
    original_size: tuple[int, int]
    mapping: CoordinateMapping = field(default_factory=CoordinateMapping)
    auto_fixes: list[AutoFixRecord] = field(default_factory=list)
    source: str = "image"


@dataclass
class DecodeResult:
    pages: list[DecodedPage]
    reason_codes: list[ReasonCode]
    file_info: dict[str, Any]

