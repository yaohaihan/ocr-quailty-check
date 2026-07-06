from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from .config import QualityConfig


def load_quality_config(path: str | Path | None = None) -> QualityConfig:
    config_path = Path(path) if path is not None else Path("config/quality.default.json")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    field_names = {field.name for field in fields(QualityConfig)}
    filtered = {key: value for key, value in data.items() if key in field_names}
    if "supported_extensions" in filtered:
        filtered["supported_extensions"] = tuple(filtered["supported_extensions"])
    return QualityConfig(**filtered)

