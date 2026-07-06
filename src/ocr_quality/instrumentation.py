from __future__ import annotations

import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self):
        self._durations: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self._durations[name] = round((time.perf_counter() - started) * 1000.0, 3)

    def to_dict(self) -> dict[str, float]:
        return dict(self._durations)

