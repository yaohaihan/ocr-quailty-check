from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .pipeline import QualityPipeline


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 4)


def summarize_durations(values: list[float]) -> dict:
    return {
        "count": len(values),
        "p50Ms": percentile(values, 50),
        "p95Ms": percentile(values, 95),
        "p99Ms": percentile(values, 99),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark OCR quality pipeline latency.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args(argv)
    pipeline = QualityPipeline()
    durations = []
    for _ in range(args.repeat):
        for path in args.paths:
            started = time.perf_counter()
            pipeline.evaluate(Path(path))
            durations.append((time.perf_counter() - started) * 1000.0)
    print(json.dumps(summarize_durations(durations), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
