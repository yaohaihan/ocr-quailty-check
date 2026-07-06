from __future__ import annotations

import argparse
import json

from .pipeline import QualityPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate whether files are ready for OCR.")
    parser.add_argument("paths", nargs="+", help="Image file paths to evaluate")
    args = parser.parse_args(argv)
    result = QualityPipeline().evaluate_many(args.paths)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

