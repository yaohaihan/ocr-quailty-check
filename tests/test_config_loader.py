import json
import tempfile
import unittest
from pathlib import Path

from ocr_quality.config_loader import load_quality_config


class ConfigLoaderTests(unittest.TestCase):
    def test_default_config_file_loads_version_and_model_flags(self):
        config = load_quality_config(Path("config/quality.default.json"))

        self.assertEqual(config.version, "ocr-quality-v2-paddle")
        self.assertTrue(config.enable_text_detection)
        self.assertTrue(config.enable_orientation_detection)
        self.assertTrue(config.enable_ocr_probe)
        self.assertEqual(config.ocr_probe_sample_size, 6)
        self.assertEqual(config.text_detector_provider, "paddleocr")

    def test_external_config_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality.json"
            path.write_text(
                json.dumps(
                    {
                        "version": "local-test",
                        "max_pages": 3,
                        "enable_ocr_probe": False,
                        "text_detector_provider": "disabled",
                        "risk_count_for_probe": 4,
                    }
                ),
                encoding="utf-8",
            )

            config = load_quality_config(path)

        self.assertEqual(config.version, "local-test")
        self.assertEqual(config.max_pages, 3)
        self.assertFalse(config.enable_ocr_probe)
        self.assertEqual(config.text_detector_provider, "disabled")
        self.assertEqual(config.risk_count_for_probe, 4)


if __name__ == "__main__":
    unittest.main()

