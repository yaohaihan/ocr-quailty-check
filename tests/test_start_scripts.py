import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StartScriptTests(unittest.TestCase):
    def test_python_start_script_launches_the_fastapi_app(self):
        script = ROOT / "start.py"

        self.assertTrue(script.exists())

        script_text = script.read_text(encoding="utf-8")

        self.assertIn("PYTHONPATH", script_text)
        self.assertIn("uvicorn", script_text)
        self.assertIn("ocr_quality.web.app:app", script_text)
        self.assertIn("OCR_QUALITY_PORT", script_text)
        self.assertIn("OCR_QUALITY_CONFIG", script_text)
        self.assertIn("PADDLE_PDX_CACHE_HOME", script_text)
        self.assertIn(".paddlex-cache", script_text)
        self.assertIn("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", script_text)
        self.assertIn("PADDLE_PDX_CPU_NUM_THREADS", script_text)
        self.assertIn("--no-browser", script_text)

    def test_readme_documents_one_command_startup(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("python start.py", readme)
        self.assertNotIn(".\\start.bat", readme)
        self.assertNotIn(".\\scripts\\start.ps1", readme)
        self.assertIn("docs/usage.md", readme)
        self.assertIn("docs/api.md", readme)
        self.assertIn("docs/development.md", readme)
        self.assertIn("docs/architecture.md", readme)

    def test_split_docs_contain_their_own_topics(self):
        usage = (ROOT / "docs" / "usage.md").read_text(encoding="utf-8")
        api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
        development = (ROOT / "docs" / "development.md").read_text(encoding="utf-8")
        architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")

        self.assertIn("python start.py", usage)
        self.assertNotIn(".\\start.bat", usage)
        self.assertIn("POST /api/quality/check", api)
        self.assertIn("unittest discover", development)
        self.assertIn("HeuristicTextDetector", architecture)


if __name__ == "__main__":
    unittest.main()
