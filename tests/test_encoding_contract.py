import unittest
from pathlib import Path

from ocr_quality.models import ReasonCode, USER_MESSAGES


ROOT = Path(__file__).resolve().parents[1]


class EncodingContractTests(unittest.TestCase):
    def test_user_messages_are_readable_chinese(self):
        self.assertEqual(USER_MESSAGES[ReasonCode.OCR_PROBE_LOW_CONFIDENCE], "OCR 抽样探测置信度较低，请重新上传更清晰文件。")
        self.assertIn("文件格式暂不支持", USER_MESSAGES[ReasonCode.UNSUPPORTED_FILE_TYPE])

    def test_frontend_contains_readable_chinese_labels(self):
        html = (ROOT / "src" / "ocr_quality" / "web" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("OCR 影像质量校验", html)
        self.assertIn("开始质检", html)
        self.assertIn("原因码", html)
        self.assertNotIn("璐", html)
        self.assertNotIn("�", html)


if __name__ == "__main__":
    unittest.main()

