import time
import unittest

from ocr_quality.instrumentation import StageTimer


class InstrumentationTests(unittest.TestCase):
    def test_stage_timer_records_named_duration(self):
        timer = StageTimer()

        with timer.measure("decode"):
            time.sleep(0.001)

        payload = timer.to_dict()
        self.assertIn("decode", payload)
        self.assertGreaterEqual(payload["decode"], 0.0)

    def test_nested_stage_names_are_preserved(self):
        timer = StageTimer()

        with timer.measure("page.0.textDetection"):
            pass

        self.assertIn("page.0.textDetection", timer.to_dict())


if __name__ == "__main__":
    unittest.main()

