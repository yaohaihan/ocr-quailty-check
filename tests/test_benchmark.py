import unittest

from ocr_quality.benchmark import percentile, summarize_durations


class BenchmarkTests(unittest.TestCase):
    def test_percentile_calculation(self):
        values = [1.0, 2.0, 3.0, 4.0]

        self.assertEqual(percentile(values, 50), 2.5)
        self.assertEqual(percentile(values, 95), 3.85)

    def test_summary_contains_required_percentiles(self):
        summary = summarize_durations([10.0, 20.0, 30.0])

        self.assertEqual(summary["count"], 3)
        self.assertIn("p50Ms", summary)
        self.assertIn("p95Ms", summary)
        self.assertIn("p99Ms", summary)


if __name__ == "__main__":
    unittest.main()

