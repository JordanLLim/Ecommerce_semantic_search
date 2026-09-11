import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_hybrid import discover_query_file, percentile


class HybridBenchmarkTests(unittest.TestCase):
    def test_percentile_bounds(self):
        values = [10.0, 20.0, 30.0, 40.0]
        self.assertEqual(percentile(values, 0.0), 10.0)
        self.assertEqual(percentile(values, 1.0), 40.0)
        self.assertAlmostEqual(percentile(values, 0.5), 25.0)

    def test_explicit_query_file_must_exist(self):
        with self.assertRaises(FileNotFoundError):
            discover_query_file(Path("definitely-missing-examples.parquet"))

    def test_explicit_query_file_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.parquet"
            path.touch()
            self.assertEqual(discover_query_file(path), path)


if __name__ == "__main__":
    unittest.main()
