import unittest
from scripts.benchmark_hybrid import percentile

class HybridBenchmarkTests(unittest.TestCase):
    def test_percentile_bounds(self):
        values = [10.0, 20.0, 30.0, 40.0]
        self.assertEqual(percentile(values, 0.0), 10.0)
        self.assertEqual(percentile(values, 1.0), 40.0)
        self.assertAlmostEqual(percentile(values, 0.5), 25.0)

if __name__ == "__main__":
    unittest.main()
