import unittest

import numpy as np

from src.search import ProductSearch


class FakeModel:
    def encode(self, text, **kwargs):
        return np.array([1.0, 0.0], dtype="float32")


class FakeIndex:
    ntotal = 2

    def search(self, vector, top_k):
        self.vector = vector
        return (
            np.array([[0.9, 0.4]], dtype="float32")[:, :top_k],
            np.array([[1, 0]])[:, :top_k],
        )


class ProductSearchTests(unittest.TestCase):
    def setUp(self):
        self.engine = ProductSearch(
            FakeModel(), FakeIndex(),
            [
                {"product_id": "p0", "product_title": "second"},
                {"product_id": "p1", "product_title": "first"},
            ],
        )

    def test_returns_aligned_ranked_products(self):
        results, latency = self.engine.search("fan", top_k=2)
        self.assertEqual([item["product_id"] for item in results], ["p1", "p0"])
        self.assertEqual([item["rank"] for item in results], [1, 2])
        self.assertGreaterEqual(latency, 0)
        self.assertEqual(self.engine.index.vector.shape, (1, 2))

    def test_top_k_is_bounded_by_catalog(self):
        results, _ = self.engine.search("fan", top_k=100)
        self.assertEqual(len(results), 2)

    def test_invalid_input_and_alignment(self):
        with self.assertRaises(ValueError):
            self.engine.search("  ")
        with self.assertRaises(ValueError):
            ProductSearch(FakeModel(), FakeIndex(), [{"product_id": "p0"}])


if __name__ == "__main__":
    unittest.main()
