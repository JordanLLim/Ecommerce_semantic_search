import unittest
from unittest.mock import patch

import numpy as np

from src.api import RankRequest, health, rank


class FakeModel:
    def encode(self, text, **kwargs):
        vectors = {"query": [1.0, 0.0], "match": [1.0, 0.0], "miss": [0.0, 1.0]}
        return np.array(vectors[text] if isinstance(text, str) else [vectors[item] for item in text])


class ApiTests(unittest.TestCase):
    def test_health_is_lazy(self):
        self.assertIn("model_loaded", health())

    @patch("src.api.get_model", return_value=FakeModel())
    def test_rank_response(self, _):
        response = rank(RankRequest(query="query", products=["miss", "match"], top_k=1))
        self.assertEqual(response["results"][0]["title"], "match")
        self.assertEqual(len(response["results"]), 1)


if __name__ == "__main__":
    unittest.main()
