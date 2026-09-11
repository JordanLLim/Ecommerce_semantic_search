import unittest
from unittest.mock import patch

import numpy as np

from src.api import RankRequest, SearchRequest, health, rank, search


class FakeModel:
    def encode(self, text, **kwargs):
        vectors = {"query": [1.0, 0.0], "match": [1.0, 0.0], "miss": [0.0, 1.0]}
        return np.array(vectors[text] if isinstance(text, str) else [vectors[item] for item in text])


class FakeSearchEngine:
    def search(self, query, top_k=10, **kwargs):
        return [{"rank": 1, "product_id": "p1", "title": "match", "score": 1.0}], 2.3456


class ApiTests(unittest.TestCase):
    def test_health_is_lazy(self):
        response = health()
        self.assertFalse(response["rank_model_loaded"])
        self.assertFalse(response["search_index_loaded"])

    @patch("src.api.get_model", return_value=FakeModel())
    def test_rank_response(self, _):
        response = rank(RankRequest(query="query", products=["miss", "match"], top_k=1))
        self.assertEqual(response["results"][0]["title"], "match")
        self.assertEqual(len(response["results"]), 1)

    @patch("src.api.get_search_engine", return_value=FakeSearchEngine())
    def test_search_response(self, _):
        response = search(SearchRequest(query="query", top_k=1))
        self.assertEqual(response["results"][0]["product_id"], "p1")
        self.assertEqual(response["retrieval_latency_ms"], 2.346)
        self.assertFalse(response["reranked"])


if __name__ == "__main__":
    unittest.main()
