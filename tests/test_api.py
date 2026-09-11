import unittest
from unittest.mock import patch

import numpy as np

from src.api import RankRequest, SearchRequest, SEARCH_CACHE, health, rank, search


class FakeModel:
    def encode(self, text, **kwargs):
        vectors = {"query": [1.0, 0.0], "match": [1.0, 0.0], "miss": [0.0, 1.0]}
        return np.array(vectors[text] if isinstance(text, str) else [vectors[item] for item in text])


class FakeSearchEngine:
    def __init__(self):
        self.calls = 0
        self.last_timing = {
            "embedding_ms": 1.0,
            "faiss_ms": 1.0,
            "bm25_ms": 0.0,
            "postprocess_ms": 0.1,
        }

    def search(self, query, top_k=10, **kwargs):
        self.calls += 1
        return [{"rank": 1, "product_id": "p1", "title": "match", "score": 1.0}], 2.3456


class ApiTests(unittest.TestCase):
    def setUp(self):
        SEARCH_CACHE._items.clear()

    def test_health_is_lazy(self):
        response = health()
        self.assertFalse(response["rank_model_loaded"])
        self.assertFalse(response["search_index_loaded"])

    @patch("src.api.get_model", return_value=FakeModel())
    def test_rank_response(self, _):
        response = rank(RankRequest(query="query", products=["miss", "match"], top_k=1))
        self.assertEqual(response["results"][0]["title"], "match")
        self.assertEqual(len(response["results"]), 1)

    def test_search_response(self):
        engine = FakeSearchEngine()
        with patch("src.api.get_search_engine", return_value=engine):
            response = search(SearchRequest(query="query", top_k=1))
        self.assertEqual(response["results"][0]["product_id"], "p1")
        self.assertEqual(response["retrieval_latency_ms"], 2.346)
        self.assertFalse(response["reranked"])
        self.assertEqual(response["faiss_latency_ms"], 1.0)

    def test_bypass_cache_runs_search_each_time(self):
        engine = FakeSearchEngine()
        request = SearchRequest(query="cache bypass query", top_k=1, bypass_cache=True)
        with patch("src.api.get_search_engine", return_value=engine):
            first = search(request)
            second = search(request)
        self.assertFalse(first["cache_hit"])
        self.assertFalse(second["cache_hit"])
        self.assertEqual(engine.calls, 2)
        self.assertEqual(SEARCH_CACHE.size, 0)


if __name__ == "__main__":
    unittest.main()
