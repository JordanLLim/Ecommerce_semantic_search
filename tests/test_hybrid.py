import unittest
from src.hybrid import BM25Retriever, reciprocal_rank_fusion


class HybridTests(unittest.TestCase):
    def test_bm25_prefers_exact_model_terms(self):
        docs = ["wireless phone case", "iphone 15 pro max 256gb case", "iphone charger"]
        results = BM25Retriever(docs).search("iphone 15 pro max 256gb", top_k=3)
        self.assertEqual(results[0][0], 1)

    def test_bm25_respects_top_k_after_heap_selection(self):
        docs = ["gaming keyboard", "gaming mouse", "gaming headset", "office chair"]
        results = BM25Retriever(docs).search("gaming", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0][1], results[1][1])

    def test_rrf_rewards_documents_seen_by_both_retrievers(self):
        fused = reciprocal_rank_fusion([[1, 2, 3], [2, 4, 1]])
        self.assertEqual(fused[0][0], 2)


if __name__ == "__main__":
    unittest.main()
