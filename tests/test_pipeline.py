import unittest
import numpy as np
import pandas as pd
from src.data import split_small_data, sample_evaluation
from src.ranking import rank_products_for_query
from src.evaluation import evaluate, candidate_ndcg
from src.triplets import build_triplets, TripletDataset

class FakeModel:
    def encode(self, text, **kwargs):
        vectors = {"query": [1., 0.], "good": [1., 0.], "bad": [0., 1.]}
        return np.array(vectors[text] if isinstance(text, str) else [vectors[t] for t in text])

def fixture():
    return pd.DataFrame([
        {"query_id": q, "query": "query", "product_id": f"{q}-{i}",
         "product_title": title, "esci_label": label, "split": split, "small_version": 1}
        for q, split in [(1, "train"), (2, "test"), (3, "test")]
        for i, (title, label) in enumerate([("good", "E"), ("bad", "I")])
    ])

class PipelineTests(unittest.TestCase):
    def test_split_and_sampling(self):
        train, test = split_small_data(fixture())
        selected, ids = sample_evaluation(test, 1)
        self.assertEqual(len(ids), 1)
        self.assertEqual(len(selected), 2)
        self.assertFalse(set(train.query_id) & set(selected.query_id))
        self.assertTrue(ids.equals(sample_evaluation(test, 1)[1]))

    def test_ranking_and_evaluation(self):
        _, test = split_small_data(fixture())
        frame, _ = sample_evaluation(test)
        results, ranks = evaluate(FakeModel(), frame)
        self.assertTrue(np.allclose(results["ndcg@10"], 1))
        self.assertEqual(len(ranks), len(frame))
        self.assertNotIn("model_score", frame)

    def test_zero_singleton_and_reverse(self):
        self.assertEqual(candidate_ndcg(np.array([0, 0]), np.array([1., 0.])), 0)
        self.assertEqual(candidate_ndcg([3], [0.2]), 1)
        self.assertEqual(candidate_ndcg([0], [0.2]), 0)
        self.assertLess(candidate_ndcg([3, 0], [0., 1.]), 1)
        self.assertTrue(np.isfinite(candidate_ndcg([3, 0], [1., 1.])))

    def test_triplets(self):
        train, test = split_small_data(fixture())
        result = build_triplets(train)
        self.assertEqual(len(result), 1)
        self.assertEqual(TripletDataset(result)[0], {"query": "query", "positive": "good", "negative": "bad"})
        self.assertTrue(result.equals(build_triplets(train)))
        with self.assertRaises(ValueError):
            build_triplets(test)

    def test_overlap_rejected(self):
        frame = fixture()
        frame.loc[frame.query_id.eq(2), "query_id"] = 1
        with self.assertRaises(ValueError):
            split_small_data(frame)

    def test_no_eligible_triplets(self):
        train, _ = split_small_data(fixture())
        self.assertTrue(build_triplets(train.loc[train.esci_label.eq("E")]).empty)

if __name__ == "__main__":
    unittest.main()

