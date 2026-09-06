import unittest
import numpy as np
import pandas as pd
from src.data import split_small_data, split_train_validation, sample_evaluation
from src.ranking import rank_products_for_query, rank_titles
from src.evaluation import evaluate, candidate_ndcg, candidate_metrics, paired_bootstrap
from src.triplets import build_triplets, TripletDataset
from src.retrieval import build_controlled_pool, known_exact_metrics, top_k_indices
from src.training import TrainingConfig

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

    def test_query_level_validation_split(self):
        train = pd.concat([fixture().loc[lambda x: x["split"].eq("train")]] * 4,
                          ignore_index=True)
        train["query_id"] = np.repeat([1, 4, 5, 6], 2)
        fit, validation = split_train_validation(train, validation_fraction=0.25)
        self.assertFalse(set(fit.query_id) & set(validation.query_id))
        self.assertEqual(validation.query_id.nunique(), 1)

    def test_online_title_ranking(self):
        ranked = rank_titles("query", ["bad", "good"], FakeModel(), top_k=1)
        self.assertEqual(ranked[0]["title"], "good")
        self.assertEqual(ranked[0]["rank"], 1)

    def test_metrics_and_bootstrap(self):
        scored = pd.DataFrame([
            {"query_id": 1, "product_id": "a", "esci_label": "E", "model_score": 1.0},
            {"query_id": 1, "product_id": "b", "esci_label": "I", "model_score": 0.0},
        ])
        metrics = candidate_metrics(scored)
        self.assertEqual(metrics.loc[0, "p@1"], 1)
        result = paired_bootstrap([1, 0.8], [0.5, 0.3], samples=20)
        self.assertAlmostEqual(result["mean_delta"], 0.5)

    def test_controlled_retrieval_pool(self):
        catalog = pd.DataFrame({"product_id": ["a", "b", "c"],
                                "product_title": ["A", "B", "C"]})
        judgments = pd.DataFrame({"product_id": ["a"]})
        pool = build_controlled_pool(catalog, judgments, pool_size=2)
        self.assertIn("a", set(pool.product_id))
        self.assertEqual(list(top_k_indices([0.1, 0.9], 1)), [1])
        metrics = known_exact_metrics(["a", "b"], {"a"})
        self.assertEqual(metrics["known_E_hit@10"], 1)

    def test_training_config_validation(self):
        self.assertEqual(TrainingConfig().validate().epochs, 1)
        with self.assertRaises(ValueError):
            TrainingConfig(epochs=0).validate()

if __name__ == "__main__":
    unittest.main()
