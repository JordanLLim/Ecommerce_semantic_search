"""Evaluate the candidate ranking once and keep per-query results."""
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
from .ranking import rank_products_for_query

def candidate_ndcg(relevance, scores, k=10):
    if k < 1 or len(relevance) == 0 or len(relevance) != len(scores):
        raise ValueError("Expected aligned, non-empty arrays and positive k.")
    # sklearn expects at least two candidates.
    if len(relevance) == 1:
        return float(relevance[0] > 0)
    return float(ndcg_score([relevance], [scores], k=k))

def evaluate(model, frame, k=10):
    if frame.empty:
        raise ValueError("No evaluation rows.")
    rows, rankings = [], []
    for query_id, group in tqdm(frame.groupby("query_id"),
                                total=frame["query_id"].nunique()):
        ranked = rank_products_for_query(group, model)
        relevance = ranked["relevance"].to_numpy()
        score = candidate_ndcg(relevance, ranked["model_score"].to_numpy(), k)
        rows.append({"query_id": query_id, "query": group["query"].iloc[0],
                     f"ndcg@{k}": score, "candidates": len(group),
                     "has_positive_gain": bool((relevance > 0).any())})
        rankings.append(ranked)
    return pd.DataFrame(rows), pd.concat(rankings, ignore_index=True)

def candidate_metrics(scored, score_column="model_score"):
    """Compute per-query graded and Exact-product ranking metrics."""
    required = {"query_id", "product_id", "esci_label", score_column}
    if scored.empty or not required.issubset(scored.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(scored.columns))}")
    rows = []
    for query_id, group in scored.groupby("query_id"):
        ranked = group.sort_values(
            [score_column, "product_id"], ascending=[False, True], kind="stable"
        )
        relevance = ranked["esci_label"].map({"E": 3, "S": 2, "C": 1, "I": 0}).to_numpy()
        exact = ranked["esci_label"].eq("E").to_numpy()
        exact_count = int(exact.sum())
        positions = np.flatnonzero(exact[:10])
        rows.append({
            "query_id": query_id,
            "ndcg@10": candidate_ndcg(relevance, ranked[score_column].to_numpy(), 10),
            "p@1": float(exact[0]),
            "p@5": float(exact[:5].sum() / min(5, len(exact))),
            "recall@10": float(exact[:10].sum() / exact_count) if exact_count else np.nan,
            "mrr@10": float(1 / (positions[0] + 1)) if len(positions) else 0.0,
            "candidate_count": len(ranked),
            "exact_count": exact_count,
        })
    return pd.DataFrame(rows)

def paired_bootstrap(left, right, samples=5000, seed=42):
    """Bootstrap the paired mean difference (left minus right)."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("Expected aligned, non-empty one-dimensional arrays.")
    differences = left - right
    if not np.isfinite(differences).all() or samples < 1:
        raise ValueError("Inputs must be finite and samples positive.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean_delta": float(differences.mean()),
            "ci_95_low": float(low), "ci_95_high": float(high),
            "samples": samples, "seed": seed}
