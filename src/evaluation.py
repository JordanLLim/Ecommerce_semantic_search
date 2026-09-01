"""Evaluate the candidate ranking once and keep per-query results."""
import pandas as pd
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
