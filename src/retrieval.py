"""Controlled-pool retrieval evaluation utilities."""

import numpy as np
import pandas as pd


def build_controlled_pool(catalog, judgments, pool_size=10_000, seed=42):
    """Include all judged products, then sample label-free distractors."""
    required_ids = set(judgments["product_id"])
    products = catalog.drop_duplicates("product_id").copy()
    required = products.loc[products["product_id"].isin(required_ids)]
    if set(required["product_id"]) != required_ids:
        raise ValueError("Some judged products are missing from the catalog.")
    if len(required) > pool_size:
        raise ValueError("pool_size is smaller than the judged candidate set.")
    remaining = products.loc[~products["product_id"].isin(required_ids)]
    extra_count = pool_size - len(required)
    if extra_count > len(remaining):
        raise ValueError("Not enough catalog products for the requested pool.")
    extra = remaining.sample(n=extra_count, random_state=seed)
    return pd.concat([required, extra], ignore_index=True).sort_values(
        "product_id", kind="stable").reset_index(drop=True)


def top_k_indices(scores, k):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all() or k < 1:
        raise ValueError("Expected finite one-dimensional scores and positive k.")
    return np.argsort(-scores, kind="stable")[:min(k, len(scores))]


def known_exact_metrics(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("At least one known Exact product is required.")
    hits = np.array([product_id in relevant for product_id in retrieved_ids])
    return {
        "known_E_recall@10": float(hits[:10].sum() / len(relevant)),
        "known_E_recall@50": float(hits[:50].sum() / len(relevant)),
        "known_E_hit@10": float(hits[:10].any()),
    }
