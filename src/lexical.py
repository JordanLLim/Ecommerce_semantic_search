"""A leakage-aware TF-IDF ranking baseline."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(training_titles, max_features=100_000):
    vectorizer = TfidfVectorizer(
        lowercase=True, token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2),
        max_features=max_features, norm="l2", dtype=np.float32,
    )
    vectorizer.fit(list(training_titles))
    return vectorizer


def score_candidates(frame, vectorizer):
    """Score each query only against its supplied candidate products."""
    groups = []
    for _, group in frame.groupby("query_id"):
        query = vectorizer.transform([group["query"].iloc[0]])
        products = vectorizer.transform(group["product_title"].fillna(""))
        scored = group.copy()
        scored["model_score"] = (products @ query.T).toarray().ravel()
        groups.append(scored)
    if not groups:
        raise ValueError("No candidates to score.")
    import pandas as pd
    return pd.concat(groups, ignore_index=True)
