"""Score supplied candidates with a sentence-transformer bi-encoder."""

import numpy as np

def rank_products_for_query(group, model):
    if group.empty or group["query"].nunique(dropna=False) != 1:
        raise ValueError("Expected one non-empty query group.")
    query = group["query"].iloc[0]
    titles = group["product_title"].fillna("").tolist()
    query_embedding = model.encode(query, normalize_embeddings=True)
    product_embeddings = model.encode(
        titles, normalize_embeddings=True, batch_size=64)
    # The dot product is cosine similarity for normalized vectors.
    result = group.copy()
    result["model_score"] = product_embeddings @ query_embedding
    return result.sort_values("model_score", ascending=False, kind="stable")

def rank_titles(query, titles, model, top_k=None):
    """Rank arbitrary titles for online inference without pandas."""
    if not query or not query.strip():
        raise ValueError("query must not be blank.")
    if not titles or any(not isinstance(title, str) for title in titles):
        raise ValueError("titles must be a non-empty list of strings.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive.")
    query_vector = np.asarray(model.encode(query, normalize_embeddings=True))
    title_vectors = np.asarray(model.encode(
        titles, normalize_embeddings=True, batch_size=64
    ))
    scores = title_vectors @ query_vector
    order = np.argsort(-scores, kind="stable")
    if top_k is not None:
        order = order[:top_k]
    return [
        {"rank": rank, "index": int(index), "title": titles[index],
         "score": float(scores[index])}
        for rank, index in enumerate(order, start=1)
    ]
