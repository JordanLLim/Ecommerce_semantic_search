"""Score the supplied candidates with normalized embeddings."""

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

