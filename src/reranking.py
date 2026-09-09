"""Optional cross-encoder reranking for retrieved candidates."""

from __future__ import annotations


class CrossEncoderReranker:
    """Lazy wrapper around a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        if not results:
            return []

        pairs = [(query, item["title"]) for item in results]
        scores = self.model.predict(pairs)
        reranked = []
        for item, score in zip(results, scores):
            enriched = dict(item)
            enriched["retrieval_score"] = enriched.pop("score", None)
            enriched["reranker_score"] = float(score)
            reranked.append(enriched)

        reranked.sort(key=lambda item: item["reranker_score"], reverse=True)
        for rank, item in enumerate(reranked[:top_k], start=1):
            item["rank"] = rank
        return reranked[:top_k]
