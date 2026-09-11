"""Load a persisted FAISS index and search product metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "") if len(token) > 1}


def _lexical_overlap(query: str, title: str) -> float:
    q = _tokens(query)
    t = _tokens(title)
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


class ProductSearch:
    """Search service with a model, vector index and aligned product metadata."""

    def __init__(self, model, index, products, settings=None):
        if not products:
            raise ValueError("Product metadata must not be empty.")
        if index.ntotal != len(products):
            raise ValueError("FAISS rows and product metadata are not aligned.")
        self.model = model
        self.index = index
        self.products = products
        self.settings = settings or {}
        self.last_timing = {}

    @classmethod
    def load(cls, model_path, artifact_dir):
        import faiss
        from sentence_transformers import SentenceTransformer

        artifact_dir = Path(artifact_dir)
        index = faiss.read_index(str(artifact_dir / "products.faiss"))
        with (artifact_dir / "products.jsonl").open(encoding="utf-8") as stream:
            products = [json.loads(line) for line in stream if line.strip()]

        settings = {}
        settings_path = artifact_dir / "metadata.json"
        if settings_path.exists():
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            nprobe = settings.get("nprobe")
            if nprobe is not None and hasattr(index, "nprobe"):
                index.nprobe = int(nprobe)

        configured_model = settings.get("model") or model_path
        return cls(SentenceTransformer(configured_model), index, products, settings=settings)

    @staticmethod
    def _matches_filters(product, brand=None, locale=None):
        if brand:
            value = str(product.get("product_brand", "")).strip().lower()
            if value != brand.strip().lower():
                return False
        if locale:
            value = str(product.get("product_locale", "")).strip().lower()
            if value != locale.strip().lower():
                return False
        return True

    def search(
        self,
        query,
        top_k=10,
        candidate_k=50,
        brand=None,
        locale=None,
        hybrid=False,
        hybrid_alpha=0.8,
        user_keywords=None,
    ):
        """Retrieve ANN candidates and optionally rescore/filter them."""
        query = query.strip()
        if not query:
            raise ValueError("query must not be blank.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not 0.0 <= hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be between 0 and 1.")

        effective_candidate_k = max(int(candidate_k or top_k), int(top_k))
        started = perf_counter()

        encode_started = perf_counter()
        vector = self.model.encode(query, normalize_embeddings=True)
        vector = np.asarray(vector, dtype="float32").reshape(1, -1)
        embedding_ms = (perf_counter() - encode_started) * 1000

        search_limit = min(effective_candidate_k, self.index.ntotal)
        faiss_started = perf_counter()
        scores, indices = self.index.search(vector, search_limit)
        faiss_ms = (perf_counter() - faiss_started) * 1000

        post_started = perf_counter()
        keyword_tokens = _tokens(" ".join(user_keywords or []))
        candidates = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            product = self.products[int(index)]
            if not self._matches_filters(product, brand=brand, locale=locale):
                continue

            dense_score = float(score)
            lexical_score = _lexical_overlap(query, str(product.get("product_title", "")))
            combined_score = dense_score
            if hybrid:
                combined_score = hybrid_alpha * dense_score + (1.0 - hybrid_alpha) * lexical_score

            personalization_boost = 0.0
            if keyword_tokens:
                title_tokens = _tokens(str(product.get("product_title", "")))
                personalization_boost = 0.02 * len(keyword_tokens & title_tokens) / max(1, len(keyword_tokens))
                combined_score += personalization_boost

            candidates.append({
                "product_id": str(product["product_id"]),
                "title": product["product_title"],
                "score": dense_score,
                "lexical_score": lexical_score,
                "combined_score": float(combined_score),
                "personalization_boost": float(personalization_boost),
                "brand": product.get("product_brand"),
                "locale": product.get("product_locale"),
            })

        if hybrid or keyword_tokens:
            candidates.sort(key=lambda item: item["combined_score"], reverse=True)

        results = candidates[: min(top_k, len(candidates))]
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank

        postprocess_ms = (perf_counter() - post_started) * 1000
        total_ms = (perf_counter() - started) * 1000
        self.last_timing = {
            "embedding_ms": round(embedding_ms, 3),
            "faiss_ms": round(faiss_ms, 3),
            "postprocess_ms": round(postprocess_ms, 3),
            "retrieval_total_ms": round(total_ms, 3),
        }
        return results, total_ms
