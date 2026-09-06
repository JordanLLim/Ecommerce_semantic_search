"""Load a persisted FAISS index and search product titles."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np


class ProductSearch:
    """Small search service with a model, vector index and aligned metadata."""

    def __init__(self, model, index, products):
        if not products:
            raise ValueError("Product metadata must not be empty.")
        if index.ntotal != len(products):
            raise ValueError("FAISS rows and product metadata are not aligned.")
        self.model = model
        self.index = index
        self.products = products

    @classmethod
    def load(cls, model_path, artifact_dir):
        import faiss
        from sentence_transformers import SentenceTransformer

        artifact_dir = Path(artifact_dir)
        index = faiss.read_index(str(artifact_dir / "products.faiss"))
        with (artifact_dir / "products.jsonl").open(encoding="utf-8") as stream:
            products = [json.loads(line) for line in stream if line.strip()]
        settings_path = artifact_dir / "metadata.json"
        if settings_path.exists():
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            nprobe = settings.get("nprobe")
            if nprobe is not None and hasattr(index, "nprobe"):
                index.nprobe = int(nprobe)
        return cls(SentenceTransformer(model_path), index, products)

    def search(self, query, top_k=10):
        query = query.strip()
        if not query:
            raise ValueError("query must not be blank.")
        if top_k < 1:
            raise ValueError("top_k must be positive.")

        started = perf_counter()
        vector = self.model.encode(query, normalize_embeddings=True)
        vector = np.asarray(vector, dtype="float32").reshape(1, -1)
        limit = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vector, limit)
        results = []
        for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1):
            if index < 0:
                continue
            product = self.products[int(index)]
            results.append({
                "rank": rank,
                "product_id": str(product["product_id"]),
                "title": product["product_title"],
                "score": float(score),
            })
        return results, (perf_counter() - started) * 1000
