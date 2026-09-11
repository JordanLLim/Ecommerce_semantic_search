"""Load a persisted FAISS index and search product metadata."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .hybrid import BM25Retriever, reciprocal_rank_fusion, tokenize

class ProductSearch:
    def __init__(self, model, index, products, settings=None):
        if not products: raise ValueError("Product metadata must not be empty.")
        if index.ntotal != len(products): raise ValueError("FAISS rows and product metadata are not aligned.")
        self.model, self.index, self.products = model, index, products
        self.settings = settings or {}
        self.last_timing = {}
        self._bm25 = None

    @classmethod
    def load(cls, model_path, artifact_dir):
        import faiss
        from sentence_transformers import SentenceTransformer
        artifact_dir = Path(artifact_dir)
        index = faiss.read_index(str(artifact_dir / "products.faiss"))
        with (artifact_dir / "products.jsonl").open(encoding="utf-8") as stream:
            products = [json.loads(line) for line in stream if line.strip()]
        settings_path = artifact_dir / "metadata.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
        if settings.get("nprobe") is not None and hasattr(index, "nprobe"):
            index.nprobe = int(settings["nprobe"])
        return cls(SentenceTransformer(settings.get("model") or model_path), index, products, settings)

    def _bm25_index(self):
        if self._bm25 is None:
            self._bm25 = BM25Retriever(str(p.get("product_title", "")) for p in self.products)
        return self._bm25

    @staticmethod
    def _matches_filters(product, brand=None, locale=None):
        if brand and str(product.get("product_brand", "")).strip().lower() != brand.strip().lower(): return False
        if locale and str(product.get("product_locale", "")).strip().lower() != locale.strip().lower(): return False
        return True

    def _result(self, doc_id, dense_score=None, bm25_score=None, fusion_score=None):
        p = self.products[int(doc_id)]
        return {"product_id": str(p["product_id"]), "title": p["product_title"], "score": dense_score, "bm25_score": bm25_score, "fusion_score": fusion_score, "brand": p.get("product_brand"), "locale": p.get("product_locale")}

    def search(self, query, top_k=10, candidate_k=50, brand=None, locale=None, hybrid=False, hybrid_alpha=0.8, user_keywords=None):
        query = query.strip()
        if not query: raise ValueError("query must not be blank.")
        if top_k < 1: raise ValueError("top_k must be at least 1.")
        candidate_k = min(max(int(candidate_k or top_k), int(top_k)), self.index.ntotal)
        started = perf_counter()
        t = perf_counter(); vector = np.asarray(self.model.encode(query, normalize_embeddings=True), dtype="float32").reshape(1, -1); embedding_ms = (perf_counter()-t)*1000
        t = perf_counter(); dense_scores, dense_ids = self.index.search(vector, candidate_k); faiss_ms = (perf_counter()-t)*1000
        dense = [(int(i), float(s)) for s, i in zip(dense_scores[0], dense_ids[0]) if i >= 0]
        bm25_ms = 0.0
        if hybrid:
            t = perf_counter(); sparse = self._bm25_index().search(query, candidate_k); bm25_ms = (perf_counter()-t)*1000
            dense_rank = [i for i, _ in dense]
            sparse_rank = [i for i, _ in sparse]
            fused = reciprocal_rank_fusion([dense_rank, sparse_rank])
            dense_map, sparse_map = dict(dense), dict(sparse)
            ranked = [(i, dense_map.get(i), sparse_map.get(i), score) for i, score in fused]
        else:
            ranked = [(i, score, None, None) for i, score in dense]
        t = perf_counter(); results = []
        keywords = set(tokenize(" ".join(user_keywords or [])))
        for doc_id, dense_score, bm25_score, fusion_score in ranked:
            p = self.products[doc_id]
            if not self._matches_filters(p, brand, locale): continue
            item = self._result(doc_id, dense_score, bm25_score, fusion_score)
            if keywords:
                item["preference_overlap"] = len(keywords & set(tokenize(str(p.get("product_title", ""))))) / max(1, len(keywords))
            results.append(item)
            if len(results) >= top_k: break
        for rank, item in enumerate(results, 1): item["rank"] = rank
        postprocess_ms = (perf_counter()-t)*1000; total_ms = (perf_counter()-started)*1000
        self.last_timing = {"embedding_ms":round(embedding_ms,3), "faiss_ms":round(faiss_ms,3), "bm25_ms":round(bm25_ms,3), "postprocess_ms":round(postprocess_ms,3), "retrieval_total_ms":round(total_ms,3)}
        return results, total_ms
