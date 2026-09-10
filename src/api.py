"""FastAPI inference service for semantic product search and ranking."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from functools import lru_cache
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .ranking import rank_titles
from .reranking import CrossEncoderReranker
from .search import ProductSearch
from .serving import LRUCache, SearchMetrics, assign_variant, cache_key, log_query

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ARTIFACT_DIR = "artifacts"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_LOG_PATH = os.getenv("QUERY_LOG_PATH", "logs/search.jsonl")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
PRELOAD_INDEX = os.getenv("PRELOAD_INDEX", "0").lower() in {"1", "true", "yes"}

SEARCH_CACHE = LRUCache(max_size=int(os.getenv("SEARCH_CACHE_SIZE", "256")))
METRICS = SearchMetrics()


class RankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    products: list[str] = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)
    candidate_k: int = Field(default=50, ge=10, le=500)
    brand: str | None = Field(default=None, max_length=200)
    locale: str | None = Field(default=None, max_length=20)
    hybrid: bool = False
    hybrid_alpha: float = Field(default=0.8, ge=0.0, le=1.0)
    rerank: bool = False
    user_keywords: list[str] = Field(default_factory=list, max_length=20)
    experiment_key: str | None = Field(default=None, max_length=200)


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(os.getenv("MODEL_PATH", DEFAULT_MODEL))


@lru_cache(maxsize=1)
def get_search_engine():
    return ProductSearch.load(
        os.getenv("MODEL_PATH", DEFAULT_MODEL),
        os.getenv("ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR),
    )


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoderReranker(os.getenv("RERANKER_MODEL", DEFAULT_RERANKER))


@asynccontextmanager
async def lifespan(_app):
    # Production deployments can preload the 1.3M-product index before receiving
    # traffic. Local development stays lazy by default so startup remains quick.
    if PRELOAD_INDEX:
        get_search_engine()
    yield


app = FastAPI(title="Findly Semantic Search", version="2.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    engine_loaded = get_search_engine.cache_info().currsize > 0
    settings = get_search_engine().settings if engine_loaded else {}
    return {
        "status": "ready" if engine_loaded else "ok",
        "version": app.version,
        "rank_model_loaded": get_model.cache_info().currsize > 0,
        "search_index_loaded": engine_loaded,
        "reranker_loaded": get_reranker.cache_info().currsize > 0,
        "artifact": settings,
    }


@app.get("/metrics")
def metrics():
    return METRICS.snapshot(cache_size=SEARCH_CACHE.size)


@app.post("/rank")
def rank(request: RankRequest):
    top_k = min(request.top_k, len(request.products))
    return {
        "query": request.query,
        "results": rank_titles(request.query, request.products, get_model(), top_k=top_k),
    }


@app.post("/search")
def search(request: SearchRequest):
    if request.candidate_k < request.top_k:
        raise HTTPException(status_code=422, detail="candidate_k must be >= top_k")

    variant = assign_variant(request.experiment_key)
    use_hybrid = request.hybrid or variant == "enhanced"
    use_reranker = request.rerank or variant == "enhanced"

    key_payload = request.model_dump()
    key_payload["resolved_variant"] = variant
    key = cache_key(key_payload)
    cached = SEARCH_CACHE.get(key)
    if cached is not None:
        METRICS.record(cached["latency_ms"], variant=variant, cache_hit=True)
        return {**cached, "cache_hit": True}

    started = perf_counter()
    try:
        retrieval_limit = max(request.candidate_k, request.top_k)
        rerank_limit = (
            min(retrieval_limit, max(request.top_k, RERANK_CANDIDATES))
            if use_reranker
            else request.top_k
        )

        results, retrieval_latency_ms = get_search_engine().search(
            request.query,
            top_k=retrieval_limit if use_reranker else request.top_k,
            candidate_k=retrieval_limit,
            brand=request.brand,
            locale=request.locale,
            hybrid=use_hybrid,
            hybrid_alpha=request.hybrid_alpha,
            user_keywords=request.user_keywords,
        )

        rerank_latency_ms = 0.0
        if use_reranker and results:
            rerank_started = perf_counter()
            results = get_reranker().rerank(request.query, results[:rerank_limit], request.top_k)
            rerank_latency_ms = (perf_counter() - rerank_started) * 1000
        else:
            results = results[: request.top_k]

        total_latency_ms = (perf_counter() - started) * 1000
        payload = {
            "query": request.query,
            "variant": variant,
            "hybrid": use_hybrid,
            "reranked": use_reranker,
            "latency_ms": round(total_latency_ms, 3),
            "retrieval_latency_ms": round(retrieval_latency_ms, 3),
            "rerank_latency_ms": round(rerank_latency_ms, 3),
            "cache_hit": False,
            "results": results,
        }
        SEARCH_CACHE.set(key, payload)
        METRICS.record(total_latency_ms, variant=variant, cache_hit=False)
        log_query(
            QUERY_LOG_PATH,
            {
                "query": request.query,
                "variant": variant,
                "top_k": request.top_k,
                "candidate_k": retrieval_limit,
                "rerank_candidates": rerank_limit if use_reranker else 0,
                "hybrid": use_hybrid,
                "reranked": use_reranker,
                "brand": request.brand,
                "locale": request.locale,
                "result_count": len(results),
                "latency_ms": round(total_latency_ms, 3),
            },
        )
        return payload
    except Exception as exc:
        METRICS.record_error()
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc
