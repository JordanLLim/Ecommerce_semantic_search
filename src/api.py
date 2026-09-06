"""FastAPI inference service for ranking a supplied product list."""

import os
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .ranking import rank_titles
from .search import ProductSearch

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ARTIFACT_DIR = "artifacts"


class RankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    products: list[str] = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)


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


app = FastAPI(title="Amazon Semantic Ranker", version="1.0.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rank_model_loaded": get_model.cache_info().currsize > 0,
        "search_index_loaded": get_search_engine.cache_info().currsize > 0,
    }


@app.post("/rank")
def rank(request: RankRequest):
    top_k = min(request.top_k, len(request.products))
    return {"query": request.query, "results": rank_titles(
        request.query, request.products, get_model(), top_k=top_k
    )}


@app.post("/search")
def search(request: SearchRequest):
    results, latency_ms = get_search_engine().search(request.query, request.top_k)
    return {
        "query": request.query,
        "latency_ms": round(latency_ms, 3),
        "results": results,
    }
