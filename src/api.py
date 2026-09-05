"""FastAPI inference service for ranking a supplied product list."""

import os
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .ranking import rank_titles

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    products: list[str] = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.getenv("MODEL_PATH", DEFAULT_MODEL))


app = FastAPI(title="Amazon Semantic Ranker", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": get_model.cache_info().currsize > 0}


@app.post("/rank")
def rank(request: RankRequest):
    top_k = min(request.top_k, len(request.products))
    return {"query": request.query, "results": rank_titles(
        request.query, request.products, get_model(), top_k=top_k
    )}
