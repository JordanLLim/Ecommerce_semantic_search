"""Compare FAISS HNSW with exact Flat search on normalized embeddings."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
import pandas as pd


def recall_at_k(reference: np.ndarray, candidate: np.ndarray) -> float:
    values = []
    for exact, approx in zip(reference, candidate):
        exact_set = {int(i) for i in exact if i >= 0}
        approx_set = {int(i) for i in approx if i >= 0}
        values.append(len(exact_set & approx_set) / max(1, len(exact_set)))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    import faiss
    from sentence_transformers import SentenceTransformer

    products = pd.read_parquet(args.catalog, columns=["product_id", "product_title"])
    products = products.dropna().drop_duplicates("product_id").head(args.limit)
    queries = [q.strip() for q in open(args.queries, encoding="utf-8") if q.strip()][: args.query_count]

    model = SentenceTransformer(args.model)
    vectors = model.encode(
        products["product_title"].astype(str).tolist(),
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    qvecs = model.encode(queries, normalize_embeddings=True)
    qvecs = np.ascontiguousarray(qvecs, dtype="float32")

    flat = faiss.IndexFlatIP(vectors.shape[1])
    flat.add(vectors)
    _, reference = flat.search(qvecs, args.top_k)

    hnsw = faiss.IndexHNSWFlat(vectors.shape[1], args.m, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = args.ef_construction
    hnsw.add(vectors)

    print(f"Benchmark: {len(vectors):,} products, {len(qvecs)} queries, top-{args.top_k}")
    for ef in args.ef_search:
        hnsw.hnsw.efSearch = ef
        started = perf_counter()
        _, indices = hnsw.search(qvecs, args.top_k)
        elapsed = (perf_counter() - started) * 1000 / len(qvecs)
        print(f"HNSW efSearch={ef:<4d} recall={recall_at_k(reference, indices):.4f} mean={elapsed:.3f} ms/query")


if __name__ == "__main__":
    main()
