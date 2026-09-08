"""Benchmark FAISS Flat and IVF search on the same normalized embeddings.

Flat search is treated as the exact nearest-neighbour reference. IVF recall@k is
measured by overlap with Flat's top-k results, while latency is measured only
for the FAISS search call so model encoding time does not hide index behaviour.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


def read_catalog(path: str, limit: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=["product_id", "product_title"])
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, usecols=["product_id", "product_title"])
    else:
        raise ValueError("Catalog must be a .csv or .parquet file.")

    frame = frame.dropna(subset=["product_id", "product_title"])
    frame = frame.drop_duplicates("product_id", keep="first").reset_index(drop=True)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive.")
        frame = frame.head(limit).copy()
    if frame.empty:
        raise ValueError("Catalog has no usable products.")
    return frame


def load_queries(path: str | None, fallback_titles: list[str], count: int, seed: int) -> list[str]:
    if path:
        query_path = Path(path)
        if query_path.suffix.lower() == ".txt":
            queries = [line.strip() for line in query_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif query_path.suffix.lower() == ".json":
            data = json.loads(query_path.read_text(encoding="utf-8"))
            queries = [str(item).strip() for item in data if str(item).strip()]
        else:
            raise ValueError("Query file must be .txt or .json.")
        if not queries:
            raise ValueError("Query file contains no usable queries.")
        return queries[:count]

    rng = np.random.default_rng(seed)
    size = min(count, len(fallback_titles))
    indices = rng.choice(len(fallback_titles), size=size, replace=False)
    return [fallback_titles[int(i)] for i in indices]


def timed_search(index, query_vectors: np.ndarray, top_k: int, repeats: int) -> tuple[np.ndarray, np.ndarray]:
    latencies = []
    last_indices = None
    for _ in range(repeats):
        started = perf_counter()
        _, indices = index.search(query_vectors, top_k)
        latencies.append((perf_counter() - started) * 1000 / len(query_vectors))
        last_indices = indices
    return last_indices, np.asarray(latencies, dtype=float)


def recall_against_flat(reference: np.ndarray, candidate: np.ndarray) -> float:
    recalls = []
    for exact, approx in zip(reference, candidate):
        exact_set = {int(i) for i in exact if i >= 0}
        approx_set = {int(i) for i in approx if i >= 0}
        recalls.append(len(exact_set & approx_set) / max(1, len(exact_set)))
    return float(np.mean(recalls))


def latency_summary(samples: np.ndarray) -> tuple[float, float]:
    return float(np.mean(samples)), float(np.percentile(samples, 95))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--queries")
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--nlist", type=int, default=224)
    parser.add_argument("--nprobes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", default="reports/faiss_benchmark.csv")
    args = parser.parse_args()

    if args.top_k < 1 or args.query_count < 1 or args.repeats < 1:
        raise ValueError("top-k, query-count and repeats must be positive.")

    import faiss
    from sentence_transformers import SentenceTransformer

    products = read_catalog(args.catalog, args.limit)
    if not 1 <= args.nlist <= len(products):
        raise ValueError("nlist must be between 1 and the catalog size.")

    model = SentenceTransformer(args.model)
    titles = products["product_title"].astype(str).tolist()
    embeddings = model.encode(
        titles,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.ascontiguousarray(embeddings, dtype="float32")

    queries = load_queries(args.queries, titles, args.query_count, args.seed)
    query_vectors = model.encode(queries, normalize_embeddings=True)
    query_vectors = np.ascontiguousarray(query_vectors, dtype="float32")

    flat = faiss.IndexFlatIP(vectors.shape[1])
    flat.add(vectors)
    reference, flat_latency = timed_search(flat, query_vectors, args.top_k, args.repeats)
    flat_mean, flat_p95 = latency_summary(flat_latency)

    rows = [{
        "products": len(products),
        "queries": len(queries),
        "index": "flat",
        "nlist": "",
        "nprobe": "",
        "recall_at_k_vs_flat": 1.0,
        "mean_latency_ms_per_query": flat_mean,
        "p95_batch_latency_ms_per_query": flat_p95,
    }]

    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    ivf = faiss.IndexIVFFlat(quantizer, vectors.shape[1], args.nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(vectors)
    ivf.add(vectors)

    for nprobe in args.nprobes:
        if not 1 <= nprobe <= args.nlist:
            raise ValueError(f"nprobe={nprobe} must be between 1 and nlist={args.nlist}.")
        ivf.nprobe = nprobe
        approximate, latency = timed_search(ivf, query_vectors, args.top_k, args.repeats)
        mean_latency, p95_latency = latency_summary(latency)
        rows.append({
            "products": len(products),
            "queries": len(queries),
            "index": "ivf",
            "nlist": args.nlist,
            "nprobe": nprobe,
            "recall_at_k_vs_flat": recall_against_flat(reference, approximate),
            "mean_latency_ms_per_query": mean_latency,
            "p95_batch_latency_ms_per_query": p95_latency,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Benchmark: {len(products):,} products, {len(queries)} queries, top-{args.top_k}")
    for row in rows:
        label = "Flat" if row["index"] == "flat" else f"IVF nprobe={row['nprobe']}"
        print(
            f"{label:16s} recall={row['recall_at_k_vs_flat']:.4f} "
            f"mean={row['mean_latency_ms_per_query']:.3f} ms/query "
            f"p95={row['p95_batch_latency_ms_per_query']:.3f} ms/query"
        )
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
