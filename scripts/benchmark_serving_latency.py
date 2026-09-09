"""Measure realistic single-query backend search latency on persisted artifacts.

Unlike the FAISS batch benchmark, this script sends one query at a time through
ProductSearch.search so query encoding and index search are both included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from src.search import ProductSearch


def load_queries(path: str, limit: int) -> list[str]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    queries = [line for line in lines if line]
    if not queries:
        raise ValueError("Query file contains no usable queries.")
    return queries[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", default="reports/serving_latency.json")
    args = parser.parse_args()

    if min(args.query_count, args.top_k, args.candidate_k) < 1:
        raise ValueError("query-count, top-k and candidate-k must be positive.")
    if args.candidate_k < args.top_k:
        raise ValueError("candidate-k must be >= top-k.")

    engine = ProductSearch.load(args.model, args.artifact_dir)
    queries = load_queries(args.queries, args.query_count)

    for query in queries[: min(args.warmup, len(queries))]:
        engine.search(query, top_k=args.top_k, candidate_k=args.candidate_k)

    latencies = []
    for query in queries:
        started = perf_counter()
        engine.search(query, top_k=args.top_k, candidate_k=args.candidate_k)
        latencies.append((perf_counter() - started) * 1000)

    values = np.asarray(latencies, dtype=float)
    result = {
        "queries": len(queries),
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "mean_ms": float(mean(latencies)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "artifact": engine.settings,
        "measurement": "single query, query embedding + FAISS search, no HTTP",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
