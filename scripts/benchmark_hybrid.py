"""Benchmark dense, hybrid RRF, and optional reranking through the search API.

By default the script samples 100 real queries from the local Amazon ESCI examples
parquet when it is available. It bypasses the serving cache, warms each mode once,
and reports mean/p50/p95/p99 for HTTP, backend, and individual backend components.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_QUERY_FILE = Path("data/raw/shopping_queries_dataset_examples.parquet")
SMOKE_QUERIES = [
    "gaming keyboard", "wireless mouse", "wireless headphones", "cooling fan",
    "toddler books", "makeup vanity", "tv stand", "baby bag", "mechanical keyboard",
    "iphone charger",
]


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * p
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def stats(values):
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def format_stats(name, values):
    s = stats(values)
    return f"{name}: mean={s['mean']:.1f} p50={s['p50']:.1f} p95={s['p95']:.1f} p99={s['p99']:.1f} ms"


def load_queries(path: Path, sample_size: int, seed: int, locale: str | None, split: str | None):
    import pandas as pd

    frame = pd.read_parquet(path)
    if "query" not in frame.columns:
        raise ValueError(f"{path} does not contain a query column")
    if locale and "product_locale" in frame.columns:
        frame = frame.loc[frame["product_locale"].eq(locale)]
    if split and "split" in frame.columns:
        frame = frame.loc[frame["split"].eq(split)]

    if "query_id" in frame.columns:
        frame = frame.drop_duplicates("query_id")
    else:
        frame = frame.drop_duplicates("query")

    queries = frame["query"].dropna().astype(str).tolist()
    queries = [q.strip() for q in queries if q.strip()]
    if not queries:
        raise ValueError("No usable queries remain after filtering")

    rng = random.Random(seed)
    rng.shuffle(queries)
    return queries[: min(sample_size, len(queries))]


def call(api, query, hybrid, rerank, candidate_k):
    payload = json.dumps({
        "query": query,
        "top_k": 10,
        "candidate_k": candidate_k,
        "hybrid": hybrid,
        "rerank": rerank,
        "bypass_cache": True,
    }).encode()
    req = Request(
        api.rstrip("/") + "/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(req, timeout=180) as response:
        body = json.load(response)
    if body.get("cache_hit"):
        raise RuntimeError("Benchmark request unexpectedly hit the search cache")
    return body, (time.perf_counter() - started) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--query-file", type=Path, default=DEFAULT_QUERY_FILE)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--locale", default="us")
    parser.add_argument("--split", default="test")
    parser.add_argument("--queries", nargs="*")
    args = parser.parse_args()

    if args.queries:
        queries = args.queries
        source = "explicit --queries"
    elif args.query_file.exists():
        queries = load_queries(args.query_file, args.sample_size, args.seed, args.locale, args.split)
        source = str(args.query_file)
    else:
        queries = SMOKE_QUERIES
        source = "built-in 10-query smoke set"

    print(f"Query source: {source}")
    print(f"Benchmark queries: {len(queries)}")

    modes = [
        ("dense", False, False),
        ("hybrid_rrf", True, False),
        ("hybrid_rrf_rerank", True, True),
    ]

    for name, hybrid, rerank in modes:
        print(f"\nWarming {name}...", flush=True)
        call(args.api, "wireless keyboard", hybrid, rerank, args.candidate_k)

        values = {
            "http": [],
            "backend": [],
            "embedding": [],
            "faiss": [],
            "bm25": [],
            "postprocess": [],
            "rerank": [],
            "transport": [],
        }

        for index, query in enumerate(queries, start=1):
            body, elapsed = call(args.api, query, hybrid, rerank, args.candidate_k)
            backend = float(body["latency_ms"])
            values["http"].append(elapsed)
            values["backend"].append(backend)
            values["embedding"].append(float(body.get("embedding_latency_ms", 0.0)))
            values["faiss"].append(float(body.get("faiss_latency_ms", 0.0)))
            values["bm25"].append(float(body.get("bm25_latency_ms", 0.0)))
            values["postprocess"].append(float(body.get("postprocess_latency_ms", 0.0)))
            values["rerank"].append(float(body.get("rerank_latency_ms", 0.0)))
            values["transport"].append(elapsed - backend)
            if index % 25 == 0 or index == len(queries):
                print(f"  {index}/{len(queries)} queries", flush=True)

        print(f"{name} ({len(queries)} queries)")
        print(format_stats("HTTP", values["http"]))
        print(format_stats("Backend", values["backend"]))
        print(format_stats("Embedding", values["embedding"]))
        print(format_stats("FAISS", values["faiss"]))
        print(format_stats("BM25", values["bm25"]))
        print(format_stats("Postprocess", values["postprocess"]))
        print(format_stats("Rerank", values["rerank"]))
        print(format_stats("HTTP outside backend", values["transport"]))


if __name__ == "__main__":
    main()
