"""Benchmark dense, BM25+FAISS RRF, and optional reranking through the search API.

Run the API with PRELOAD_INDEX=1 first. This script reports observed HTTP latency
and does not claim relevance improvements; use the offline evaluation pipeline for that.
"""
from __future__ import annotations
import argparse, json, statistics, time
from urllib.request import Request, urlopen

DEFAULT_QUERIES = [
    "gaming keyboard", "wireless mouse", "wireless headphones", "cooling fan",
    "toddler books", "makeup vanity", "tv stand", "baby bag", "mechanical keyboard",
    "iphone charger",
]

def percentile(values, p):
    values = sorted(values)
    if not values: return 0.0
    pos = (len(values)-1) * p
    lo, hi = int(pos), min(int(pos)+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo]) * (pos-lo)

def call(api, query, hybrid, rerank, candidate_k):
    payload = json.dumps({"query": query, "top_k": 10, "candidate_k": candidate_k, "hybrid": hybrid, "rerank": rerank}).encode()
    req = Request(api.rstrip("/")+"/search", data=payload, headers={"Content-Type":"application/json"}, method="POST")
    started = time.perf_counter()
    with urlopen(req, timeout=180) as response:
        body = json.load(response)
    return body, (time.perf_counter()-started)*1000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    args = parser.parse_args()
    modes = [("dense",False,False),("hybrid_rrf",True,False),("hybrid_rrf_rerank",True,True)]
    for name, hybrid, rerank in modes:
        http, backend, embedding, faiss, bm25, rerank_ms = [], [], [], [], [], []
        for i, q in enumerate(args.queries):
            # suffix keeps repeated benchmark runs out of the exact-request cache
            body, elapsed = call(args.api, q+f" benchmark{i}", hybrid, rerank, args.candidate_k)
            http.append(elapsed); backend.append(body["latency_ms"]); embedding.append(body.get("embedding_latency_ms",0)); faiss.append(body.get("faiss_latency_ms",0)); bm25.append(body.get("bm25_latency_ms",0)); rerank_ms.append(body.get("rerank_latency_ms",0))
        print(f"\n{name} ({len(http)} queries)")
        print(f"HTTP mean/p50/p95/p99: {statistics.mean(http):.1f}/{percentile(http,.5):.1f}/{percentile(http,.95):.1f}/{percentile(http,.99):.1f} ms")
        print(f"Backend mean: {statistics.mean(backend):.1f} ms | embedding {statistics.mean(embedding):.1f} | FAISS {statistics.mean(faiss):.1f} | BM25 {statistics.mean(bm25):.1f} | rerank {statistics.mean(rerank_ms):.1f}")

if __name__ == "__main__": main()
