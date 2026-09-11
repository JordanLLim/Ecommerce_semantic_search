# Hybrid benchmark protocol

The serving comparison uses the same query list and `candidate_k` for three modes:

1. Dense FAISS retrieval.
2. Independent BM25 + FAISS retrieval fused with RRF.
3. BM25 + FAISS + RRF followed by the existing CrossEncoder reranker.

Run the API in a preloaded/warmed state first, then execute:

```powershell
python scripts/benchmark_hybrid.py --api http://localhost:8000 --candidate-k 50
```

The script reports HTTP mean/p50/p95/p99 and mean backend component timings. These measurements are serving-performance measurements, not relevance metrics.

Do not claim that hybrid search or reranking improves relevance from this latency benchmark. Relevance must be evaluated separately on held-out ESCI judgments with the same query set and ranking metrics such as NDCG@10 and MRR@10.

Do not compare the HTTP percentiles directly with the batch-normalized FAISS benchmark: the latter excludes query encoding, Python processing and HTTP overhead.