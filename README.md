# Amazon Semantic Search and Product Ranking

End-to-end ecommerce search on the Amazon Shopping Queries ESCI dataset, covering offline relevance experiments, 1.3M-product ANN retrieval, true sparse+dense hybrid search, cross-encoder reranking, FastAPI serving, Streamlit UI, latency profiling, Docker Compose and reproducible benchmarks.

> Independent portfolio project; not affiliated with Amazon. Raw data, credentials, generated indexes and model weights are not committed.

## Architecture

```text
Amazon catalog
   |                         offline
   +-> MiniLM embeddings -> FAISS Flat / IVF artifacts

User query
   |
   +-----------------> BM25 sparse retrieval --------+
   |                                                  |
   +-> MiniLM query embedding -> FAISS dense ANN -----+-> RRF fusion
                                                         |
                                                   hard filters
                                                         |
                                              optional CrossEncoder
                                                         |
                                                     top-k JSON
                                                         |
                                                FastAPI -> Streamlit
```

BM25 and FAISS are independent first-stage retrievers. Reciprocal Rank Fusion (RRF) combines their ranked candidate lists without requiring dense and sparse scores to share a scale. The CrossEncoder remains an optional second-stage reranker.

## Offline relevance results

All candidate-ranking methods use the same 2,089 held-out queries and 42,185 query-product pairs. ESCI relevance is Exact=3, Substitute=2, Complement=1, Irrelevant=0.

| Model | NDCG@10 | P@1 | P@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | 0.8103 | 0.5577 | 0.5114 | 0.6417 | 0.7007 |
| Frozen MiniLM | 0.8339 | 0.6276 | 0.5469 | 0.6710 | 0.7538 |
| Fine-tuned A, lr=2e-5 | **0.8458** | **0.6496** | **0.5661** | **0.6852** | **0.7704** |
| Fine-tuned B, lr=1e-5 | 0.8433 | **0.6496** | 0.5649 | 0.6832 | 0.7677 |

Experiment A improved NDCG@10 by 0.0119 over frozen MiniLM. The paired query-level bootstrap 95% interval for that change was [0.0084, 0.0156].

### Controlled retrieval

A separate 10K-product experiment retained judged products and added catalog distractors. Because relevance labels are incomplete outside the supplied ESCI pairs, these are reported as known-E metrics rather than full-catalog relevance ground truth.

| Model | Known-E Recall@10 | Known-E Recall@50 | Known-E Hit@10 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.4283 | 0.7224 | 0.800 |
| Frozen MiniLM | 0.5084 | 0.8036 | 0.885 |
| Fine-tuned A | 0.5056 | 0.7968 | 0.880 |
| Fine-tuned B | **0.5183** | **0.8194** | **0.890** |

Full saved experiment values are in `reports/experiment_results.json`.

## 1.3M-product FAISS benchmark

FAISS IVF is compared against exact `IndexFlatIP` using the same normalized embeddings. ANN Recall@10 below means overlap with Flat's exact top-10 neighbors; it is an index-fidelity metric, not ESCI relevance.

| Index | nlist | nprobe | ANN Recall@10 | Batch-normalized FAISS time/query |
| --- | ---: | ---: | ---: | ---: |
| Flat | - | - | 1.0000 | 5.129 ms |
| IVF | 4096 | 16 | 0.8618 | 0.411 ms |
| IVF | 4096 | 32 | 0.9044 | 0.711 ms |
| IVF | 4096 | 64 | 0.9379 | 1.375 ms |
| **IVF** | **4096** | **128** | **0.9592** | **2.615 ms** |
| IVF | 4096 | 256 | 0.9753 | 5.097 ms |

`nprobe=128` is the selected operating point from this benchmark: 95.92% top-10 overlap with Flat at about half its batch-normalized FAISS search time. These timings exclude query encoding and HTTP overhead.

## Search pipeline

The serving path supports persisted FAISS Flat/IVF indexes, independent in-process BM25 retrieval, RRF fusion, exact brand/locale filters, optional CrossEncoder reranking, deterministic experiment assignment, process-local LRU caching, JSONL query telemetry, `/metrics`, and component latency profiling.

The current BM25 implementation is intentionally dependency-free for a reproducible portfolio deployment. It demonstrates a real independent sparse retrieval branch, but it is not presented as a replacement for distributed OpenSearch/Elasticsearch infrastructure at production traffic scale.

## Index lifecycle

```text
Offline: catalog -> clean/deduplicate -> encode titles -> build FAISS -> save aligned artifacts
Online: startup -> load metadata/model/FAISS -> warm inference -> ready -> query -> retrieve/fuse/rerank -> JSON
```

The same embedding model must be used to build and query an index. Catalog/model changes require a compatible rebuild or a deliberately designed incremental update path.

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Build an index

Small exact example:

```bash
python scripts/build_index.py --catalog data/raw/shopping_queries_dataset_products.parquet --limit 10000 --index-type flat --output-dir artifacts
```

Large IVF example:

```bash
python scripts/build_index.py --catalog data/raw/shopping_queries_dataset_products.parquet --limit 1300000 --index-type ivf --nlist 4096 --nprobe 128 --output-dir artifacts
```

Generated artifacts are kept outside Git.

## Run locally

For the large catalog, **preload and warm the backend before sending requests**. Lazy startup can make the first request include model/index cold-start work and should not be used as a steady-state latency measurement.

### Windows PowerShell

```powershell
$env:PRELOAD_INDEX="1"; uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete.` before starting the UI or running benchmarks. In another PowerShell window:

```powershell
$env:API_URL="http://127.0.0.1:8000"; streamlit run app.py
```

On the Windows machine used for local profiling, `http://localhost:8000/health` incurred about 2 seconds of transport overhead while `http://127.0.0.1:8000/health` returned in about 12 ms on average. For local Windows benchmarks, use `127.0.0.1` so proxy/PAC or hostname-resolution behavior is not mixed into search latency.

### Linux / macOS

```bash
PRELOAD_INDEX=1 uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Then:

```bash
API_URL=http://localhost:8000 streamlit run app.py
```

Docker Compose already enables backend preloading. More detail is in `docs/LOCAL_SERVING.md`.

### API example

```bash
curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" -d '{"query":"quiet cooling fan","top_k":10,"candidate_k":50,"hybrid":true,"rerank":true}'
```

Useful endpoints: `GET /health`, `GET /metrics`, `POST /rank`, `POST /search`.

## Benchmarks

- `scripts/benchmark_faiss.py` — Flat vs IVF ANN approximation and batch timing.
- `scripts/benchmark_serving_latency.py` — one-query-at-a-time backend latency.
- `scripts/benchmark_hybrid.py` — Dense vs BM25+FAISS RRF vs hybrid+CrossEncoder HTTP latency. Defaults to `http://127.0.0.1:8000` for reliable local Windows timing.
- `scripts/benchmark_local_http.py` — diagnostic for localhost vs 127.0.0.1 transport overhead.
- `scripts/benchmark_hnsw.py` — experimental Flat/HNSW comparison.

### Local 100-query hybrid serving profile

A final local profile used 100 real held-out Amazon ESCI test queries, a preloaded FastAPI process, `127.0.0.1`, explicit cache bypass, `candidate_k=50`, and a 1.3M-product catalog. The table below is one coherent run with 10 CrossEncoder rerank candidates; it is a local engineering profile, not a production SLA.

| Mode | Backend mean | Backend p50 | Backend p95 | Backend p99 |
| --- | ---: | ---: | ---: | ---: |
| Dense FAISS | 139.4 ms | 87.6 ms | 350.6 ms | 548.3 ms |
| BM25 + FAISS RRF | 231.1 ms | 161.7 ms | 621.2 ms | 900.7 ms |
| Hybrid + CrossEncoder | 366.1 ms | 300.0 ms | 757.9 ms | 957.8 ms |

For the hybrid+rereanker path in that run, the CrossEncoder itself used 206.2 ms mean / 198.4 ms p50 / 328.6 ms p95. A separate run with 20 rerank candidates measured 352.8 ms mean / 332.5 ms p50 / 566.9 ms p95 for the reranker component. This shows the expected latency cost of widening the second-stage candidate set, but it does **not** establish a relevance winner because reranker quality was not separately evaluated for 10 vs 20 candidates. The service therefore keeps 20 as the quality-oriented default while exposing a request-level override for latency experiments.

The in-process BM25 path also shows broad-term tail-latency variability on the 1.3M catalog. This is documented as a deployment boundary rather than hidden: a production-scale system would normally move sparse retrieval to OpenSearch/Elasticsearch or another dedicated search engine instead of relying on a Python in-process postings index.

Relevance and latency are deliberately evaluated separately. A faster ANN index does not prove better relevance, and an offline NDCG improvement does not prove higher CTR or conversion.

## Docker Compose

Put generated index files under `artifacts/`, then run:

```bash
docker compose up --build
```

FastAPI is exposed on `localhost:8000` and Streamlit on `localhost:8501`. Artifacts are mounted read-only rather than baked into the image.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/data.py` | loading, validation and query-level splits |
| `src/training.py` / `src/triplets.py` | encoder fine-tuning |
| `src/evaluation.py` | NDCG, precision, recall, MRR and bootstrap utilities |
| `src/retrieval.py` | controlled retrieval evaluation |
| `src/search.py` | online FAISS/BM25 retrieval, fusion and filtering |
| `src/hybrid.py` | BM25 sparse index and RRF |
| `src/reranking.py` | optional CrossEncoder reranker |
| `src/serving.py` | cache, telemetry and experiment utilities |
| `src/api.py` | FastAPI service |
| `app.py` | Streamlit client |
| `scripts/` | index building and benchmarks |
| `tests/` | unit/integration-oriented tests |

## Evaluation boundaries and limitations

The project keeps four concepts separate: ESCI relevance metrics measure ranking quality; ANN Recall measures approximate-index fidelity; latency measures system performance; real user satisfaction would require online metrics such as CTR, add-to-cart, conversion, reformulation and abandonment.

Current limitations include title-only dense embeddings, post-retrieval brand/locale filtering rather than vector-database-native pre-filtering, process-local cache/metrics, an in-process BM25 index, and no claimed online business impact. Hybrid retrieval and reranking must be benchmarked on the same held-out queries before claiming relevance gains. A live cloud deployment still requires generated artifacts and deployment credentials.