# Amazon Semantic Search and Product Ranking

End-to-end semantic product search on the Amazon Shopping Queries ESCI dataset. The project covers offline relevance experiments, large-catalog ANN retrieval, a FastAPI serving layer, optional reranking and lexical fusion, a Streamlit client, observability utilities, Docker deployment and reproducible benchmarks.

> Independent portfolio project; not affiliated with Amazon. Raw data, credentials, generated indexes and model weights are not committed.

## System architecture

```text
Amazon product catalog
        |
        v
Sentence Transformer embeddings
        |
        v
FAISS index artifacts --------------------------+
                                                |
Streamlit UI -- HTTP --> FastAPI /search -------+
                                                |
                                  query embedding
                                                |
                                           FAISS IVF
                                                |
                                  metadata filtering
                                                |
                          optional lexical fusion / preferences
                                                |
                              optional cross-encoder reranker
                                                |
                                           top-k JSON
```

The UI and retrieval service are intentionally separated. Streamlit is a client; model/index loading and search logic remain behind the FastAPI API.

## Offline relevance results

### Candidate ranking

All methods use the same 2,089 held-out validation queries and 42,185 supplied query-product pairs. Graded relevance is Exact=3, Substitute=2, Complement=1 and Irrelevant=0.

| Model | NDCG@10 | P@1 | P@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | 0.8103 | 0.5577 | 0.5114 | 0.6417 | 0.7007 |
| Frozen MiniLM | 0.8339 | 0.6276 | 0.5469 | 0.6710 | 0.7538 |
| Fine-tuned A, lr=2e-5 | **0.8458** | **0.6496** | **0.5661** | **0.6852** | **0.7704** |
| Fine-tuned B, lr=1e-5 | 0.8433 | **0.6496** | 0.5649 | 0.6787 | 0.7698 |

Experiment A improved NDCG@10 by **0.0119** over the frozen model. The paired bootstrap 95% interval was **[0.0084, 0.0156]**. Across individual queries, 51.89% improved, 41.41% worsened and 6.70% were unchanged.

### Controlled 10K-product retrieval

Candidate ranking is not catalog retrieval. A separate test sampled 200 validation queries, retained all 3,941 judged products and added 6,059 label-free catalog distractors.

| Model | Known-E Recall@10 | Known-E Recall@50 | Known-E Hit@10 |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.4283 | 0.7224 | 0.800 |
| Frozen MiniLM | 0.5084 | 0.8036 | 0.885 |
| Fine-tuned A | 0.5056 | 0.7968 | 0.880 |
| Fine-tuned B | **0.5183** | **0.8194** | **0.890** |

I use the name **known-E recall** because ESCI does not label every randomly sampled catalog product. With pool seeds 17, 43 and 97, Experiment B averaged 0.5242 Recall@10 and 0.8223 Recall@50. Full saved values are in [`reports/experiment_results.json`](reports/experiment_results.json).

## ANN scalability benchmark

FAISS IVF is compared with exact `IndexFlatIP` using the same normalized MiniLM embeddings. ANN Recall@10 is the overlap between IVF and Flat top-10 neighbours. It measures index approximation quality, **not product relevance or model accuracy**.

The current benchmark function searches a batch of queries in one FAISS call and divides total search time by the number of queries. The timing below is therefore **batch-normalized FAISS search time/query**, not production HTTP latency and not true single-request p95 latency.

### 1.3M-product benchmark, 1,000 real Amazon queries

| Index | nlist | nprobe | ANN Recall@10 vs Flat | Batch-normalized search time/query |
| --- | ---: | ---: | ---: | ---: |
| Flat | - | - | 1.0000 | 5.129 ms |
| IVF | 4096 | 16 | 0.8618 | 0.411 ms |
| IVF | 4096 | 32 | 0.9044 | 0.711 ms |
| IVF | 4096 | 64 | 0.9379 | 1.375 ms |
| **IVF** | **4096** | **128** | **0.9592** | **2.615 ms** |
| IVF | 4096 | 256 | 0.9753 | 5.097 ms |

`nprobe=128` is the selected operating point among these runs: it retained **95.92%** of Flat's top-10 neighbours while using roughly **half the batch-normalized FAISS search time**. `nprobe=256` improved recall to 97.53% but removed almost all of the timing advantage.

An earlier `nlist=1100` run made IVF slower than Flat at high `nprobe`; increasing the number of inverted lists to 4096 produced a much better speed-recall trade-off. This is kept as an engineering finding rather than assuming IVF is automatically faster for every index configuration.

Run [`scripts/benchmark_faiss.py`](scripts/benchmark_faiss.py) for Flat/IVF approximation tests. Run [`scripts/benchmark_serving_latency.py`](scripts/benchmark_serving_latency.py) for one-query-at-a-time backend latency including query embedding. [`scripts/benchmark_hnsw.py`](scripts/benchmark_hnsw.py) is provided to compare HNSW before making a production index choice.
## Design decisions

The project was developed in stages rather than treating semantic search as a single modelling problem. Each stage answers a different question: whether semantic representations improve ranking, whether task-specific fine-tuning improves those representations, and whether the approach can support retrieval from a much larger product catalog.

### TF-IDF vs MiniLM

TF-IDF provides a simple lexical baseline and works well when query and product titles share the same terms. It is useful because it shows how much of the task can be solved by exact or near-exact word overlap.

Frozen MiniLM is used as the first semantic baseline. This separates gains from pretrained semantic representations from gains introduced later by task-specific fine-tuning.

### Why triplet loss?

The goal is not to predict an ESCI class directly, but to improve the embedding space used for ranking and retrieval. Triplet training encourages an Exact product to move closer to its query while pushing an Irrelevant product farther away.

The first version uses a simple Exact/Irrelevant sampling strategy to keep the experiment interpretable and reproducible. Hard-negative mining is a possible extension, but is kept separate so its effect can be measured rather than mixed into the baseline experiment.

### Candidate ranking vs catalog retrieval

Ranking a supplied candidate set and searching a catalog are different problems.

Candidate ranking asks: **given a known set of products, can the model order them correctly?**

Catalog retrieval asks: **can the system find relevant products when they are mixed with many unrelated products?**

For this reason, NDCG, Precision, Recall and MRR are first measured on the fixed ESCI query-product pairs. A separate retrieval experiment then adds thousands of catalog distractors while retaining every judged product.

### Why repeat retrieval with multiple seeds?

A retrieval result based on one random distractor pool can depend on which products happened to be sampled. Repeating the experiment with multiple seeds provides a simple robustness check and shows whether model comparisons remain stable when the candidate pool changes.

### Why paired bootstrap confidence intervals?

A small improvement in average NDCG does not necessarily mean one model is consistently better. Paired bootstrap resampling is therefore performed at the query level, with every model evaluated on the same queries.

This is especially useful here because fine-tuning improves some queries while degrading others.

### Why keep both fine-tuned experiments?

Experiment A achieves the strongest candidate-ranking NDCG@10, while Experiment B performs better on the controlled known-E retrieval test.

Both are retained because the model that ranks a small candidate set best is not necessarily the model that retrieves best from a larger pool. Keeping both results makes that trade-off visible instead of selecting a winner from a single metric.

### Exact search and approximate search

The current controlled retrieval experiment is small enough for exhaustive similarity search. The next systems step is to add FAISS and compare exact and approximate nearest-neighbour indexes.

**IndexFlatIP** performs exhaustive search over all indexed embeddings. It is slower as the catalog grows, but provides an exact reference for the embedding model.

**IndexIVFFlat** first partitions the embedding space into clusters and searches only selected clusters. It reduces the amount of the catalog examined for each query, but introduces a recall-latency trade-off controlled partly by `nprobe`.

Flat search is therefore not made obsolete by IVF. It acts as the exact reference used to measure how much retrieval quality is lost when approximate search is introduced.

The intended comparison is:

```text
IndexFlatIP   -> exact-search reference
IndexIVFFlat  -> approximate candidate retrieval
```

The useful production question is not simply whether IVF is faster, but how much recall is retained for the latency improvement.

### Why not immediately treat the full 1.2M-product catalog as evaluation ground truth?

The full US catalog contains more than 1.2 million unique products, but ESCI relevance judgments are incomplete outside the supplied query-product pairs. A randomly retrieved product with no ESCI label is not necessarily irrelevant.

The current controlled pools therefore retain all known judged products and use additional catalog products as distractors. Full-catalog ANN search can still be useful for scalability and latency experiments, but incomplete labels should not be treated as reliable negative relevance judgments.

### Current engineering direction

The planned retrieval path is:

```text
User query
    |
    v
MiniLM query encoder
    |
    v
FAISS candidate retrieval
    |
    v
Top-K product candidates
    |
    v
Semantic ranking
    |
    v
Ranked results
```

The next benchmark should measure retrieval recall and query latency together across different FAISS configurations. The objective is to find an operating point where approximate retrieval reduces search cost while preserving most of the exact-search result quality.

## Leakage controls

## Search pipeline

The online pipeline supports:

- FAISS Flat or IVF indexes persisted to disk.
- Configurable `nlist`/`nprobe` stored with index metadata.
- Optional brand and locale filtering when those fields exist in the product catalog.
- Dense + lexical candidate fusion. This is post-ANN lexical fusion, **not a separate BM25 candidate generator**.
- Optional cross-encoder reranking of retrieved candidates.
- Small preference-keyword boosts to demonstrate a personalization hook.
- Deterministic dense/enhanced A/B assignment from a stable experiment key.
- In-process LRU caching for repeated identical requests.
- JSONL query telemetry and lightweight `/metrics` counters.
- Model/index artifact version metadata.
- Configurable embedding and reranker model paths. A multilingual embedding model can be substituted, but multilingual relevance should be evaluated before making quality claims.

## Index lifecycle

Offline index build:

```text
catalog -> clean/deduplicate -> encode product titles -> train/build FAISS -> save index + aligned metadata + settings
```

Online serving:

```text
service startup -> load model/index once -> encode each query -> retrieve -> optional post-processing -> return JSON
```

Catalog/model changes require a compatible rebuild or a deliberately designed incremental update path. The serving code does not rebuild the index per request.

## Layout

| Path | Purpose |
| --- | --- |
| `src/data.py` | Validated loading, query-level splits, relevance mapping |
| `src/triplets.py` | Deterministic Exact/Irrelevant triplets |
| `src/training.py` | Configured cosine-triplet training loop |
| `src/ranking.py` | Candidate and online ranking |
| `src/lexical.py` | TF-IDF baseline |
| `src/evaluation.py` | NDCG, P@K, Recall, MRR and paired bootstrap |
| `src/retrieval.py` | Controlled pools and known-E metrics |
| `src/search.py` | Persisted FAISS search, filtering and lexical fusion |
| `src/reranking.py` | Optional cross-encoder reranker |
| `src/serving.py` | Cache, telemetry, metrics and A/B utilities |
| `src/api.py` | FastAPI inference service |
| `app.py` | Streamlit HTTP client |
| `scripts/build_index.py` | Product embedding and FAISS artifact builder |
| `scripts/benchmark_faiss.py` | Flat/IVF ANN recall and batched timing benchmark |
| `scripts/benchmark_serving_latency.py` | Single-query backend latency benchmark |
| `scripts/benchmark_hnsw.py` | Flat/HNSW comparison experiment |
| `tests/` | Offline and serving utility tests |

## Data

Place the Amazon Shopping Queries files below in `data/raw/`, or set `AMAZON_DATA_DIR`:

- `shopping_queries_dataset_examples.parquet`
- `shopping_queries_dataset_products.parquet`

The earlier US experiment merge contained 1,818,825 query-product rows, 97,345 queries and 1,215,851 products. The ANN benchmark above used a separately loaded 1.3M-product catalog benchmark set, so those counts should not be treated as the same filtered dataset.

## Setup and tests

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

CI compiles Python sources and runs the unit test suite on pushes and pull requests.

## Build an index

Small exact index:

```bash
python scripts/build_index.py \
  --catalog data/raw/shopping_queries_dataset_products.parquet \
  --limit 10000 \
  --index-type flat \
  --output-dir artifacts
```

Large IVF example using the selected benchmark configuration:

```bash
python scripts/build_index.py \
  --catalog data/raw/shopping_queries_dataset_products.parquet \
  --limit 1300000 \
  --index-type ivf \
  --nlist 4096 \
  --nprobe 128 \
  --output-dir artifacts
```

The same embedding model must be used to build and query an index.

## Run locally

Backend:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

UI:

```bash
API_URL=http://localhost:8000 streamlit run app.py
```

Open the Streamlit URL and compare dense retrieval, lexical fusion and cross-encoder reranking interactively.

### Search API

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":"quiet cooling fan",
    "top_k":10,
    "candidate_k":50,
    "hybrid":true,
    "rerank":true
  }'
```

Useful endpoints:

- `GET /health`: service and loaded-artifact information.
- `GET /metrics`: request/error/cache/variant counters.
- `POST /rank`: rank a supplied product list.
- `POST /search`: retrieve from the persisted catalog index.

Environment variables include `MODEL_PATH`, `RERANKER_MODEL`, `ARTIFACT_DIR`, `QUERY_LOG_PATH`, `SEARCH_CACHE_SIZE` and `API_URL` for the UI.

## Docker Compose

Put the generated index files under `artifacts/`, then run:

```bash
docker compose up --build
```

The API is exposed on `localhost:8000` and Streamlit on `localhost:8501`. The UI container calls the API container through the Compose network. Generated indexes remain outside the image and are mounted read-only into the API container.

## Evaluation boundaries

Two different layers are intentionally reported separately:

1. **Semantic relevance** — NDCG, precision, MRR and known-E retrieval metrics using ESCI judgments.
2. **ANN approximation** — overlap of approximate IVF/HNSW neighbours with exact Flat neighbours.

High ANN recall does not prove high product relevance. Likewise, offline NDCG does not directly measure clicks, conversion or business value.

## Reproducing experiments

1. Load and validate the Amazon data.
2. Split training/validation by query ID, never by individual rows.
3. Fit lexical baselines on the fit split only.
4. Construct triplets without validation leakage.
5. Train independent encoder experiments from the same initialization.
6. Evaluate identical held-out candidate sets and run paired bootstrap analysis.
7. Build controlled retrieval pools retaining judged products.
8. Build persisted Flat/IVF artifacts for serving.
9. Benchmark ANN recall against Flat with real Amazon queries.
10. Measure single-query serving latency separately from batch FAISS throughput.
11. Run API/UI integration with the generated artifacts.

## Current limitations

- Title-only dense representation remains the main retrieval signal; descriptions and attributes are stored when present but are not yet embedded jointly.
- The lexical fusion step reranks ANN candidates; it is not a fully independent BM25 retrieval branch.
- Cross-encoder reranking increases quality capacity but also adds serving latency and still requires relevance evaluation on the target candidate set.
- Preference keywords are a demonstration hook, not learned personalization.
- The cache and metrics are process-local; a distributed deployment would use shared infrastructure such as Redis and a dedicated metrics stack.
- JSONL query logging is suitable for a portfolio deployment, not a high-volume production logging pipeline.
- The 1.3M ANN figures are hardware-dependent benchmark measurements, not universal FAISS performance claims.
- A live cloud deployment still requires generated artifacts and deployment credentials; neither is committed to the repository.
- The expanded serving path in this branch has not yet been validated end-to-end against the user's generated 1.3M artifacts. CI covers syntax and unit tests; real integration validation must use those local artifacts before merge/resume claims.
