# Amazon Semantic Search and Product Ranking

Experiments with lexical and semantic product ranking on the Amazon Shopping Queries ESCI dataset. The project compares TF-IDF with MiniLM, fine-tunes the encoder with triplet loss, and tests both candidate ranking and retrieval from a larger product pool.

> Independent portfolio project; not affiliated with Amazon. Raw data, credentials and model weights are not committed.

## Results

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

I use the name **known-E recall** because ESCI does not label every randomly sampled catalog product. With pool seeds 17, 43 and 97, Experiment B averaged 0.5242 Recall@10 and 0.8223 Recall@50. Full values are in [`reports/experiment_results.json`](reports/experiment_results.json).

### FAISS scalability benchmark

The serving benchmark compares IVF approximate search with exact `IndexFlatIP` search using normalized MiniLM embeddings. It uses 200 sampled Amazon queries and reports ANN Recall@10 as overlap with Flat's exact top-10 neighbors. This is an index approximation metric, not ESCI relevance or model accuracy. Embedding time is excluded so the latency numbers isolate FAISS search behavior.

| Catalog | Index | nlist | nprobe | ANN Recall@10 vs Flat | Mean latency/query | Speedup vs Flat |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 50K | Flat | - | - | 1.0000 | 2.140 ms | 1.0x |
| 50K | IVF | 224 | 64 | 0.9550 | 0.645 ms | 3.3x |
| 500K | Flat | - | - | 1.0000 | 29.614 ms | 1.0x |
| 500K | IVF | 707 | 96 | 0.9475 | 4.161 ms | 7.1x |
| **500K** | **IVF** | **707** | **104** | **0.9515** | **4.479 ms** | **6.6x** |
| 500K | IVF | 707 | 128 | 0.9570 | 5.536 ms | 5.3x |

For the 500K benchmark, `nprobe=104` was the first tested setting to cross a 95% ANN Recall@10 target. It retained **95.15%** of Flat's exact top-10 neighbors while reducing mean search latency from **29.614 ms to 4.479 ms**, about **6.6x faster**. Higher `nprobe` values improved recall further but with diminishing returns in latency.

Run the benchmark with [`scripts/benchmark_faiss.py`](scripts/benchmark_faiss.py). Results are machine-dependent, so the table records the measured experiment rather than a general FAISS performance claim.

## Leakage controls

- Train, validation and test boundaries are defined by **query ID**, not rows.
- TF-IDF vocabulary is fit only on fit-split product titles.
- Validation queries are excluded before triplet construction.
- Models use identical evaluation pairs and deterministic tie handling.
- The initially inspected test sample is documented as exposed and is not used for model selection.

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
| `src/search.py` | Persisted FAISS index loading and search |
| `src/api.py` | FastAPI inference service |
| `scripts/build_index.py` | Product embedding and FAISS index builder |
| `scripts/benchmark_faiss.py` | Flat/IVF ANN recall and latency benchmark |
| `tests/test_pipeline.py` | Offline synthetic tests |
| `notebooks/amazon_ranking.ipynb` | Compact baseline walkthrough |

## Data

Place the Amazon Shopping Queries files below in `data/raw/`, or set `AMAZON_DATA_DIR`:

- `shopping_queries_dataset_examples.parquet`
- `shopping_queries_dataset_products.parquet`

The full US merge contained 1,818,825 rows, 97,345 queries and 1,215,851 products. The main experiment used the provided small-version split.

## Setup and tests

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
jupyter lab
```

Model downloads require internet access or a local Hugging Face cache. GPU training is practical on Kaggle or Colab; utilities and synthetic tests run offline.

## API

Build a small exact-search index first. Use `--model` with a local Experiment B checkpoint when it is available.

```bash
python scripts/build_index.py \
  --catalog data/raw/shopping_queries_dataset_products.parquet \
  --limit 10000 \
  --index-type flat \
  --output-dir artifacts
```

For a larger catalog, IVF avoids comparing the query with every product. `nprobe` controls the speed-recall trade-off by setting how many clusters are searched.

```bash
python scripts/build_index.py \
  --catalog data/raw/shopping_queries_dataset_products.parquet \
  --limit 500000 \
  --index-type ivf \
  --nlist 707 \
  --nprobe 104 \
  --output-dir artifacts
```

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{"query":"quiet cooling fan","products":["USB cabinet fan","Tower fan with remote","Automotive cooling fan assembly"],"top_k":3}'
```

`/rank` orders a supplied product list. `/search` retrieves products from the persisted index:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"quiet cooling fan","top_k":10}'
```

Set `MODEL_PATH` to use a local fine-tuned checkpoint and `ARTIFACT_DIR` when the index is stored elsewhere. The same model must be used to build and query an index. `/health` reports whether the rank model and search index have been loaded.

## Docker

```bash
docker build -t amazon-semantic-ranker .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts:/app/artifacts:ro" \
  amazon-semantic-ranker
```

For a fixed deployment, set `MODEL_PATH` to a saved local checkpoint instead of downloading a model at startup.

## Reproducing experiments

1. Load the US merge and create the provided small split.
2. Split training rows by query ID with `split_train_validation`.
3. Fit TF-IDF only on unique fit-split titles.
4. Build triplets only from the fit split.
5. Train independent A/B models from the same initialization.
6. Score the fixed validation pairs and compute candidate metrics.
7. Align queries and run a paired bootstrap.
8. Build controlled pools retaining every judged product.
9. Repeat with multiple distractor seeds.
10. Benchmark Flat and IVF on sampled real queries, keeping embedding time outside the FAISS latency measurement.

The values in the JSON report were copied from saved notebook outputs. They were not rerun during this refactor because the raw data and checkpoints are not committed. FAISS benchmark values were measured separately on the experiment machine and are hardware-dependent.

## Limitations

- Title-only representation; descriptions and structured attributes were excluded.
- One sampled Exact/Irrelevant pair per eligible query; no hard-negative mining.
- One epoch and two learning rates are not exhaustive tuning.
- The 10K pool is a controlled stress test, not full-catalog relevance evaluation.
- The FAISS serving benchmark currently scales to 500K indexed products; full-catalog ANN benchmarking remains future work.
- Offline NDCG does not directly measure clicks or conversion.
