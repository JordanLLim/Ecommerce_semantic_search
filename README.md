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
| `src/api.py` | FastAPI inference service |
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

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{"query":"quiet cooling fan","products":["USB cabinet fan","Tower fan with remote","Automotive cooling fan assembly"],"top_k":3}'
```

Set `MODEL_PATH` to use a local fine-tuned checkpoint. `/health` does not load the model, keeping startup observable.

## Docker

```bash
docker build -t amazon-semantic-ranker .
docker run --rm -p 8000:8000 amazon-semantic-ranker
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

The values in the JSON report were copied from saved notebook outputs. They were not rerun during this refactor because the raw data and checkpoints are not committed.

## Limitations

- Title-only representation; descriptions and structured attributes were excluded.
- One sampled Exact/Irrelevant pair per eligible query; no hard-negative mining.
- One epoch and two learning rates are not exhaustive tuning.
- The 10K pool is a controlled stress test, not full-catalog evaluation.
- Offline NDCG does not directly measure clicks or conversion.
