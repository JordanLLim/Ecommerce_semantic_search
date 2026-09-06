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
