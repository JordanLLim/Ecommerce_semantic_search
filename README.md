# Amazon Shopping Query Ranking

A title-only semantic ranking baseline using the Amazon Shopping Queries (ESCI) dataset and frozen all-MiniLM-L6-v2 embeddings.

## Current scope

The notebook ranks the **provided products for each query**. It does not search the full product catalog. It includes data checks, a frozen baseline, per-query error analysis, and training-triplet preparation.

Fine-tuning, a cross-encoder reranker, an ANN index, and deployment are not implemented.

## Recorded result

The uploaded source notebook reports **mean NDCG@10 = 0.8415** on 1,000 sampled US small-version test queries (seed 42). This is a historical output, not a fresh run of this refactor. See `reports/source_run.json`.

- Relevance values: Exact = 3, Substitute = 2, Complement = 1, Irrelevant = 0.
- NDCG uses scikit-learn's linear gains and default tie handling.
- Scores are averaged equally across queries. All-zero relevance groups receive zero.
- This is a project-specific setup, not a claim of an official benchmark score.
- The original notebook prepared 11,833 triplets. It did not train a model.
- Exact dependency versions, model revision, sampled IDs and a checkpoint were not saved in the source notebook, so exact reproduction is not guaranteed.

## Run locally

Use Python 3.11 in a virtual environment:

```bash
python -m venv .venv
# Activate the environment for your operating system.
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/amazon_ranking.ipynb` and run from the top.

Place these files in `data/raw/` or set `AMAZON_DATA_DIR` to their folder:

- shopping_queries_dataset_examples.parquet
- shopping_queries_dataset_products.parquet

Use the Amazon Shopping Queries dataset already attached to your Kaggle notebook. Data is not included here; check its redistribution terms before uploading any data or derived text. Model loading requires internet access or a local model cache.

## Run on Kaggle

Upload the project ZIP as a private input, extract it into a writable working folder, and set `AMAZON_PROJECT_ROOT` to the extracted `amazon-ranking` directory before the setup cell. Keep the original ESCI dataset attached.

The notebook defaults to the original Kaggle dataset path when running under `/kaggle`; override `AMAZON_DATA_DIR` if your attachment uses a different path. Outputs default to `/kaggle/working/amazon-ranking-outputs`.

Downloading only the notebook is not enough: its `src/` folder is required. After a successful run, download the outputs and save a Kaggle version that includes them; writing to the session filesystem is not a durable backup.

## Files

| Path | Purpose |
| --- | --- |
| notebooks/amazon_ranking.ipynb | Exploration, experiment settings and interpretation |
| src/data.py | Loading, merge checks, split and evaluation sampling |
| src/ranking.py | Candidate scoring with normalized embeddings |
| src/evaluation.py | NDCG and per-query evaluation |
| src/triplets.py | Training-only triplet preparation and dataset adapter |
| tests/test_pipeline.py | Offline tests with small synthetic data |
| reports/source_run.json | Historical values read from the uploaded notebook |
| SPLIT_GUIDE.md | Original-cell mapping and changes |

The notebook writes sampled query IDs, configuration, package versions, per-query metrics, candidate scores and triplets into the output directory. These outputs are ignored by Git because they may contain dataset text.

## Validation

```bash
python -m unittest discover -s tests -v
```

Offline tests cover helper logic without downloading a model. A full dataset/model run is still required after this refactor. The requirements file is not a tested lockfile.

## Before fine-tuning

Split the training queries into train/validation groups before building training triplets. Use validation for sampling choices, hyperparameters and error-driven iteration. Do not feed inspected test examples into training.

The existing test set has already been inspected for errors. Document that exposure; do not describe it as an untouched final holdout. Future comparisons must use the same candidate set, query IDs and gain mapping. Add a simple lexical or random baseline before interpreting how strong 0.8415 is.

## Next work

- Query-level validation split and a training loop.
- Saved checkpoints and baseline/fine-tuned comparisons.
- Lexical baseline and controlled ablations.
- Full-catalog retrieval evaluated separately from candidate ranking.

This repository is an independent dataset project, not an Amazon affiliation.

