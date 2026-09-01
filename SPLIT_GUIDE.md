# Refactor guide

Cell numbers below are one-based positions in the uploaded notebook, not execution counts.

| Original cells | Destination | Reason |
| --- | --- | --- |
| 1 | Removed | Kaggle's recursive file listing is setup noise |
| 2-4, 11, 14, 23-24 | src/data.py plus notebook setup | Reusable loading and split logic |
| 5-13 | Notebook: data overview | Counts, missing values and exploration belong next to their interpretation |
| 15-17 | Requirements plus notebook model setup | Separate environment installation from analysis |
| 18-21, 25 | src/ranking.py plus sample display | Use one scoring function |
| 22, 26-29 | src/data.py and src/evaluation.py | Keep the original gains; evaluate only once |
| 30-31 | Notebook: error analysis | Inspect cached rankings instead of encoding again |
| 32-38 | src/triplets.py | Group once rather than repeatedly scanning the training dataframe |
| 39-40 | src/triplets.py and notebook | Dataset adapter and preview |
| 41 | Notebook: DataLoader preview | Define train_loader before using it |

## What changed

- Added short English section notes and comments.
- Kept one notebook so the analysis remains easy to follow.
- Removed repeated split/distribution cells and the second full evaluation pass.
- Preserved title-only scoring, gain mapping, sampling seed and per-query E/I sampling.
- Added merge, label, duplicate and train/test query-ID checks.
- Added safe evaluation handling for singleton candidate groups.
- Replaced the undefined loader with an explicit seeded DataLoader.
- Added output exports and run metadata. Historical output is kept separately; edited code cells have no inherited outputs.
- No training loop, trained checkpoint, retrieval index or new performance claim was added.

## Important distinctions

Triplet preparation is not fine-tuning. MiniLM here is a bi-encoder, not a cross-encoder reranker. The metric measures supplied-candidate ranking, not catalog-wide retrieval.

Missing product titles use an empty string during ranking, as in the original evaluation helper. Triplet construction instead raises an error on missing text. Inspect missing data before making a new cleaning decision.

Do not split into separate training or API modules yet: there is no implemented training or API code to move. Do not commit datasets, credentials, model weights or the generated text exports.

## Limits of this handoff

The uploaded file has 41 code cells and no markdown cells; its final cell has a NameError. The source outputs include the baseline result and prepared-triplet count. No dataset or model weights were attached for a full rerun.

