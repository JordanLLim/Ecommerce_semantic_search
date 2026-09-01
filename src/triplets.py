"""Prepare one Exact/Irrelevant pair per eligible training query."""
import pandas as pd

def build_triplets(train, seed=42, max_triplets=20000):
    if max_triplets < 1 or not train["split"].eq("train").all():
        raise ValueError("Use training rows only and a positive triplet limit.")
    rows = []
    eligible = train.loc[train["esci_label"].isin(["E", "I"])]
    for query_id, group in eligible.groupby("query_id"):
        positives = group.loc[group["esci_label"].eq("E")]
        negatives = group.loc[group["esci_label"].eq("I")]
        if positives.empty or negatives.empty:
            continue
        # Keep the original per-group sampling rule for reproducibility.
        positive = positives.sample(n=1, random_state=seed).iloc[0]
        negative = negatives.sample(n=1, random_state=seed).iloc[0]
        rows.append({"query_id": query_id, "query": positive["query"],
                     "positive": positive["product_title"],
                     "negative": negative["product_title"]})
    result = pd.DataFrame(rows, columns=["query_id", "query", "positive", "negative"])
    if result[["query", "positive", "negative"]].isna().any().any():
        raise ValueError("Missing triplet text; inspect before training.")
    return result.sample(n=min(max_triplets, len(result)),
                         random_state=seed).reset_index(drop=True)

class TripletDataset:
    """A map-style dataset accepted by PyTorch DataLoader."""
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx][["query", "positive", "negative"]].to_dict()

