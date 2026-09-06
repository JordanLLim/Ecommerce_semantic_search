"""Data loading and query-level splitting utilities."""
from pathlib import Path
import pandas as pd

RELEVANCE = {"E": 3, "S": 2, "C": 1, "I": 0}

def load_us_data(data_dir):
    data_dir = Path(data_dir)
    examples = pd.read_parquet(data_dir / "shopping_queries_dataset_examples.parquet")
    products = pd.read_parquet(data_dir / "shopping_queries_dataset_products.parquet")
    examples = examples.loc[examples["product_locale"].eq("us")].copy()
    products = products.loc[products["product_locale"].eq("us")].copy()
    df = examples.merge(products, on=["product_locale", "product_id"],
                        how="left", validate="many_to_one", indicator=True)
    if not df["_merge"].eq("both").all():
        raise ValueError("Some examples have no matching product.")
    df = df.drop(columns="_merge")
    if df[["query_id", "query", "product_id", "esci_label"]].isna().any().any():
        raise ValueError("Missing query, product ID, or label.")
    if not df["esci_label"].isin(RELEVANCE).all():
        raise ValueError("Unknown ESCI label.")
    if df.duplicated(["query_id", "product_id"]).any():
        raise ValueError("Duplicate query-product pairs; inspect before continuing.")
    return df

def split_small_data(df):
    small = df.loc[df["small_version"].eq(1)].copy()
    if not small["split"].isin(["train", "test"]).all():
        raise ValueError("Unexpected split.")
    train = small.loc[small["split"].eq("train")].copy()
    test = small.loc[small["split"].eq("test")].copy()
    if train.empty or test.empty:
        raise ValueError("Both train and test data are required.")
    if set(train["query_id"]) & set(test["query_id"]):
        raise ValueError("Query IDs overlap between train and test.")
    return train, test

def split_train_validation(train, validation_fraction=0.10, seed=42):
    """Split training data by query ID to prevent query leakage."""
    if train.empty or not train["split"].eq("train").all():
        raise ValueError("Expected non-empty rows from the training split.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    query_ids = train["query_id"].drop_duplicates()
    validation_ids = query_ids.sample(frac=validation_fraction, random_state=seed)
    validation = train.loc[train["query_id"].isin(validation_ids)].copy()
    fit = train.loc[~train["query_id"].isin(validation_ids)].copy()
    if set(fit["query_id"]) & set(validation["query_id"]):
        raise AssertionError("Query leakage between fit and validation splits.")
    return fit, validation

def add_relevance(frame):
    """Return a copy with graded ESCI relevance values."""
    result = frame.copy()
    result["relevance"] = result["esci_label"].map(RELEVANCE)
    if result["relevance"].isna().any():
        raise ValueError("Unknown ESCI label.")
    return result

def sample_evaluation(test, n_queries=1000, seed=42):
    if n_queries < 1 or test.empty:
        raise ValueError("Evaluation requires queries.")
    ids = test["query_id"].drop_duplicates().sample(
        n=min(n_queries, test["query_id"].nunique()), random_state=seed)
    result = test.loc[test["query_id"].isin(ids)].copy()
    result = add_relevance(result)
    return result, ids
