"""Build a FAISS product index from a CSV or Parquet catalog."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["product_id", "product_title"]
OPTIONAL_COLUMNS = ["product_brand", "product_color", "product_locale"]


def read_catalog(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema.names)
        columns = REQUIRED_COLUMNS + [name for name in OPTIONAL_COLUMNS if name in available]
        frame = pd.read_parquet(path, columns=columns)
    elif path.suffix.lower() == ".csv":
        header = pd.read_csv(path, nrows=0)
        available = set(header.columns)
        columns = REQUIRED_COLUMNS + [name for name in OPTIONAL_COLUMNS if name in available]
        frame = pd.read_csv(path, usecols=columns)
    else:
        raise ValueError("Catalog must be a .csv or .parquet file.")

    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Catalog is missing required columns: {missing}")

    frame = frame.dropna(subset=REQUIRED_COLUMNS)
    frame = frame.drop_duplicates("product_id", keep="first").reset_index(drop=True)
    frame["product_id"] = frame["product_id"].astype(str)
    frame["product_title"] = frame["product_title"].astype(str)
    if frame.empty:
        raise ValueError("Catalog has no usable products.")
    return frame


def encode_titles(model, titles, batch_size, show_progress_bar=False):
    embeddings = model.encode(
        titles,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
    return np.ascontiguousarray(embeddings, dtype="float32")


def build_index_chunked(
    products,
    model,
    index_type="flat",
    batch_size=256,
    chunk_size=50000,
    train_size=100000,
    nlist=100,
    nprobe=10,
):
    import faiss

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    total = len(products)
    sample_size = min(train_size, total)
    if sample_size < 1:
        raise ValueError("train_size must be positive.")

    if index_type == "ivf" and not 1 <= nlist <= sample_size:
        raise ValueError("nlist must be between 1 and the IVF training sample size.")

    # Encode a single title first so the index can be created without holding
    # the full catalog embedding matrix in memory.
    probe = encode_titles(model, [products.iloc[0]["product_title"]], batch_size=1)
    dimension = int(probe.shape[1])
    del probe

    if index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "ivf":
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(dimension), dimension, nlist, faiss.METRIC_INNER_PRODUCT
        )

        # Use a deterministic spread across the catalog for IVF training.
        train_indices = np.linspace(0, total - 1, num=sample_size, dtype=np.int64)
        train_titles = products.iloc[train_indices]["product_title"].tolist()
        print(f"Training IVF on {sample_size:,} sampled products...")
        train_vectors = encode_titles(
            model, train_titles, batch_size=batch_size, show_progress_bar=True
        )
        index.train(train_vectors)
        index.nprobe = min(nprobe, nlist)
        del train_vectors, train_titles, train_indices
        gc.collect()
        print("IVF training complete.")
    else:
        raise ValueError("index_type must be 'flat' or 'ivf'.")

    chunks = (total + chunk_size - 1) // chunk_size
    for chunk_number, start in enumerate(range(0, total, chunk_size), start=1):
        end = min(start + chunk_size, total)
        titles = products.iloc[start:end]["product_title"].tolist()
        print(
            f"Indexing chunk {chunk_number}/{chunks}: "
            f"products {start:,}-{end - 1:,} ({end - start:,})"
        )
        vectors = encode_titles(
            model, titles, batch_size=batch_size, show_progress_bar=True
        )
        index.add(vectors)
        del vectors, titles
        gc.collect()
        print(f"FAISS index now contains {index.ntotal:,} products.")

    return index, dimension


def write_artifacts(frame, index, output_dir, settings):
    import faiss

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "products.faiss"))
    frame.to_json(
        output_dir / "products.jsonl", orient="records", lines=True, force_ascii=False
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--index-type", choices=["flat", "ivf"], default="flat")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--nlist", type=int, default=100)
    parser.add_argument("--nprobe", type=int, default=10)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    products = read_catalog(args.catalog)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive.")
        products = products.head(args.limit).copy()

    print(f"Loaded {len(products):,} unique products.")
    model = SentenceTransformer(args.model)
    index, dimension = build_index_chunked(
        products=products,
        model=model,
        index_type=args.index_type,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        train_size=args.train_size,
        nlist=args.nlist,
        nprobe=args.nprobe,
    )

    settings = {
        "artifact_version": 2,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "index_type": args.index_type,
        "products": len(products),
        "dimension": dimension,
        "nlist": args.nlist if args.index_type == "ivf" else None,
        "nprobe": min(args.nprobe, args.nlist) if args.index_type == "ivf" else None,
        "chunk_size": args.chunk_size,
        "train_size": args.train_size if args.index_type == "ivf" else None,
        "metadata_columns": list(products.columns),
    }
    write_artifacts(products, index, args.output_dir, settings)
    print(f"Indexed {len(products):,} products in {args.output_dir}")


if __name__ == "__main__":
    main()
