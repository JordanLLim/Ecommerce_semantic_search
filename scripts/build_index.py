"""Build a FAISS product index from a CSV or Parquet catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_catalog(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=["product_id", "product_title"])
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, usecols=["product_id", "product_title"])
    else:
        raise ValueError("Catalog must be a .csv or .parquet file.")
    frame = frame.dropna(subset=["product_id", "product_title"])
    frame = frame.drop_duplicates("product_id", keep="first").reset_index(drop=True)
    frame["product_id"] = frame["product_id"].astype(str)
    frame["product_title"] = frame["product_title"].astype(str)
    if frame.empty:
        raise ValueError("Catalog has no usable products.")
    return frame


def build_faiss_index(embeddings, index_type="flat", nlist=100, nprobe=10):
    import faiss

    vectors = np.ascontiguousarray(embeddings, dtype="float32")
    if vectors.ndim != 2 or not len(vectors):
        raise ValueError("Expected a non-empty embedding matrix.")
    dimension = vectors.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "ivf":
        if not 1 <= nlist <= len(vectors):
            raise ValueError("nlist must be between 1 and the catalog size.")
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(dimension), dimension, nlist, faiss.METRIC_INNER_PRODUCT
        )
        index.train(vectors)
        index.nprobe = min(nprobe, nlist)
    else:
        raise ValueError("index_type must be 'flat' or 'ivf'.")
    index.add(vectors)
    return index


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
    parser.add_argument("--nlist", type=int, default=100)
    parser.add_argument("--nprobe", type=int, default=10)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    products = read_catalog(args.catalog)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive.")
        products = products.head(args.limit).copy()
    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        products["product_title"].tolist(),
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    index = build_faiss_index(embeddings, args.index_type, args.nlist, args.nprobe)
    settings = {
        "model": args.model,
        "index_type": args.index_type,
        "products": len(products),
        "dimension": int(np.asarray(embeddings).shape[1]),
        "nlist": args.nlist if args.index_type == "ivf" else None,
        "nprobe": min(args.nprobe, args.nlist) if args.index_type == "ivf" else None,
    }
    write_artifacts(products, index, args.output_dir, settings)
    print(f"Indexed {len(products):,} products in {args.output_dir}")


if __name__ == "__main__":
    main()
