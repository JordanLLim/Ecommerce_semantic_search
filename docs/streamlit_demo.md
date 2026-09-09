# Streamlit search demo

The Streamlit app is a thin client for the existing FastAPI search service.
It does not load the embedding model or FAISS index itself.

## Architecture

```text
Streamlit UI
    |
    | HTTP POST /search
    v
FastAPI
    |
    +-- Sentence Transformer query embedding
    |
    +-- FAISS product index
    v
Top-k product results
```

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API from the repository root:

```bash
uvicorn src.api:app --reload
```

The API expects the persisted search artifacts under `artifacts/` by default.
Set `ARTIFACT_DIR` and `MODEL_PATH` if they live elsewhere.

In a second terminal, start Streamlit:

```bash
streamlit run app.py
```

By default the UI calls `http://localhost:8000`. To use another backend:

```bash
API_URL=https://your-api.example.com streamlit run app.py
```

## What the UI measures

- **Backend search** is the latency returned by the FastAPI `/search` endpoint. In the current backend this includes query embedding plus the FAISS search operation.
- **HTTP round trip** is measured by the Streamlit client around the complete POST request, so it also includes request/response and network overhead.

These values are different from the offline FAISS benchmark, which measures index search behavior separately.
