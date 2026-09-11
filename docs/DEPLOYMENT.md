# Deployment

The serving stack runs as two containers:

- FastAPI API on port `8000`
- Streamlit storefront on port `8501`

Large 1.3M-product artifacts are mounted at runtime rather than committed to Git.

## Runtime artifacts

```text
artifacts/
├── products.faiss
├── products.jsonl
└── metadata.json
```

## Local Docker validation

```bash
docker compose build
docker compose up
```

Open:

- storefront: `http://localhost:8501`
- API health: `http://localhost:8000/health`

Docker Compose enables `PRELOAD_INDEX=1`, so the search index is loaded during API startup instead of making the first shopper request pay the full initialization cost.

## Important environment variables

| Variable | Purpose |
| --- | --- |
| `ARTIFACT_DIR` | Persisted FAISS and product metadata artifacts |
| `QUERY_LOG_PATH` | Query telemetry JSONL |
| `SEARCH_CACHE_SIZE` | In-process response-cache capacity |
| `PRELOAD_INDEX` | Preload the search engine during API startup |
| `RERANK_CANDIDATES` | Maximum first-stage candidates sent to the CrossEncoder |
| `API_URL` | Streamlit-to-FastAPI endpoint |

## EC2 deployment shape

```text
Browser
  ↓
Streamlit storefront
  ↓ internal HTTP
FastAPI
  ↓
MiniLM query encoder
  ↓
FAISS IVF 1.3M index
  ↓
optional lexical fusion
  ↓
optional CrossEncoder reranking
```

For a public demo, keep the API port private when possible and expose the storefront through a reverse proxy with TLS.

## Deployment checklist

1. Copy the `artifacts/` directory to the host.
2. Pull the application source.
3. Confirm the files are directly under `artifacts/`, not `artifacts/artifacts/`.
4. Run `docker compose build`.
5. Run `docker compose up -d`.
6. Check `/health` and confirm `search_index_loaded` becomes true.
7. Run a dense query with reranking disabled.
8. Check `/metrics`.
9. Only then expose the storefront publicly.

The current in-process cache and JSONL telemetry are intentionally lightweight portfolio infrastructure. A multi-instance production system would normally move shared state and observability to services such as Redis and a monitoring stack.
