# Deployment

The serving stack is split into two containers:

- FastAPI API on port `8000`
- Streamlit storefront on port `8501`

The API mounts the generated artifacts instead of storing the 1.3M-product index in Git.

## Required local artifacts

The repository should contain this runtime layout before deployment:

```text
artifacts/
├── products.faiss
├── products.jsonl
└── metadata.json
```

The large artifact files should stay out of Git. Store or transfer them separately, for example through object storage or directly onto the deployment host.

## Local Docker validation

```bash
docker compose build
docker compose up
```

Then open:

- storefront: `http://localhost:8501`
- API health: `http://localhost:8000/health`

The Docker API enables `PRELOAD_INDEX=1`, so startup waits for the search engine to load instead of making the first shopper request pay the full initialization cost.

## Environment variables

| Variable | Default in compose | Purpose |
| --- | --- | --- |
| `ARTIFACT_DIR` | `/app/artifacts` | Persisted FAISS and metadata artifacts |
| `QUERY_LOG_PATH` | `/app/logs/search.jsonl` | Query telemetry |
| `SEARCH_CACHE_SIZE` | `256` | In-process response-cache capacity |
| `PRELOAD_INDEX` | `1` | Load search artifacts during API startup |
| `RERANK_CANDIDATES` | `20` | Maximum first-stage candidates sent to the CrossEncoder |
| `API_URL` | `http://api:8000` | Streamlit-to-FastAPI endpoint |

## EC2 deployment shape

A simple portfolio deployment can run both Docker services on one EC2 host:

```text
Browser
  ↓ :8501
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

For a public demo, restrict EC2 security-group ingress to the ports you deliberately expose. A stronger production setup would put a reverse proxy and TLS in front of the application, keep port 8000 internal, and expose only the public web endpoint.

## Deployment checklist

1. Copy the `artifacts/` directory to the host.
2. Pull the application source.
3. Confirm the artifacts are not nested as `artifacts/artifacts/...`.
4. Run `docker compose build`.
5. Run `docker compose up -d`.
6. Check `/health` and confirm `search_index_loaded` is true after startup.
7. Run a dense query with reranking disabled.
8. Run a second query and inspect `/metrics`.
9. Only then expose the storefront publicly.

The current in-process cache and JSONL metrics are intentionally lightweight portfolio infrastructure. A multi-instance production service would typically move shared cache and telemetry to dedicated services such as Redis and a monitoring stack.
