# Local serving

For the large catalog, preload and warm the backend before sending user traffic. This loads the model, FAISS artifacts and metadata once, then executes a warm-up query so the first real request does not pay cold-start inference cost.

## Windows PowerShell

```powershell
$env:PRELOAD_INDEX="1"; uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete.` before starting the UI or benchmarking requests.

Then, in another PowerShell window:

```powershell
$env:API_URL="http://localhost:8000"; streamlit run app.py
```

## Linux / macOS

```bash
PRELOAD_INDEX=1 uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Then:

```bash
API_URL=http://localhost:8000 streamlit run app.py
```

## Why preload?

Without `PRELOAD_INDEX=1`, local development uses lazy loading. The first search can include model/index initialization and cold inference and therefore is not representative of steady-state serving latency. Docker Compose already enables preloading for the API container.

For latency measurements, benchmark only after startup completes and use multiple distinct queries. Keep cold-start time separate from steady-state request latency.
