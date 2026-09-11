# SearchVerse demo

Animated hackathon-style frontend for the existing FastAPI search backend.

## Run

Start the backend from the repository root:

```powershell
$env:PRELOAD_INDEX="1"; uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Then start the demo in another terminal:

```powershell
cd demo
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

The demo sends a real hybrid search request (`FAISS + BM25 + RRF + CrossEncoder`) to `/search` and renders the returned products and component timings. The animated retrieval universe is a presentation layer rather than a literal plot of all 1.3M embeddings.

To point the frontend at another backend:

```powershell
$env:VITE_API_URL="http://127.0.0.1:8000"; npm run dev
```
