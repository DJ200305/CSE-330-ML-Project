# Log Anomaly Web Server

This repo contains a FastAPI backend and a small React UI for LogBERT + Mistral log anomaly analysis.

## Backend

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```

Place your trained artifacts as described in `backend/README.md`.

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI calls `http://localhost:8000` by default. Override with `VITE_API_URL` if needed.
