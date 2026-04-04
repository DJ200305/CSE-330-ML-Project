# Log Anomaly Web Server

This repo contains a FastAPI backend and a small React UI for LogBERT + Meta-Llama-3-8B LLM log anomaly analysis with possible fixes

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

## Kaggle LLM Backend
This backend is set up using ngrok. `inference-ngrok(1)` notebook is used.

The UI calls `http://localhost:8000` by default. Override with `VITE_API_URL` if needed.
