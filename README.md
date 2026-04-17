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

## LogBERT + LLM Anomaly Pipeline

### What This Is
Two-stage log anomaly detection pipeline combining LogBERT 
(Guo et al., IJCNN 2021) with an open-source LLM for root-cause explanation.

Identified limitation of LLM-LADE (Zhang et al., 2025): routing all logs 
through the LLM is prohibitively expensive at scale for 7B–8B models.

Our fix: LogBERT acts as a gatekeeper — only flagged sequences reach the LLM,
drastically reducing inference load while preserving explanation quality.

### Results (BGL Dataset)
| Model                        | Precision | Recall | F1    |
|------------------------------|-----------|--------|-------|
| LogBERT original (4.7M logs) | 0.894     | 0.923  | 0.908 |
| Ours (1M logs)               | 0.879     | 0.866  | 0.873 |

LLM explanations aligned with LogBERT on severe events (RTS panic, 
retransmission errors). Filtering reduced LLM calls to anomalous sequences only.
