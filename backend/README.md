# Backend (FastAPI)

## Expected model files
Place your trained artifacts here:
- `backend/app/resources/logbert.pt`
- `backend/app/resources/event2id.json`
- `backend/app/resources/id_to_template.json`
- `backend/app/resources/llm_lade_results/` (Meta-Llama-3-8B fine-tuned model directory)

You can also override paths with environment variables:
- `LOGBERT_PATH`
- `EVENT2ID_PATH`
- `ID_TO_TEMPLATE_PATH`
- `LLM_DIR`

Optional settings:
- `WINDOW_SIZE` (default `128`)
- `MASK_RATIO` (default `0.15`)
- `THRESHOLD` (default `15`)
- `MAX_ANOMALIES` (default `50`)

## Run
```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```
