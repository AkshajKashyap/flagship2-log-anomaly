# Week 9: Ship the Log Anomaly Detector as a Service (FastAPI + Docker)

## Goal
Turn the anomaly scoring pipeline into a runnable service with:
- a scoring endpoint that returns anomaly scores + flags
- basic monitoring hooks (latency, error counts, anomaly rate, recent score snapshot)
- Docker packaging so the service can be deployed consistently

## What was shipped

### 1) FastAPI service
A lightweight API that loads pre-trained artifacts on startup and exposes endpoints:

- `GET /health`
  - Purpose: sanity check that the service is alive and artifacts loaded.
  - Example response includes:
    - `models_loaded`: list of available models (iforest, ocsvm)
    - `n_features`: number of expected input feature columns

- `GET /metrics`
  - Purpose: minimal monitoring hooks.
  - Tracks:
    - request count, error count
    - total rows scored, rows flagged
    - anomaly rate (rows_flagged / rows_scored)
    - average and max latency
    - recent score distribution snapshot (p50/p90/p99) as a drift signal

- `POST /score_batch`
  - Purpose: score a batch of feature rows.
  - Input:
    - `model`: `"iforest"` or `"ocsvm"`
    - `rows`: list of objects containing:
      - `group`
      - `window_start` (optional)
      - `features`: dict of numeric features
  - Output:
    - anomaly_score and is_anomaly per row
    - threshold used for the model

### 2) Service artifacts (what the API depends on)
The API loads these files at startup:

- `artifacts/iforest.joblib`
- `artifacts/ocsvm.joblib`
- `artifacts/thresholds.json`
- `artifacts/feature_schema.json`
  - stores the ordered list of feature columns expected by the model
  - ensures request feature dicts map into the correct model input vector order

### 3) Dockerized deployment
A Docker image that:
- installs Python deps from `requirements.txt`
- copies `src/` and `artifacts/`
- runs the API via `uvicorn`
- exposes port 8000

## How to run (local dev)
From the repo root:

1) Ensure artifacts exist:
- `artifacts/iforest.joblib`, `artifacts/ocsvm.joblib`, `artifacts/thresholds.json`, `artifacts/feature_schema.json`

2) Run server:
```bash
pip install fastapi uvicorn
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

3) Test:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
curl -s http://127.0.0.1:8000/metrics | python -m json.tool
```

## How to run (Docker)

1) Build:

```bash
docker build -t log-anomaly-api .
```

2) Run:

```bash
docker run --rm -p 8000:8000 log-anomaly-api
```

3) Test:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
curl -s http://127.0.0.1:8000/metrics | python -m json.tool
```

## Notes / limitations (current design)

- The service expects feature vectors, not raw log lines.

- Raw log ingestion + window feature generation remains an offline batch pipeline (Weeks 6-7).

- Drift monitoring is a lightweight proxy using recent score quantiles.

- Future improvements could include feature distribution drift checks and per-group baselines.

- Thresholds are loaded from `thresholds.json` and applied consistently across requests.

## Next (Week 10+)

- Add request schema validation and clearer error messages for missing feature keys.

- Add an optional endpoint to accept raw log lines and perform on-the-fly feature generation for a window.

- Add more robust monitoring (Prometheus-format metrics, structured logging, dashboards).
