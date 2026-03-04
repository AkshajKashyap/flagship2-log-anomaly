# src/api.py
from __future__ import annotations

import json
import time
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from log_anomaly.modeling import anomaly_scores

logger = logging.getLogger("log_anomaly_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ARTIFACTS = Path("artifacts")


class ScoreRow(BaseModel):
    group: str
    window_start: Optional[str] = None
    features: Dict[str, float] = Field(..., description="Feature dict matching feature_schema.json")


class ScoreBatchRequest(BaseModel):
    model: str = Field(..., description="iforest or ocsvm")
    rows: List[ScoreRow]


class ScoreBatchResponseRow(BaseModel):
    group: str
    window_start: Optional[str]
    anomaly_score: float
    is_anomaly: bool


class ScoreBatchResponse(BaseModel):
    model: str
    threshold: float
    rows: List[ScoreBatchResponseRow]


app = FastAPI(title="Log Anomaly Scoring Service", version="0.1.0")

# In-memory monitoring
STATE: Dict[str, Any] = {
    "requests": 0,
    "errors": 0,
    "rows_scored": 0,
    "rows_flagged": 0,
    "latency_ms_sum": 0.0,
    "latency_ms_max": 0.0,
    "recent_scores": deque(maxlen=500),  # for drift snapshots
}

MODELS: Dict[str, Any] = {}
THRESHOLDS: Dict[str, float] = {}
FEATURE_COLS: List[str] = []


def load_artifacts() -> None:
    global MODELS, THRESHOLDS, FEATURE_COLS

    if not ARTIFACTS.exists():
        raise RuntimeError("artifacts/ not found")

    # models
    MODELS["iforest"] = joblib.load(ARTIFACTS / "iforest.joblib")
    MODELS["ocsvm"] = joblib.load(ARTIFACTS / "ocsvm.joblib")

    # thresholds
    thresholds = json.loads((ARTIFACTS / "thresholds.json").read_text(encoding="utf-8"))
    THRESHOLDS["iforest"] = float(thresholds["iforest"]["threshold"])
    THRESHOLDS["ocsvm"] = float(thresholds["ocsvm"]["threshold"])

    # feature schema
    schema = json.loads((ARTIFACTS / "feature_schema.json").read_text(encoding="utf-8"))
    FEATURE_COLS = list(schema["feature_columns"])


@app.on_event("startup")
def _startup() -> None:
    load_artifacts()
    logger.info("Loaded artifacts: models=%s, n_features=%d", list(MODELS.keys()), len(FEATURE_COLS))


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "models_loaded": list(MODELS.keys()),
        "n_features": len(FEATURE_COLS),
    }


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    req = max(int(STATE["requests"]), 1)
    avg_latency = float(STATE["latency_ms_sum"]) / req
    anomaly_rate = float(STATE["rows_flagged"]) / max(int(STATE["rows_scored"]), 1)

    recent = list(STATE["recent_scores"])
    if recent:
        recent = np.array(recent, dtype=float)
        score_snapshot = {
            "n": int(len(recent)),
            "p50": float(np.percentile(recent, 50)),
            "p90": float(np.percentile(recent, 90)),
            "p99": float(np.percentile(recent, 99)),
        }
    else:
        score_snapshot = {"n": 0}

    return {
        "requests": int(STATE["requests"]),
        "errors": int(STATE["errors"]),
        "rows_scored": int(STATE["rows_scored"]),
        "rows_flagged": int(STATE["rows_flagged"]),
        "anomaly_rate": anomaly_rate,
        "avg_latency_ms": avg_latency,
        "max_latency_ms": float(STATE["latency_ms_max"]),
        "score_snapshot_recent": score_snapshot,
    }


@app.post("/score_batch", response_model=ScoreBatchResponse)
def score_batch(req: ScoreBatchRequest) -> ScoreBatchResponse:
    t0 = time.time()
    STATE["requests"] += 1

    model_name = req.model.strip().lower()
    if model_name not in MODELS:
        STATE["errors"] += 1
        raise HTTPException(status_code=400, detail="model must be one of: iforest, ocsvm")

    model = MODELS[model_name]
    threshold = THRESHOLDS[model_name]

    # Build X in strict column order
    X_list = []
    for r in req.rows:
        feats = r.features
        try:
            X_list.append([float(feats.get(c, 0.0)) for c in FEATURE_COLS])
        except Exception:
            STATE["errors"] += 1
            raise HTTPException(status_code=400, detail="invalid feature values")

    X = np.asarray(X_list, dtype=float)
    scores = anomaly_scores(model, X)
    flags = scores >= threshold

    # monitoring counters
    latency_ms = (time.time() - t0) * 1000.0
    STATE["rows_scored"] += int(len(req.rows))
    STATE["rows_flagged"] += int(flags.sum())
    STATE["latency_ms_sum"] += float(latency_ms)
    STATE["latency_ms_max"] = max(float(STATE["latency_ms_max"]), float(latency_ms))
    for s in scores.tolist():
        STATE["recent_scores"].append(float(s))

    logger.info(
        "score_batch model=%s n_rows=%d flagged=%d latency_ms=%.2f",
        model_name,
        len(req.rows),
        int(flags.sum()),
        latency_ms,
    )

    out_rows = []
    for r, s, f in zip(req.rows, scores.tolist(), flags.tolist()):
        out_rows.append(
            ScoreBatchResponseRow(
                group=r.group,
                window_start=r.window_start,
                anomaly_score=float(s),
                is_anomaly=bool(f),
            )
        )

    return ScoreBatchResponse(model=model_name, threshold=float(threshold), rows=out_rows)