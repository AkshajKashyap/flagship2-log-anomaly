from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values(["window_start", "group"]).reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.7) -> SplitData:
    # Split by time to reduce leakage from adjacent windows
    unique_times = df["window_start"].sort_values().unique()
    cut = int(len(unique_times) * train_frac)
    train_times = set(unique_times[:cut])
    train = df[df["window_start"].isin(train_times)].copy()
    val = df[~df["window_start"].isin(train_times)].copy()
    return SplitData(train=train, val=val)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    drop_cols = ["group", "window_start"]
    feat_cols = [c for c in df.columns if c not in drop_cols]
    return df[feat_cols].to_numpy(dtype=float)


def train_iforest(X_train: np.ndarray, random_state: int = 42) -> Pipeline:
    # Scaling helps OCSVM more, but keep consistent for both models
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", IsolationForest(
                n_estimators=300,
                max_samples="auto",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]
    )
    pipe.fit(X_train)
    return pipe


def train_ocsvm(X_train: np.ndarray) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")),
        ]
    )
    pipe.fit(X_train)
    return pipe


def anomaly_scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    # Higher score = more anomalous
    core = model.named_steps["model"]
    if hasattr(core, "score_samples"):
        # IsolationForest: higher score_samples means more normal, so negate
        return -core.score_samples(model.named_steps["scaler"].transform(X))
    # OneClassSVM: decision_function higher means more normal, so negate
    return -core.decision_function(model.named_steps["scaler"].transform(X))


def threshold_percentile(train_scores: np.ndarray, percentile: float) -> float:
    return float(np.percentile(train_scores, percentile))


def threshold_budget_per_day(df_train: pd.DataFrame, train_scores: np.ndarray, budget_per_day: int) -> float:
    # Choose a threshold that yields ~budget_per_day flagged windows on training data.
    tmp = df_train[["window_start"]].copy()
    tmp["score"] = train_scores
    tmp["day"] = tmp["window_start"].dt.date
    n_days = tmp["day"].nunique()
    if n_days == 0:
        return float(np.max(train_scores))

    target_total = budget_per_day * n_days
    # Flag the top target_total windows by score
    k = min(max(int(target_total), 1), len(tmp))
    cutoff = float(np.sort(train_scores)[-k])
    return cutoff


def threshold_stability(df_train: pd.DataFrame, train_scores: np.ndarray, candidate_percentiles: Tuple[float, ...] = (99.0, 99.2, 99.4, 99.6, 99.7, 99.8, 99.9)) -> Dict[str, float]:
    # Pick the percentile that gives the most stable daily anomaly rate.
    tmp = df_train[["window_start"]].copy()
    tmp["score"] = train_scores
    tmp["day"] = tmp["window_start"].dt.date

    best = {"percentile": float(candidate_percentiles[0]), "threshold": float(np.percentile(train_scores, candidate_percentiles[0])), "daily_cv": float("inf")}
    for p in candidate_percentiles:
        thr = float(np.percentile(train_scores, p))
        tmp["is_anom"] = tmp["score"] >= thr
        daily_rate = tmp.groupby("day")["is_anom"].mean()
        if len(daily_rate) <= 1:
            cv = 0.0
        else:
            m = float(daily_rate.mean())
            s = float(daily_rate.std(ddof=1))
            cv = (s / (m + 1e-9))
        if cv < best["daily_cv"]:
            best = {"percentile": float(p), "threshold": thr, "daily_cv": float(cv)}
    return best


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def score_dataframe(df: pd.DataFrame, scores: np.ndarray, threshold: float, model_name: str) -> pd.DataFrame:
    out = df[["group", "window_start"]].copy()
    out["model"] = model_name
    out["anomaly_score"] = scores
    out["is_anomaly"] = out["anomaly_score"] >= threshold
    return out
