from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE_FEATURES = [
    "total_events",
    "unique_templates",
    "burstiness",
    "other_template_count",
    "error_ratio",
]

def add_group_percentiles(features: pd.DataFrame) -> pd.DataFrame:
    out = features[["group", "window_start"] + BASE_FEATURES].copy()
    for col in BASE_FEATURES:
        out[f"p_{col}"] = out.groupby("group")[col].rank(pct=True)
    return out[["group", "window_start"] + BASE_FEATURES + [f"p_{c}" for c in BASE_FEATURES]]

def main() -> None:
    p = argparse.ArgumentParser(description="Export review batch with context features")
    p.add_argument("--scored", type=str, default="data/processed/scored_windows.parquet")
    p.add_argument("--features", type=str, default="data/processed/features_full.parquet")
    p.add_argument("--model", type=str, required=True, choices=["iforest", "ocsvm"])
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--only-flagged", action="store_true")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    scored = pd.read_parquet(args.scored)
    df = scored[(scored["model"] == args.model) & (scored["split"] == args.split)].copy()

    if args.only_flagged:
        df = df[df["is_anomaly"] == True]

    df = df.sort_values("anomaly_score", ascending=False).head(args.k)

    feat = pd.read_parquet(args.features)
    context = add_group_percentiles(feat)

    merged = df.merge(context, on=["group", "window_start"], how="left")
    merged["label"] = ""
    merged["notes"] = ""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print("Wrote review batch:", out, "rows=", len(merged))

if __name__ == "__main__":
    main()
