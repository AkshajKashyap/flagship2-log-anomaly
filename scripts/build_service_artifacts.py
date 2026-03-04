# scripts/build_service_artifacts.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ARTIFACTS = Path("artifacts")

def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet("data/processed/features_full.parquet")
    drop_cols = {"group", "window_start"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    (ARTIFACTS / "feature_schema.json").write_text(
        json.dumps({"feature_columns": feature_cols}, indent=2),
        encoding="utf-8",
    )
    print("Wrote:", ARTIFACTS / "feature_schema.json", "n_features=", len(feature_cols))

if __name__ == "__main__":
    main()