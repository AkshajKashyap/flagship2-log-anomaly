from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from log_anomaly.feedback import connect, upsert_label, LabelRow


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest labeled review CSV into sqlite DB")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--db", type=str, default="data/feedback/feedback.sqlite")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    required = {"model","split","group","window_start","anomaly_score","is_anomaly","label","notes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Normalize label column: treat empty/whitespace as missing
    df["label"] = df["label"].astype(str).str.strip()
    df.loc[df["label"].isin(["", "nan", "None"]), "label"] = pd.NA

    # Keep only rows with an actual label
    df = df[df["label"].notna()].copy()
    if len(df) == 0:
        print("No labeled rows found. Fill label with 1 or 0 in the CSV and re-run.")
        return

    # Convert and validate
    df["label"] = df["label"].astype(int)
    if not df["label"].isin([0, 1]).all():
        bad = df[~df["label"].isin([0, 1])]["label"].unique()
        raise ValueError(f"Label must be 0 or 1. Found: {bad}")

    conn = connect(Path(args.db))
    n = 0
    for _, r in df.iterrows():
        upsert_label(
            conn,
            LabelRow(
                model=str(r["model"]),
                grp=str(r["group"]),
                window_start=str(r["window_start"]),
                anomaly_score=float(r["anomaly_score"]),
                is_anomaly=int(bool(r["is_anomaly"])),
                label=int(r["label"]),
                notes=str(r.get("notes","") or ""),
            ),
        )
        n += 1

    print(f"Ingested {n} labels into {args.db}")


if __name__ == "__main__":
    main()
