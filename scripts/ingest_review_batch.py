from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from log_anomaly.feedback import connect, upsert_label, LabelRow


def parse_bool(x) -> int:
    # Accept True/False, "True"/"False", 1/0, "1"/"0"
    if isinstance(x, bool):
        return int(x)
    s = str(x).strip().lower()
    if s in {"true", "1", "1.0"}:
        return 1
    if s in {"false", "0", "0.0"}:
        return 0
    # default fallback
    return int(bool(x))


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest labeled review CSV into sqlite DB")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--db", type=str, default="data/feedback/feedback.sqlite")
    args = p.parse_args()

    # Read everything as strings to avoid weird type inference (Excel often writes 1.0)
    df = pd.read_csv(args.csv, dtype=str)

    required = {"model","split","group","window_start","anomaly_score","is_anomaly","label","notes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Normalize label: strip, empty -> NA, numeric parse (handles 1, 0, 1.0, 0.0)
    df["label"] = df["label"].astype(str).str.strip()
    df.loc[df["label"].isin(["", "nan", "none", "na"]), "label"] = pd.NA
    df = df[df["label"].notna()].copy()
    if len(df) == 0:
        print("No labeled rows found. Fill label with 1 or 0 in the CSV and re-run.")
        return

    df["label_num"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label_num"].notna()].copy()
    if len(df) == 0:
        print("Labels could not be parsed as numbers. Use 0/1 (or 0.0/1.0).")
        return

    df["label_int"] = df["label_num"].round().astype(int)
    if not df["label_int"].isin([0, 1]).all():
        bad = df[~df["label_int"].isin([0, 1])]["label"].unique()
        raise ValueError(f"Label must be 0 or 1. Found: {bad}")

    # Parse anomaly_score and is_anomaly
    df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce")
    df = df[df["anomaly_score"].notna()].copy()

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
                is_anomaly=parse_bool(r["is_anomaly"]),
                label=int(r["label_int"]),
                notes=str(r.get("notes", "") or ""),
            ),
        )
        n += 1

    print(f"Ingested {n} labels into {args.db}")


if __name__ == "__main__":
    main()
