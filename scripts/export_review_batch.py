from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

def main() -> None:
    p = argparse.ArgumentParser(description="Export top anomaly windows for human review labeling")
    p.add_argument("--scored", type=str, default="data/processed/scored_windows.parquet")
    p.add_argument("--model", type=str, required=True, choices=["iforest", "ocsvm"])
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--only-flagged", action="store_true", help="export only rows where is_anomaly=True")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    s = pd.read_parquet(args.scored)
    df = s[(s["model"] == args.model) & (s["split"] == args.split)].copy()

    if args.only_flagged:
        df = df[df["is_anomaly"] == True]

    df = df.sort_values("anomaly_score", ascending=False).head(args.k)

    df["label"] = ""   # fill with 1 (true anomaly) or 0 (false alarm)
    df["notes"] = ""   # optional
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("Wrote review batch:", out, "rows=", len(df))

if __name__ == "__main__":
    main()
