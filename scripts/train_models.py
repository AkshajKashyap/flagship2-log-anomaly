from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from log_anomaly.modeling import (
    load_features,
    time_split,
    feature_matrix,
    train_iforest,
    train_ocsvm,
    anomaly_scores,
    threshold_percentile,
    threshold_budget_per_day,
    threshold_stability,
    save_model,
    save_json,
    score_dataframe,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Week 7: Train unsupervised log anomaly models + thresholds")
    p.add_argument("--features", type=str, default="data/processed/features_full.parquet")
    p.add_argument("--out-scored", type=str, default="data/processed/scored_windows.parquet")
    p.add_argument("--train-frac", type=float, default=0.7)

    # Threshold strategy knobs
    p.add_argument("--threshold-strategy", type=str, choices=["percentile", "budget", "stability"], default="percentile")
    p.add_argument("--percentile", type=float, default=99.7)
    p.add_argument("--budget-per-day", type=int, default=20)

    args = p.parse_args()

    df = load_features(Path(args.features))
    split = time_split(df, train_frac=args.train_frac)

    X_train = feature_matrix(split.train)
    X_val = feature_matrix(split.val)

    models = {
        "iforest": train_iforest(X_train),
        "ocsvm": train_ocsvm(X_train),
    }

    artifacts = Path("artifacts")
    thresholds = {}

    scored_all = []

    for name, model in models.items():
        train_scores = anomaly_scores(model, X_train)
        val_scores = anomaly_scores(model, X_val)

        if args.threshold_strategy == "percentile":
            thr = threshold_percentile(train_scores, args.percentile)
            meta = {"strategy": "percentile", "percentile": args.percentile}
        elif args.threshold_strategy == "budget":
            thr = threshold_budget_per_day(split.train, train_scores, args.budget_per_day)
            meta = {"strategy": "budget", "budget_per_day": args.budget_per_day}
        else:
            best = threshold_stability(split.train, train_scores)
            thr = best["threshold"]
            meta = {"strategy": "stability", **best}

        thresholds[name] = {"threshold": float(thr), **meta}

        save_model(model, artifacts / f"{name}.joblib")

        scored_train = score_dataframe(split.train, train_scores, thr, model_name=name)
        scored_train["split"] = "train"
        scored_val = score_dataframe(split.val, val_scores, thr, model_name=name)
        scored_val["split"] = "val"
        scored_all.append(scored_train)
        scored_all.append(scored_val)

    save_json(thresholds, artifacts / "thresholds.json")

    scored = pd.concat(scored_all, ignore_index=True)
    out_path = Path(args.out_scored)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out_path, index=False)

    print("Saved models to:", artifacts)
    print("Saved thresholds to:", artifacts / "thresholds.json")
    print("Saved scored windows to:", out_path)

    # Quick proxy metrics
    summary = scored.groupby(["model", "split"])["is_anomaly"].mean().reset_index()
    print("\nAnomaly rate by model/split:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
