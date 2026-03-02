# Week 7: Unsupervised Models + Thresholding (HDFS)

## Goal
Train baseline unsupervised anomaly detectors on window-level HDFS log features and choose a practical thresholding policy.

## Inputs
- Features: `data/processed/features_full.parquet`
  - Row key: (group/component, 5-minute window)
  - Columns: base metrics + top-200 template counts (`tmpl_0 ... tmpl_199`)
- Template vocab: `data/processed/template_vocab.json`

## Models
1) Isolation Forest (baseline)
2) One-Class SVM (OCSVM) (comparison)

Split:
- Time-based split (earlier windows = train, later windows = val)

## Threshold strategy
Final choice: **budget-per-day**, budget = **10 anomalies/day** (calibrated on train split).
Reason: keeps alert volume in a human-reviewable range, aligns with operational monitoring constraints.

Observed anomaly rates after budget=10/day:
- Isolation Forest:
  - train: 20 / 1214 (1.65%)
  - val: 12 / 820 (1.46%)
- OCSVM:
  - train: 20 / 1214 (1.65%)
  - val: 7 / 820 (0.85%)

## What the models flag
Both models primarily flag `dfs.FSDataset` windows that represent bursty behavior.
These anomalies are driven by extreme increases in event volume and repeated normal templates, not by novel templates.

### Example anomaly (top IF val anomaly)
- Window: 2008-11-11 04:45:00
- Group: dfs.FSDataset
- total_events: 194,827
- unique_templates: 66
- burstiness: 4.79
- other_template_count: 0
- error_ratio: ~0.0028

Baseline comparison for dfs.FSDataset:
- Median (50%) total_events: 8
- 90% total_events: ~4,695
- 99% total_events: ~147,016
This window (194,827) exceeds the 99th percentile.

Top templates in the anomaly window:
- 12,214: `<NUM>.<NUM>.<NUM>.<NUM>:<NUM>:Transmitted block <BLOCK> to <IP>`
- 3,816: `BLOCK* NameSystem.allocateBlock: ... part-<NUM>. <BLOCK>`
- 3,804: `BLOCK* NameSystem.allocateBlock: ... part-<NUM>. <BLOCK>`
- 3,714: `BLOCK* NameSystem.allocateBlock: ... part-<NUM>. <BLOCK>`
- 3,614: `BLOCK* NameSystem.allocateBlock: ... part-<NUM>. <BLOCK>`

These templates have a median count of 0 in typical dfs.FSDataset windows, indicating a burst of block transmissions and allocations rather than novelty.

## Artifacts
- Models: `artifacts/iforest.joblib`, `artifacts/ocsvm.joblib`
- Threshold policy: `artifacts/thresholds.json`
- Scored windows: `data/processed/scored_windows.parquet`

## Next (Week 8)
Add a feedback loop for reviewing anomalies (store flagged windows, optional labeling), and start building monitoring hooks for score distributions and drift.
