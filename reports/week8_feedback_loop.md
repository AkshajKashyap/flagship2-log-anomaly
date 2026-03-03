# Week 8: Feedback Loop + Evaluation (HDFS Log Anomalies)

## Goal
Add a minimal human-in-the-loop workflow for reviewing flagged anomaly windows, storing labels, and evaluating alert quality with simple proxy metrics.

## Workflow implemented
1) Export a review batch from scored windows to CSV
   - File includes: group, window_start, model, anomaly_score, is_anomaly, split
   - Reviewer fills: label (1=true anomaly, 0=false alarm) and optional notes
2) Ingest labeled CSV into a local SQLite database
3) Evaluate:
   - precision@k on reviewed items (top-k/day on validation)
   - daily anomaly-rate stability (mean/std/CV)
   - score drift between train vs val distributions (KS distance)

## Storage
- SQLite DB: `data/feedback/feedback.sqlite`
- Table stores (model, group, window_start) + anomaly_score + is_anomaly + human label + notes

## Metrics (current run)
Model: iforest
- precision@10 on reviewed top-k/day (val): 1.0 (n_reviewed=10)
- val daily anomaly-rate mean: 0.007389
- val daily anomaly-rate CV: 1.414
- score drift KS(train vs val): 0.1229

## Notes / interpretation
- precision@10 is high on the initial reviewed batch, suggesting the top-ranked anomalies are meaningful.
- daily anomaly-rate variability is high (CV > 1), which is realistic for bursty system behavior and motivates exploring stability-oriented thresholds or per-group thresholds later.
- train vs val score distribution shows some shift (KS ~0.12), indicating non-stationarity over time (drift).

## Artifacts / scripts
- Export review batch: `scripts/export_review_batch.py`
- Ingest labels: `scripts/ingest_review_batch.py`
- Evaluate: `scripts/evaluate_feedback.py`
- Feedback storage module: `src/log_anomaly/feedback.py`

## Next (Week 9 preview)
- Add a simple monitoring/reporting output that summarizes anomaly rates and drift over time.
- Expand labeling coverage (more days, include OCSVM) and compute precision@k with larger n.
