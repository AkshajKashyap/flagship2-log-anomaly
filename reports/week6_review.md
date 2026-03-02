# Week 6 Review: Log Anomaly Detection (HDFS) — Features Pipeline

## What Week 6 produced (1–2 sentences)
This week built the end-to-end preprocessing pipeline for Project 2: convert raw HDFS system logs into a window-level feature table suitable for unsupervised anomaly detection. The output artifacts are a Parquet feature table and a JSON template vocabulary.

---

## Dataset
- Name: LogPai / Loghub HDFS (HDFS_v1)
- Where it lives locally: `data/raw/hdfs/HDFS.log` (not committed to git)
- What it represents: chronological log messages from a Hadoop Distributed File System (HDFS) cluster (DataNode/NameNode subsystems).

---

## One-line pipeline summary
Raw log lines → parsed events → normalized templates → 5-minute windows (grouped by component) → numeric feature table + template vocabulary.

---

## Key concepts (plain English)
- **Event**: one parsed log line with fields like timestamp, level, component, content.
- **Template**: normalized “event type” derived from message content by replacing variable tokens (block IDs, IPs, numbers) with placeholders.
- **Window-level**: features are computed per 5-minute time bucket (not per individual log line).
- **Group**: we compute separate windows per component (e.g., dfs.DataNode, dfs.FSNamesystem) so behavior is modeled relative to each subsystem’s own baseline.
- **Vocab**: the top-K most common templates (K=200) that define stable feature columns `tmpl_0 ... tmpl_199`.

---

## Main files and what each is responsible for
### Entrypoints
- `src/train.py`
  - Purpose: the official “one command” entrypoint for generating features.
  - Knobs: input path, output path, window size, top_k, max_lines (optional).

- `scripts/build_features.py`
  - Purpose: a convenience CLI wrapper that calls the same underlying feature builder module.

- `scripts/peek_hdfs.py`
  - Purpose: quick sanity check tool to inspect a few parsed/templated events from the raw log file.

### Core library (source of truth)
- `src/log_anomaly/feature_builder.py`
  - Purpose: canonical implementation of feature generation.
  - Pass 1: scan events to count templates and build the top-K template vocabulary.
  - Pass 2: scan again to aggregate window-level features using the fixed vocab, then write outputs.

### Building blocks
- `src/log_anomaly/parsing.py`
  - Purpose: parse raw HDFS log lines into structured events (timestamp, level, component, content).
  - Approach: regex-based parsing; malformed lines are skipped.

- `src/log_anomaly/templating.py`
  - Purpose: turn event content into a template by replacing variable tokens with placeholders.

- `src/log_anomaly/windowing.py`
  - Purpose: map timestamps to fixed-size window buckets (default: 300 seconds).

---

## Feature table schema (what each row/column means)
### Row key
Each row corresponds to: **(group/component, window_start)**.

### Base features
- `total_events`: number of events in the window
- `unique_templates`: number of distinct templates observed in the window
- `error_ratio`: fraction of non-INFO events in the window (expected to be near 0 for HDFS)
- `burstiness`: relative spike score comparing this window to recent typical volume (EMA-based)
- `other_template_count`: count of events whose templates are outside the top-K vocab

### Template count features
- `tmpl_0 ... tmpl_199`: counts of each vocab template within the window

---

## Output artifacts
- Feature table: `data/processed/features_full.parquet`
- Template vocabulary: `data/processed/template_vocab.json`

Notes:
- Raw data is ignored via `.gitignore`
- Processed outputs are reproducible by rerunning the feature builder

---

## Sanity checks performed
- Parser success rate on a sample: 100% on the first 2000 lines.
- Feature distributions on a subset looked non-trivial:
  - `total_events` and `unique_templates` varied widely across components/windows.
  - `error_ratio` was near zero, consistent with INFO-heavy HDFS logs.
- Full run completed successfully:
  - ~11.17M lines processed, output ~2034 rows and 207 columns.

---

## Where to change key knobs (for future weeks)
- Window size: `--window-seconds` (default 300)
- Vocab size: `--top-k` (default 200)
- Grouping key: currently component; could be changed in feature builder aggregation
- Error definition: currently non-INFO; could be tightened to WARN+ERROR only

---

## What’s next (Week 7 preview)
Use `features_full.parquet` to train an unsupervised anomaly model (Isolation Forest baseline) and define a threshold/alerting strategy. Save model artifacts and produce a ranked list of anomalous windows for inspection.
