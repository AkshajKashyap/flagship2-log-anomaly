from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS anomaly_labels (
  model TEXT NOT NULL,
  grp   TEXT NOT NULL,
  window_start TEXT NOT NULL,
  anomaly_score REAL,
  is_anomaly INTEGER,
  label INTEGER,                 -- 1=true anomaly, 0=false alarm
  notes TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (model, grp, window_start)
);
"""

@dataclass(frozen=True)
class LabelRow:
    model: str
    grp: str
    window_start: str
    anomaly_score: float
    is_anomaly: int
    label: int
    notes: str

def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(DB_SCHEMA)
    conn.commit()
    return conn

def upsert_label(conn: sqlite3.Connection, row: LabelRow) -> None:
    conn.execute(
        """
        INSERT INTO anomaly_labels (model, grp, window_start, anomaly_score, is_anomaly, label, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model, grp, window_start) DO UPDATE SET
          anomaly_score=excluded.anomaly_score,
          is_anomaly=excluded.is_anomaly,
          label=excluded.label,
          notes=excluded.notes
        """,
        (row.model, row.grp, row.window_start, row.anomaly_score, row.is_anomaly, row.label, row.notes),
    )
    conn.commit()
