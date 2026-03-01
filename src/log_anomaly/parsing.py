from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Optional

_HDFS_RE = re.compile(
    r"^(?P<date>\d{6})\s+"
    r"(?P<time>\d{6})\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<component>[^:]+):\s*"
    r"(?P<content>.*)$"
)

@dataclass(frozen=True)
class LogEvent:
    ts: datetime
    pid: int
    level: str
    component: str
    content: str
    raw: str

def parse_hdfs_line(line: str) -> Optional[LogEvent]:
    line = line.rstrip("\n")
    m = _HDFS_RE.match(line)
    if not m:
        return None

    # date is YYMMDD, treat as 20YY-MM-DD for this dataset era
    yymmdd = m.group("date")
    hhmmss = m.group("time")
    ts = datetime.strptime("20" + yymmdd + hhmmss, "%Y%m%d%H%M%S")

    return LogEvent(
        ts=ts,
        pid=int(m.group("pid")),
        level=m.group("level"),
        component=m.group("component"),
        content=m.group("content"),
        raw=line,
    )
