"""Event-time daily aggregation with deduplication, watermarks and a late ledger."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta


@dataclass(frozen=True)
class Event:
    event_id: str
    event_time: datetime
    amount: float


class DailyRevenueAggregator:
    def __init__(self, allowed_lateness: timedelta = timedelta(hours=6),
                 dedup_retention: timedelta = timedelta(days=7)):
        self.allowed_lateness = allowed_lateness
        # how long event ids are remembered; must cover how late a retry can arrive
        self.dedup_retention = dedup_retention
        self.open_windows: dict[date, float] = defaultdict(float)
        self.finalised: dict[date, float] = {}
        self.late_ledger: list[Event] = []
        self.duplicates_dropped = 0
        self._seen: dict[str, datetime] = {}       # event_id -> event_time
        self._max_event_time: datetime | None = None

    @property
    def watermark(self) -> datetime | None:
        if self._max_event_time is None:
            return None
        return self._max_event_time - self.allowed_lateness

    def process(self, e: Event) -> None:
        if e.event_id in self._seen:
            self.duplicates_dropped += 1
            return
        self._seen[e.event_id] = e.event_time
        day = e.event_time.date()
        if day in self.finalised:
            self.late_ledger.append(e)             # reconcile, never silently change
            return
        self.open_windows[day] += e.amount
        if self._max_event_time is None or e.event_time > self._max_event_time:
            self._max_event_time = e.event_time
        self._advance()

    def _advance(self) -> None:
        wm = self.watermark
        for day in sorted(self.open_windows):
            window_end = datetime.combine(day + timedelta(days=1), datetime.min.time())
            if wm >= window_end:
                self.finalised[day] = round(self.open_windows.pop(day), 2)
        # bound dedup state: forget ids older than the retention horizon
        cutoff = wm - self.dedup_retention
        for eid in [k for k, t in self._seen.items() if t < cutoff]:
            del self._seen[eid]

    def late_adjustments(self) -> dict[date, float]:
        adj: dict[date, float] = defaultdict(float)
        for e in self.late_ledger:                 # ledger entries are already de-duplicated
            adj[e.event_time.date()] += e.amount
        return dict(adj)
