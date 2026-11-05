"""Event-time daily aggregation with deduplication, watermarks and a late ledger.

Events whose time is too far ahead of processing time (a phone with a wrong
clock) are quarantined instead of advancing the watermark, so one bad event
cannot close today's window early or wipe the dedup memory.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# How far ahead of processing time an event may claim to be before we treat
# its timestamp as a broken device clock rather than real time.
DEFAULT_MAX_FUTURE_SKEW = timedelta(minutes=5)


@dataclass(frozen=True)
class Event:
    event_id: str
    event_time: datetime
    amount: float


class DailyRevenueAggregator:
    def __init__(self, allowed_lateness: timedelta = timedelta(hours=6),
                 dedup_retention: timedelta = timedelta(days=7),
                 clock: Callable[[], datetime] | None = None,
                 max_future_skew: timedelta = DEFAULT_MAX_FUTURE_SKEW):
        self.allowed_lateness = allowed_lateness
        # how long event ids are remembered; must cover how late a retry can arrive
        self.dedup_retention = dedup_retention
        self.open_windows: dict[date, float] = defaultdict(float)
        self.finalised: dict[date, float] = {}
        self.late_ledger: list[Event] = []
        # events with impossible future timestamps, kept for inspection
        self.future_quarantine: list[Event] = []
        # clock returns processing time; None disables the future-time check
        self.clock = clock
        self.max_future_skew = max_future_skew
        self.duplicates_dropped = 0
        self._seen: dict[str, datetime] = {}       # event_id -> event_time
        self._max_event_time: datetime | None = None

    @property
    def watermark(self) -> datetime | None:
        if self._max_event_time is None:
            return None
        return self._max_event_time - self.allowed_lateness

    def _is_from_the_future(self, e: Event) -> bool:
        """True if the event claims a time beyond processing time + skew."""
        if self.clock is None:
            return False
        return e.event_time > self.clock() + self.max_future_skew

    def process(self, e: Event) -> None:
        # Checked first: a future-dated event must not touch dedup state,
        # windows or the watermark.
        if self._is_from_the_future(e):
            self.future_quarantine.append(e)
            return
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
