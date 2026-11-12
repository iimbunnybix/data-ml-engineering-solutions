"""Build user sessions from inactivity gaps instead of clock-hour buckets.

The original job defined a session as "all of a user's events in the same
clock hour". A visit from 10:50 to 11:10 crosses the hour and became two
sessions, while two separate visits at 14:05 and 14:55 were merged into one.
Session counts were inflated and session lengths deflated. This module keeps
that version and fixes it the standard way: per user, sort events by time and
start a new session only after more than 30 minutes without activity.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
# Industry-standard web analytics timeout: 30 minutes of inactivity ends a session.
INACTIVITY_GAP_SECONDS = 30 * SECONDS_PER_MINUTE

# Simulated traffic: short pauses inside a visit, long ones between visits.
MIN_EVENTS_PER_VISIT, MAX_EVENTS_PER_VISIT = 2, 12
MIN_PAUSE_IN_VISIT, MAX_PAUSE_IN_VISIT = 10, 5 * SECONDS_PER_MINUTE
MIN_PAUSE_BETWEEN_VISITS, MAX_PAUSE_BETWEEN_VISITS = 45 * SECONDS_PER_MINUTE, 3 * SECONDS_PER_HOUR
MAX_VISITS_PER_USER = 4
FIRST_VISIT_WINDOW = 12 * SECONDS_PER_HOUR


@dataclass(frozen=True)
class Event:
    """One tracked user action. timestamp is epoch seconds (event time, not arrival time)."""

    user_id: str
    timestamp: int
    event_id: str


@dataclass(frozen=True)
class Session:
    """A run of one user's events with no long pause inside it."""

    user_id: str
    start: int
    end: int
    event_count: int

    @property
    def duration_seconds(self) -> int:
        return self.end - self.start


def naive_hourly_sessions(events: list[Event]) -> list[Session]:
    """Original job: one session per user per clock hour."""
    buckets: dict[tuple[str, int], list[int]] = {}
    for event in events:
        hour = event.timestamp // SECONDS_PER_HOUR
        buckets.setdefault((event.user_id, hour), []).append(event.timestamp)
    sessions = []
    for (user_id, _hour), timestamps in sorted(buckets.items()):
        sessions.append(Session(user_id, min(timestamps), max(timestamps), len(timestamps)))
    return sessions


def events_by_user(events: list[Event]) -> dict[str, list[Event]]:
    """Group events per user, each list sorted by event time.

    Events arrive out of order (mobile clients upload in batches), so sorting
    is required before gaps mean anything. event_id breaks ties so events with
    the same timestamp always come out in the same order on every run.
    """
    grouped: dict[str, list[Event]] = {}
    for event in events:
        grouped.setdefault(event.user_id, []).append(event)
    for user_events in grouped.values():
        user_events.sort(key=sort_key)
    return grouped


def sort_key(event: Event) -> tuple[int, str]:
    """Event time first, then event_id as a deterministic tie-breaker."""
    return (event.timestamp, event.event_id)


def split_on_gaps(user_events: list[Event], gap_seconds: int) -> list[Session]:
    """Cut one user's sorted events wherever the pause is longer than gap_seconds."""
    sessions = []
    start = user_events[0].timestamp
    previous = start
    count = 0
    for event in user_events:
        # A pause of exactly gap_seconds still counts as the same visit.
        if event.timestamp - previous > gap_seconds:
            sessions.append(Session(event.user_id, start, previous, count))
            start = event.timestamp
            count = 0
        previous = event.timestamp
        count += 1
    sessions.append(Session(user_events[0].user_id, start, previous, count))
    return sessions


def gap_sessions(events: list[Event], gap_seconds: int = INACTIVITY_GAP_SECONDS) -> list[Session]:
    """Fixed job: sessions end after more than gap_seconds of inactivity, per user."""
    if gap_seconds <= 0:
        raise ValueError(f"gap_seconds must be positive, got {gap_seconds}")
    grouped = events_by_user(events)
    sessions = []
    for user_id in sorted(grouped):
        sessions.extend(split_on_gaps(grouped[user_id], gap_seconds))
    return sessions


def average_duration_minutes(sessions: list[Session]) -> float:
    """The dashboard's "average session length"."""
    total_seconds = 0
    for session in sessions:
        total_seconds += session.duration_seconds
    return total_seconds / len(sessions) / SECONDS_PER_MINUTE


def make_visit(rng: random.Random, user_id: str, start: int, first_index: int) -> list[Event]:
    """One real visit: a few events, each a short pause after the previous one."""
    events = []
    timestamp = start
    for offset in range(rng.randint(MIN_EVENTS_PER_VISIT, MAX_EVENTS_PER_VISIT)):
        events.append(Event(user_id, timestamp, f"{user_id}-{first_index + offset}"))
        timestamp += rng.randint(MIN_PAUSE_IN_VISIT, MAX_PAUSE_IN_VISIT)
    return events


def make_traffic(rng: random.Random, user_count: int, day_start: int) -> tuple[list[Event], int]:
    """Deterministic shuffled events plus the true number of visits.

    Pauses inside a visit are at most 5 minutes and pauses between visits are
    at least 45, so the true visit count is known exactly. The events are
    shuffled because that is how batched mobile uploads arrive.
    """
    events: list[Event] = []
    visit_count = 0
    for user_number in range(user_count):
        user_id = f"user{user_number:04d}"
        visit_start = day_start + rng.randint(0, FIRST_VISIT_WINDOW)
        for _ in range(rng.randint(1, MAX_VISITS_PER_USER)):
            visit = make_visit(rng, user_id, visit_start, len(events))
            events.extend(visit)
            visit_count += 1
            visit_start = visit[-1].timestamp + rng.randint(MIN_PAUSE_BETWEEN_VISITS,
                                                            MAX_PAUSE_BETWEEN_VISITS)
    rng.shuffle(events)
    return events, visit_count
