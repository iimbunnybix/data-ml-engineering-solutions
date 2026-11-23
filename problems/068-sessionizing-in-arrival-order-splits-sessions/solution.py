"""Build user sessions from event time, not from the order events arrived in.

The analytics job grouped app events into sessions (a new session after 30
minutes of inactivity) by walking each user's events in the order they
reached the pipeline. Mobile apps buffer events while offline and upload
them later, so an old event can arrive after newer ones. The gap from that
old event to the next one looks huge, the job starts a new session, and one
visit is counted as two, with a negative duration on top.

The fix sorts each user's events by event time (with the event id as a tie
breaker) before applying the inactivity rule, so the result does not depend
on arrival order at all.
"""
from __future__ import annotations

from dataclasses import dataclass

SECONDS_PER_MINUTE = 60
# Standard web-analytics inactivity timeout: 30 minutes without an event.
SESSION_TIMEOUT_SECONDS = 30 * SECONDS_PER_MINUTE


@dataclass(frozen=True)
class Event:
    user_id: str
    event_time: int  # seconds, when it happened on the device
    event_id: str


@dataclass(frozen=True)
class Session:
    user_id: str
    start: int
    end: int
    event_count: int

    @property
    def duration_seconds(self) -> int:
        return self.end - self.start


def group_by_user(events: list[Event]) -> dict[str, list[Event]]:
    """Each user's events, keeping the order they were given in."""
    by_user: dict[str, list[Event]] = {}
    for event in events:
        by_user.setdefault(event.user_id, []).append(event)
    return by_user


# --- the original, buggy version ---------------------------------------------
def naive_sessions_for_user(user_events: list[Event]) -> list[Session]:
    """Original: walks events in arrival order and compares to the previous one."""
    sessions: list[Session] = []
    first = previous = user_events[0]
    count = 1
    for event in user_events[1:]:
        # An old, late event makes this gap negative; the event after it then
        # looks far away from the "previous" one and opens a new session.
        if event.event_time - previous.event_time > SESSION_TIMEOUT_SECONDS:
            sessions.append(Session(first.user_id, first.event_time, previous.event_time, count))
            first, count = event, 0
        previous = event
        count += 1
    sessions.append(Session(first.user_id, first.event_time, previous.event_time, count))
    return sessions


def naive_sessionize(events_in_arrival_order: list[Event]) -> list[Session]:
    """Original job over all users."""
    sessions: list[Session] = []
    for user_events in group_by_user(events_in_arrival_order).values():
        sessions.extend(naive_sessions_for_user(user_events))
    return sessions


# --- the fix -----------------------------------------------------------------
def event_order(event: Event) -> tuple[int, str]:
    """Sort key: event time, then event id so equal times are deterministic."""
    return (event.event_time, event.event_id)


def deduplicate(user_events: list[Event]) -> list[Event]:
    """Drop re-uploaded copies of the same event (offline retries resend batches)."""
    seen: set[str] = set()
    unique: list[Event] = []
    for event in user_events:
        if event.event_id not in seen:
            seen.add(event.event_id)
            unique.append(event)
    return unique


def sessions_for_user(user_events: list[Event]) -> list[Session]:
    """Split one user's events into sessions by gaps in event time."""
    ordered = sorted(deduplicate(user_events), key=event_order)
    sessions: list[Session] = []
    start = end = ordered[0].event_time
    count = 1
    for event in ordered[1:]:
        if event.event_time - end > SESSION_TIMEOUT_SECONDS:
            sessions.append(Session(event.user_id, start, end, count))
            start, count = event.event_time, 0
        end = event.event_time
        count += 1
    sessions.append(Session(ordered[0].user_id, start, end, count))
    return sessions


def sessionize(events: list[Event]) -> list[Session]:
    """Fixed job: same sessions whatever order the events arrived in."""
    sessions: list[Session] = []
    by_user = group_by_user(events)
    for user_id in sorted(by_user):
        sessions.extend(sessions_for_user(by_user[user_id]))
    return sessions
