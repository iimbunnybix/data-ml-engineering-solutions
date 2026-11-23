# 068 · Sessions per user jumped 26% after the offline-mode release, and some sessions had negative length

## Problem
A shopping app's analytics pipeline groups app events into **sessions**: a
new session starts after 30 minutes without any event, the usual web
analytics rule. "Sessions per user" and "average session length" are on the
product team's weekly dashboard and feed the engagement goal.

The week the app shipped offline mode (events are buffered on the phone and
uploaded when the connection returns), sessions per user rose sharply while
orders stayed flat. The product team was about to report a big engagement
win. An analyst then found sessions with a **negative** duration, which
cannot happen, and the jump was traced to the pipeline.

## Root cause
`naive_sessionize` walked each user's events **in the order they reached
the pipeline** and compared each one to the previous one. With offline
buffering, an older event arrives after newer ones:

```
arrival order:  10:10  10:20  10:40  10:05 (late)  10:45
```

The late 10:05 event gives a negative gap, so it is added to the current
session and becomes "previous". The next event, 10:45, is then 40 minutes
after "previous" and opens a **new session**. One visit counts as two, and
the first one "ends" at 10:05, five minutes before it started.

In the replay in `test_solution.py` (1,000 users, 3 visits each, 10% of
events uploaded five events late), the true number of sessions is
**3,000**; the naive job reports **3,787** (+26%), and **825** of its
sessions have a negative duration.

## Solution
`sessionize` makes the result a function of the events, not of their
arrival:

1. **Group by user, then sort by event time** (`event_order`), with the
   event id as a tie breaker so equal timestamps are ordered the same way
   on every run.
2. **Drop re-uploaded duplicates** by event id (`deduplicate`); an offline
   retry often resends a whole batch.
3. **Apply the 30-minute rule on the sorted events** (`sessions_for_user`).
   A gap of exactly 30 minutes stays in the session; one second more starts
   a new one.

On the replay, the fixed job returns exactly the same sessions for the
shuffled arrival order as for the true order: 3,000 sessions, none
negative.

## Trade-offs
- Sorting needs all of a user's events for the period. In a daily batch,
  events uploaded after the day was processed are still missed; reprocess
  the last few days (for example with a 3-day lookback) so late uploads are
  folded in, and accept that recent days change.
- In a streaming job, the same fix means event-time session windows with a
  watermark and allowed lateness. Late data past the allowed lateness is
  dropped or sent to a side output, so streaming numbers are only final
  after that delay.
- Device clocks can be wrong. Event time from a phone with a bad clock can
  merge or split sessions; clamping to the server receive time, or using a
  clock offset measured at upload, limits the damage.
- Deduplicating by event id requires the app to generate a stable id per
  event before it is buffered. Without one, retries double-count events
  even after sorting.

## Run
```bash
python -m unittest -v test_solution.py
```
