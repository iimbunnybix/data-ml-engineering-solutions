# 048 · The product dashboard showed 26% more sessions than real visits, and sessions that looked 3.6 minutes shorter

## Problem
A shopping app's product analytics job builds daily sessions from the
clickstream: page views, searches, add-to-cart events. The job defined a
session as "all of a user's events in the same clock hour" (`GROUP BY
user_id, date_trunc('hour', event_time)`). The growth team noticed that
"sessions per user" jumped whenever people shopped in the evening around the
hour, that a single checkout often appeared in two sessions (cart in one,
payment in the next), and that the funnel conversion per session was too low
compared with the order count.

In the simulation (1,000 users, 2,485 real visits, 17,413 events arriving
shuffled, seed 48) the hourly job reported **3,135 sessions (+26%)** with an
average length of **12.1 minutes**. The real visits averaged **15.7 minutes**.
671 visits crossed an hour boundary, and 21 hourly "sessions" were really two
separate visits merged together.

## Root cause
A clock hour has nothing to do with user behaviour. A visit from 10:50 to
11:10 is cut at 11:00 into two short sessions, inflating counts and shortening
durations. In the other direction, a visit at 14:02 and a return visit at
14:50 fall into the same bucket and are counted as one session that seems to
last at least 48 minutes. Which way the error goes depends on when people happen to use
the app, so the metric also moves with traffic mix rather than with the product.

## Solution
- `naive_hourly_sessions` keeps the original clock-hour grouping.
- `gap_sessions` sessionises by inactivity, the definition most analytics
  tools use: per user, sort events by event time, and start a new session when
  the pause since the previous event is **longer than 30 minutes**. A pause of
  exactly 30 minutes continues the session.
- Input order does not matter: `events_by_user` sorts each user's events by
  `(timestamp, event_id)`, so batched or out-of-order uploads give the same
  sessions, and events with identical timestamps are ordered deterministically
  and stay in the same session.
- On the simulated day the fix finds exactly the 2,485 real visits.

## Trade-offs
- 30 minutes is a convention, not a law. Pick the timeout from the data (look
  at the distribution of pauses between events) and keep it fixed, because
  changing it changes every historical metric.
- Sessions depend on all of a user's events, so a late event can merge two
  sessions or extend one. Streaming jobs need session windows with an allowed
  lateness and must update or retract earlier results; batch jobs should
  recompute the day (and the evening before, for sessions crossing midnight).
- Sorting per user is `O(n log n)` and needs a user's events together. In SQL
  this is `LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time,
  event_id)` plus a running sum of "gap > 30 min" flags; in Spark or Flink, a
  session window.
- Event time comes from client clocks, which can be wrong. Very skewed clocks
  still produce odd sessions; clamp or flag timestamps far from server receive
  time.
- Some products also end a session at midnight or on a campaign change. Add
  those rules explicitly rather than reintroducing fixed buckets.

## Run
```bash
python -m unittest -v test_solution.py
```
