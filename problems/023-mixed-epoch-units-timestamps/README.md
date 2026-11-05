# 023 · After merging the mobile and partner event feeds, a third of yesterday's events were dated January 1970 and the nightly job crashed on "year 57813"

## Problem
A product-analytics pipeline merges click and purchase events from three
sources into one `events` table, partitioned by UTC day. The web tracker
sends `event_time` as epoch **milliseconds**, which the pipeline assumed for
every row. After the mobile SDK (epoch **seconds**) was merged in, its events
showed up in a `1970-01-21` partition and the daily active users chart for
the real day dropped. A week later a partner feed started sending epoch
**microseconds**, and the nightly job failed outright with
`ValueError: year 57813 is out of range`, so no dashboards refreshed at all.

## Root cause
An epoch timestamp is just a number; nothing in the value says which unit it
uses. Dividing every value by 1,000 is correct for exactly one source:

| Source | Raw value for 2026-11-04 09:30 UTC | Read as milliseconds |
|---|---|---|
| mobile (seconds) | 1762248600 | 1970-01-21 09:30:48.6 |
| web (milliseconds) | 1762248600000 | 2026-11-04 09:30 (correct) |
| partner (microseconds) | 1762248600000000 | crash: year 57813 is out of range |

## Solution
`solution.py` decides the unit per row instead of assuming one:

- **Try every common unit** (seconds, milliseconds, microseconds,
  nanoseconds) and keep the ones that give a date inside a **plausible
  window**: not before the product launched (2015-01-01) and at most one day
  ahead of the ingestion clock. The check is done in plain seconds
  arithmetic so a value in the year 57813 is rejected instead of crashing.
- Units differ by a factor of 1,000 and the window covers far less than
  that, so for real data exactly one unit fits. If **no unit** fits
  (`0`, `42`, `10**30`, a device clock two days ahead) or **more than one**
  fits (only possible with a window starting in January 1970), the row goes to a
  **quarantine** list with its raw value and a reason, instead of being
  silently placed on the wrong day.
- `unit_mix` counts events per detected unit, so a source that suddenly
  switches format is visible on a dashboard.

In the tests the same moment sent in all three units parses to
`2026-11-04 09:30 UTC` and lands in a single `2026-11-04` partition.

## Trade-offs
- Magnitude detection is a safety net, not a contract. The real fix is a
  schema with an explicit type (`timestamp-millis` in Avro, a proper
  `TIMESTAMP` column) or a declared unit per source.
- The plausible window must be maintained: a backfill of pre-2015 history
  needs a different `earliest` date. Two units can only both fit if the
  window starts within about three weeks of 1970-01-01 (the code then
  quarantines the row as ambiguous).
- Detecting per row allows a single source to mix units. Detecting per
  source per batch (and alerting if it changes) is stricter and cheaper.
- Quarantined rows still need an owner and a replay path, otherwise they are
  just a quieter way of losing data.

## Run
```bash
python -m unittest -v test_solution.py
```
