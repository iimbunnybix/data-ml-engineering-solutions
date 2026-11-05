# 016 · Every dashboard showed a 50% drop in Tuesday's events, but users and revenue were normal

## Problem
A clickstream pipeline loads one `events` partition per day (about 1 million
rows on a weekday, about 600,000 on a Sunday) and then refreshes the product
and growth dashboards. One Tuesday the upstream export job was killed at noon
by a node restart and retried only the "success" marker, so the partition
landed with 498,000 rows. The load step's only check was "partition is not
empty", so it published. Product managers opened an incident about a 50%
engagement drop, a growth experiment was paused, and the partition had to be
backfilled and every downstream table rebuilt.

## Root cause
The pipeline validated that data existed, not that the **amount** of data was
plausible. Partial deliveries, double loads and filter bugs all produce
partitions with valid schemas and wrong volumes, and none of them fail a
schema or non-null check.

## Solution
`check_partition_volume` runs after load and before publish:

- **Same-weekday baseline.** The new count is compared with the same weekday
  in the previous 8 weeks, so the normal weekly cycle (Sundays are ~40% lower
  in the fixture) is not mistaken for an anomaly. The fixture's all-days
  median is 795,000, while the Sunday median is 601,500.
- **Robust score.** The modified z-score `0.6745 × (count − median) / MAD`
  with the usual 3.5 cut-off (Iglewicz & Hoaglin). Median and MAD are not
  dragged around by a past outage: with one earlier 300,000-row Tuesday in
  the history, a mean/standard-deviation z-score for today's 498,000 rows is
  only −1.7 (would pass), while the robust score is −23.1 (blocked).
- **Both directions.** Too-high counts (a double load) are blocked too.
- **Guards.** Empty partitions are blocked; MAD is floored at 1% of the
  median so a perfectly flat history does not divide by zero; with fewer than
  4 comparable days the partition is published with an explicit warning.

On the fixture, the half-day partition scores −32.2 against a median of
999,000 and is blocked; a normal 1,004,000-row Tuesday scores 0.3.

## Trade-offs
- Real changes (a launch, a holiday, a marketing push) will be blocked too;
  the operator needs a documented "accept and publish" override, and holidays
  can be excluded from the baseline.
- Weekly seasonality is handled; strong trends (fast growth) slowly push the
  count away from an 8-week median. A shorter window or a forecast model
  (e.g. seasonal decomposition) handles trend better at the cost of
  complexity.
- Volume checks catch missing rows, not wrong rows; keep schema, null-rate
  and freshness checks alongside, and compare with the source's own count
  when the upstream can publish one (the strongest check of all).

## Run
```bash
python -m unittest -v test_solution.py
```
