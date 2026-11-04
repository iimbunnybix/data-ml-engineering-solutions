# 010 · Daily order counts disagreed with the restaurants' totals all summer, and the November hourly chart merged two 1 a.m. hours

## Problem
A food-delivery company in New York stores every order with a UTC timestamp
and reports "orders per business day" and an hourly demand chart in local time
(America/New_York). Restaurant payouts, courier staffing and the finance close
all use these numbers. For most of the year, daily counts disagreed with the
point-of-sale totals the operations team pulled from restaurants, while in
January everything matched. On the first Sunday of November the hourly chart
showed one 1 a.m. bar about twice the usual height, and the staffing model
trained on it planned too many couriers for that hour the next year.

## Root cause
Two time-zone shortcuts:

1. **A fixed offset instead of a time zone.** The job computed the local day
   as `(ts_utc - 5h).date()`. New York is UTC−5 only in standard time; during
   daylight saving time (second Sunday of March to first Sunday of November)
   it is UTC−4. So every order placed between 00:00 and 01:00 local time was
   counted on the *previous* day. With orders spread evenly over 28 summer
   days, **4.2%** of orders (≈ 1/24) are assigned to the wrong day; for a
   late-night business the share is larger.
2. **Grouping by local wall-clock hour.** On the "fall back" day the clock
   shows 01:00–02:00 twice (EDT, then EST). Keying the hourly chart by the
   string `YYYY-MM-DD 01` merges both hours into one bar (120 minutes of
   orders), and the day appears to have 24 hours instead of 25. The spring
   day really has only 23 hours, so "hour 02" is legitimately missing there.

## Solution
`solution.py`:

- **Convert with the IANA database** (`zoneinfo`): `local_day` uses the
  offset that was in effect *at that instant*. Naive (offset-less) timestamps
  are rejected so local time can never be mistaken for UTC.
- **Hourly buckets keyed by UTC hour start**, which is unambiguous; the
  display label includes the zone abbreviation (`01:00 EDT`, `01:00 EST`).
  On 2026-11-01 the chart has 25 buckets of 60 minutes each; on 2026-03-08
  it has 23.
- **Local-day bounds in UTC** (`local_day_bounds_utc`) give the exact
  `[start, end)` instant range of a business day (23, 24 or 25 hours), and
  `utc_partitions_for_local_day` lists which UTC date partitions a query must
  scan: a New York day always spans two.
- Tests check that every event falls inside the bounds of the day it is
  assigned to, across 250 days that include both transitions.

## Trade-offs
- Time-zone rules change by law; keep `tzdata` (the OS package, or the
  `tzdata` PyPI package where the OS has none) up to date, or old rules are
  applied to new dates.
- A business day that is not a calendar day (e.g. a restaurant "day" ending at
  4 a.m.) needs an explicit cutoff on top of this, defined in local time.
- Keying by UTC hour is right for storage; charts must still render 23/25-hour
  days honestly. Comparing "same hour last year" across a transition needs an
  explicit rule.
- Multi-region businesses should store the zone per entity (restaurant,
  city) rather than one global zone, and report each region's day in its own zone.
- In SQL warehouses the equivalent is `AT TIME ZONE 'America/New_York'` (or
  `CONVERT_TIMEZONE`) instead of interval arithmetic with a fixed offset.

## Run
```bash
python -m unittest -v test_solution.py
```
