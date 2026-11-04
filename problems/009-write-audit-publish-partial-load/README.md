# 009 · The morning revenue dashboard showed sales down 59%, and leadership opened an incident over a half-finished file

## Problem
Every night a job loads the previous day's order export (~10,000 orders) from
the e-commerce platform into the `orders` table, and the 8 a.m. dashboards and
the leadership email are built on it. One night the upstream export job ran
out of disk and wrote a truncated file. The loader picked it up, the job
finished "green", and the 8 a.m. email reported revenue down 59% day over day.
Leadership pulled the commercial team into an incident call before anyone
checked the data; the email went out with numbers that were wrong, and trust
in the dashboard took weeks to rebuild.

## Root cause
**The pipeline published whatever arrived, with no gate between "loaded" and
"visible".** Success meant "the job did not crash", not "the data is
plausible". A truncated file is valid CSV, so nothing failed. In the
reproduction (seeded days of 9,619–10,466 orders), loading the first 41% of
a 10,000-order day shows **40.7% of the true revenue**, i.e. "down 59%".

## Solution
**Write-Audit-Publish** (`WapLoader` in `solution.py`):

1. **Write** the day's rows to a staging area that no dashboard reads.
2. **Audit** the staged data:
   - *row count* within 0.5×–2× the **median** of the last 7 published days
     (median so one earlier bad day does not move the baseline);
   - *null rate*: `order_id` 0%, `customer_id` ≤ 1% (catches broken joins);
   - *uniqueness* of `order_id` (catches a file loaded twice, which a row-count
     check at exactly 2.0× would miss);
   - *range*: no negative amounts.
3. **Publish** with an atomic pointer swap only if every audit passes. On
   failure, the staged data moves to **quarantine** for inspection, the
   published table keeps its last good state, and the report is raised as an
   alert. A re-run with the full file publishes normally and clears quarantine;
   `force=True` exists for a human who has confirmed an unusual day is real.

In the tests, the truncated load fails with `4100 rows = 0.42x median 9816`,
while a 1.8× sale day passes, and duplicate, null and negative-amount
partitions are each caught by exactly the audit meant for them.

## Trade-offs
- **Late data instead of wrong data.** A blocked publish means the dashboard
  is a day stale until someone acts. That is the point, but it needs an owner
  and an alert route, otherwise it becomes a silent outage.
- **Thresholds are business-specific.** 0.5×–2× suits a stable daily volume;
  launches, holidays and weekly seasonality need wider bands, a same-weekday
  baseline, or a calendar of expected events. With fewer than 3 days of
  history the count audit is skipped and says so.
- Audits catch *implausible* data, not *wrong-but-plausible* data (e.g. a 10%
  shortfall). Reconciling totals against the source system's own count or
  control total, sent alongside the file, is stronger when the producer can
  provide one.
- In real warehouses the publish step is a table-format feature: Iceberg
  branches / WAP, Delta or Hudi commits, or a view swap. Tools like Great
  Expectations, Soda or dbt tests can provide the audits.

## Run
```bash
python -m unittest -v test_solution.py
```
