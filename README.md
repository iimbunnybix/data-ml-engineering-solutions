# Data & ML Engineering Solutions

Real problems from data pipelines and ML systems (data quality, late and duplicate events, drift, model serving), each with root cause, a working Python fix and tests.

Every problem folder has the same structure:

| File | Contents |
|---|---|
| `README.md` | The incident or requirement, root cause, solution and trade-offs |
| `solution.py` | Working implementation (Python standard library only) |
| `test_solution.py` | Tests that reproduce the problem and prove the fix |

## Problems

<!-- INDEX:START -->
| # | Problem |
|---|---|
| 001 | [Upstream renamed a column and the churn model silently scored garbage for 9 days](problems/001-schema-drift-guard) |
| 002 | [Daily revenue dashboard over-reports by 3% and changes after it is published](problems/002-late-events-dedup-aggregation) |
| 003 | [Fraud model scored near-perfect offline but caught almost nothing after launch](problems/003-point-in-time-feature-join) |
| 004 | [Payment-risk model started declining good customers after an upstream release, and no alert fired](problems/004-feature-drift-psi-monitor) |
| 005 | [Re-running a failed backfill inflated September revenue by 60% on the finance dashboard](problems/005-idempotent-backfill-partition-overwrite) |
| 006 | [Nightly merchant-enrichment job stuck at 199/200 tasks for hours after onboarding one large marketplace](problems/006-skewed-join-key-salting) |
| 007 | [Fraud model's live scores disagreed with offline scores for the exact same transactions](problems/007-training-serving-skew-parity) |
| 008 | [Shipped orders showed as "pending" in the warehouse, and some cancelled orders came back](problems/008-cdc-out-of-order-upserts) |
| 009 | [The morning revenue dashboard showed sales down 59%, and leadership opened an incident over a half-finished file](problems/009-write-audit-publish-partial-load) |
| 010 | [Daily order counts disagreed with the restaurants' totals all summer, and the November hourly chart merged two 1 a.m. hours](problems/010-dst-daily-rollup-timezone) |
| 011 | [Ticket-escalation model scored 84% in validation, then did no better than a coin flip on new customers](problems/011-entity-leakage-random-split) |
| 012 | [Revenue by region jumped 15% overnight with no change in orders, and the sales team started celebrating](problems/012-join-fanout-duplicate-dimension) |
| 013 | [The warehouse quietly disagreed with production: a few orders per day were missing or stuck at an old status, forever](problems/013-incremental-watermark-missed-rows) |
| 014 | [The new chargeback model's merchant feature scored 0.77 AUC in validation, and 0.55 in production](problems/014-target-encoding-leakage) |
| 015 | [The daily revenue report showed 42% of the day's sales, and keyless orders were booked to "Test accounts"](problems/015-null-join-keys-dropped-rows) |
| 016 | [Every dashboard showed a 50% drop in Tuesday's events, but users and revenue were normal](problems/016-partition-volume-anomaly-check) |
| 017 | [Sales table showed a day's revenue twice after the nightly load job was restarted](problems/017-file-reingestion-double-count) |
| 018 | [Fraud alerts went from "84% are real" in testing to 19 out of 20 false alarms in production](problems/018-balanced-sample-threshold-precision) |
<!-- INDEX:END -->

## Run the tests

```bash
python run_tests.py                       # all problems
cd problems/<folder> && python -m unittest -v
```

Requires Python 3.10+. No third-party packages.

## Author

Hrishikesh Jadhav
