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
| 019 | [Revenue by customer tier was higher than total revenue after a CRM backfill](problems/019-scd2-late-change-overlapping-versions) |
| 020 | [Dashboard queries on the clickstream table got slower every day although data volume was flat](problems/020-small-files-compaction-planner) |
| 021 | [Fraud model recall dropped after every nightly retrain, while offline metrics looked fine](problems/021-label-maturity-training-window) |
| 022 | [Customer dashboard said the average customer was 43 and earned ₹9.6 lakh; the real numbers were 49 and ₹6.8 lakh](problems/022-sentinel-values-skew-aggregates) |
| 023 | [After merging the mobile and partner event feeds, a third of yesterday's events were dated January 1970 and the nightly job crashed on "year 57813"](problems/023-mixed-epoch-units-timestamps) |
| 024 | [Fraud scores for iOS users jumped to the Android level whenever a serving batch had no Android traffic](problems/024-category-encoding-column-order) |
| 025 | [Products with commas or line breaks in their description quietly vanished from the catalogue after every nightly import](problems/025-csv-quoted-newlines-row-loss) |
| 026 | [A fraud model retrained on a "random sample" was great on Asia-Pacific traffic and useless everywhere else](problems/026-head-sample-bias-reservoir-sampling) |
| 027 | [The revenue dashboard showed 35x more money than the bank received after we launched in India](problems/027-mixed-currency-revenue-fx-conversion) |
| 028 | [A "9% conversion lift" from a new checkout page vanished after launch because the 50/50 test was really 52/48](problems/028-ab-test-sample-ratio-mismatch) |
| 029 | [Delivery ETAs were 20 minutes too optimistic all evening while every dashboard stayed green](problems/029-stale-online-features-freshness-ttl) |
| 030 | [Churn model's launch-review accuracy kept coming in about 3 points lower in production](problems/030-test-set-reuse-model-selection) |
| 031 | [The retention team spent a week calling happy customers while the churn model's real high-risk list went uncalled](problems/031-batch-scores-joined-by-row-position) |
| 032 | [After a merchant-onboarding push, customers of 220 honest shops had their cards declined as fraud](problems/032-feature-hashing-merchant-collisions) |
| 033 | [The retention team staffed for 13,400 churners and 6,000 left; most customers flagged "more likely than not to leave" stayed](problems/033-uncalibrated-scores-capacity-forecast) |
| 034 | [The daily wire-total reconciliation alerted every morning over two cents, and each rerun of the job gave a slightly different total](problems/034-order-dependent-float-revenue-totals) |
| 035 | [A tenth of yesterday's revenue was booked to region "Unknown", although every customer had a region in the CRM](problems/035-early-arriving-facts-unknown-region) |
<!-- INDEX:END -->

## Run the tests

```bash
python run_tests.py                       # all problems
cd problems/<folder> && python -m unittest -v
```

Requires Python 3.10+. No third-party packages.

## Author

Hrishikesh Jadhav
