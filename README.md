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
<!-- INDEX:END -->

## Run the tests

```bash
python run_tests.py                       # all problems
cd problems/<folder> && python -m unittest -v
```

Requires Python 3.10+. No third-party packages.

## Author

Hrishikesh Jadhav
