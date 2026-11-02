# 001 · Upstream renamed a column and the churn model silently scored garbage for 9 days

## Problem
A nightly pipeline loads `customers_YYYYMMDD.csv` from the CRM team and
scores churn risk. The CRM team renamed `monthly_spend` to
`monthly_spend_inr` and changed `signup_date` from `YYYY-MM-DD` to
`DD/MM/YYYY`. `pandas.read_csv` happily loaded the file, the missing
column became all-NaN, the model imputed the median for everyone, and the
retention team called the wrong 2,000 customers before anyone noticed.

## Root cause
The pipeline trusted the file's shape. There was no **data contract**:
nothing checked required columns, types, allowed ranges or null rates
before the data reached the model, and no step failed loudly.

## Solution
A small, dependency-free **schema contract** validated at ingestion:

- Required columns, with optional **aliases** for known renames
  (`monthly_spend_inr → monthly_spend`) so planned changes don't break.
- Per-column type checks, allowed ranges and categorical values.
- Date format check with an explicit format string.
- **Null-rate thresholds** — a column that is suddenly 100% null fails.
- Unknown new columns produce a warning, not a failure.

Validation returns a report with every problem found (not just the first),
and the pipeline stops before scoring if there are any errors.

## Trade-offs
- Libraries like Great Expectations / pandera do this at scale; this shows the
  mechanism with no dependencies.
- Too-strict contracts break on harmless changes; aliases and warnings for
  extra columns keep it practical.
- The contract itself must be versioned and agreed with the upstream team.

## Run
```bash
python -m unittest -v test_solution.py
```
