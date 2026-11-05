# 022 · Customer dashboard said the average customer was 43 and earned ₹9.6 lakh; the real numbers were 49 and ₹6.8 lakh

## Problem
A lending company loads a nightly customer export from a partner CRM
(20,000 rows in the test data) into its warehouse. The data feeds a
customer-profile dashboard and the features of a credit model. Analysts
noticed that average age looked too young, average income looked far too
high, and the model's "tenure" feature had a mean above the 120-month
maximum the business allows. No load job failed: every value was a valid
number.

## Root cause
The CRM never writes nulls. Unknown values are written as **sentinels**:
age `0` or `-1`, annual income `9999999`, tenure `-1`. A newer CRM version
also started writing `999` for unknown tenure, and nobody was told. The
pipeline treated these as real numbers, so every mean, percentile and model
feature included them.

Measured on the seeded test data:

| Column | Naive mean | Mean after cleaning | Share that was a sentinel |
|---|---|---|---|
| age | 43.1 | 49.2 | 12.4% (`0`, `-1`) |
| annual_income | 957,996 | 678,827 | 3.0% (`9999999`) |
| tenure_months | 157.0 | 59.8 (after declaring `999`) | 10.3% (`999`) |

## Solution
- **Declared sentinel registry** (`SENTINELS_BY_COLUMN`): agreed with the
  data owner, one set of "unknown" values per column. `clean_rows` replaces
  them with `None` before anything else runs. The registry is per column on
  purpose: tenure `0` is a real value ("joined this month") while age `0` is
  not.
- **Report missing rate next to the mean** (`mean_and_missing_rate`), so the
  12% of unknown ages are visible instead of silently dropped.
- **Spike detector** (`find_spikes`) for sentinels nobody declared: in a
  column with at least 20 distinct values, the most common value is flagged
  if it holds at least 2% of rows and at least 5× the median share of a
  distinct value. On the cleaned test data it flags only `tenure_months =
  999` (10.3% of rows vs a typical 0.74%). After adding `999` to the
  registry, no spikes remain and the tenure mean is 59.8.

## Trade-offs
- The detector looks at one value per column. Two sentinels that split the
  share (age `0` and `-1`, about 6% each on raw data) can each stay under
  the 5× ratio and pass unflagged, which is why the registry is the primary
  defence and the detector is a backstop.
- Real data can have legitimate spikes (a minimum wage, a default plan
  price, round numbers people type). A flag should open a question to the
  data owner, not auto-null values.
- Low-cardinality columns are skipped because a few values naturally
  dominate them; sentinels there need the registry.
- The real fix is upstream: ask the producer to emit nulls, or add a data
  contract that rejects sentinel values at ingestion.

## Run
```bash
python -m unittest -v test_solution.py
```
