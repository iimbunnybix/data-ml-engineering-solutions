# 066 · The delivery-pricing report showed 10,064 trips in its distance bands for a day with 10,000 deliveries

## Problem
A food-delivery company prices courier pay by distance band and reviews a
daily report of trips per band (0-5 km, 5-10 km, 10-20 km). Finance noticed
the band totals added up to more trips than were delivered: in the seeded
simulation of 10,000 trips the report showed 10,064. Couriers on 5 km and
10 km trips were also paid twice in a reconciliation script built on the
same band logic.

An engineer "fixed" it by changing the bands to 0-4.9, 5-9.9 and 10-20.
Now the report showed 9,943 trips: 57 trips of 4.95 km or 9.95 km fell
between two bands and vanished from the report and the pay calculation.

## Root cause
**Bands written as (low, high) pairs with `low <= distance <= high` share
their edges.** The courier app reports distances rounded to 50 m, so
exactly 5.00 km and 10.00 km are common (64 of the 10,000 trips). Each of
those matched two bands. Subtracting 0.1 from every upper bound removed the
overlap but opened gaps, because the data has finer resolution than the
gap. Both bugs come from describing contiguous bands with two independent
numbers per band, and nothing checked that the band totals equal the number
of input rows.

## Solution
`band_counts(values, edges)`:

1. **Bands come from one sorted list of edges** (`[0, 5, 10, 20]`), so they
   are contiguous with no overlaps or gaps by construction.
   `validate_edges` rejects unsorted, repeated or non-finite edges.
2. **Half-open intervals** `[low, high)`: `bisect_right` puts a value equal
   to an edge into the band that starts there. The last band is closed,
   `[10, 20]`, so the maximum value is not lost. Labels say which end is
   included.
3. **Out-of-range values are reported**, not dropped: negatives, values
   above the top edge and NaN go to `out_of_range`. NaN gets an explicit
   check because every comparison with NaN is False.
4. **Reconciliation check**: counts plus out-of-range must equal the number
   of inputs, otherwise the function raises.

On the simulation the bands count 2,437 / 2,606 / 4,957 = exactly 10,000.

## Trade-offs
- Which band an edge belongs to is a business rule: `[low, high)` means a
  5.00 km trip is paid at the 5-10 km rate. If the contract says "up to 5 km",
  use `(low, high]` instead, but pick one rule and apply it everywhere.
- Float edges work here because distances are rounded to two decimals; for
  money use `Decimal` or integer cents so `0.1 + 0.2` style errors cannot
  push a value across an edge.
- `bisect` is O(log bands) per value; with pandas, `pd.cut(..., right=False,
  include_lowest=True)` or `numpy.digitize` do the same thing vectorised,
  but still check the totals.
- The reconciliation check is cheap and catches future regressions in any
  binning code, not just this one.

## Run
```bash
python -m unittest -v test_solution.py
```
