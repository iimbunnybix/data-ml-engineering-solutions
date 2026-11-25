# 072 · A stuck fridge sensor sent 85 °C readings for hours and the "remove outliers" step kept every one of them

## Problem
A grocery chain's cold-chain pipeline collects fridge temperatures from its
stores and, before the data reaches the spoilage model, drops faulty sensor
readings with a classic rule: remove any reading more than 3 standard
deviations from the batch mean. One night a sensor got stuck and reported
85 °C for several hours, 60 of the 500 readings in its batch
(`test_solution.py`, seeded). The filter removed **none** of them. The
cleaned batch averaged **13.7 °C**, well above the 8 °C spoilage limit, so the
store was told to throw away a fridge of good stock that had held about 4 °C
all night. The same rule had worked in testing, where a sensor glitch was a
handful of readings, not a sustained fault.

## Root cause
**The outliers were used to measure what "normal" is.** The mean and standard
deviation are computed from the whole batch, bad readings included, and both
are pulled hard by extreme values. With 12% of readings at 85 °C the mean
moves from 4 °C to 13.7 °C and the standard deviation grows from 0.5 °C to
26.3 °C, so an 85 °C reading is only **2.70** deviations from the mean: the
outliers mask themselves. With 40 stuck readings (8%) the same rule still
catches all of them, which is why small tests passed.

The rule is also blind on small batches: with `n` values and the sample
standard deviation, no z-score can exceed `(n - 1) / sqrt(n)`, which is
**2.85** for 10 readings. A single 85 °C spike among nine 4 °C readings can
never be flagged.

## Solution
**Use a robust centre and spread: the median and the median absolute
deviation (MAD)** (`outlier_indexes`, the Iglewicz-Hoaglin modified z-score):

1. Distance is measured from the median, which does not move while fewer
   than half of the readings are bad.
2. It is scaled by MAD / 0.6745, which estimates the standard deviation for
   normal data but is not inflated by the outliers.
3. Readings with a modified z-score above 3.5 (the published recommendation)
   are dropped.
4. When more than half the readings are identical MAD is 0, so the code falls
   back to the mean absolute deviation (times 1.2533) instead of dividing by
   zero or flagging every reading that differs.

The fixed filter flags exactly the 60 stuck readings, the cleaned batch
averages 4.0 °C, and the single spike in the 10-reading batch is flagged.

## Trade-offs
- It still fails once half or more of a batch is bad: the median then
  belongs to the fault. Sensor health checks (a value repeated for an hour,
  a physically impossible value) catch that case better than statistics.
- On clean normal data a 3.5 limit still drops the occasional genuine tail
  reading: 1 of 500 in the seeded clean batch (the old rule also drops 1).
- MAD assumes roughly symmetric data. For skewed metrics (latencies, order
  values) use quantile limits or score the log of the values.
- A fixed physical range (fridges cannot read 85 °C) is simpler and should
  run first; the statistical filter is for faults inside the plausible range.

## Run
```bash
python -m unittest -v test_solution.py
```
