# 059 · A freezer full of stock thawed overnight while its temperature chart showed a flat, perfect -19 °C

## Problem
A cold-storage company monitors a few thousand freezers. Each sensor sends a
temperature reading every 5 minutes. A pipeline resamples every sensor onto
a fixed 5-minute grid and fills gaps with the last known value, because the
alarm rules and the anomaly model both expect a complete series, and radio
packets are lost every few minutes.

One night a sensor's battery died at 03:00 while its freezer's compressor was
also failing. The morning shift found thawed stock. The dashboard showed a
flat -19 °C until 12:00, and no alarm had fired. In the test reproduction the
pipeline filled **108 grid points** (9 hours) with the sensor's last reading,
and the "warmer than -12 °C" rule raised **zero** alerts all shift.

## Root cause
Forward-fill with no limit turns "we have not heard from this sensor" into
"the temperature is exactly what it was last time". That is fine for one lost
packet, but a dead sensor becomes indistinguishable from a perfectly stable
freezer, and every rule downstream only looks at temperatures, never at how
old they are. The missing data was not missing any more by the time it
reached the alarm.

## Solution
- **Bounded forward-fill** (`resample`): a reading is carried forward only
  while it is at most `MAX_STALENESS_MINUTES` (15) old. Two lost readings
  are still filled; after that the grid point has `celsius=None`.
- **Keep the age.** Every grid point carries `age_minutes`, so a model can
  use "minutes since the last reading" as a feature and dashboards can grey
  out old values instead of drawing a confident flat line.
- **Alert on staleness as a first-class condition** (`alerts_for`): one
  `sensor_stale` alert when a sensor goes quiet, and another only if it
  recovers and goes quiet again. In the test the dead sensor raises one
  alert at 03:20, 20 minutes after its last reading (03:15 is still within
  the 15-minute limit), instead of nothing for 9 hours.

## Trade-offs
- The staleness limit is a judgement call per signal: too short and a flaky
  radio pages people all night, too long and real outages are hidden. Base
  it on the sensor's normal reporting interval and observed packet loss.
- A stale alert says "we do not know", not "the freezer is warm". It needs a
  runbook (check the sensor, take a manual reading) or it will be ignored.
- Interpolation or model-based imputation can fill short gaps more smoothly,
  but they also invent data; they still need the same staleness bound.
- The same bug appears with price feeds, exchange rates and inventory
  counts: any "last known value" join needs a maximum age.

## Run
```bash
python -m unittest -v test_solution.py
```
