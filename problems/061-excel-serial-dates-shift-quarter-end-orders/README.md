# 061 · Q1 revenue came in ₹65,000 short for one region because its last two days of March were reported as April

## Problem
A retailer's finance pipeline loads monthly order sheets from regional sales
teams. Most regions export CSV from the ERP, but two upload sheets saved from
Excel. When a date column loses its date formatting, Excel writes the
underlying **serial number** instead of the date, for example `46112`
instead of `2026-03-31`, and the loader had a fallback to convert serials.

At quarter close, one region's Q1 revenue was short and its Q2 opened with
orders "from the future" (dated 1 and 2 April before April had started).
In the test data, ₹65,000 of orders placed on 30 and 31 March appear in Q2.
Finance had to reopen the quarter and the region's sales bonuses were
calculated on the wrong numbers.

## Root cause
`naive_serial_to_date` computed `1900-01-01 + serial days`. That is wrong in
two ways that add up:

- Excel counts **1 January 1900 as day 1**, not day 0 (one day late).
- Excel's 1900 date system includes **29 February 1900**, which never
  existed (1900 is not a leap year), a bug kept from Lotus 1-2-3 for
  compatibility (serial 60). Every date from 1 March 1900 on is one more
  day late.

So every modern date came out exactly two days late; it went unnoticed
because the shift only changes totals near month and quarter boundaries.

## Solution
`serial_to_datetime(serial, date_system)`:

- **1900 system (Windows Excel):** serials from 61 use the epoch
  1899-12-30, serials 1-59 use 1899-12-31, so `46112` is 2026-03-31, `1` is
  1900-01-01 and `61` is 1900-03-01. Serial `60` (the fake leap day) and
  serials below 1 raise `InvalidSerialDate` instead of producing a date.
- **1904 system (old Mac Excel workbooks):** day 0 is 1904-01-01; the same
  day has a serial 1,462 lower than in the 1900 system. The workbook's date
  system must be passed in, because the number alone cannot tell you.
- The fraction is the time of day (`.75` is 18:00), rounded to whole
  seconds so float noise does not produce 17:59:59.999.

With the fix, the test region's Q1 total is ₹75,000 and Q2 ₹5,000, instead
of ₹10,000 and ₹70,000.

## Trade-offs
- The best fix is upstream: read the workbook with a library that applies
  the file's own date system and cell formats, or require ISO dates
  (`2026-03-31`) in uploads and reject bare numbers.
- A number column cannot always be recognised as a date. A plausible range
  check (for example serials that fall within the last few years) helps
  catch a date column that arrived as numbers, but it can also misfire on
  amounts or IDs in that range.
- Dates before 1 March 1900 hardly matter for orders; they are handled to
  match Excel exactly so the converter has no surprising corner.

## Run
```bash
python -m unittest -v test_solution.py
```
