# 070 · "Sales down 50% vs last year" paged merchandising every Monday, while the business was flat

## Problem
A retailer's daily sales dashboard shows each day "vs last year", and an
alert pages the merchandising on-call when a day is more than 20% below last
year. In January the alert started firing **every Monday**. Fridays and
Saturdays showed +36% and +47% "growth" that the weekly totals did not
support. On the Saturday after Black Friday the alert called it a 63% crash,
and Black Friday itself was reported as +445%. People stopped trusting the
dashboard and muted the alert, which is how a real outage day later went
unnoticed. The job also had a known crash on 29 February.

## Root cause
`naive_last_year_date` compared each day with **the same calendar date one
year earlier** (`day.replace(year=day.year - 1)`). A year is 52 weeks plus
one day (two in a leap year), so the comparison day is a different weekday:
1 January 2026 is a Thursday, 1 January 2025 was a Wednesday. Retail sales
have a strong weekly pattern, so every Monday was compared with a Sunday
(about half the sales), every Saturday with a Friday, and so on. Moving
events like Black Friday (fourth Friday of November) land on different dates
each year, so they were compared with ordinary days. `replace` also raises
`ValueError` for 29 February, because the previous year has no such date.

In the replay in `test_solution.py`, a business with **exactly zero growth**
(same weekday pattern every year, Black Friday four times a normal Friday)
pages on **53 days of 2026**: all 52 Mondays (-50%) plus the day after Black
Friday (-63%). Another 103 days show more than +20%.

## Solution
`comparable_last_year_date` compares with the day **364 days (exactly 52
weeks) earlier**:

- It is always the **same weekday**, so the weekly pattern cancels out.
- Black Friday 2026 (27 Nov) maps to Black Friday 2025 (28 Nov). This holds
  in most years, but not all: in 2024 and 2030 Black Friday is a week later
  in the calendar than 364 days after the previous one.
- It is always a valid date, so 29 February no longer crashes.

`year_over_year_change` returns `None` when the baseline is missing or zero
instead of dividing by zero. On the replay the flat business never pages,
and a real 50% drop on one day still pages.

## Trade-offs
- The comparison day is one or two calendar days earlier than "the same
  date", so month-to-date and year-to-date totals built from it cover a
  slightly different date range than finance's calendar totals. Large
  retailers use a **retail calendar** (4-5-4, with a 53rd week every five or
  six years) and an explicit event calendar for days like Black Friday,
  which the 364-day rule misses in years such as 2024 and 2030.
- Fixed-date holidays (Christmas, New Year, national days) now compare with
  a different date. Treat them through a holiday table, not the 364-day
  rule.
- Daily YoY is noisy even when aligned. Weekly or 7-day rolling comparisons
  are more stable for alerting; keep the daily view for investigation.
- A same-weekday comparison still says nothing about promotions or price
  changes in either year; annotate those on the dashboard.

## Run
```bash
python -m unittest -v test_solution.py
```
