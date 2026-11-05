# 035 · A tenth of yesterday's revenue was booked to region "Unknown", although every customer had a region in the CRM

## Problem
An e-commerce company streams orders into `fct_orders` within seconds, and
loads `dim_customer` from the CRM with an hourly sync. Regional sales
dashboards and the regional managers' bonus pool are computed from
`fct_orders` joined to `dim_customer`. After a marketing push that brought in
many new customers, the "Unknown" region grew to about a tenth of daily
revenue and never shrank, even days later when every one of those customers
was present in `dim_customer` with a valid region. Regional managers were
underpaid, and analysts started "fixing" the numbers by hand in spreadsheets.

## Root cause
**Facts arrived before their dimension row, and the loader resolved the
customer once, at load time.** A new customer orders minutes after signing
up, but their CRM row only lands at the next hourly sync. The loader looked
the customer up, found nothing, and stored the shared `UNKNOWN_KEY` (-1) on the
order. That row is shared by every unresolved order, so when the customer
arrived later there was no way to tell which Unknown orders were theirs, and
nothing ever revisited them.

The reproduction (300 new customers in a day, 5 orders each over the 5 hours
after signup, hourly CRM sync) pins **155 of 1,500 orders**, **10.2% of the
day's revenue**, to Unknown permanently.

## Solution
`solution.py` uses the **inferred member** (early-arriving fact) pattern:

1. `CustomerDimension.key_for_fact` gives an unseen customer a placeholder
   row with its **own surrogate key** (`inferred=True`, `region=None`) and the
   fact stores that key instead of -1.
2. `CustomerDimension.upsert` fills the placeholder **in place** when the CRM
   row arrives, keeping the key. Facts join to the dimension at query time,
   so every earlier order now reports the correct region, with no rewrite of
   the fact table.
3. Until then the order reports as Unknown, which is honest, and
   `inferred_customers()` lists placeholders still waiting, so a customer the
   CRM never sends shows up in monitoring instead of hiding in a bucket.

In the reproduction, Unknown revenue goes from 10.2% to **0** once the day's
CRM syncs have run, and the regional totals add up to total revenue.

## Trade-offs
- Filling a placeholder in place is SCD type 1 behaviour: later CRM changes
  also rewrite the region of old orders (a test shows this). If history must
  keep the region *at order time*, use SCD type 2 rows and treat the first
  real version as starting at the placeholder's creation.
- Placeholders can be created for garbage IDs (typos, test traffic). Alert on
  inferred members older than the expected sync delay and clean them up.
- Alternative: hold early facts in a "pending" table and retry the lookup on
  each sync. That keeps the dimension clean but delays the orders from
  appearing at all, and needs its own expiry and alerting.
- Alternative: re-resolve Unknown facts in a nightly batch. That works only if
  the fact still carries the natural key (`customer_id`), and the dashboards
  are wrong until the batch runs.

## Run
```bash
python -m unittest -v test_solution.py
```
