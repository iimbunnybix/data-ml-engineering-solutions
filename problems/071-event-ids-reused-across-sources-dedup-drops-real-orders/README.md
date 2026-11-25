# 071 · Finance saw 37% of order revenue vanish the week the US checkout went live, with no errors anywhere

## Problem
An online retailer's order pipeline reads events from a message broker and
writes them into the orders table that finance reports revenue from. The
broker delivers at least once, so the pipeline drops retries by remembering
every `event_id` it has seen. For years only the EU checkout service produced
events. When a second checkout service launched in the US, the orders table
started coming up short: in the replay (`test_solution.py`: 10,000 EU and
6,000 US orders, 3% delivered twice, regions interleaved) the pipeline keeps
**10,000 of 16,000 real orders** and reports **62.6%** of the real revenue.
The job raised no errors, the dedup counter just looked busier, and both
regions lost orders, so neither regional team saw a clean "US is missing" signal.

## Root cause
**The dedup key was not globally unique.** Each checkout service numbers its
events from its own sequence starting at 1, so `event_id` is unique only
*within one producer*. Order `"7"` from the EU and order `"7"` from the US are
different orders, but `naive_dedup` keyed on `event_id` alone and treated
whichever arrived second as a retry of the first. All 6,000 US ids overlap
the EU range, so 6,000 real orders were dropped: in this replay 2,995 EU and
3,005 US orders, depending on which region's copy of an id arrived first.
With a single producer the ids happened to be unique, so tests and the
first years of production never showed the bug.

## Solution
**Identify an event by `(source, event_id)`** (`dedup_key`, `dedup`):

1. The key is a tuple of the producer name and its id, so ids are only
   compared within the producer that issued them. A real retry has the same
   source and id and is still dropped.
2. The key is a tuple, not a joined string: `"eu" + "-" + "1-2"` and
   `"eu-1" + "-" + "2"` would otherwise be the same key.
3. An event with no source raises `MissingSourceError` rather than being
   deduplicated against an unknown namespace.

With the fix every one of the 16,000 orders is kept exactly once, the 484
redelivered copies are dropped, and revenue matches.

## Trade-offs
- The pipeline now depends on every producer sending a stable source name.
  Renaming a service (`checkout-us` to `checkout-na`) makes its old retries
  look new; source names should be treated as part of the event contract.
- Producer-generated UUIDs (or ULIDs) avoid the problem without a namespace,
  but only for new producers; existing sequential ids still need the source.
- The seen-key set grows forever here. Production dedup keeps keys for a
  bounded window (the broker's maximum redelivery delay) in a TTL store.
- This catches broker retries only. A client that submits the same order
  twice with two different event ids needs a business idempotency key
  (for example the checkout session id) instead.

## Run
```bash
python -m unittest -v test_solution.py
```
