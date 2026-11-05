# 008 · Shipped orders showed as "pending" in the warehouse, and some cancelled orders came back

## Problem
A CDC connector streams every row change from the production orders database
(PostgreSQL) through a message queue into the analytics warehouse. Operations
dashboards and the customer-support tool read the warehouse copy. After a
week with several sink timeouts and one consumer-group rebalance, support saw
orders that had shipped days ago still marked "pending", and a few orders
customers had cancelled were back as active. Row counts matched closely
enough that the nightly reconciliation job raised nothing.

## Root cause
The sink applied events **in arrival order** (insert/update → upsert, delete
→ delete), which is correct only with exactly-once, in-order delivery. Real
CDC delivery is *at-least-once*: when the sink times out, the batch is retried;
after a rebalance, the consumer replays from its last committed offset. Older
batches therefore arrive *after* newer ones and:

- an old update overwrites a newer one (status rolls back), and
- an old update arriving after a delete **re-creates** the deleted row.

In the simulation (3,000 changes over 200 keys, delivered in batches of 50
with 10 older batches replayed, 3,500 deliveries in total), the naive sink
ends with **29 of 186 keys wrong**, including 1 resurrected deleted row.

## Solution
**LSN-guarded, idempotent apply with tombstones.**

1. Every change carries its source **LSN** (log sequence number), which is
   strictly increasing in the source database's write-ahead log.
2. The sink stores the LSN of the last change applied to each key and ignores
   any event with `lsn <= stored_lsn`. Re-deliveries and stale events become
   no-ops. In SQL this is `MERGE ... WHEN MATCHED AND src.lsn > tgt.lsn`.
   LSNs must be integers: PostgreSQL's textual form (`"0/9"`, `"0/10"`)
   compares wrongly as a string, so the sink rejects non-integer LSNs and
   `parse_pg_lsn("X/Y")` converts the text form first.
3. Deletes are stored as **tombstones** (key + delete LSN) instead of
   removing the row, so a late, older update is recognised as stale.
4. `purge_tombstones(safe_lsn)` removes tombstones older than a low watermark
   (the oldest LSN that could still be redelivered). The test shows that
   purging above the watermark reopens the resurrection bug.

With the same delivery stream, the versioned sink matches the source exactly,
applies each of the 3,000 changes once and skips the 500 duplicates. It also
converges for random shuffles of the stream.

## Trade-offs
- Per-key "last LSN wins" is correct only for **full row images**. If the
  connector emits partial updates (only changed columns), you must apply them
  in order or fetch the full row.
- LSNs are only comparable within one source database. After a failover to a
  new primary or when merging several sources, you need a composite version
  (e.g. epoch + LSN) or a source-assigned version column.
- Tombstones cost storage until purged, and choosing the watermark needs
  knowledge of consumer offsets and retention.
- Alternative: an append-only change log plus a periodic "latest per key"
  compaction (`ROW_NUMBER() OVER (PARTITION BY key ORDER BY lsn DESC)`). It is
  simpler to reason about, but reads are only as fresh as the last compaction.

## Run
```bash
python -m unittest -v test_solution.py
```
