# 052 · Northeast orders vanished from the regional dashboard and one SKU's revenue doubled overnight

## Problem
A retailer ingests daily order files from a marketplace partner as CSV:
`order_id, zip_code, sku, amount`. After the loader was rewritten to use
automatic type inference ("save everyone from declaring schemas"), the
regional dashboard showed Massachusetts, New Jersey and Connecticut collapsing
to almost nothing, while an "unknown" region appeared. At the same time the
best-selling SKU `7` roughly doubled its revenue and SKU `007`, a different
product, disappeared from reports.

In the seeded test file (1,000 orders over 10 ZIP codes, 3 of which start
with `0`, and 4 SKUs), the inferring loader leaves **290 of 1,000 orders**
in the unknown region, and reports **$53,071** for SKU `7` instead of its
true **$26,743** (the rest belongs to `007`). Total revenue is unchanged,
so the daily total check stayed green.

## Root cause
Type inference decides a column's type from what a value *looks like*.
`"02134".isdigit()` is true, so the ZIP becomes the integer `2134`, and the
leading zero, which carries meaning, is gone. The region table is keyed by
the five-character ZIP text, so `"2134"` never matches `"02134"`. SKUs `"007"`
and `"7"` both become `7` and are summed as one product. ZIP codes, SKUs,
phone numbers, account numbers and IBANs are identifiers: digits, but not
numbers. Spreadsheets do the same thing when someone opens and re-saves a file.

## Solution
- `load_orders` uses an explicit `ORDER_SCHEMA`: only `amount` is parsed as a
  number; identifier columns stay exactly as written.
- `check_columns` rejects files whose header differs from the schema instead
  of silently guessing types for unknown columns.
- `validate_zip_code` requires exactly five digits, so a file where zeros were
  already stripped upstream (for example, re-saved in a spreadsheet) fails the
  load instead of quietly producing unknown regions.
- The tests show 290 unknown-region orders and the merged SKU with the naive
  loader, and zero unknown regions and four distinct SKUs with the fix.

## Trade-offs
- Every file format needs a declared schema, and a new partner column fails
  the load until someone adds it. That is the intended behaviour for
  financial data, but it is more upkeep than inference.
- Padding with `zfill(5)` can repair ZIPs already stripped upstream, but it
  cannot repair SKUs: `7` could have been `07` or `007`. Fix the source.
- In pandas the equivalent is `read_csv(..., dtype={"zip_code": str, "sku":
  str})`; in Spark, pass a schema instead of `inferSchema=true`.
- Validating ZIP codes as five digits is US-specific; other countries need
  their own rules or no format check at all.

## Run
```bash
python -m unittest -v test_solution.py
```
