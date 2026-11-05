# 025 · Products with commas or line breaks in their description quietly vanished from the catalogue after every nightly import

## Problem
An online marketplace imports its supplier catalogue every night from a CSV
export (about 400,000 products) into the search index and the pricing
tables. Merchandisers kept reporting that specific products "disappeared"
from search overnight, then came back after someone re-uploaded them by
hand. The import job always finished green. An audit later found that
roughly 3% of the catalogue had never been loaded, and those were mostly
premium products, because they had the longest, most detailed descriptions.

## Root cause
`naive_parse` read the file with `splitlines()` and split each line on `,`.
CSV is not "one record per line, fields separated by commas": a quoted field
may contain commas, doubled quotes (`""`) and even line breaks (RFC 4180).
So:

- `"Ceramic, dishwasher safe"` became two fields, giving the row 5 columns.
- A description with a newline split one record across two physical lines,
  each with too few columns.

The loader treated a wrong column count as "a malformed row, skip it" and
logged nothing, so the loss was invisible. In the test, 2 of 4 products are
dropped and the function returns normally.

## Solution
- **Parse with a real CSV parser.** `csv.reader` over a stream opened with
  `newline=""` handles quoting, doubled quotes, embedded commas and embedded
  newlines, for both `\r\n` and `\n` files. All 4 test products round-trip
  exactly, including the newline inside the teapot's description.
- **Never skip, fail.** A record with the wrong number of columns raises
  `MalformedRecord` with the record number; a changed header (renamed or
  reordered column) is rejected before any row is read.
- **Reconcile counts.** `load_products` compares the parsed row count with
  the row count in the vendor's manifest file and raises `RowCountMismatch`
  if they differ, which also catches an upload that was cut short.

## Trade-offs
- Failing the whole import on one bad record is strict. If the vendor's data
  is often dirty, write bad records to a quarantine file and fail only if
  they exceed a small threshold, but always count and report them.
- A row-count manifest needs the vendor's cooperation. Without one, compare
  against yesterday's count with a tolerance (see problem 016's volume
  check) or ask for a checksum.
- `csv` decides quoting by the dialect. Exports that use a different
  delimiter, escape character (`\"` instead of `""`) or encoding need an
  explicit dialect; `csv.Sniffer` guesses but should not be trusted blindly.
- For large files, columnar formats (Parquet) avoid text-quoting issues
  entirely and carry their own schema.

## Run
```bash
python -m unittest -v test_solution.py
```
