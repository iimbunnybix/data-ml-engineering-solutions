# 039 · Revenue attributed to account managers fell by a third after billing changed its export, while the bank received the same money

## Problem
A B2B company pays account-manager commission from a monthly job that joins
CRM customers (`customer_key`, `account_manager`) to billing accounts
(`account_key`, `revenue_cents`) on the customer key. The billing team moved
its export to a new tool. The next month, attributed revenue dropped by about
a third while bank deposits were flat. Sales leadership opened a "churn"
investigation, and four account managers were told their book had shrunk.

Looking at the unmatched rows, the keys were "the same" to any human:
`ACME-0001 ` (trailing space), `ACME-0001` followed by a no-break space,
`acme-0001`, `ＡＣＭＥ－０００１` (full-width characters from an East-Asian
locale) and `ACME-0001` with an invisible zero-width space pasted from a web
page. About 3% of CRM keys also had a trailing space.

In the simulation (2,000 customers, 2,010 billing accounts including 10 trial
accounts with no customer, seed 7), the exact join matched **1,367** accounts
and attributed **68.7%** of revenue: 158,982,614 of 507,647,849 cents went
nowhere, and the job reported success.

## Root cause
- **The join compared raw strings.** Whitespace, case and Unicode form are
  not meaningful in a customer code, but string equality treats every
  difference as a different customer.
- **A partial clean-up is not enough.** `strip().upper()` (the fix in 015 for
  ASCII whitespace and case) matches 1,789 accounts here, still losing over
  200: full-width characters and zero-width spaces survive it.
- **Nobody measured the join.** An inner join returns fewer rows without any
  error, so losing 31% of revenue looked like a normal month.

## Solution
`solution.py`:

- **One `normalise_key`, applied to both sides.** Unicode NFKC (full-width →
  ASCII, no-break space → space, decomposed accents → composed), remove
  zero-width and BOM characters, trim and collapse whitespace, then
  `casefold()`. With it, all 2,000 customers match.
- **Match-rate report.** `normalised_join` returns a `JoinReport` with the
  matched pairs, unmatched billing accounts and unmatched CRM customers. Here
  the match rate is 99.5% (the 10 trial accounts), and their revenue goes to
  an explicit `UNMATCHED` bucket, so attributed revenue adds up to the billing
  total.
- **Collision detection.** Normalisation must not merge distinct keys. If two
  rows on one side normalise to the same key (`ACME-01` and `acme-01` as two
  CRM customers), it is reported as a `KeyCollision` and that key is not
  matched at all, instead of crediting one manager by chance.
- **Quality gate.** `check_join_quality` raises `JoinQualityError` on any
  collision or when the match rate is below 98%, before commission is paid.

## Trade-offs
- Case-folding and NFKC are only correct if the key is a code where those
  differences carry no meaning. Confirm with both owning systems; for keys
  that are case-sensitive (some base-62 IDs), normalise only whitespace and
  Unicode form.
- NFKC is aggressive (`①` → `1`, `ﬁ` → `fi`). That is fine for codes, and
  collision detection catches the rare case where it merges real keys.
- Normalising in every join is slow and easy to forget. Better: store the
  normalised key as its own column at ingestion and keep the raw key for
  display and audits.
- The 98% threshold is a business choice. A small, stable unmatched set
  (trials, test accounts) is normal; a sudden jump is the signal to alert on.
- Related: 015 covers NULL and blank join keys; 012 covers fan-out from
  duplicate dimension keys.

## Run
```bash
python -m unittest -v test_solution.py
```
