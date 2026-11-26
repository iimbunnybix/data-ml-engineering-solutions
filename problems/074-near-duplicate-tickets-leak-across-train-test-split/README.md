# 074 · The ticket classifier scored 77% offline but routed most new tickets to the wrong team

## Problem
A support team trains a classifier that routes incoming tickets to billing,
login, shipping or refunds. The training set is a helpdesk export where many
customers had sent the same complaint more than once: a "Re: Fwd:" resend, a
new ticket number from the helpdesk, the earlier mail quoted underneath,
sometimes in capitals. The pipeline shuffled the rows and held out 20% as a
test set.

The model's offline test accuracy looked healthy, so it shipped. In
production, routing accuracy on new tickets was far lower, agents
re-assigned tickets by hand all day, and first-response times went up. The
offline number had been measuring something else.

## Root cause
**Copies of the same ticket landed on both sides of the split.** Row-level
random splitting treats a resend as an independent example. In the model
(400 distinct complaints, 160 of them resent 1-3 extra times: 724 rows),
**64.8%** of test rows had a copy of themselves in training. A "model" that
only memorises training tickets and guesses the most common category
otherwise scores **76.6%** on that split. It has learned nothing that
transfers to a new complaint, and on a split without leaked copies it scores
**12.4%**. Exact-match deduplication does not help, because the copies
differ in prefix, ticket number, case, spacing and quoted text.

This differs from entity leakage (problem 011, the same *customer* in both
sets) and from oversampling before the split (problem 053, where the copies
are made by the pipeline and share a row id). Here the rows are different
strings, written by customers, that are the same example.

## Solution
1. **Normalise to what the customer asked.** `normalize` applies Unicode NFKC,
   drops quoted (`>`) lines, case-folds, removes `Re:`/`Fwd:` prefixes and
   helpdesk ticket references, and collapses whitespace.
2. **Fingerprint** the normalised text with SHA-256, so the id is the same on
   every machine and every run (Python's built-in `hash()` is salted per
   process).
3. **Split by group, not by row.** `group_split` sends every ticket whose
   fingerprint falls in hash buckets 0-19 to test and the rest to training.
   All copies of a complaint therefore land on the same side, and a copy that
   arrives next month joins its earlier copies' side.

With the group split, 0% of test tickets have a copy in training and the
memoriser's score falls from 76.6% to 12.4%. The test number now measures
generalisation to complaints the model has never seen.

## Trade-offs
- **Normalisation rules are domain knowledge.** These handle our mail client
  and helpdesk noise. Paraphrased resends ("I was charged two times") still
  slip through; catching those needs MinHash/SimHash or embedding similarity
  with a threshold, which brings false merges of genuinely different tickets.
- **Group splits are uneven.** 129 test rows instead of 145, and category
  mix varies with which groups hash where. For small datasets, stratify by
  category within the group assignment.
- **Keep duplicates in training?** Repeated complaints are real signal about
  frequency, so we keep them, but only on one side of the split. Some teams
  deduplicate training too, to stop the model over-weighting a few loud
  customers.
- A time-based split (train on older tickets, test on newer) also limits
  leakage and mirrors production, but resends that span the cut-off still
  leak unless they are grouped as well.

## Run
```bash
python -m unittest -v test_solution.py
```
