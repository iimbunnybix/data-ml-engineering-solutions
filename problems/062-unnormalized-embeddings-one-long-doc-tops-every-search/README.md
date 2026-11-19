# 062 · Help-centre search returned the "Terms of Service" page as the top answer to almost every question

## Problem
A company replaced keyword search in its help centre with semantic search:
every article is turned into an embedding vector, stored in a vector index,
and each customer question is embedded and matched against it. The index
was configured for **inner product** ("dot product") similarity because it
was the fastest option.

Within a week, support saw contact volume rise: customers asking "how do I
get my money back" or "where is my parcel" were shown the long Terms of
Service page first and gave up. Search click-through on the first result
fell, and offline spot checks found the same few long articles at the top
of most result lists.

## Root cause
A dot product is `|a| * |b| * cos(angle)`: it grows with the **length** of
the vectors as well as with how closely they point the same way. The
embedding model used here does not return unit-length vectors, and long
documents that mention many topics end up with long vectors. In the test,
the Terms of Service vector `[2, 2, 2]` scores 2.0 against the refund
question `[1, 0, 0]`, while the refund policy `[0.9, 0.1, 0]` scores only
0.9, so the legal page is first for all three test queries.

## Solution
`CosineIndex` normalises every vector to length 1 when it is **added**, and
normalises every query before **searching**. For unit vectors the dot
product equals cosine similarity, so only direction (meaning) counts and
scores stay between -1 and 1:

| Query | Raw dot product, first result | Cosine, first result |
|---|---|---|
| how do I get my money back | terms-of-service (2.0) | refund-policy (0.99) |
| where is my parcel | terms-of-service (2.2) | track-delivery (0.99) |
| forgot my password | terms-of-service (2.0) | reset-password (0.99) |

Normalising at write time keeps search a plain dot product, so the fast
inner-product index can still be used. Zero vectors (an empty article or a
failed embedding call) raise `ZeroVectorError` instead of dividing by zero
or quietly scoring 0, and equal scores are ordered by document id so
results are stable.

## Trade-offs
- Check what the model expects. Many embedding models are trained for
  cosine similarity and some already return unit vectors; a few are trained
  so that length carries meaning (for example "how relevant or popular"),
  and normalising would throw that away. Measure with a labelled query set
  (recall@k, MRR) before and after.
- Re-normalising means re-indexing every stored vector, and query and index
  must be normalised the same way; mixing old and new vectors silently
  brings the problem back.
- Long documents are often better split into chunks of a few hundred words
  and embedded separately; one vector for a 20-page document blurs all its
  topics together whatever the similarity measure.

## Run
```bash
python -m unittest -v test_solution.py
```
