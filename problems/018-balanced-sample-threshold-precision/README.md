# 018 · Fraud alerts went from "84% are real" in testing to 19 out of 20 false alarms in production

## Problem
A payments team trained a fraud model on a **50/50 sample** of fraud and
normal payments (fraud is rare, so they down-sampled normal payments to make
training easier) and tuned the alert threshold at 0.5 on that same sample. In
the simulation in `solution.py`, that sample gives **84% precision and 84%
recall**. In production, where only **1%** of payments are fraud, the same
threshold gives **4.9% precision** on 50,000 payments: about 19 of every 20
blocked payments were legitimate. The review queue overflowed and good
customers had cards declined.

## Root cause
A classifier's probability depends on the **class prior it was trained
under**. Bayes' rule: posterior odds = likelihood ratio × prior odds. Training
on 50/50 data bakes in prior odds of 1:1; production prior odds are 1:99. The
model's ranking is still fine, but its probabilities are inflated (average
predicted fraud probability on production traffic is about 0.23 against a
real rate of 0.01), and a threshold chosen on balanced data does not mean the
same thing on live traffic. Precision depends directly on the base rate,
while recall does not, which is why recall looked unchanged and nobody noticed
until the alerts arrived.

## Solution
1. **Prior-shift correction** (`adjust_for_base_rate`): in log-odds,
   subtract the training prior and add the production prior. A balanced score
   of 0.5 becomes 0.01 — "no evidence" now means "the normal fraud rate".
   After correction the average predicted probability is about 0.010, matching
   reality, so the scores can be read as real probabilities again.
2. **Choose the threshold on data with the production base rate**
   (`choose_threshold_for_precision`): one sorted pass finds the lowest
   threshold that still meets the precision target, giving the most recall
   for that precision. With a 50% precision target it picks a corrected
   threshold of about 0.32 (recall 16%), versus 9% recall when simply using
   0.5 on the corrected scores (precision 68%).

The correction is monotonic, so it never changes which payment ranks above
another; it fixes what the numbers *mean* and where the cut-off goes.

## Trade-offs
- The production fraud rate must be known and kept current; if it drifts,
  the correction drifts with it. Monitor it (see problem 004).
- Prior-shift correction assumes only the class mix changed, not how fraud
  looks (`P(x | fraud)`). If fraud patterns change, retrain.
- With ~1% positives, a 50,000-payment tuning set has only ~480 fraud cases,
  so the chosen threshold is noisy: on a fresh seeded sample the same
  threshold gave 44% precision instead of 50%. Use more labelled data, a
  confidence margin, or re-tune regularly.
- Alternatives: train with class weights instead of down-sampling, or fit
  an isotonic / Platt calibrator on production-rate validation data.

## Run
```bash
python -m unittest -v test_solution.py
```
