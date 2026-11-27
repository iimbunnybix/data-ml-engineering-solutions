"""Click-through-rate feature computed with integer division.

The feature SQL was `clicks / impressions` on two INTEGER columns, which
truncates like Python's `clicks // impressions`. Every ad with fewer clicks
than impressions (almost all of them) got CTR 0, and a few tiny ads with one
impression and one click got CTR 1, so the ranking model had nothing to learn
from. The fix divides as floats, guards zero impressions, smooths small ads
toward the global CTR, and adds a data-quality check that fails when a
feature collapses to a handful of distinct values.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

SEED = 2
AD_COUNT = 1000
MIN_TRUE_CTR = 0.005
MAX_TRUE_CTR = 0.08
NEW_AD_SHARE = 0.2                # ads launched today: 0-3 impressions
MAX_NEW_AD_IMPRESSIONS = 3
MIN_IMPRESSIONS = 50
MAX_IMPRESSIONS = 5000
PRIOR_STRENGTH = 100.0            # smoothing acts like 100 extra impressions
TOP_K = 50                        # ads shown in the top placement
MIN_DISTINCT_VALUES = 10          # fewer distinct values than this = collapsed
MAX_SINGLE_VALUE_SHARE = 0.95     # one value covering more rows = collapsed


@dataclass(frozen=True)
class AdStats:
    """One ad's counts for the day, plus its true CTR (known only in simulation)."""
    ad_id: int
    impressions: int
    clicks: int
    true_ctr: float


# ---------------------------------------------------------------- data
def build_ads(seed: int = SEED) -> list[AdStats]:
    """Mostly established ads, plus new ads with almost no impressions."""
    rng = random.Random(seed)
    ads: list[AdStats] = []
    for ad_id in range(AD_COUNT):
        true_ctr = rng.uniform(MIN_TRUE_CTR, MAX_TRUE_CTR)
        if rng.random() < NEW_AD_SHARE:
            impressions = rng.randint(0, MAX_NEW_AD_IMPRESSIONS)
        else:
            impressions = rng.randint(MIN_IMPRESSIONS, MAX_IMPRESSIONS)
        clicks = count_clicks(impressions, true_ctr, rng)
        ads.append(AdStats(ad_id, impressions, clicks, true_ctr))
    return ads


def count_clicks(impressions: int, true_ctr: float, rng: random.Random) -> int:
    """Each impression is clicked with probability true_ctr."""
    clicks = 0
    for _ in range(impressions):
        if rng.random() < true_ctr:
            clicks += 1
    return clicks


# ---------------------------------------------------------------- CTR feature
def original_ctr(ad: AdStats) -> int:
    """Original: INTEGER / INTEGER in SQL truncates, exactly like // here.

    The SQL used NULLIF(impressions, 0) and COALESCE(..., 0), so ads with no
    impressions got 0 instead of an error.
    """
    if ad.impressions == 0:
        return 0
    return ad.clicks // ad.impressions


def global_ctr(ads: list[AdStats]) -> float:
    """All clicks over all impressions: the CTR we assume for an unknown ad."""
    total_clicks = sum(ad.clicks for ad in ads)
    total_impressions = sum(ad.impressions for ad in ads)
    if total_impressions == 0:
        return 0.0
    return total_clicks / total_impressions


def smoothed_ctr(ad: AdStats, prior_ctr: float, prior_strength: float = PRIOR_STRENGTH) -> float:
    """Fix: float division, pulled toward prior_ctr for ads with few impressions.

    This is the mean of a Beta prior updated with the ad's clicks: it adds
    prior_strength "virtual impressions" at prior_ctr. An ad with 1 click in
    1 impression gets about the prior, not 100%; an ad with thousands of
    impressions keeps (almost) its own CTR. Zero impressions gives the prior.
    """
    virtual_clicks = prior_ctr * prior_strength
    denominator = ad.impressions + prior_strength
    # Only possible with no smoothing and no impressions: nothing is known.
    if denominator == 0:
        return prior_ctr
    return (ad.clicks + virtual_clicks) / (ad.impressions + prior_strength)


def raw_float_ctr(ad: AdStats) -> float:
    """Float division without smoothing; 0.0 for ads with no impressions."""
    if ad.impressions == 0:
        return 0.0
    return ad.clicks / ad.impressions


def ctr_feature(ads: list[AdStats]) -> list[float]:
    """The fixed feature column, one value per ad."""
    prior = global_ctr(ads)
    return [smoothed_ctr(ad, prior) for ad in ads]


# ---------------------------------------------------------------- data quality
def feature_collapse_problems(values: list[float]) -> list[str]:
    """Return why a feature column looks collapsed; an empty list means OK.

    A real CTR has hundreds of distinct values. A handful of values, or one
    value on nearly every row, means the computation broke (truncation, a
    default filled everywhere, a wrong join) even though nothing errored.
    """
    if not values:
        return ["feature column is empty"]
    problems: list[str] = []
    distinct = len(set(values))
    if distinct < MIN_DISTINCT_VALUES:
        problems.append(f"only {distinct} distinct values (minimum {MIN_DISTINCT_VALUES})")
    counts: dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    top_value = max(counts, key=counts.get)
    top_share = counts[top_value] / len(values)
    if top_share > MAX_SINGLE_VALUE_SHARE:
        problems.append(f"value {top_value} on {top_share:.1%} of rows (maximum {MAX_SINGLE_VALUE_SHARE:.0%})")
    return problems


# ---------------------------------------------------------------- ranking
def top_ads(ads: list[AdStats], feature: list[float], k: int = TOP_K) -> list[AdStats]:
    """The k ads with the highest feature value.

    Ties keep the input order (Python's sort is stable), as a model would
    break ties arbitrarily when the feature carries no information.
    """
    order = sorted(range(len(ads)), key=lambda index: feature[index], reverse=True)
    return [ads[index] for index in order[:k]]


def mean_true_ctr_of_top_ads(ads: list[AdStats], feature: list[float], k: int = TOP_K) -> float:
    """Real CTR of the top-k ads: how good a ranking by this feature is."""
    top = top_ads(ads, feature, k)
    return sum(ad.true_ctr for ad in top) / len(top)


def new_ads_in_top(ads: list[AdStats], feature: list[float], k: int = TOP_K) -> int:
    """How many top-k ads have too few impressions to judge (lucky clicks)."""
    top = top_ads(ads, feature, k)
    return sum(1 for ad in top if ad.impressions < MIN_IMPRESSIONS)
