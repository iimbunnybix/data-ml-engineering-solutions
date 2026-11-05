"""Block A/B results when the traffic split is not what was configured.

A 50/50 checkout experiment ended up with fewer logged users in the new
variant: its page redirect sometimes fired before the tracking call, so users
who bounced during the redirect were never logged. Those users rarely buy, so
the variant's conversion rate looked higher than it really was, and the team
shipped a "winner" that had no real effect.

The fix is a sample ratio mismatch (SRM) check run before anyone may read the
results: a chi-square goodness-of-fit test of the observed user counts
against the configured split. If the p-value is below 0.001 the split is
almost certainly broken by a bug, and reading the results raises an error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Strict on purpose: SRM is checked on every experiment, often daily, so a
# loose threshold would block many healthy tests by chance.
SRM_P_VALUE_THRESHOLD = 0.001
SIGNIFICANCE_LEVEL = 0.05


@dataclass(frozen=True)
class VariantStats:
    name: str
    users: int
    conversions: int

    @property
    def conversion_rate(self) -> float:
        return self.conversions / self.users if self.users else 0.0


@dataclass(frozen=True)
class SrmResult:
    chi_square: float
    p_value: float
    observed_shares: list[float]

    @property
    def has_mismatch(self) -> bool:
        return self.p_value < SRM_P_VALUE_THRESHOLD


@dataclass(frozen=True)
class ExperimentReadout:
    winner: str
    lift: float          # relative change of treatment vs control
    p_value: float       # two-proportion z-test
    significant: bool


class SampleRatioMismatchError(Exception):
    """The observed split does not match the configured one; results are biased."""


# --- statistics (standard library only) -------------------------------------
def chi_square_statistic(observed: list[int], expected_shares: list[float]) -> float:
    """Pearson chi-square: sum of (observed - expected)^2 / expected over groups."""
    if len(observed) != len(expected_shares):
        raise ValueError("need one expected share per group")
    if any(count < 0 for count in observed):
        raise ValueError("counts must be >= 0")
    if any(share <= 0 for share in expected_shares) or not math.isclose(sum(expected_shares), 1.0):
        raise ValueError("expected shares must be positive and sum to 1")
    total = sum(observed)
    if total == 0:
        raise ValueError("no users observed")
    statistic = 0.0
    for count, share in zip(observed, expected_shares):
        expected = total * share
        statistic += (count - expected) ** 2 / expected
    return statistic


def chi_square_p_value_1dof(statistic: float) -> float:
    """P(X >= statistic) for a chi-square with 1 degree of freedom.

    A chi-square with 1 dof is the square of a standard normal Z, so
    P(Z^2 >= x) = P(|Z| >= sqrt(x)) = erfc(sqrt(x / 2)).
    """
    return math.erfc(math.sqrt(statistic / 2))


def two_proportion_p_value(control: VariantStats, treatment: VariantStats) -> float:
    """Two-sided p-value of a pooled two-proportion z-test on conversion rates."""
    pooled_rate = (control.conversions + treatment.conversions) / (control.users + treatment.users)
    standard_error = math.sqrt(pooled_rate * (1 - pooled_rate) * (1 / control.users + 1 / treatment.users))
    if standard_error == 0:
        return 1.0
    z_score = (treatment.conversion_rate - control.conversion_rate) / standard_error
    # Two-sided normal tail: P(|Z| >= |z|) = erfc(|z| / sqrt(2)).
    return math.erfc(abs(z_score) / math.sqrt(2))


# --- the original readout and the guarded one --------------------------------
def naive_read_results(control: VariantStats, treatment: VariantStats) -> ExperimentReadout:
    """Original dashboard: compares conversion rates without checking the split."""
    p_value = two_proportion_p_value(control, treatment)
    lift = treatment.conversion_rate / control.conversion_rate - 1
    winner = treatment.name if lift > 0 else control.name
    return ExperimentReadout(winner, lift, p_value, p_value < SIGNIFICANCE_LEVEL)


def check_srm(control: VariantStats, treatment: VariantStats,
              expected_control_share: float = 0.5) -> SrmResult:
    """Chi-square goodness-of-fit of the two user counts against the configured split."""
    observed = [control.users, treatment.users]
    expected_shares = [expected_control_share, 1 - expected_control_share]
    statistic = chi_square_statistic(observed, expected_shares)
    total = sum(observed)
    shares = [count / total for count in observed]
    return SrmResult(statistic, chi_square_p_value_1dof(statistic), shares)


def read_results(control: VariantStats, treatment: VariantStats,
                 expected_control_share: float = 0.5) -> ExperimentReadout:
    """Readout that refuses to report a winner when the split is broken."""
    srm = check_srm(control, treatment, expected_control_share)
    if srm.has_mismatch:
        raise SampleRatioMismatchError(
            f"sample ratio mismatch: observed {srm.observed_shares[0]:.2%} / {srm.observed_shares[1]:.2%}, "
            f"expected {expected_control_share:.0%} / {1 - expected_control_share:.0%} "
            f"(chi-square {srm.chi_square:.1f}, p = {srm.p_value:.2g}); find the logging bug first")
    return naive_read_results(control, treatment)
