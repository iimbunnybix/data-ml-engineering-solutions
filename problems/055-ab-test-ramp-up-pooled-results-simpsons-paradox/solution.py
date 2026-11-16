"""Analyse an A/B test whose traffic split changed half-way through.

The experiment started with 10% of users on the new checkout and was ramped
to 50% in its second week, which happened to be a holiday sale when everyone
converted more. The original analysis pooled all users of each arm, so the
treatment arm was mostly "sale week" users and looked like a big win, even
though it lost to control inside every week (Simpson's paradox). This module
compares the arms inside each ramp phase, where the split was constant, and
combines those per-phase differences with fixed weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# 1.96 standard errors either side of the estimate is a 95% interval.
Z_95 = 1.96
# Allocation shares that differ by less than this are treated as "unchanged".
ALLOCATION_TOLERANCE = 0.01


@dataclass(frozen=True)
class ArmCounts:
    users: int
    conversions: int

    def rate(self) -> float:
        """Conversion rate of this arm; callers must check users > 0 first."""
        return self.conversions / self.users


@dataclass(frozen=True)
class Phase:
    """One period with a constant traffic split, e.g. one week of the ramp."""
    name: str
    control: ArmCounts
    treatment: ArmCounts

    def total_users(self) -> int:
        return self.control.users + self.treatment.users

    def treatment_share(self) -> float:
        return self.treatment.users / self.total_users()


@dataclass(frozen=True)
class StratifiedResult:
    difference: float          # treatment rate minus control rate, absolute
    standard_error: float
    interval_low: float
    interval_high: float
    excluded_phases: list[str]


def naive_pooled_difference(phases: list[Phase]) -> float:
    """Original analysis: add up every phase per arm, then compare the rates.

    Wrong when the split changes over time, because each arm then contains a
    different mix of phases and the phases have different baselines.
    """
    control_users = sum(phase.control.users for phase in phases)
    control_conversions = sum(phase.control.conversions for phase in phases)
    treatment_users = sum(phase.treatment.users for phase in phases)
    treatment_conversions = sum(phase.treatment.conversions for phase in phases)
    control_rate = control_conversions / control_users
    treatment_rate = treatment_conversions / treatment_users
    return treatment_rate - control_rate


def allocation_changed(phases: list[Phase]) -> bool:
    """True when the treatment share differs between phases.

    Pooling is only safe with a constant split, so this is the check that
    should run before any pooled number is shown on a dashboard.
    """
    shares = [phase.treatment_share() for phase in phases if phase.total_users() > 0]
    if not shares:
        return False
    return max(shares) - min(shares) > ALLOCATION_TOLERANCE


def phase_difference(phase: Phase) -> tuple[float, float]:
    """Return (treatment - control rate, its variance) inside one phase."""
    control_rate = phase.control.rate()
    treatment_rate = phase.treatment.rate()
    # Standard variance of a difference of two independent proportions.
    variance = (treatment_rate * (1 - treatment_rate) / phase.treatment.users
                + control_rate * (1 - control_rate) / phase.control.users)
    return treatment_rate - control_rate, variance


def usable_phases(phases: list[Phase]) -> tuple[list[Phase], list[str]]:
    """Split phases into those with users in both arms and those without.

    A phase where one arm got no traffic (for example a paused arm) has no
    within-phase comparison, so it is excluded and reported, not guessed.
    """
    usable: list[Phase] = []
    excluded: list[str] = []
    for phase in phases:
        if phase.control.users > 0 and phase.treatment.users > 0:
            usable.append(phase)
        else:
            excluded.append(phase.name)
    return usable, excluded


def stratified_difference(phases: list[Phase]) -> StratifiedResult:
    """Average the within-phase differences, weighted by each phase's traffic.

    Weighting by total users per phase (not per arm) answers "what would the
    effect be if everyone in the test period got the treatment", and the
    weights are the same for both arms, so the phase mix cannot bias them.
    """
    usable, excluded = usable_phases(phases)
    if not usable:
        raise ValueError("no phase has users in both arms")
    total_users = sum(phase.total_users() for phase in usable)
    difference = 0.0
    variance = 0.0
    for phase in usable:
        weight = phase.total_users() / total_users
        phase_diff, phase_variance = phase_difference(phase)
        difference += weight * phase_diff
        variance += weight * weight * phase_variance
    standard_error = math.sqrt(variance)
    return StratifiedResult(
        difference=difference,
        standard_error=standard_error,
        interval_low=difference - Z_95 * standard_error,
        interval_high=difference + Z_95 * standard_error,
        excluded_phases=excluded,
    )
