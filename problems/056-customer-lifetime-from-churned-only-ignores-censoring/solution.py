"""Estimate customer lifetime without ignoring customers who have not churned yet.

Marketing set its acquisition budget from "average customer lifetime", which
an analyst computed as the mean tenure of customers who had already
cancelled. Customers who stay longest are exactly the ones who have not
cancelled yet, so they were left out and the lifetime came out far too
short. (Averaging the tenure of only the current customers is the mirror
mistake.) This module uses the Kaplan-Meier estimator, which counts every
customer for the months they were actually observed, cancelled or not.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# Survival probability at which the median lifetime is read off the curve.
MEDIAN_SURVIVAL = 0.5


@dataclass(frozen=True)
class Customer:
    customer_id: int
    tenure_months: int   # months observed so far (until churn or until today)
    churned: bool        # False means still active: lifetime is at least tenure


def simulate_customers(cohorts: int, cohort_size: int, monthly_churn: float,
                       rng: random.Random) -> list[Customer]:
    """One cohort signs up each month; today is the end of the last month.

    The oldest cohort has been observed for `cohorts` months, the newest for
    one month. Each month an active customer cancels with `monthly_churn`.
    """
    customers: list[Customer] = []
    for cohort in range(cohorts):
        months_observable = cohorts - cohort
        for _ in range(cohort_size):
            tenure, churned = simulate_one(months_observable, monthly_churn, rng)
            customers.append(Customer(len(customers), tenure, churned))
    return customers


def simulate_one(months_observable: int, monthly_churn: float,
                 rng: random.Random) -> tuple[int, bool]:
    """Return (tenure, churned) for one customer observed for at most N months."""
    for month in range(1, months_observable + 1):
        if rng.random() < monthly_churn:
            return month, True
    return months_observable, False


def true_median_lifetime(monthly_churn: float) -> int:
    """Median of the generating process: first month where survival <= 50%."""
    survival = 1.0
    month = 0
    while survival > MEDIAN_SURVIVAL:
        month += 1
        survival *= 1 - monthly_churn
    return month


def naive_lifetime_from_churned(customers: list[Customer]) -> float:
    """Original report: mean tenure of cancelled customers only.

    Active customers are dropped, and they are the long-lived ones, so the
    estimate is biased short.
    """
    tenures = [customer.tenure_months for customer in customers if customer.churned]
    return sum(tenures) / len(tenures)


def kaplan_meier(customers: list[Customer]) -> dict[int, float]:
    """Return {month: probability a customer is still active after that month}.

    In month t, the customers "at risk" are everyone observed for at least t
    months; only those who cancelled in month t count as events. A still-
    active customer contributes to the denominator for every month we saw
    them, which is the information the naive report threw away.
    """
    if not customers:
        raise ValueError("no customers")
    last_month = max(customer.tenure_months for customer in customers)
    curve: dict[int, float] = {}
    survival = 1.0
    for month in range(1, last_month + 1):
        at_risk, churned = counts_for_month(customers, month)
        if at_risk > 0:
            survival *= 1 - churned / at_risk
        curve[month] = survival
    return curve


def counts_for_month(customers: list[Customer], month: int) -> tuple[int, int]:
    """Return (customers observed at least `month` months, of whom cancelled then)."""
    at_risk = 0
    churned = 0
    for customer in customers:
        if customer.tenure_months >= month:
            at_risk += 1
            if customer.churned and customer.tenure_months == month:
                churned += 1
    return at_risk, churned


def median_lifetime(curve: dict[int, float]) -> int | None:
    """First month where survival drops to 50% or below.

    Returns None when the curve never gets there: more than half of the
    customers outlived the observation window, so the data cannot say yet.
    """
    for month in sorted(curve):
        if curve[month] <= MEDIAN_SURVIVAL:
            return month
    return None
