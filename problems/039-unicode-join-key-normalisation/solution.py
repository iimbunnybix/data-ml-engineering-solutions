"""Join CRM customers to billing accounts on keys that differ only in form.

The revenue-attribution job joined CRM customers to billing accounts on the raw
key string. The billing export started sending the same keys with trailing or
no-break spaces, lower case, full-width characters (``ＡＣＭＥ－０１``) and
invisible zero-width characters, so exact matching silently lost a large share
of accounts and their revenue. This module normalises keys the same way on
both sides (Unicode NFKC, invisible characters removed, whitespace collapsed,
case folded), reports the match rate and what did not match, and refuses to
merge two different raw keys that normalise to the same value (a collision).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

# NFKC maps compatibility characters (full-width letters and digits, no-break
# space, ligatures) to their plain form and composes accents (NFD -> NFC).
UNICODE_FORM = "NFKC"
# Format characters that are invisible in every UI but break string equality.
INVISIBLE_CHARACTERS = ("​", "‌", "‍", "⁠", "﻿")
MIN_MATCH_RATE = 0.98
UNMATCHED_OWNER = "UNMATCHED"


class JoinQualityError(Exception):
    """Raised when the join is too lossy or ambiguous to publish."""


@dataclass(frozen=True)
class KeyCollision:
    """Several rows on one side that normalise to the same key."""

    side: str
    normalised_key: str
    raw_keys: tuple[str, ...]


@dataclass
class JoinReport:
    """Matched pairs plus everything needed to see what the join lost."""

    matched: list[tuple[dict, dict]] = field(default_factory=list)
    unmatched_accounts: list[dict] = field(default_factory=list)
    unmatched_customers: list[dict] = field(default_factory=list)
    collisions: list[KeyCollision] = field(default_factory=list)

    def match_rate(self) -> float:
        """Share of billing accounts that found exactly one CRM customer."""
        total = len(self.matched) + len(self.unmatched_accounts)
        if total == 0:
            return 1.0
        return len(self.matched) / total


# --- Original job: exact match on the raw key --------------------------------

def naive_join(customers: list[dict], accounts: list[dict]) -> list[tuple[dict, dict]]:
    """Inner join on the raw strings; every non-identical key is silently lost."""
    customer_by_key = {}
    for customer in customers:
        customer_by_key[customer["customer_key"]] = customer
    pairs = []
    for account in accounts:
        customer = customer_by_key.get(account["account_key"])
        if customer is not None:
            pairs.append((customer, account))
    return pairs


# --- Fixed job ----------------------------------------------------------------

def normalise_key(raw_key: str) -> str:
    """Canonical form of a key, applied identically to both sides of the join."""
    text = unicodedata.normalize(UNICODE_FORM, raw_key)
    for character in INVISIBLE_CHARACTERS:
        text = text.replace(character, "")
    # split() with no argument splits on any Unicode whitespace, so this
    # trims the ends and collapses inner runs to one ordinary space.
    text = " ".join(text.split())
    # casefold() is the Unicode-correct case-insensitive form (ß -> ss).
    return text.casefold()


def group_by_normalised_key(rows: list[dict], key_field: str) -> dict[str, list[dict]]:
    """Group rows by their normalised key (normally one row per key)."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(normalise_key(row[key_field]), []).append(row)
    return groups


def find_collisions(groups: dict[str, list[dict]], key_field: str,
                    side: str) -> list[KeyCollision]:
    """Report every normalised key that more than one row on this side shares."""
    collisions = []
    for key, group in groups.items():
        if len(group) > 1:
            raw_keys = tuple(row[key_field] for row in group)
            collisions.append(KeyCollision(side, key, raw_keys))
    return collisions


def normalised_join(customers: list[dict], accounts: list[dict]) -> JoinReport:
    """Join on normalised keys and record matches, misses and collisions."""
    customer_groups = group_by_normalised_key(customers, "customer_key")
    account_groups = group_by_normalised_key(accounts, "account_key")
    report = JoinReport(collisions=find_collisions(customer_groups, "customer_key", "crm")
                        + find_collisions(account_groups, "account_key", "billing"))
    collided_keys = {collision.normalised_key for collision in report.collisions}

    for key, account_rows in account_groups.items():
        # A collided key is never matched: picking one of the rows would
        # attribute revenue to a customer by chance.
        if key in collided_keys or key not in customer_groups:
            report.unmatched_accounts.extend(account_rows)
        else:
            report.matched.append((customer_groups[key][0], account_rows[0]))

    for key, customer_rows in customer_groups.items():
        if key in collided_keys or key not in account_groups:
            report.unmatched_customers.extend(customer_rows)
    return report


def check_join_quality(report: JoinReport, min_match_rate: float = MIN_MATCH_RATE) -> None:
    """Fail the job on any collision or when too few accounts matched."""
    if report.collisions:
        first = report.collisions[0]
        raise JoinQualityError(
            f"{len(report.collisions)} key collision(s), e.g. {first.side} "
            f"keys {first.raw_keys!r} all normalise to {first.normalised_key!r}")
    rate = report.match_rate()
    if rate < min_match_rate:
        raise JoinQualityError(
            f"match rate {rate:.1%} is below {min_match_rate:.1%}; "
            f"{len(report.unmatched_accounts)} billing accounts unmatched")


def revenue_by_owner(report: JoinReport) -> dict[str, int]:
    """Revenue in cents per account manager, with misses in an explicit bucket."""
    revenue: dict[str, int] = {}
    for customer, account in report.matched:
        owner = customer["account_manager"]
        revenue[owner] = revenue.get(owner, 0) + account["revenue_cents"]
    for account in report.unmatched_accounts:
        revenue[UNMATCHED_OWNER] = revenue.get(UNMATCHED_OWNER, 0) + account["revenue_cents"]
    return revenue
