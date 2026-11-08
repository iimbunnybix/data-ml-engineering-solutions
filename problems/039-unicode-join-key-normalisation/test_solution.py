import random
import unittest

from solution import (UNMATCHED_OWNER, JoinQualityError, check_join_quality,
                      naive_join, normalise_key, normalised_join, revenue_by_owner)

CUSTOMER_COUNT = 2_000
UNKNOWN_ACCOUNT_COUNT = 10
MANAGERS = ("Asha", "Ben", "Chen", "Dana")
FULL_WIDTH_OFFSET = 0xFEE0  # "A" (0x41) + offset = "Ａ" (0xFF21)


def to_full_width(text: str) -> str:
    """What an East-Asian IME or some PDF exports produce for ASCII text."""
    characters = []
    for character in text:
        if "!" <= character <= "~":
            characters.append(chr(ord(character) + FULL_WIDTH_OFFSET))
        else:
            characters.append(character)
    return "".join(characters)


def billing_variant(key: str, rng: random.Random) -> str:
    """How the new billing export spells a key: mostly clean, sometimes not."""
    roll = rng.random()
    if roll < 0.70:
        return key
    if roll < 0.78:
        return key + " "
    if roll < 0.82:
        return key + " "           # no-break space from a spreadsheet
    if roll < 0.90:
        return key.lower()
    if roll < 0.96:
        return to_full_width(key)
    return key + "​"               # zero-width space copied from a web page


def make_world(seed: int = 7) -> tuple[list[dict], list[dict]]:
    """2,000 CRM customers, their billing accounts, and 10 accounts with no customer."""
    rng = random.Random(seed)
    customers, accounts = [], []
    for number in range(CUSTOMER_COUNT):
        key = f"ACME-{number:04d}"
        # The CRM side is not perfectly clean either.
        crm_key = key + " " if rng.random() < 0.03 else key
        customers.append({"customer_key": crm_key, "account_manager": rng.choice(MANAGERS)})
        accounts.append({"account_key": billing_variant(key, rng),
                         "revenue_cents": rng.randint(10_000, 500_000)})
    for number in range(UNKNOWN_ACCOUNT_COUNT):
        accounts.append({"account_key": f"TRIAL-{number:02d}", "revenue_cents": 50_000})
    return customers, accounts


def total_revenue(accounts: list[dict]) -> int:
    """Revenue the bank actually received."""
    return sum(account["revenue_cents"] for account in accounts)


class ReproduceProblem(unittest.TestCase):
    def setUp(self):
        self.customers, self.accounts = make_world()

    def test_exact_join_loses_about_a_third_of_accounts(self):
        pairs = naive_join(self.customers, self.accounts)
        self.assertEqual((len(pairs), len(self.accounts)), (1367, 2010))

    def test_exact_join_attributes_about_two_thirds_of_revenue(self):
        attributed = sum(account["revenue_cents"]
                         for _, account in naive_join(self.customers, self.accounts))
        share = attributed / total_revenue(self.accounts)
        self.assertEqual(round(share, 3), 0.687)

    def test_strip_and_upper_alone_still_lose_over_200_accounts(self):
        crm_keys = {customer["customer_key"].strip().upper() for customer in self.customers}
        matched = [account for account in self.accounts
                   if account["account_key"].strip().upper() in crm_keys]
        self.assertEqual(len(matched), 1789)

    def test_strip_and_upper_alone_still_miss_full_width_and_zero_width_keys(self):
        self.assertNotEqual(to_full_width("ACME-0001").strip().upper(), "ACME-0001")
        self.assertNotEqual("ACME-0001​".strip().upper(), "ACME-0001")


class FixNormaliseKey(unittest.TestCase):
    def test_whitespace_and_case_variants_normalise_together(self):
        for raw in ("ACME-01 ", " acme-01", "ACME-01 ", "Acme-01\t"):
            self.assertEqual(normalise_key(raw), "acme-01")

    def test_full_width_characters_normalise_to_ascii(self):
        self.assertEqual(normalise_key("ＡＣＭＥ－０１"), "acme-01")

    def test_zero_width_characters_are_removed(self):
        self.assertEqual(normalise_key("ACME​-01﻿"), "acme-01")

    def test_decomposed_accents_match_composed_ones(self):
        self.assertEqual(normalise_key("Café-01"), normalise_key("Café-01"))

    def test_inner_whitespace_runs_collapse_to_one_space(self):
        self.assertEqual(normalise_key("ACME   01"), "acme 01")


class FixNormalisedJoin(unittest.TestCase):
    def setUp(self):
        self.customers, self.accounts = make_world()
        self.report = normalised_join(self.customers, self.accounts)

    def test_every_real_customer_is_matched(self):
        self.assertEqual(len(self.report.matched), CUSTOMER_COUNT)
        self.assertEqual(self.report.unmatched_customers, [])

    def test_accounts_without_a_customer_are_reported_not_dropped(self):
        unmatched_keys = sorted(a["account_key"] for a in self.report.unmatched_accounts)
        self.assertEqual(len(unmatched_keys), UNKNOWN_ACCOUNT_COUNT)
        self.assertTrue(all(key.startswith("TRIAL-") for key in unmatched_keys))

    def test_match_rate_is_reported(self):
        self.assertEqual(round(self.report.match_rate(), 4), 0.995)

    def test_revenue_by_owner_adds_up_to_billing_total(self):
        revenue = revenue_by_owner(self.report)
        self.assertEqual(sum(revenue.values()), total_revenue(self.accounts))
        self.assertEqual(revenue[UNMATCHED_OWNER], UNKNOWN_ACCOUNT_COUNT * 50_000)

    def test_healthy_join_passes_the_quality_check(self):
        check_join_quality(self.report)

    def test_low_match_rate_fails_the_quality_check(self):
        lossy = normalised_join(self.customers[:1_000], self.accounts)
        with self.assertRaises(JoinQualityError):
            check_join_quality(lossy)

    def test_empty_inputs_give_an_empty_report(self):
        report = normalised_join([], [])
        self.assertEqual(report.matched, [])
        self.assertEqual(report.match_rate(), 1.0)


class FixCollisionDetection(unittest.TestCase):
    def setUp(self):
        # Two different CRM customers whose keys only differ in case.
        self.customers = [
            {"customer_key": "ACME-01", "account_manager": "Asha"},
            {"customer_key": "acme-01", "account_manager": "Ben"},
            {"customer_key": "ACME-02", "account_manager": "Chen"},
        ]
        self.accounts = [
            {"account_key": "ACME-01", "revenue_cents": 100_000},
            {"account_key": "ACME-02", "revenue_cents": 200_000},
        ]
        self.report = normalised_join(self.customers, self.accounts)

    def test_collision_is_reported_with_both_raw_keys(self):
        self.assertEqual(len(self.report.collisions), 1)
        collision = self.report.collisions[0]
        self.assertEqual((collision.side, collision.normalised_key), ("crm", "acme-01"))
        self.assertEqual(collision.raw_keys, ("ACME-01", "acme-01"))

    def test_collided_key_is_not_matched_to_either_customer(self):
        matched_keys = [account["account_key"] for _, account in self.report.matched]
        self.assertEqual(matched_keys, ["ACME-02"])
        self.assertEqual(revenue_by_owner(self.report)[UNMATCHED_OWNER], 100_000)

    def test_collision_fails_the_quality_check_even_with_high_match_rate(self):
        with self.assertRaises(JoinQualityError):
            check_join_quality(self.report, min_match_rate=0.0)

    def test_billing_side_collisions_are_detected_too(self):
        accounts = self.accounts + [{"account_key": "ＡＣＭＥ－０２", "revenue_cents": 5}]
        report = normalised_join(self.customers[2:], accounts)
        self.assertEqual([c.side for c in report.collisions], ["billing"])
        self.assertEqual(len(report.unmatched_accounts), 3)


if __name__ == "__main__":
    unittest.main()
