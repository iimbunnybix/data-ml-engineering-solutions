import random
import unittest

from solution import (UNKNOWN_KEY, UNKNOWN_REGION, CustomerDimension, CustomerRecord,
                      Order, load_order, naive_load_order, revenue_by_region)

CUSTOMERS = 300
ORDERS_PER_CUSTOMER = 5
MINUTES_PER_DAY = 24 * 60
CRM_SYNC_EVERY_MINUTES = 60
REGIONS = ["North", "South", "East", "West"]


def make_day(seed: int = 7) -> list[tuple[int, int, object]]:
    """One day of events as (minute, priority, event), CRM rows before orders on ties.

    Each customer signs up at a random minute; the hourly CRM sync loads them
    at the next full hour; their orders arrive over the following 5 hours.
    """
    rng = random.Random(seed)
    events: list[tuple[int, int, object]] = []
    order_id = 0
    for number in range(CUSTOMERS):
        customer_id = f"cust-{number}"
        signup = rng.randrange(MINUTES_PER_DAY - 300)
        next_sync = (signup // CRM_SYNC_EVERY_MINUTES + 1) * CRM_SYNC_EVERY_MINUTES
        events.append((next_sync, 0, CustomerRecord(customer_id, rng.choice(REGIONS))))
        for _ in range(ORDERS_PER_CUSTOMER):
            order_id += 1
            minute = signup + rng.randrange(300)
            events.append((minute, 1, Order(order_id, customer_id, rng.randint(500, 20_000))))
    events.sort(key=lambda event: (event[0], event[1]))
    return events


def replay(loader, events) -> tuple[list, CustomerDimension]:
    """Feed the day's events through a loader and return facts and dimension."""
    dimension = CustomerDimension()
    facts = []
    for _, _, event in events:
        if isinstance(event, CustomerRecord):
            dimension.upsert(event)
        else:
            facts.append(loader(event, dimension))
    return facts, dimension


def total_cents(events) -> int:
    return sum(event.amount_cents for _, _, event in events if isinstance(event, Order))


class ReproduceProblem(unittest.TestCase):
    def test_orders_before_crm_sync_stay_unknown_after_the_customer_arrives(self):
        events = make_day()
        facts, dimension = replay(naive_load_order, events)
        unknown = revenue_by_region(facts, dimension).get(UNKNOWN_REGION, 0)
        share = unknown / total_cents(events)
        self.assertGreater(share, 0.08)          # 10.2% of the day's revenue
        self.assertEqual(dimension.inferred_customers(), [])   # every customer did arrive

    def test_single_early_order_is_pinned_to_unknown_key(self):
        dimension = CustomerDimension()
        fact = naive_load_order(Order(1, "cust-a", 1_000), dimension)
        dimension.upsert(CustomerRecord("cust-a", "North"))
        self.assertEqual(fact.customer_key, UNKNOWN_KEY)
        self.assertEqual(revenue_by_region([fact], dimension), {UNKNOWN_REGION: 1_000})


class FixInferredMembers(unittest.TestCase):
    def test_no_revenue_is_unknown_once_all_customers_are_synced(self):
        events = make_day()
        facts, dimension = replay(load_order, events)
        totals = revenue_by_region(facts, dimension)
        self.assertNotIn(UNKNOWN_REGION, totals)
        self.assertEqual(sum(totals.values()), total_cents(events))

    def test_early_order_picks_up_region_when_customer_row_arrives(self):
        dimension = CustomerDimension()
        fact = load_order(Order(1, "cust-a", 1_000), dimension)
        self.assertEqual(revenue_by_region([fact], dimension), {UNKNOWN_REGION: 1_000})
        dimension.upsert(CustomerRecord("cust-a", "North"))
        self.assertEqual(revenue_by_region([fact], dimension), {"North": 1_000})

    def test_placeholder_keeps_its_key_when_filled(self):
        dimension = CustomerDimension()
        placeholder_key = dimension.key_for_fact("cust-a")
        self.assertEqual(dimension.upsert(CustomerRecord("cust-a", "East")), placeholder_key)

    def test_customer_who_never_arrives_is_listed_for_follow_up(self):
        dimension = CustomerDimension()
        load_order(Order(1, "ghost", 500), dimension)
        load_order(Order(2, "cust-b", 700), dimension)
        dimension.upsert(CustomerRecord("cust-b", "West"))
        self.assertEqual(dimension.inferred_customers(), ["ghost"])

    def test_repeated_early_orders_share_one_placeholder(self):
        dimension = CustomerDimension()
        first = load_order(Order(1, "cust-a", 100), dimension)
        second = load_order(Order(2, "cust-a", 200), dimension)
        self.assertEqual(first.customer_key, second.customer_key)
        self.assertEqual(len(dimension.rows_by_key), 1)

    def test_known_customer_is_not_marked_inferred(self):
        dimension = CustomerDimension()
        dimension.upsert(CustomerRecord("cust-a", "South"))
        fact = load_order(Order(1, "cust-a", 300), dimension)
        self.assertEqual(revenue_by_region([fact], dimension), {"South": 300})
        self.assertEqual(dimension.inferred_customers(), [])

    def test_later_crm_update_changes_region_of_existing_facts(self):
        dimension = CustomerDimension()
        dimension.upsert(CustomerRecord("cust-a", "South"))
        fact = load_order(Order(1, "cust-a", 300), dimension)
        dimension.upsert(CustomerRecord("cust-a", "North"))
        self.assertEqual(revenue_by_region([fact], dimension), {"North": 300})


if __name__ == "__main__":
    unittest.main()
