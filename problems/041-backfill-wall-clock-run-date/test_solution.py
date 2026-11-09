import random
import unittest
from datetime import date, timedelta

from solution import (Event, IncompleteDateError, backfill, build_partition,
                      naive_build_partition, window_for)

BACKFILL_START = date(2026, 10, 1)
BACKFILL_END = date(2026, 10, 30)
BACKFILL_RUN_DAY = date(2026, 11, 9)   # the morning the engineer ran the backfill


class FakeClock:
    """A clock whose 'today' is set by the test."""

    def __init__(self, today: date):
        self.current = today

    def today(self) -> date:
        return self.current


def make_events(seed: int = 41) -> list[Event]:
    """Seeded activity for September to November with a growing user base."""
    rng = random.Random(seed)
    events = []
    day = date(2026, 9, 1)
    while day < BACKFILL_RUN_DAY:
        # Product launched a referral campaign: more distinct users every day.
        users_today = 200 + (day - date(2026, 9, 1)).days * 10
        for _ in range(users_today):
            user_number = rng.randrange(0, users_today * 3)
            events.append(Event(user_id=f"u{user_number}", day=day))
        day += timedelta(days=1)
    return events


def counts(rows: list[dict]) -> list[int]:
    return [row["active_users_7d"] for row in rows]


class ReproduceProblem(unittest.TestCase):
    def setUp(self):
        self.events = make_events()

    def test_scheduled_run_the_next_morning_looks_correct(self):
        partition_date = date(2026, 11, 5)
        clock = FakeClock(partition_date + timedelta(days=1))
        naive = naive_build_partition(self.events, partition_date, clock)
        fixed = build_partition(self.events, partition_date, clock)
        self.assertEqual(naive, fixed)

    def test_backfill_writes_the_same_number_to_every_day(self):
        rows = backfill(self.events, BACKFILL_START, BACKFILL_END,
                        FakeClock(BACKFILL_RUN_DAY), build=naive_build_partition)
        self.assertEqual(len(rows), 30)
        self.assertEqual(len(set(counts(rows))), 1)

    def test_backfilled_october_shows_the_week_before_the_backfill_ran(self):
        clock = FakeClock(BACKFILL_RUN_DAY)
        rows = backfill(self.events, BACKFILL_START, BACKFILL_END, clock,
                        build=naive_build_partition)
        week_before_run = build_partition(self.events, date(2026, 11, 8), clock)
        self.assertEqual(rows[0]["partition_date"], BACKFILL_START)
        self.assertEqual(rows[0]["active_users_7d"], week_before_run["active_users_7d"])

    def test_retry_two_days_late_shifts_the_window(self):
        partition_date = date(2026, 11, 5)
        on_time = naive_build_partition(self.events, partition_date,
                                        FakeClock(date(2026, 11, 6)))
        late = naive_build_partition(self.events, partition_date,
                                     FakeClock(date(2026, 11, 8)))
        self.assertNotEqual(on_time["active_users_7d"], late["active_users_7d"])


class FixLogicalDateWindow(unittest.TestCase):
    def setUp(self):
        self.events = make_events()

    def test_backfill_gives_a_different_value_per_day(self):
        rows = backfill(self.events, BACKFILL_START, BACKFILL_END,
                        FakeClock(BACKFILL_RUN_DAY))
        self.assertEqual(len(set(counts(rows))), 30)

    def test_backfill_matches_what_daily_scheduled_runs_produced(self):
        backfilled = backfill(self.events, BACKFILL_START, BACKFILL_END,
                              FakeClock(BACKFILL_RUN_DAY))
        for row in backfilled:
            next_morning = FakeClock(row["partition_date"] + timedelta(days=1))
            scheduled = naive_build_partition(self.events, row["partition_date"], next_morning)
            self.assertEqual(row, scheduled)

    def test_retry_on_any_later_day_gives_the_same_result(self):
        partition_date = date(2026, 11, 5)
        results = set()
        for run_day in (date(2026, 11, 6), date(2026, 11, 8), date(2027, 1, 1)):
            row = build_partition(self.events, partition_date, FakeClock(run_day))
            results.add(row["active_users_7d"])
        self.assertEqual(len(results), 1)

    def test_window_includes_partition_date_and_six_days_before(self):
        self.assertEqual(window_for(date(2026, 11, 5)),
                         (date(2026, 10, 30), date(2026, 11, 5)))

    def test_todays_partition_is_refused_because_the_day_is_not_over(self):
        with self.assertRaises(IncompleteDateError):
            build_partition(self.events, BACKFILL_RUN_DAY, FakeClock(BACKFILL_RUN_DAY))

    def test_future_partition_is_refused(self):
        with self.assertRaises(IncompleteDateError):
            build_partition(self.events, date(2026, 12, 1), FakeClock(BACKFILL_RUN_DAY))

    def test_day_with_no_events_reports_zero(self):
        row = build_partition([], date(2026, 10, 1), FakeClock(BACKFILL_RUN_DAY))
        self.assertEqual(row["active_users_7d"], 0)

    def test_user_active_on_several_days_is_counted_once(self):
        events = [Event("u1", date(2026, 10, 1)), Event("u1", date(2026, 10, 3)),
                  Event("u2", date(2026, 10, 3))]
        row = build_partition(events, date(2026, 10, 3), FakeClock(BACKFILL_RUN_DAY))
        self.assertEqual(row["active_users_7d"], 2)


if __name__ == "__main__":
    unittest.main()
