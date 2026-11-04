import random
import unittest
from datetime import date, datetime, timedelta, timezone

from solution import (daily_counts, hour_label, hourly_counts, local_day, local_day_bounds_utc,
                      local_hours_in_day, naive_hourly, naive_local_day,
                      utc_partitions_for_local_day)
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


def uniform_events(rng, start_local, days, n):
    """n events uniformly spread over `days` local days, returned as UTC timestamps."""
    start = start_local.astimezone(UTC)
    span = days * 86_400
    return [start + timedelta(seconds=rng.randrange(span)) for _ in range(n)]


class ReproduceFixedOffsetBug(unittest.TestCase):
    def test_summer_orders_after_midnight_land_on_previous_day(self):
        ts = datetime(2026, 7, 4, 4, 30, tzinfo=UTC)          # 00:30 EDT on July 4
        self.assertEqual(naive_local_day(ts), date(2026, 7, 3))
        self.assertEqual(local_day(ts, NY), date(2026, 7, 4))

    def test_share_of_summer_orders_misattributed(self):
        rng = random.Random(11)
        events = uniform_events(rng, datetime(2026, 7, 1, tzinfo=NY), 28, 100_000)
        wrong = sum(1 for e in events if naive_local_day(e) != local_day(e, NY))
        self.assertAlmostEqual(wrong / len(events), 1 / 24, delta=0.003)  # ~4.2%

    def test_winter_is_unaffected(self):
        rng = random.Random(11)
        events = uniform_events(rng, datetime(2026, 1, 5, tzinfo=NY), 28, 10_000)
        self.assertTrue(all(naive_local_day(e) == local_day(e, NY) for e in events))

    def test_fall_back_hour_is_merged_on_naive_chart(self):
        # One event per minute across local 2026-11-01 (a 25-hour day).
        start, end = local_day_bounds_utc(date(2026, 11, 1), NY)
        events = [start + timedelta(minutes=m) for m in range(int((end - start).total_seconds() // 60))]
        naive = naive_hourly(events, NY)
        self.assertEqual(naive["2026-11-01 01"], 120)          # two hours shown as one
        self.assertEqual(len([k for k in naive if k.startswith("2026-11-01")]), 24)


class ZoneAwareRollups(unittest.TestCase):
    def test_day_lengths(self):
        lengths = {d: (lambda s, e: (e - s).total_seconds() / 3600)(*local_day_bounds_utc(d, NY))
                   for d in (date(2026, 3, 8), date(2026, 7, 4), date(2026, 11, 1))}
        self.assertEqual(lengths, {date(2026, 3, 8): 23, date(2026, 7, 4): 24,
                                   date(2026, 11, 1): 25})

    def test_fall_back_hours_stay_separate(self):
        start, end = local_day_bounds_utc(date(2026, 11, 1), NY)
        events = [start + timedelta(minutes=m) for m in range(int((end - start).total_seconds() // 60))]
        hourly = hourly_counts(events, NY)
        self.assertEqual(len(hourly), 25)
        self.assertTrue(all(v == 60 for v in hourly.values()))
        labels = [hour_label(b, NY) for b in hourly]
        self.assertIn("2026-11-01 01:00 EDT", labels)
        self.assertIn("2026-11-01 01:00 EST", labels)

    def test_spring_forward_has_no_2am(self):
        labels = [hour_label(b, NY) for b in local_hours_in_day(date(2026, 3, 8), NY)]
        self.assertEqual(len(labels), 23)
        self.assertFalse(any(" 02:00 " in label for label in labels))

    def test_daily_totals_preserved(self):
        rng = random.Random(3)
        events = uniform_events(rng, datetime(2026, 10, 25, tzinfo=NY), 14, 20_000)
        counts = daily_counts(events, NY)
        self.assertEqual(sum(counts.values()), 20_000)
        self.assertEqual(min(counts), date(2026, 10, 25))
        self.assertEqual(max(counts), date(2026, 11, 7))

    def test_utc_partitions_to_scan(self):
        self.assertEqual(utc_partitions_for_local_day(date(2026, 7, 4), NY),
                         [date(2026, 7, 4), date(2026, 7, 5)])

    def test_bounds_match_event_assignment(self):
        rng = random.Random(5)
        for e in uniform_events(rng, datetime(2026, 3, 1, tzinfo=NY), 250, 5_000):
            s, t = local_day_bounds_utc(local_day(e, NY), NY)
            self.assertTrue(s <= e < t)

    def test_naive_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            local_day(datetime(2026, 7, 4, 12), NY)


if __name__ == "__main__":
    unittest.main()
