import random
import unittest
from datetime import date, datetime, timedelta

from solution import DailyRevenueAggregator, Event


def ev(i, day, hour, amount=100.0):
    return Event(f"o{i}", datetime(2026, 11, day, hour), amount)


class AggregatorTests(unittest.TestCase):
    def test_duplicates_are_counted_once(self):
        agg = DailyRevenueAggregator()
        for e in [ev(1, 1, 10), ev(1, 1, 10), ev(1, 1, 10), ev(2, 1, 11)]:
            agg.process(e)
        self.assertEqual(agg.open_windows[date(2026, 11, 1)], 200.0)
        self.assertEqual(agg.duplicates_dropped, 2)

    def test_window_finalises_after_watermark(self):
        agg = DailyRevenueAggregator(allowed_lateness=timedelta(hours=6))
        agg.process(ev(1, 1, 20))
        agg.process(ev(2, 2, 5))                    # watermark = Nov 1 23:00
        self.assertNotIn(date(2026, 11, 1), agg.finalised)
        agg.process(ev(3, 2, 6))                    # watermark = Nov 2 00:00
        self.assertEqual(agg.finalised[date(2026, 11, 1)], 100.0)

    def test_slightly_late_event_still_counted(self):
        agg = DailyRevenueAggregator(allowed_lateness=timedelta(hours=6))
        agg.process(ev(1, 1, 22))
        agg.process(ev(2, 2, 3))
        agg.process(ev(3, 1, 23))                   # late but within lateness
        agg.process(ev(4, 2, 7))
        self.assertEqual(agg.finalised[date(2026, 11, 1)], 200.0)

    def test_very_late_event_goes_to_ledger(self):
        agg = DailyRevenueAggregator(allowed_lateness=timedelta(hours=6))
        agg.process(ev(1, 1, 10))
        agg.process(ev(2, 2, 12))
        agg.process(ev(3, 1, 9, amount=55))         # day already finalised
        self.assertEqual(agg.finalised[date(2026, 11, 1)], 100.0)
        agg.process(ev(3, 1, 9, amount=55))         # retry of the late event
        self.assertEqual(agg.late_adjustments(), {date(2026, 11, 1): 55})

    def test_out_of_order_stream_matches_ground_truth(self):
        rng = random.Random(7)
        events = []
        for i in range(2000):
            t = datetime(2026, 11, 1) + timedelta(minutes=rng.randint(0, 5 * 24 * 60))
            events.append(Event(f"e{i}", t, rng.randint(1, 500)))
        truth = {}
        for e in events:
            truth[e.event_time.date()] = truth.get(e.event_time.date(), 0) + e.amount
        stream = sorted(events, key=lambda e: e.event_time + timedelta(minutes=rng.randint(0, 180)))
        stream += rng.sample(stream, 200)            # retries
        agg = DailyRevenueAggregator(allowed_lateness=timedelta(hours=4))
        for e in stream:
            agg.process(e)
        for day, total in agg.finalised.items():
            self.assertAlmostEqual(total + agg.late_adjustments().get(day, 0), truth[day])

    def test_dedup_state_is_bounded(self):
        agg = DailyRevenueAggregator(allowed_lateness=timedelta(hours=1),
                                     dedup_retention=timedelta(days=1))
        for i in range(24 * 30):
            agg.process(Event(f"x{i}", datetime(2026, 1, 1) + timedelta(hours=i), 1))
        self.assertLess(len(agg._seen), 60)


class FakeClock:
    """Processing-time clock the test moves by hand."""

    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class FutureDatedEventTests(unittest.TestCase):
    def make_aggregator_with_bad_phone_event(self):
        clock = FakeClock(datetime(2026, 11, 1, 12))
        agg = DailyRevenueAggregator(clock=clock)
        agg.process(ev(1, 1, 10, 100.0))
        agg.process(ev(2, 1, 11, 50.0))
        # phone clock one year ahead
        agg.process(Event("bad", datetime(2027, 11, 1, 10), 1.0))
        return agg

    def test_without_clock_future_event_closes_todays_window(self):
        agg = DailyRevenueAggregator()
        agg.process(ev(1, 1, 10))
        agg.process(Event("bad", datetime(2027, 11, 1, 10), 1.0))
        self.assertIn(date(2026, 11, 1), agg.finalised)

    def test_future_event_is_quarantined(self):
        agg = self.make_aggregator_with_bad_phone_event()
        self.assertEqual([e.event_id for e in agg.future_quarantine], ["bad"])

    def test_future_event_does_not_finalise_open_window(self):
        agg = self.make_aggregator_with_bad_phone_event()
        self.assertNotIn(date(2026, 11, 1), agg.finalised)
        self.assertEqual(agg.open_windows[date(2026, 11, 1)], 150.0)

    def test_retry_after_future_event_is_still_deduplicated(self):
        agg = self.make_aggregator_with_bad_phone_event()
        agg.process(ev(1, 1, 10, 100.0))
        agg.process(ev(3, 1, 12, 70.0))
        self.assertEqual(agg.duplicates_dropped, 1)
        self.assertEqual(agg.open_windows[date(2026, 11, 1)], 220.0)
        self.assertEqual(agg.late_adjustments(), {})

    def test_event_within_skew_tolerance_is_accepted(self):
        clock = FakeClock(datetime(2026, 11, 1, 12))
        agg = DailyRevenueAggregator(clock=clock)
        agg.process(Event("o1", datetime(2026, 11, 1, 12, 4), 10.0))
        self.assertEqual(agg.future_quarantine, [])
        self.assertEqual(agg.open_windows[date(2026, 11, 1)], 10.0)


if __name__ == "__main__":
    unittest.main()
