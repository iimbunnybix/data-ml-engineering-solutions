import unittest
from datetime import datetime, timezone

from solution import (
    daily_counts,
    naive_parse_millis,
    parse_event_time,
    parse_events,
    ParsedEvent,
    QuarantinedEvent,
    unit_mix,
)

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
EVENT_TIME = datetime(2026, 11, 4, 9, 30, tzinfo=timezone.utc)
EVENT_SECONDS = int(EVENT_TIME.timestamp())


def merged_feed() -> list[tuple[str, float]]:
    """The same moment sent by three sources in three different units."""
    return [
        ("mobile-1", EVENT_SECONDS),                     # seconds
        ("web-1", EVENT_SECONDS * 1_000),                # milliseconds
        ("partner-1", EVENT_SECONDS * 1_000_000),        # microseconds
    ]


class ReproduceProblem(unittest.TestCase):
    def test_seconds_read_as_milliseconds_land_in_1970(self):
        self.assertEqual(naive_parse_millis(EVENT_SECONDS).year, 1970)

    def test_microseconds_read_as_milliseconds_crash_the_job(self):
        with self.assertRaises((ValueError, OverflowError, OSError)):
            naive_parse_millis(EVENT_SECONDS * 1_000_000)


class FixDetectsUnit(unittest.TestCase):
    def test_all_three_units_parse_to_the_same_moment(self):
        result = parse_events(merged_feed(), NOW)
        self.assertEqual([event.timestamp for event in result.events], [EVENT_TIME] * 3)
        self.assertEqual(result.quarantined, [])

    def test_daily_partition_has_every_event_on_the_right_day(self):
        result = parse_events(merged_feed(), NOW)
        timestamps = [event.timestamp for event in result.events]
        self.assertEqual(daily_counts(timestamps), {"2026-11-04": 3})

    def test_unit_mix_reports_each_source_format(self):
        result = parse_events(merged_feed(), NOW)
        self.assertEqual(unit_mix(result.events),
                         {"seconds": 1, "milliseconds": 1, "microseconds": 1})

    def test_nanoseconds_are_detected(self):
        parsed = parse_event_time("n", EVENT_SECONDS * 1_000_000_000, NOW)
        self.assertIsInstance(parsed, ParsedEvent)
        self.assertEqual(parsed.unit, "nanoseconds")

    def test_fractional_milliseconds_keep_sub_second_precision(self):
        parsed = parse_event_time("w", EVENT_SECONDS * 1_000 + 250, NOW)
        self.assertEqual(parsed.timestamp.microsecond, 250_000)


class FixQuarantinesImplausibleRows(unittest.TestCase):
    def test_zero_timestamp_is_quarantined_not_dated_1970(self):
        parsed = parse_event_time("bad", 0, NOW)
        self.assertIsInstance(parsed, QuarantinedEvent)

    def test_far_future_value_is_quarantined_without_crashing(self):
        parsed = parse_event_time("bad", 10 ** 30, NOW)
        self.assertIsInstance(parsed, QuarantinedEvent)

    def test_event_two_days_in_the_future_is_quarantined(self):
        parsed = parse_event_time("skew", NOW.timestamp() + 2 * 86_400, NOW)
        self.assertIsInstance(parsed, QuarantinedEvent)

    def test_event_slightly_in_the_future_is_accepted(self):
        parsed = parse_event_time("skew", NOW.timestamp() + 3_600, NOW)
        self.assertIsInstance(parsed, ParsedEvent)

    def test_window_wide_enough_for_two_units_is_reported_as_ambiguous(self):
        very_early = datetime(1970, 1, 2, tzinfo=timezone.utc)
        parsed = parse_event_time("x", EVENT_SECONDS * 1_000, NOW, earliest=very_early)
        self.assertIsInstance(parsed, QuarantinedEvent)
        self.assertIn("ambiguous", parsed.reason)

    def test_quarantined_rows_are_kept_with_their_raw_value(self):
        result = parse_events(merged_feed() + [("bad", 42)], NOW)
        self.assertEqual(len(result.events), 3)
        self.assertEqual(result.quarantined[0].raw_value, 42)


if __name__ == "__main__":
    unittest.main()
