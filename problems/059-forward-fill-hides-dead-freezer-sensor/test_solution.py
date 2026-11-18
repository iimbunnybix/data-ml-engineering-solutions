import unittest

from solution import (Alert, GridPoint, MAX_STALENESS_MINUTES, Reading,
                      alerts_for, naive_alerts, naive_resample, resample)

SHIFT_START = 0          # 00:00
SHIFT_END = 720          # 12:00
BATTERY_DIES_AT = 180    # 03:00, the last reading the sensor ever sends
HEALTHY_CELSIUS = -19.0


def readings_until(last_minute: int, celsius: float = HEALTHY_CELSIUS,
                   first_minute: int = SHIFT_START) -> list[Reading]:
    """A sensor reporting every 5 minutes from first_minute to last_minute."""
    readings = []
    for minute in range(first_minute, last_minute + 1, 5):
        readings.append(Reading(minute, celsius))
    return readings


def dead_sensor_readings() -> list[Reading]:
    """Healthy readings until the battery dies, then silence for 9 hours."""
    return readings_until(BATTERY_DIES_AT)


class ReproduceProblem(unittest.TestCase):
    def test_forward_fill_repeats_the_last_reading_for_nine_hours(self):
        values = naive_resample(dead_sensor_readings(), SHIFT_START, SHIFT_END)
        after_death = values[BATTERY_DIES_AT // 5 + 1:]
        self.assertEqual(len(after_death), 108)
        self.assertEqual(set(after_death), {HEALTHY_CELSIUS})

    def test_no_alarm_fires_for_the_whole_shift(self):
        values = naive_resample(dead_sensor_readings(), SHIFT_START, SHIFT_END)
        self.assertEqual(naive_alerts(values, SHIFT_START), [])


class FixBoundsForwardFill(unittest.TestCase):
    def test_short_gap_is_still_filled(self):
        readings = [Reading(0, -19.0), Reading(15, -18.5)]   # two readings lost
        points = resample(readings, 0, 15)
        self.assertEqual([point.celsius for point in points], [-19.0, -19.0, -19.0, -18.5])

    def test_value_exactly_at_the_staleness_limit_is_kept(self):
        points = resample([Reading(0, -19.0)], 0, MAX_STALENESS_MINUTES)
        self.assertEqual(points[-1], GridPoint(MAX_STALENESS_MINUTES, -19.0, 15))

    def test_value_older_than_the_limit_becomes_missing_with_its_age(self):
        points = resample([Reading(0, -19.0)], 0, 20)
        self.assertEqual(points[-1], GridPoint(20, None, 20))

    def test_out_of_order_readings_are_sorted_first(self):
        readings = [Reading(10, -18.0), Reading(0, -19.0), Reading(5, -18.5)]
        points = resample(readings, 0, 10)
        self.assertEqual([point.celsius for point in points], [-19.0, -18.5, -18.0])

    def test_no_reading_yet_is_missing_with_unknown_age(self):
        points = resample([Reading(10, -19.0)], 0, 10)
        self.assertEqual(points[0], GridPoint(0, None, None))

    def test_empty_input_gives_all_missing_points(self):
        points = resample([], 0, 10)
        self.assertEqual([point.celsius for point in points], [None, None, None])


class FixAlertsOnStaleSensors(unittest.TestCase):
    def test_dead_sensor_raises_one_stale_alert_twenty_minutes_after_last_reading(self):
        points = resample(dead_sensor_readings(), SHIFT_START, SHIFT_END)
        self.assertEqual(alerts_for(points), [Alert(200, "sensor_stale")])

    def test_warm_freezer_still_raises_too_warm(self):
        readings = readings_until(10) + [Reading(15, -10.0)]
        points = resample(readings, 0, 15)
        self.assertEqual(alerts_for(points), [Alert(15, "too_warm")])

    def test_sensor_that_recovers_and_dies_again_alerts_twice(self):
        readings = readings_until(0) + readings_until(60, first_minute=40)
        points = resample(readings, 0, 90)
        self.assertEqual(alerts_for(points),
                         [Alert(20, "sensor_stale"), Alert(80, "sensor_stale")])

    def test_healthy_sensor_raises_no_alerts(self):
        points = resample(readings_until(SHIFT_END), SHIFT_START, SHIFT_END)
        self.assertEqual(alerts_for(points), [])


if __name__ == "__main__":
    unittest.main()
