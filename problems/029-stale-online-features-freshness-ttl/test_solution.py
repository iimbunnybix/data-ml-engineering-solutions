import unittest

from solution import (ACTIVE_COURIERS, ETA_FEATURES, PENDING_ORDERS, SECONDS_PER_MINUTE, FeatureStore,
                      FreshnessPolicy, StalenessMetrics, naive_read_features, predict_eta_minutes,
                      read_fresh_features)

ZONE = "zone-7"
EVENING_MINUTES = 300           # 16:00 to 21:00, one request per minute
STALL_MINUTE = 60               # streaming job stops at 17:00
PEAK_START, PEAK_END = 120, 240  # dinner peak 18:00 to 20:00
QUIET_PENDING, PEAK_PENDING, COURIERS = 20.0, 60.0, 20.0
MAX_AGE_SECONDS = 5 * SECONDS_PER_MINUTE
# Fallbacks are the evening's historical averages, not today's real values.
POLICIES = {
    PENDING_ORDERS: FreshnessPolicy(MAX_AGE_SECONDS, fallback=45.0),
    ACTIVE_COURIERS: FreshnessPolicy(MAX_AGE_SECONDS, fallback=20.0),
}


def true_features(minute: int) -> dict[str, float]:
    """What the zone really looks like at ``minute``."""
    pending = PEAK_PENDING if PEAK_START <= minute < PEAK_END else QUIET_PENDING
    return {PENDING_ORDERS: pending, ACTIVE_COURIERS: COURIERS}


def write_minute(store: FeatureStore, minute: int) -> None:
    """The streaming job's per-minute write."""
    for name, value in true_features(minute).items():
        store.write(ZONE, name, value, written_at=minute * SECONDS_PER_MINUTE)


def simulate_evening(use_fix: bool) -> tuple[list[float], StalenessMetrics]:
    """Score one request per minute; return the absolute ETA error per peak minute."""
    store, metrics = FeatureStore(), StalenessMetrics()
    peak_errors = []
    for minute in range(EVENING_MINUTES):
        if minute < STALL_MINUTE:
            write_minute(store, minute)
        now = minute * SECONDS_PER_MINUTE
        if use_fix:
            served = read_fresh_features(store, ZONE, POLICIES, now, metrics)
        else:
            served = naive_read_features(store, ZONE, ETA_FEATURES)
        if PEAK_START <= minute < PEAK_END:
            error = abs(predict_eta_minutes(served) - predict_eta_minutes(true_features(minute)))
            peak_errors.append(error)
    return peak_errors, metrics


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


class ReproduceProblem(unittest.TestCase):
    def test_stalled_pipeline_makes_peak_etas_20_minutes_too_short(self):
        errors, _ = simulate_evening(use_fix=False)
        self.assertEqual(mean(errors), 20.0)

    def test_naive_read_returns_a_four_hour_old_value_without_complaint(self):
        store = FeatureStore()
        store.write(ZONE, PENDING_ORDERS, 20.0, written_at=0.0)
        self.assertEqual(naive_read_features(store, ZONE, (PENDING_ORDERS,)), {PENDING_ORDERS: 20.0})

    def test_naive_read_turns_a_missing_feature_into_zero(self):
        self.assertEqual(naive_read_features(FeatureStore(), ZONE, (PENDING_ORDERS,)), {PENDING_ORDERS: 0.0})


class FixFreshnessPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FeatureStore()
        self.metrics = StalenessMetrics()
        self.store.write(ZONE, PENDING_ORDERS, 33.0, written_at=1000.0)
        self.store.write(ZONE, ACTIVE_COURIERS, 11.0, written_at=1000.0)

    def read_at(self, now: float) -> dict[str, float]:
        return read_fresh_features(self.store, ZONE, POLICIES, now, self.metrics)

    def test_fresh_values_are_served_as_stored(self):
        self.assertEqual(self.read_at(1000.0 + MAX_AGE_SECONDS), {PENDING_ORDERS: 33.0, ACTIVE_COURIERS: 11.0})

    def test_value_older_than_max_age_is_replaced_by_fallback(self):
        served = self.read_at(1000.0 + MAX_AGE_SECONDS + 1)
        self.assertEqual(served, {PENDING_ORDERS: 45.0, ACTIVE_COURIERS: 20.0})

    def test_missing_value_uses_fallback_and_counts_as_stale(self):
        served = read_fresh_features(FeatureStore(), ZONE, POLICIES, 0.0, self.metrics)
        self.assertEqual(served[PENDING_ORDERS], 45.0)
        self.assertEqual(self.metrics.stale_rate(PENDING_ORDERS), 1.0)

    def test_stale_rate_counts_only_stale_reads(self):
        self.read_at(1000.0)
        self.read_at(1000.0 + MAX_AGE_SECONDS + 1)
        self.assertEqual(self.metrics.stale_rate(PENDING_ORDERS), 0.5)

    def test_no_reads_means_no_alert(self):
        self.assertEqual(StalenessMetrics().alerting_features(), [])


class FixEveningReplay(unittest.TestCase):
    def test_fallbacks_cut_the_peak_eta_error_to_7_5_minutes(self):
        errors, _ = simulate_evening(use_fix=True)
        self.assertEqual(mean(errors), 7.5)

    def test_staleness_alert_fires_for_both_features(self):
        _, metrics = simulate_evening(use_fix=True)
        self.assertEqual(metrics.alerting_features(), [ACTIVE_COURIERS, PENDING_ORDERS])

    def test_stale_reads_start_once_values_exceed_max_age(self):
        _, metrics = simulate_evening(use_fix=True)
        stale_minutes = EVENING_MINUTES - STALL_MINUTE - 5  # minutes 60..64 are still within 5 min
        self.assertEqual(metrics.stale_reads[PENDING_ORDERS], stale_minutes)
        self.assertAlmostEqual(metrics.stale_rate(PENDING_ORDERS), stale_minutes / EVENING_MINUTES)


if __name__ == "__main__":
    unittest.main()
