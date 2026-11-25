import random
import statistics
import unittest

from solution import naive_outlier_indexes, outlier_indexes, robust_scores, without_indexes

SEED = 72
BATCH_SIZE = 500
STUCK_READINGS = 60          # 12% of the batch: the sensor was stuck for a while
STUCK_VALUE_C = 85.0         # the faulty value the sensor reported
FRIDGE_MEAN_C = 4.0
FRIDGE_SPREAD_C = 0.5
SPOILAGE_LIMIT_C = 8.0       # a batch whose average is above this is flagged as spoiled


def sensor_batch(stuck_count: int) -> list[float]:
    """One batch of fridge readings with ``stuck_count`` faulty values mixed in."""
    rng = random.Random(SEED)
    good = [rng.gauss(FRIDGE_MEAN_C, FRIDGE_SPREAD_C) for _ in range(BATCH_SIZE - stuck_count)]
    readings = good + [STUCK_VALUE_C] * stuck_count
    rng.shuffle(readings)
    return readings


def stuck_indexes(readings: list[float]) -> list[int]:
    return [index for index, value in enumerate(readings) if value == STUCK_VALUE_C]


def small_batch_with_one_spike() -> list[float]:
    """Ten readings, one of them a single huge spike."""
    return [4.1, 3.9, 4.0, 4.2, 3.8, 4.0, 4.1, 3.9, 4.0, 85.0]


class ReproduceProblem(unittest.TestCase):
    def test_stuck_sensor_readings_are_not_flagged(self):
        self.assertEqual(naive_outlier_indexes(sensor_batch(STUCK_READINGS)), [])

    def test_filtered_batch_still_looks_spoiled(self):
        readings = sensor_batch(STUCK_READINGS)
        kept = without_indexes(readings, naive_outlier_indexes(readings))
        self.assertGreater(statistics.fmean(kept), SPOILAGE_LIMIT_C)

    def test_filter_works_when_only_a_few_readings_are_bad(self):
        readings = sensor_batch(40)
        self.assertEqual(naive_outlier_indexes(readings), stuck_indexes(readings))

    def test_one_spike_in_ten_readings_can_never_reach_three_deviations(self):
        # With n values the largest possible z-score is (n - 1) / sqrt(n): 2.85 for n = 10.
        self.assertEqual(naive_outlier_indexes(small_batch_with_one_spike()), [])


class FixRobustZScore(unittest.TestCase):
    def test_every_stuck_reading_is_flagged(self):
        readings = sensor_batch(STUCK_READINGS)
        self.assertEqual(outlier_indexes(readings), stuck_indexes(readings))

    def test_filtered_batch_has_the_real_fridge_temperature(self):
        readings = sensor_batch(STUCK_READINGS)
        kept = without_indexes(readings, outlier_indexes(readings))
        self.assertAlmostEqual(statistics.fmean(kept), FRIDGE_MEAN_C, delta=0.1)

    def test_single_spike_in_a_small_batch_is_flagged(self):
        self.assertEqual(outlier_indexes(small_batch_with_one_spike()), [9])

    def test_clean_batch_loses_almost_no_readings(self):
        # A 3.5 limit still flags the odd genuine tail reading in normal data.
        flagged = outlier_indexes(sensor_batch(0))
        self.assertLessEqual(len(flagged), BATCH_SIZE // 100)

    def test_spike_is_found_when_most_readings_are_identical(self):
        # MAD is 0 here, so the mean-absolute-deviation fallback is used.
        readings = [4.0] * 8 + [4.5, 40.0]
        self.assertEqual(outlier_indexes(readings), [9])

    def test_constant_readings_have_no_outliers(self):
        self.assertEqual(robust_scores([4.0, 4.0, 4.0]), [0.0, 0.0, 0.0])
        self.assertEqual(outlier_indexes([4.0, 4.0, 4.0]), [])

    def test_empty_batch_has_no_outliers(self):
        self.assertEqual(outlier_indexes([]), [])


if __name__ == "__main__":
    unittest.main()
