import math
import random
import unittest

from solution import band_counts, band_index, band_labels, naive_band_counts, validate_edges

TRIPS = 10_000
MAX_DISTANCE_KM = 20.0
GPS_RESOLUTION_KM = 0.05
SEED = 3
OVERLAPPING_BANDS = [(0, 5), (5, 10), (10, 20)]
GAPPED_BANDS = [(0, 4.9), (5, 9.9), (10, 20)]
EDGES = [0, 5, 10, 20]


def trip_distances(seed: int = SEED) -> list[float]:
    """Distances rounded to 50 m, as the courier app reports them."""
    rng = random.Random(seed)
    distances = []
    for _ in range(TRIPS):
        raw = rng.uniform(0, MAX_DISTANCE_KM)
        distances.append(round(round(raw / GPS_RESOLUTION_KM) * GPS_RESOLUTION_KM, 2))
    return distances


class ReproduceProblem(unittest.TestCase):
    def test_boundary_value_is_counted_in_two_bands(self):
        counts = naive_band_counts([5.0], OVERLAPPING_BANDS)
        self.assertEqual(counts, {"0-5": 1, "5-10": 1, "10-20": 0})

    def test_overlapping_bands_total_more_than_the_trips(self):
        counts = naive_band_counts(trip_distances(), OVERLAPPING_BANDS)
        self.assertGreater(sum(counts.values()), TRIPS)

    def test_gapped_bands_drop_trips_between_bands(self):
        counts = naive_band_counts(trip_distances(), GAPPED_BANDS)
        self.assertLess(sum(counts.values()), TRIPS)
        self.assertEqual(naive_band_counts([4.95], GAPPED_BANDS), {"0-4.9": 0, "5-9.9": 0, "10-20": 0})


class FixHalfOpenBands(unittest.TestCase):
    def test_boundary_value_goes_to_the_upper_band_only(self):
        self.assertEqual(band_index(5.0, EDGES), 1)

    def test_value_just_below_an_edge_stays_in_the_lower_band(self):
        self.assertEqual(band_index(4.95, EDGES), 0)

    def test_top_edge_belongs_to_the_last_band(self):
        self.assertEqual(band_index(20.0, EDGES), 2)

    def test_bottom_edge_belongs_to_the_first_band(self):
        self.assertEqual(band_index(0.0, EDGES), 0)

    def test_counts_reconcile_with_the_number_of_trips(self):
        report = band_counts(trip_distances(), EDGES)
        self.assertEqual(sum(report.counts.values()), TRIPS)
        self.assertEqual(report.out_of_range, [])

    def test_labels_show_which_end_is_included(self):
        self.assertEqual(band_labels(EDGES), ["[0, 5)", "[5, 10)", "[10, 20]"])

    def test_values_outside_the_range_are_reported_not_dropped(self):
        report = band_counts([-1.0, 3.0, 25.0], EDGES)
        self.assertEqual(report.out_of_range, [-1.0, 25.0])
        self.assertEqual(report.total, 3)

    def test_nan_is_reported_as_out_of_range(self):
        report = band_counts([float("nan")], EDGES)
        self.assertEqual(len(report.out_of_range), 1)
        self.assertTrue(math.isnan(report.out_of_range[0]))

    def test_empty_input_gives_zero_counts(self):
        self.assertEqual(band_counts([], EDGES).counts, {"[0, 5)": 0, "[5, 10)": 0, "[10, 20]": 0})


class FixValidateEdges(unittest.TestCase):
    def test_unsorted_edges_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_edges([0, 10, 5])

    def test_repeated_edge_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_edges([0, 5, 5, 10])

    def test_single_edge_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_edges([5])

    def test_infinite_edge_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_edges([0, float("inf")])


if __name__ == "__main__":
    unittest.main()
