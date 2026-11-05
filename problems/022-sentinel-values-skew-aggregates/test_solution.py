import random
import unittest

from solution import (
    MIN_DISTINCT_VALUES,
    SENTINELS_BY_COLUMN,
    clean_rows,
    find_spike,
    find_spikes,
    mean_and_missing_rate,
    naive_mean,
    replace_sentinels,
    simulate_crm_export,
)

COLUMNS = ["age", "annual_income", "tenure_months"]


def crm_export() -> list[dict]:
    return simulate_crm_export(20_000, random.Random(22))


def registry_with_tenure_999() -> dict:
    registry = dict(SENTINELS_BY_COLUMN)
    registry["tenure_months"] = frozenset({-1, 999})
    return registry


class ReproduceProblem(unittest.TestCase):
    def test_age_sentinels_pull_the_average_age_down(self):
        rows = crm_export()
        cleaned_mean, _ = mean_and_missing_rate(clean_rows(rows), "age")
        self.assertLess(naive_mean(rows, "age"), cleaned_mean - 5)

    def test_income_sentinel_inflates_the_average_income(self):
        rows = crm_export()
        cleaned_mean, _ = mean_and_missing_rate(clean_rows(rows), "annual_income")
        self.assertGreater(naive_mean(rows, "annual_income"), 1.3 * cleaned_mean)


class FixDeclaredSentinels(unittest.TestCase):
    def test_declared_sentinels_become_none(self):
        row = {"age": 0, "annual_income": 9_999_999, "tenure_months": -1}
        cleaned = replace_sentinels(row, SENTINELS_BY_COLUMN)
        self.assertEqual(cleaned, {"age": None, "annual_income": None, "tenure_months": None})

    def test_real_values_are_kept(self):
        row = {"age": 34, "annual_income": 650_000, "tenure_months": 0}
        self.assertEqual(replace_sentinels(row, SENTINELS_BY_COLUMN), row)

    def test_zero_is_only_a_sentinel_where_declared(self):
        # tenure 0 means "joined this month", not "unknown".
        cleaned = replace_sentinels({"tenure_months": 0}, SENTINELS_BY_COLUMN)
        self.assertEqual(cleaned["tenure_months"], 0)

    def test_cleaning_does_not_modify_the_input_rows(self):
        rows = [{"age": -1}]
        clean_rows(rows)
        self.assertEqual(rows, [{"age": -1}])

    def test_missing_rate_reports_the_sentinel_share(self):
        _, missing_rate = mean_and_missing_rate(clean_rows(crm_export()), "age")
        self.assertAlmostEqual(missing_rate, 0.12, delta=0.01)

    def test_cleaned_age_mean_is_the_middle_of_the_real_range(self):
        cleaned_mean, _ = mean_and_missing_rate(clean_rows(crm_export()), "age")
        self.assertAlmostEqual(cleaned_mean, 49.0, delta=0.5)   # uniform 18..80

    def test_all_missing_column_has_no_mean(self):
        self.assertEqual(mean_and_missing_rate([{"age": None}], "age"), (None, 1.0))

    def test_empty_input_has_no_mean_and_no_missing(self):
        self.assertEqual(mean_and_missing_rate([], "age"), (None, 0.0))


class FixSpikeDetector(unittest.TestCase):
    def test_undeclared_tenure_sentinel_is_flagged_after_cleaning(self):
        spikes = find_spikes(clean_rows(crm_export()), COLUMNS)
        self.assertEqual([(spike.column, spike.value) for spike in spikes],
                         [("tenure_months", 999)])

    def test_declaring_the_flagged_value_fixes_the_tenure_mean(self):
        cleaned = clean_rows(crm_export(), registry_with_tenure_999())
        tenure_mean, _ = mean_and_missing_rate(cleaned, "tenure_months")
        self.assertAlmostEqual(tenure_mean, 60.0, delta=1.0)    # uniform 0..120
        self.assertEqual(find_spikes(cleaned, COLUMNS), [])

    def test_income_sentinel_is_flagged_on_raw_data(self):
        spikes = find_spikes(crm_export(), ["annual_income"])
        self.assertEqual(spikes[0].value, 9_999_999)

    def test_low_cardinality_columns_are_not_checked(self):
        values = [1, 2] * 50 + [3] * 900
        self.assertLess(len(set(values)), MIN_DISTINCT_VALUES)
        self.assertIsNone(find_spike("plan_tier", values))

    def test_evenly_spread_values_have_no_spike(self):
        self.assertIsNone(find_spike("score", list(range(100)) * 10))


if __name__ == "__main__":
    unittest.main()
