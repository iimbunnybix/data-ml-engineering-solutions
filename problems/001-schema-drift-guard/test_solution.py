import unittest

from solution import CUSTOMER_CONTRACT, SchemaViolation, load_csv

GOOD = """customer_id,monthly_spend,signup_date,plan,tenure_months
C1,499.0,2024-01-15,basic,12
C2,1299.5,2023-06-01,pro,30
"""


class SchemaGuardTests(unittest.TestCase):
    def test_valid_file_loads_with_types(self):
        rows = load_csv(GOOD, CUSTOMER_CONTRACT)
        self.assertEqual(rows[1]["monthly_spend"], 1299.5)
        self.assertEqual(rows[0]["tenure_months"], 12)

    def test_the_actual_incident_is_caught(self):
        drifted = """customer_id,spend,signup_date,plan,tenure_months
C1,499.0,15/01/2024,basic,12
"""
        with self.assertRaises(SchemaViolation) as ctx:
            load_csv(drifted, CUSTOMER_CONTRACT)
        errors = " ".join(ctx.exception.report.errors)
        self.assertIn("missing required column 'monthly_spend'", errors)
        self.assertIn("'signup_date' has 1 invalid", errors)

    def test_known_rename_is_accepted_via_alias(self):
        renamed = GOOD.replace("monthly_spend", "monthly_spend_inr")
        rows = load_csv(renamed, CUSTOMER_CONTRACT)
        self.assertIn("monthly_spend", rows[0])

    def test_all_null_column_fails(self):
        text = """customer_id,monthly_spend,signup_date,plan,tenure_months
C1,,2024-01-15,basic,12
C2,,2024-01-16,plus,3
"""
        with self.assertRaisesRegex(SchemaViolation, "null rate 100%"):
            load_csv(text, CUSTOMER_CONTRACT)

    def test_out_of_range_and_unknown_category(self):
        text = """customer_id,monthly_spend,signup_date,plan,tenure_months
C1,-5,2024-01-15,gold,12
"""
        with self.assertRaises(SchemaViolation) as ctx:
            load_csv(text, CUSTOMER_CONTRACT)
        self.assertEqual(len(ctx.exception.report.errors), 2)

    def test_new_extra_column_only_warns(self):
        from solution import validate
        import csv, io
        text = GOOD.replace("tenure_months\n", "tenure_months,city\n").replace(",12\n", ",12,Pune\n") \
                   .replace(",30\n", ",30,Nagpur\n")
        r = csv.DictReader(io.StringIO(text))
        _, report = validate(list(r), r.fieldnames, CUSTOMER_CONTRACT)
        self.assertTrue(report.ok)
        self.assertIn("unexpected new column 'city'", report.warnings)

    def test_all_nan_column_counts_as_null_and_fails(self):
        text = """customer_id,monthly_spend,signup_date,plan,tenure_months
C1,nan,2024-01-15,basic,12
C2,NaN,2023-06-01,pro,30
"""
        with self.assertRaisesRegex(SchemaViolation, "null rate 100%"):
            load_csv(text, CUSTOMER_CONTRACT)

    def test_infinite_float_is_rejected_as_invalid(self):
        text = """customer_id,monthly_spend,signup_date,plan,tenure_months
C1,inf,2024-01-15,basic,12
"""
        with self.assertRaisesRegex(SchemaViolation, "'monthly_spend' has 1 invalid"):
            load_csv(text, CUSTOMER_CONTRACT)


if __name__ == "__main__":
    unittest.main()
