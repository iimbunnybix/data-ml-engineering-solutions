import unittest
from datetime import date, datetime

from solution import (
    DATE_SYSTEM_1904,
    InvalidSerialDate,
    naive_serial_to_date,
    revenue_by_quarter,
    serial_to_date,
    serial_to_datetime,
)

MARCH_30_2026 = 46111
MARCH_31_2026 = 46112
APRIL_1_2026 = 46113

# (serial, amount in rupees) from a regional order sheet.
ORDERS = [(46080, 10_000), (MARCH_30_2026, 25_000), (MARCH_31_2026, 40_000), (APRIL_1_2026, 5_000)]


class ReproduceProblem(unittest.TestCase):
    def test_naive_conversion_is_two_days_late(self):
        self.assertEqual(naive_serial_to_date(MARCH_31_2026), date(2026, 4, 2))

    def test_last_two_days_of_march_move_into_q2(self):
        totals = revenue_by_quarter(ORDERS, naive_serial_to_date)
        self.assertEqual(totals, {"2026-Q1": 10_000, "2026-Q2": 70_000})


class FixModernDates(unittest.TestCase):
    def test_quarter_end_serial_is_march_31(self):
        self.assertEqual(serial_to_date(MARCH_31_2026), date(2026, 3, 31))

    def test_new_year_serial_matches_excel(self):
        self.assertEqual(serial_to_date(45658), date(2025, 1, 1))

    def test_quarter_end_orders_stay_in_q1(self):
        totals = revenue_by_quarter(ORDERS, serial_to_date)
        self.assertEqual(totals, {"2026-Q1": 75_000, "2026-Q2": 5_000})

    def test_fraction_is_the_time_of_day(self):
        self.assertEqual(serial_to_datetime(MARCH_31_2026 + 0.75), datetime(2026, 3, 31, 18, 0))


class FixEarly1900Serials(unittest.TestCase):
    def test_serial_one_is_january_first_1900(self):
        self.assertEqual(serial_to_date(1), date(1900, 1, 1))

    def test_serial_59_is_february_28_1900(self):
        self.assertEqual(serial_to_date(59), date(1900, 2, 28))

    def test_serial_61_is_march_first_1900(self):
        self.assertEqual(serial_to_date(61), date(1900, 3, 1))

    def test_fake_leap_day_is_rejected(self):
        with self.assertRaises(InvalidSerialDate):
            serial_to_date(60)

    def test_serial_zero_is_rejected(self):
        with self.assertRaises(InvalidSerialDate):
            serial_to_date(0)


class Fix1904DateSystem(unittest.TestCase):
    def test_serial_zero_is_january_first_1904(self):
        self.assertEqual(serial_to_date(0, DATE_SYSTEM_1904), date(1904, 1, 1))

    def test_same_day_is_1462_lower_in_the_1904_system(self):
        self.assertEqual(serial_to_date(MARCH_31_2026 - 1462, DATE_SYSTEM_1904), date(2026, 3, 31))

    def test_unknown_date_system_is_rejected(self):
        with self.assertRaises(ValueError):
            serial_to_date(MARCH_31_2026, "1970")


if __name__ == "__main__":
    unittest.main()
