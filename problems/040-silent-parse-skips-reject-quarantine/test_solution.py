import json
import random
import unittest

from solution import (BatchRejectedError, RowRejected, ingest_batch, naive_ingest,
                      parse_batch, parse_event)

BATCH_SIZE = 10_000
NEW_APP_SHARE = 0.30          # share of traffic on the app release that changed `amount`
CORRUPT_LINE_SHARE = 0.002    # background noise: truncated lines from the queue


def event_line(number: int, rng: random.Random, amount_as_text: bool) -> str:
    """One JSON line; the new app release sends amount as formatted text."""
    amount = round(rng.uniform(1, 5_000), 2)
    record = {
        "event_id": f"evt-{number:06d}",
        "amount": f"{amount:,.2f}" if amount_as_text else amount,
        "currency": "INR",
        "created_at": f"2026-11-07T{number % 24:02d}:15:00+05:30",
    }
    return json.dumps(record)


def make_batch(seed: int = 11, new_app_share: float = NEW_APP_SHARE,
               size: int = BATCH_SIZE) -> list[str]:
    """A day's batch of events, with optional new-format rows and rare corrupt lines."""
    rng = random.Random(seed)
    lines = []
    for number in range(size):
        line = event_line(number, rng, amount_as_text=rng.random() < new_app_share)
        if rng.random() < CORRUPT_LINE_SHARE:
            line = line[: len(line) // 2]
        lines.append(line)
    return lines


def good_line(event_id: str = "e1") -> str:
    """A valid row for small, hand-written batches."""
    return json.dumps({"event_id": event_id, "amount": 10.0, "currency": "INR",
                       "created_at": "2026-11-07T10:00:00+05:30"})


class ReproduceProblem(unittest.TestCase):
    def test_naive_ingest_silently_drops_new_format_rows(self):
        lines = make_batch()
        events = naive_ingest(lines)
        self.assertEqual((len(events), len(lines)), (6898, 10_000))

    def test_naive_ingest_on_an_unreadable_batch_returns_normally(self):
        self.assertEqual(naive_ingest(["not json"] * 50), [])


class FixParseEvent(unittest.TestCase):
    def assertRejectedWith(self, line: str, reason: str) -> None:
        with self.assertRaises(RowRejected) as caught:
            parse_event(line)
        self.assertEqual(caught.exception.reason, reason)

    def test_valid_row_is_parsed(self):
        self.assertEqual(parse_event(good_line()).amount, 10.0)

    def test_formatted_amount_string_is_rejected_with_reason(self):
        line = good_line().replace("10.0", '"1,299.00"')
        self.assertRejectedWith(line, "amount_not_a_number")

    def test_truncated_line_is_invalid_json(self):
        self.assertRejectedWith(good_line()[:20], "invalid_json")

    def test_missing_field_is_named_in_reason(self):
        self.assertRejectedWith('{"event_id": "e1", "amount": 1}', "missing_currency")

    def test_nan_and_boolean_amounts_are_rejected(self):
        self.assertRejectedWith(good_line().replace("10.0", "NaN"), "amount_not_finite")
        self.assertRejectedWith(good_line().replace("10.0", "true"), "amount_not_a_number")

    def test_bad_timestamp_is_rejected(self):
        line = good_line().replace("2026-11-07T10:00:00+05:30", "07/11/2026 10:00")
        self.assertRejectedWith(line, "bad_created_at")

    def test_json_array_is_not_an_event(self):
        self.assertRejectedWith("[1, 2]", "not_an_object")


class FixQuarantine(unittest.TestCase):
    def setUp(self):
        self.lines = make_batch()
        self.result = parse_batch(self.lines)

    def test_every_input_row_is_either_accepted_or_quarantined(self):
        self.assertEqual(self.result.total_rows(), len(self.lines))

    def test_quarantine_counts_rejects_by_reason(self):
        self.assertEqual(dict(self.result.reasons()),
                         {"amount_not_a_number": 3087, "invalid_json": 15})

    def test_rejected_row_keeps_raw_text_and_line_number(self):
        first = self.result.rejected[0]
        self.assertEqual(self.lines[first.line_number - 1], first.raw)


class FixRejectRateGate(unittest.TestCase):
    def test_format_change_fails_the_batch(self):
        with self.assertRaises(BatchRejectedError) as caught:
            ingest_batch(make_batch())
        self.assertIn("31.0%", str(caught.exception))
        self.assertEqual(len(caught.exception.result.rejected), 3102)

    def test_normal_background_noise_passes_the_gate(self):
        result = ingest_batch(make_batch(new_app_share=0.0))
        self.assertEqual(len(result.rejected), 15)
        self.assertEqual(round(result.reject_rate(), 4), 0.0015)

    def test_small_batch_with_one_bad_row_is_not_failed_by_its_rate(self):
        lines = [good_line(f"e{n}") for n in range(9)] + ["broken"]
        result = ingest_batch(lines)     # 10% rate, but only one reject
        self.assertEqual(len(result.rejected), 1)

    def test_small_batch_with_many_bad_rows_fails(self):
        lines = [good_line(f"e{n}") for n in range(20)] + ["broken"] * 6
        with self.assertRaises(BatchRejectedError):
            ingest_batch(lines)

    def test_small_batch_where_every_row_fails_is_rejected(self):
        with self.assertRaises(BatchRejectedError):
            ingest_batch(["broken", "also broken"])

    def test_empty_batch_passes_with_nothing_to_load(self):
        result = ingest_batch([])
        self.assertEqual((result.total_rows(), result.reject_rate()), (0, 0.0))


if __name__ == "__main__":
    unittest.main()
