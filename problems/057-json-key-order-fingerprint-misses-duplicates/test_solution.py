import json
import unittest

from solution import (canonical_json, deduplicate, fingerprint, naive_fingerprint,
                      total_amount)

ORIGINAL = '{"event_id": "pay-1", "amount": 250, "currency": "INR", "ingested_at": "10:00:01"}'


def retried_copies() -> list[str]:
    """The same payment as re-sent by a retrying producer in four different shapes."""
    return [
        '{"currency": "INR", "event_id": "pay-1", "amount": 250, "ingested_at": "10:00:01"}',
        '{"event_id":"pay-1","amount":250,"currency":"INR","ingested_at":"10:00:01"}',
        '{"event_id": "pay-1", "amount": 250.0, "currency": "INR", "ingested_at": "10:00:01"}',
        '{"event_id": "pay-1", "amount": 250, "currency": "INR", "ingested_at": "10:05:42"}',
    ]


class ReproduceProblem(unittest.TestCase):
    def test_every_retried_copy_gets_a_different_raw_hash(self):
        hashes = {naive_fingerprint(raw) for raw in [ORIGINAL] + retried_copies()}
        self.assertEqual(len(hashes), 5)

    def test_revenue_is_counted_five_times(self):
        kept = deduplicate([ORIGINAL] + retried_copies(), naive_fingerprint)
        self.assertEqual(total_amount(kept), 1250)


class FixCanonicalFingerprint(unittest.TestCase):
    def test_all_copies_collapse_to_one_event(self):
        kept = deduplicate([ORIGINAL] + retried_copies())
        self.assertEqual(kept, [ORIGINAL])
        self.assertEqual(total_amount(kept), 250)

    def test_nested_key_order_does_not_matter(self):
        first = '{"id": 1, "card": {"brand": "visa", "last4": "4242"}}'
        second = '{"card": {"last4": "4242", "brand": "visa"}, "id": 1}'
        self.assertEqual(fingerprint(first), fingerprint(second))

    def test_unicode_escape_and_literal_character_match(self):
        self.assertEqual(fingerprint('{"name": "Jos\\u00e9"}'), fingerprint('{"name": "José"}'))

    def test_different_amount_is_a_different_event(self):
        self.assertNotEqual(fingerprint('{"id": 1, "amount": 250}'),
                            fingerprint('{"id": 1, "amount": 250.5}'))

    def test_list_order_is_significant(self):
        self.assertNotEqual(fingerprint('{"items": ["a", "b"]}'),
                            fingerprint('{"items": ["b", "a"]}'))

    def test_true_is_not_confused_with_one(self):
        self.assertNotEqual(fingerprint('{"flag": true}'), fingerprint('{"flag": 1}'))

    def test_volatile_fields_are_dropped_from_canonical_form(self):
        self.assertEqual(canonical_json(json.loads(ORIGINAL)),
                         '{"amount":250,"currency":"INR","event_id":"pay-1"}')

    def test_distinct_events_are_all_kept_in_arrival_order(self):
        events = ['{"id": 2}', '{"id": 1}', '{"id": 2}', '{"id": 3}']
        self.assertEqual(deduplicate(events), ['{"id": 2}', '{"id": 1}', '{"id": 3}'])


class FixRejectsBadInput(unittest.TestCase):
    def test_nan_amount_is_rejected(self):
        with self.assertRaises(ValueError):
            fingerprint('{"amount": NaN}')

    def test_non_object_event_is_rejected(self):
        with self.assertRaises(ValueError):
            fingerprint('[1, 2, 3]')


if __name__ == "__main__":
    unittest.main()
