import unittest

from solution import (Event, Truth, blocklist_warnings, exclude_blocked, make_events,
                      naive_exclude_blocked, sql_not_in)

USER_COUNT = 1_000
EVENTS_PER_USER = 10
BLOCKED_USERS = list(range(1, 21))
EXPECTED_KEPT = (USER_COUNT - len(BLOCKED_USERS)) * EVENTS_PER_USER


def daily_events() -> list[Event]:
    """10,000 events from 1,000 users."""
    return make_events(USER_COUNT, EVENTS_PER_USER)


def blocklist_with_null() -> list[int | None]:
    """20 blocked users plus one row someone inserted without a user id."""
    return BLOCKED_USERS + [None]


class ReproduceProblem(unittest.TestCase):
    def test_not_in_works_while_blocklist_has_no_nulls(self):
        kept = naive_exclude_blocked(daily_events(), BLOCKED_USERS)
        self.assertEqual(len(kept), EXPECTED_KEPT)

    def test_one_null_in_blocklist_drops_every_event(self):
        kept = naive_exclude_blocked(daily_events(), blocklist_with_null())
        self.assertEqual(kept, [])

    def test_not_in_with_null_is_unknown_not_true(self):
        self.assertIs(sql_not_in(500, [1, 2, None]), Truth.UNKNOWN)

    def test_not_in_with_a_match_is_false_even_with_null(self):
        self.assertIs(sql_not_in(1, [1, None]), Truth.FALSE)


class FixWithNotExists(unittest.TestCase):
    def test_null_in_blocklist_does_not_remove_unblocked_users(self):
        kept = exclude_blocked(daily_events(), blocklist_with_null())
        self.assertEqual(len(kept), EXPECTED_KEPT)

    def test_blocked_users_are_still_removed(self):
        kept = exclude_blocked(daily_events(), blocklist_with_null())
        kept_users = {event.user_id for event in kept}
        self.assertTrue(kept_users.isdisjoint(BLOCKED_USERS))

    def test_matches_not_in_when_blocklist_has_no_nulls(self):
        events = daily_events()
        self.assertEqual(exclude_blocked(events, BLOCKED_USERS),
                         naive_exclude_blocked(events, BLOCKED_USERS))

    def test_event_without_user_id_is_kept(self):
        anonymous = Event(event_id=1, user_id=None)
        self.assertEqual(exclude_blocked([anonymous], BLOCKED_USERS), [anonymous])

    def test_empty_blocklist_keeps_everything(self):
        events = daily_events()
        self.assertEqual(exclude_blocked(events, []), events)


class FixBlocklistGuard(unittest.TestCase):
    def test_null_blocklist_row_raises_a_warning(self):
        warnings = blocklist_warnings(blocklist_with_null())
        self.assertEqual(len(warnings), 1)
        self.assertIn("1 row(s) with NULL user_id", warnings[0])

    def test_clean_blocklist_has_no_warnings(self):
        self.assertEqual(blocklist_warnings(BLOCKED_USERS), [])


if __name__ == "__main__":
    unittest.main()
