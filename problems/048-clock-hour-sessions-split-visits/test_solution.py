import random
import unittest

from solution import (INACTIVITY_GAP_SECONDS, SECONDS_PER_MINUTE, Event, Session,
                      average_duration_minutes, gap_sessions, make_traffic,
                      naive_hourly_sessions)

SEED = 48
USER_COUNT = 1_000
# 2023-11-14 00:00:00 UTC, a whole hour, so clock-hour buckets start on it.
DAY_START = 1_699_920_000
TEN_AM = DAY_START + 10 * 3600
MINUTE = SECONDS_PER_MINUTE


def events_at(user_id: str, minutes_after_ten: list[int]) -> list[Event]:
    """Events for one user at the given minutes after 10:00."""
    events = []
    for index, minutes in enumerate(minutes_after_ten):
        events.append(Event(user_id, TEN_AM + minutes * MINUTE, f"{user_id}-{index}"))
    return events


def visit_across_eleven_oclock() -> list[Event]:
    """One continuous visit from 10:50 to 11:10, a click every 5 minutes."""
    return events_at("alice", [50, 55, 60, 65, 70])


def two_visits_in_one_hour() -> list[Event]:
    """A visit at 14:02-14:06 and another at 14:50-14:55, with 44 idle minutes between."""
    return events_at("bob", [242, 246, 290, 295])


def simulated_day() -> tuple[list[Event], int]:
    """Shuffled events for 1,000 users and the true number of visits."""
    return make_traffic(random.Random(SEED), USER_COUNT, DAY_START)


class ReproduceProblem(unittest.TestCase):
    def test_visit_across_the_hour_becomes_two_sessions(self):
        self.assertEqual(len(naive_hourly_sessions(visit_across_eleven_oclock())), 2)

    def test_two_visits_in_the_same_hour_become_one_session(self):
        self.assertEqual(len(naive_hourly_sessions(two_visits_in_one_hour())), 1)

    def test_daily_session_count_is_inflated(self):
        events, true_visits = simulated_day()
        self.assertGreater(len(naive_hourly_sessions(events)), true_visits * 1.2)


class FixWithInactivityGap(unittest.TestCase):
    def test_visit_across_the_hour_is_one_session(self):
        sessions = gap_sessions(visit_across_eleven_oclock())
        self.assertEqual(sessions, [Session("alice", TEN_AM + 50 * MINUTE, TEN_AM + 70 * MINUTE, 5)])

    def test_two_visits_in_the_same_hour_are_two_sessions(self):
        sessions = gap_sessions(two_visits_in_one_hour())
        self.assertEqual([session.event_count for session in sessions], [2, 2])

    def test_daily_session_count_matches_true_visits(self):
        events, true_visits = simulated_day()
        self.assertEqual(len(gap_sessions(events)), true_visits)

    def test_sessions_are_longer_than_the_hourly_version(self):
        events, _ = simulated_day()
        self.assertGreater(average_duration_minutes(gap_sessions(events)),
                           average_duration_minutes(naive_hourly_sessions(events)))

    def test_out_of_order_input_gives_the_same_sessions(self):
        events = visit_across_eleven_oclock() + two_visits_in_one_hour()
        shuffled = list(events)
        random.Random(1).shuffle(shuffled)
        self.assertEqual(gap_sessions(shuffled), gap_sessions(events))

    def test_events_with_the_same_timestamp_stay_in_one_session(self):
        events = events_at("carol", [0, 0, 0, 10])
        self.assertEqual(gap_sessions(events), [Session("carol", TEN_AM, TEN_AM + 10 * MINUTE, 4)])

    def test_pause_of_exactly_the_timeout_continues_the_session(self):
        timeout_minutes = INACTIVITY_GAP_SECONDS // MINUTE
        self.assertEqual(len(gap_sessions(events_at("dave", [0, timeout_minutes]))), 1)

    def test_pause_one_second_longer_than_the_timeout_starts_a_new_session(self):
        events = [Event("erin", TEN_AM, "e1"),
                  Event("erin", TEN_AM + INACTIVITY_GAP_SECONDS + 1, "e2")]
        self.assertEqual(len(gap_sessions(events)), 2)

    def test_users_are_sessionised_separately(self):
        # Interleaved activity from two users must not keep each other's sessions alive.
        events = events_at("frank", [0, 40]) + events_at("grace", [20, 60])
        self.assertEqual(len(gap_sessions(events)), 4)

    def test_no_events_gives_no_sessions(self):
        self.assertEqual(gap_sessions([]), [])

    def test_non_positive_gap_is_rejected(self):
        with self.assertRaises(ValueError):
            gap_sessions(visit_across_eleven_oclock(), gap_seconds=0)


if __name__ == "__main__":
    unittest.main()
