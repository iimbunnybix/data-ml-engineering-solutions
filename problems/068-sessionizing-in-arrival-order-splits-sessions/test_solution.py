import random
import unittest

from solution import (SECONDS_PER_MINUTE, SESSION_TIMEOUT_SECONDS, Event, naive_sessionize,
                      sessionize)

MINUTE = SECONDS_PER_MINUTE
USERS = 1_000
VISITS_PER_USER = 3
EVENTS_PER_VISIT = 8
OFFLINE_SHARE = 0.10          # share of events buffered on the phone and uploaded late
UPLOAD_DELAY_EVENTS = 5       # a buffered event arrives this many events late
SEED = 7


def event(user_id, minute, event_id):
    return Event(user_id, minute * MINUTE, event_id)


def generate_events(rng):
    """True visits: events 1-5 minutes apart, visits 2-6 hours apart."""
    events = []
    for user_number in range(USERS):
        user_id = f"user-{user_number}"
        clock = rng.randint(0, 600) * MINUTE
        for visit in range(VISITS_PER_USER):
            for step in range(EVENTS_PER_VISIT):
                events.append(Event(user_id, clock, f"{user_id}-{visit}-{step}"))
                clock += rng.randint(1, 5) * MINUTE
            clock += rng.randint(120, 360) * MINUTE
    return events


def arrival_order(events, rng):
    """Most events arrive in order; buffered ones arrive a few events later."""
    arrivals = []
    for position, item in enumerate(events):
        delay = UPLOAD_DELAY_EVENTS if rng.random() < OFFLINE_SHARE else 0
        arrivals.append((position + delay, position, item))
    arrivals.sort()
    return [item for _, _, item in arrivals]


def replay():
    rng = random.Random(SEED)
    events = generate_events(rng)
    return events, arrival_order(events, rng)


def late_upload_example():
    """Visit from minute 10 to 45; the minute-5 event was buffered offline and arrives late."""
    return [event("u", 10, "a"), event("u", 20, "b"), event("u", 40, "c"),
            event("u", 5, "late"), event("u", 45, "d")]


class ReproduceProblem(unittest.TestCase):
    def test_one_late_event_splits_a_visit_in_two(self):
        arrived = late_upload_example()
        self.assertEqual(len(naive_sessionize(arrived)), 2)

    def test_late_event_gives_a_session_with_negative_duration(self):
        arrived = late_upload_example()
        durations = [session.duration_seconds for session in naive_sessionize(arrived)]
        self.assertLess(min(durations), 0)

    def test_offline_buffering_inflates_session_counts(self):
        events, arrived = replay()
        true_count = len(sessionize(events))
        self.assertEqual(true_count, USERS * VISITS_PER_USER)
        self.assertGreater(len(naive_sessionize(arrived)), true_count * 1.1)


class FixSortByEventTime(unittest.TestCase):
    def test_late_event_joins_its_visit(self):
        arrived = late_upload_example()
        sessions = sessionize(arrived)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].event_count, 5)
        self.assertEqual(sessions[0].duration_seconds, 40 * MINUTE)

    def test_result_does_not_depend_on_arrival_order(self):
        events, arrived = replay()
        self.assertEqual(sessionize(arrived), sessionize(events))

    def test_gap_of_exactly_the_timeout_stays_in_one_session(self):
        arrived = [Event("u", 0, "a"), Event("u", SESSION_TIMEOUT_SECONDS, "b")]
        self.assertEqual(len(sessionize(arrived)), 1)

    def test_gap_just_over_the_timeout_starts_a_new_session(self):
        arrived = [Event("u", 0, "a"), Event("u", SESSION_TIMEOUT_SECONDS + 1, "b")]
        self.assertEqual(len(sessionize(arrived)), 2)

    def test_reuploaded_duplicate_is_counted_once(self):
        arrived = [event("u", 0, "a"), event("u", 5, "b"), event("u", 0, "a")]
        self.assertEqual(sessionize(arrived)[0].event_count, 2)

    def test_users_are_sessionized_independently(self):
        arrived = [event("u1", 0, "a"), event("u2", 10, "b"), event("u1", 20, "c")]
        sessions = sessionize(arrived)
        self.assertEqual([(s.user_id, s.event_count) for s in sessions], [("u1", 2), ("u2", 1)])

    def test_no_session_has_negative_duration(self):
        _, arrived = replay()
        self.assertTrue(all(session.duration_seconds >= 0 for session in sessionize(arrived)))


if __name__ == "__main__":
    unittest.main()
