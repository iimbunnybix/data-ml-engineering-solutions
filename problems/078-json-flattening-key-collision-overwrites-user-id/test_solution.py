import unittest

from solution import FlattenCollision, flatten, joinable_share, load_events, original_flatten

ACCOUNT_IDS = {101, 102, 103}


def legacy_event(account_id: int) -> dict:
    return {"event": "page_view", "user_id": account_id, "page": {"url": "/home"}}


def new_sdk_event(account_id: int) -> dict:
    """New SDK: keeps the legacy field and adds a nested user object after it."""
    return {"event": "page_view", "user_id": account_id,
            "user": {"id": f"ext-{account_id * 7919}", "plan": "pro"}}


class ReproduceProblem(unittest.TestCase):
    def test_nested_user_id_overwrites_the_account_id(self):
        row = original_flatten(new_sdk_event(101))
        self.assertEqual(row["user_id"], "ext-799819")

    def test_overwritten_rows_no_longer_join_to_accounts(self):
        events = [legacy_event(101), new_sdk_event(102), new_sdk_event(103), legacy_event(103)]
        rows = [original_flatten(event) for event in events]
        self.assertEqual(joinable_share(rows, ACCOUNT_IDS), 0.5)

    def test_key_order_decides_which_value_survives(self):
        reordered = {"user": {"id": "ext-1"}, "user_id": 101}
        self.assertEqual(original_flatten(reordered)["user_id"], 101)


class FixDetectCollisions(unittest.TestCase):
    def test_collision_is_raised_with_both_paths(self):
        with self.assertRaises(FlattenCollision) as caught:
            flatten(new_sdk_event(101))
        self.assertEqual(caught.exception.column, "user_id")
        self.assertEqual(caught.exception.paths, ("user_id", "user.id"))

    def test_collision_is_raised_whatever_the_key_order(self):
        with self.assertRaises(FlattenCollision):
            flatten({"user": {"id": "ext-1"}, "user_id": 101})

    def test_colliding_events_are_quarantined_not_loaded(self):
        events = [legacy_event(101), new_sdk_event(102), legacy_event(103)]
        result = load_events(events)
        self.assertEqual(len(result.rows), 2)
        self.assertEqual(len(result.quarantined), 1)
        self.assertIn("'user.id'", result.quarantined[0][1])

    def test_every_loaded_row_still_joins(self):
        events = [legacy_event(101), new_sdk_event(102), legacy_event(103)]
        self.assertEqual(joinable_share(load_events(events).rows, ACCOUNT_IDS), 1.0)

    def test_nested_keys_without_collision_flatten_normally(self):
        self.assertEqual(flatten(legacy_event(101)),
                         {"event": "page_view", "user_id": 101, "page_url": "/home"})

    def test_deep_nesting_collides_with_a_shallower_key(self):
        with self.assertRaises(FlattenCollision):
            flatten({"a": {"b_c": 1, "b": {"c": 2}}})

    def test_empty_nested_object_produces_no_columns(self):
        self.assertEqual(flatten({"event": "x", "meta": {}}), {"event": "x"})

    def test_lists_are_kept_as_values(self):
        self.assertEqual(flatten({"tags": ["a", "b"]}), {"tags": ["a", "b"]})


if __name__ == "__main__":
    unittest.main()
