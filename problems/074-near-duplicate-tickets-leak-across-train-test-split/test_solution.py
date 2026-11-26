import unittest

from solution import (Ticket, build_tickets, fingerprint, group_split, leaked_share,
                      memorizer_accuracy, naive_random_split, normalize)


class ReproduceProblem(unittest.TestCase):
    def setUp(self):
        self.train, self.test = naive_random_split(build_tickets())

    def test_random_split_puts_copies_of_most_test_tickets_in_training(self):
        self.assertAlmostEqual(leaked_share(self.train, self.test), 0.648, places=3)

    def test_memorizing_model_looks_good_on_the_leaky_split(self):
        self.assertAlmostEqual(memorizer_accuracy(self.train, self.test), 0.766, places=3)


class FixNormalizeText(unittest.TestCase):
    def test_reply_prefixes_are_removed(self):
        self.assertEqual(normalize("Re: Fwd: card charged twice"), "card charged twice")

    def test_case_and_spacing_are_ignored(self):
        self.assertEqual(normalize("  CARD   charged\ttwice "), "card charged twice")

    def test_ticket_numbers_are_removed(self):
        self.assertEqual(normalize("card charged twice (ticket #48213)"), "card charged twice")

    def test_quoted_earlier_mail_is_dropped(self):
        self.assertEqual(normalize("card charged twice\n> On Monday you wrote"), "card charged twice")

    def test_full_width_characters_match_ascii(self):
        self.assertEqual(normalize("ｃａｒｄ charged"), "card charged")

    def test_different_complaints_keep_different_fingerprints(self):
        self.assertNotEqual(fingerprint("card charged twice"), fingerprint("parcel late"))

    def test_resent_copy_has_the_original_fingerprint(self):
        resent = "Re: Fwd:  CARD CHARGED TWICE  (ticket #55555)\n> previous message"
        self.assertEqual(fingerprint(resent), fingerprint("card charged twice"))


class FixGroupSplit(unittest.TestCase):
    def setUp(self):
        self.tickets = build_tickets()
        self.train, self.test = group_split(self.tickets)

    def test_no_test_ticket_has_a_copy_in_training(self):
        self.assertEqual(leaked_share(self.train, self.test), 0.0)

    def test_memorizing_model_no_longer_scores_well(self):
        self.assertAlmostEqual(memorizer_accuracy(self.train, self.test), 0.124, places=3)

    def test_every_ticket_is_assigned_exactly_once(self):
        self.assertEqual(len(self.train) + len(self.test), len(self.tickets))

    def test_split_is_the_same_on_every_run(self):
        self.assertEqual(group_split(self.tickets), (self.train, self.test))

    def test_a_new_copy_joins_its_earlier_copies_side(self):
        original = self.test[0]
        late_copy = Ticket("Fwd: " + original.text.upper() + " (ticket #1)", original.category)
        _, test_again = group_split([late_copy])
        self.assertEqual(test_again, [late_copy])


if __name__ == "__main__":
    unittest.main()
