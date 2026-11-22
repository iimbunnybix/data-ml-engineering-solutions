import random
import unittest

from solution import (TIE, TOO_FEW_VOTES, Vote, aggregate_labels, label_share,
                      naive_majority_labels, plurality_label)

ITEMS = 2_000
TRUE_TOXIC_SHARE = 0.30
SENIOR_MISSES_TOXIC = 0.25     # senior labels a toxic comment "ok" this often
JUNIOR_ERROR_RATE = 0.05       # junior flips the true label this often
SEED = 7


def simulated_votes(seed: int = SEED) -> tuple[list[Vote], dict[str, str]]:
    """Two votes per comment, senior first; returns (votes, true labels)."""
    rng = random.Random(seed)
    votes, truth = [], {}
    for index in range(ITEMS):
        item_id = f"c{index}"
        true_label = "toxic" if rng.random() < TRUE_TOXIC_SHARE else "ok"
        truth[item_id] = true_label
        senior = "ok" if true_label == "toxic" and rng.random() < SENIOR_MISSES_TOXIC else true_label
        junior = true_label
        if rng.random() < JUNIOR_ERROR_RATE:
            junior = "ok" if true_label == "toxic" else "toxic"
        votes.append(Vote(item_id, "senior", senior))
        votes.append(Vote(item_id, "junior", junior))
    return votes, truth


def votes_for(item_id: str, *annotator_labels: tuple[str, str]) -> list[Vote]:
    return [Vote(item_id, annotator, label) for annotator, label in annotator_labels]


class ReproduceProblem(unittest.TestCase):
    def test_tie_goes_to_whoever_voted_first(self):
        first_ok = votes_for("c1", ("senior", "ok"), ("junior", "toxic"))
        first_toxic = votes_for("c1", ("junior", "toxic"), ("senior", "ok"))
        self.assertEqual(naive_majority_labels(first_ok), {"c1": "ok"})
        self.assertEqual(naive_majority_labels(first_toxic), {"c1": "toxic"})

    def test_naive_labels_undercount_toxic_comments(self):
        votes, truth = simulated_votes()
        naive_share = label_share(naive_majority_labels(votes), "toxic")
        true_share = label_share(truth, "toxic")
        self.assertLess(naive_share, true_share - 0.06)

    def test_naive_labels_match_the_senior_annotator_on_every_item(self):
        votes, _ = simulated_votes()
        senior = {vote.item_id: vote.label for vote in votes if vote.annotator_id == "senior"}
        self.assertEqual(naive_majority_labels(votes), senior)


class FixAdjudicateTies(unittest.TestCase):
    def test_tie_goes_to_adjudication_in_any_vote_order(self):
        for votes in (votes_for("c1", ("senior", "ok"), ("junior", "toxic")),
                      votes_for("c1", ("junior", "toxic"), ("senior", "ok"))):
            result = aggregate_labels(votes)
            self.assertEqual(result.labels, {})
            self.assertEqual(result.adjudication, {"c1": TIE})

    def test_agreement_is_accepted(self):
        result = aggregate_labels(votes_for("c1", ("senior", "toxic"), ("junior", "toxic")))
        self.assertEqual(result.labels, {"c1": "toxic"})

    def test_strict_plurality_wins_with_three_votes(self):
        votes = votes_for("c1", ("a", "ok"), ("b", "toxic"), ("c", "toxic"))
        self.assertEqual(aggregate_labels(votes).labels, {"c1": "toxic"})

    def test_three_way_split_is_a_tie(self):
        self.assertIsNone(plurality_label(["ok", "toxic", "spam"]))

    def test_single_vote_is_not_enough(self):
        result = aggregate_labels(votes_for("c1", ("senior", "ok")))
        self.assertEqual(result.adjudication, {"c1": TOO_FEW_VOTES})

    def test_repeat_vote_by_one_annotator_counts_once(self):
        votes = votes_for("c1", ("senior", "ok"), ("senior", "ok"), ("junior", "toxic"))
        self.assertEqual(aggregate_labels(votes).adjudication, {"c1": TIE})

    def test_accepted_labels_no_longer_undercount_toxic(self):
        votes, truth = simulated_votes()
        result = aggregate_labels(votes)
        accepted_truth = {item: truth[item] for item in result.labels}
        self.assertAlmostEqual(label_share(result.labels, "toxic"),
                               label_share(accepted_truth, "toxic"), delta=0.02)

    def test_every_item_is_either_labelled_or_queued(self):
        votes, truth = simulated_votes()
        result = aggregate_labels(votes)
        self.assertEqual(len(result.labels) + len(result.adjudication), len(truth))

    def test_agreement_rate_is_reported(self):
        votes = (votes_for("c1", ("a", "ok"), ("b", "ok"))
                 + votes_for("c2", ("a", "ok"), ("b", "toxic")))
        self.assertEqual(aggregate_labels(votes).agreement_rate, 0.5)

    def test_no_votes_gives_empty_result(self):
        result = aggregate_labels([])
        self.assertEqual((result.labels, result.adjudication, result.agreement_rate), ({}, {}, 0.0))


if __name__ == "__main__":
    unittest.main()
