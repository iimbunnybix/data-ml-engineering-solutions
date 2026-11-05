import unittest

from solution import (
    ColumnMismatch,
    LinearModel,
    UNKNOWN,
    Vocabulary,
    naive_score_batch,
    score_batch,
)

TRAINING_DEVICES = ["android", "ios", "web", "ios", "android"]
# Risk weight per device as learned in training; unknown devices get a cautious weight.
WEIGHTS = {"android": 0.5, "ios": 0.1, "web": 0.9, UNKNOWN: 0.7}


def trained() -> tuple[LinearModel, Vocabulary]:
    vocabulary = Vocabulary.fit("device", TRAINING_DEVICES)
    names = vocabulary.columns()
    weights = []
    for category in list(vocabulary.categories) + [UNKNOWN]:
        weights.append(WEIGHTS[category])
    return LinearModel(tuple(names), tuple(weights), bias=0.0), vocabulary


class ReproduceProblem(unittest.TestCase):
    def test_batch_without_android_scores_ios_users_with_the_android_weight(self):
        model, _ = trained()
        self.assertEqual(naive_score_batch(model, ["ios", "web"]), [0.5, 0.1])

    def test_unseen_category_shifts_web_users_onto_the_ios_weight(self):
        model, _ = trained()
        # Batch columns are [tv, web]: web users land in column 1, the model's "ios".
        self.assertEqual(naive_score_batch(model, ["web", "tv"]), [0.1, 0.5])


class FixUsesTrainingVocabulary(unittest.TestCase):
    def test_scores_do_not_depend_on_which_categories_are_in_the_batch(self):
        model, vocabulary = trained()
        self.assertEqual(score_batch(model, vocabulary, ["ios", "web"]), [0.1, 0.9])
        self.assertEqual(score_batch(model, vocabulary, ["web"]), [0.9])

    def test_single_row_batch_scores_like_a_large_batch(self):
        model, vocabulary = trained()
        large = score_batch(model, vocabulary, ["android", "ios", "web"])
        self.assertEqual(score_batch(model, vocabulary, ["ios"]), [large[1]])

    def test_unseen_category_uses_the_unknown_bucket(self):
        model, vocabulary = trained()
        self.assertEqual(score_batch(model, vocabulary, ["tv", "web"]), [0.7, 0.9])

    def test_encoding_width_is_fixed(self):
        _, vocabulary = trained()
        self.assertEqual(len(vocabulary.encode(["tv"])[0]), 4)
        self.assertEqual(vocabulary.columns()[-1], "device=__unknown__")

    def test_unknown_share_is_reported(self):
        _, vocabulary = trained()
        self.assertEqual(vocabulary.unknown_share(["tv", "ios", "car", "web"]), 0.5)
        self.assertEqual(vocabulary.unknown_share([]), 0.0)

    def test_empty_batch_returns_no_scores(self):
        model, vocabulary = trained()
        self.assertEqual(score_batch(model, vocabulary, []), [])


class FixRejectsColumnMismatch(unittest.TestCase):
    def test_reordered_columns_are_rejected(self):
        model, vocabulary = trained()
        columns = vocabulary.columns()
        columns[0], columns[1] = columns[1], columns[0]
        with self.assertRaises(ColumnMismatch):
            model.score([1.0, 0.0, 0.0, 0.0], columns)

    def test_vocabulary_from_another_training_run_is_rejected(self):
        model, _ = trained()
        retrained = Vocabulary.fit("device", TRAINING_DEVICES + ["tv"])
        with self.assertRaises(ColumnMismatch):
            score_batch(model, retrained, ["ios"])

    def test_row_with_wrong_length_is_rejected(self):
        model, vocabulary = trained()
        with self.assertRaises(ColumnMismatch):
            model.score([1.0, 0.0, 0.0], vocabulary.columns())


if __name__ == "__main__":
    unittest.main()
