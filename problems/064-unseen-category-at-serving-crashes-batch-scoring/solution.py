"""Encode categories so a value never seen in training cannot crash or mis-score a batch.

The nightly fraud-scoring job turned each categorical value into an integer
index with a dict built from training data. When a new payment method
launched, `index[value]` raised KeyError on the first such row and the whole
job failed. The "quick fix" `index.get(value, 0)` stopped the crash but sent
every new value to index 0, which is a real category ("apple_pay"), so those
payments were silently scored as if they were Apple Pay.

The fix reserves index 0 for an explicit "__unknown__" bucket. Training values
seen fewer than a minimum number of times are folded into it, so the model
learns a real weight for "rare or new". At serving time unseen values go to
that bucket, unknowns are counted per column, and a column whose unknown rate
crosses a threshold raises an alert instead of failing the job.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

Row = dict[str, str | None]

UNKNOWN = "__unknown__"
UNKNOWN_INDEX = 0
# Values rarer than this in training get no index of their own: their weight
# would be learned from a handful of rows.
DEFAULT_MIN_COUNT = 20
# Above this share of unknowns in a column, the model is scoring blind on it.
DEFAULT_ALERT_UNKNOWN_RATE = 0.05


def naive_fit_index(rows: list[Row], column: str) -> dict[str, int]:
    """The original encoder: every training value gets an index, starting at 0."""
    values = sorted({row[column] for row in rows if row.get(column) is not None})
    return {value: position for position, value in enumerate(values)}


def naive_encode(index: dict[str, int], value: str) -> int:
    """Original lookup: raises KeyError on any value not seen in training."""
    return index[value]


def naive_encode_with_default(index: dict[str, int], value: str) -> int:
    """The hotfix: no crash, but unseen values silently become category 0."""
    return index.get(value, 0)


@dataclass
class UnknownReport:
    """Per-column counts of values that fell into the unknown bucket."""
    rows: int = 0
    unknown_counts: Counter[str] = field(default_factory=Counter)
    unseen_values: dict[str, Counter[str]] = field(default_factory=dict)

    def unknown_rate(self, column: str) -> float:
        """Share of rows in this batch whose value for the column was unknown."""
        if self.rows == 0:
            return 0.0
        return self.unknown_counts[column] / self.rows


@dataclass(frozen=True)
class CategoryEncoder:
    """Value-to-index maps frozen at training time; index 0 is always the unknown bucket."""
    indexes: dict[str, dict[str, int]]

    @classmethod
    def fit(cls, rows: list[Row], columns: list[str],
            min_count: int = DEFAULT_MIN_COUNT) -> "CategoryEncoder":
        """Give each value seen at least min_count times its own index, from 1 upwards."""
        indexes: dict[str, dict[str, int]] = {}
        for column in columns:
            counts = Counter(row.get(column) for row in rows)
            frequent = sorted(value for value, count in counts.items()
                              if value is not None and value != UNKNOWN and count >= min_count)
            # Index 0 is reserved, so no real category can be confused with "unknown".
            indexes[column] = {value: position + 1 for position, value in enumerate(frequent)}
        return cls(indexes)

    def encode_value(self, column: str, value: str | None) -> int:
        """Index of a value, or UNKNOWN_INDEX for missing, rare or unseen values."""
        return self.indexes[column].get(value, UNKNOWN_INDEX) if value is not None else UNKNOWN_INDEX

    def encode(self, rows: list[Row], report: UnknownReport | None = None) -> list[dict[str, int]]:
        """Encode every row; if a report is given, count unknowns per column into it."""
        encoded = []
        for row in rows:
            encoded_row = {}
            for column in self.indexes:
                value = row.get(column)
                encoded_row[column] = self.encode_value(column, value)
                if report is not None and encoded_row[column] == UNKNOWN_INDEX:
                    report.unknown_counts[column] += 1
                    report.unseen_values.setdefault(column, Counter())[str(value)] += 1
            encoded.append(encoded_row)
        if report is not None:
            report.rows += len(rows)
        return encoded


def unknown_rate_alerts(report: UnknownReport,
                        threshold: float = DEFAULT_ALERT_UNKNOWN_RATE) -> list[str]:
    """One message per column whose unknown rate is above the threshold."""
    alerts = []
    for column in sorted(report.unknown_counts):
        rate = report.unknown_rate(column)
        if rate > threshold:
            top_values = report.unseen_values[column].most_common(3)
            alerts.append(f"{column}: {rate:.1%} unknown (threshold {threshold:.0%}), "
                          f"top values {top_values}")
    return alerts


@dataclass(frozen=True)
class FraudRateModel:
    """Tiny stand-in model: the training fraud rate for each (column, index)."""
    rates: dict[str, dict[int, float]]

    @classmethod
    def fit(cls, encoded_rows: list[dict[str, int]], labels: list[bool]) -> "FraudRateModel":
        """Fraud rate per encoded index, learned from the training rows."""
        totals: dict[str, Counter[int]] = {}
        frauds: dict[str, Counter[int]] = {}
        for encoded_row, is_fraud in zip(encoded_rows, labels, strict=True):
            for column, index in encoded_row.items():
                totals.setdefault(column, Counter())[index] += 1
                frauds.setdefault(column, Counter())[index] += int(is_fraud)
        rates = {column: {index: frauds[column][index] / count for index, count in counts.items()}
                 for column, counts in totals.items()}
        return cls(rates)

    def score(self, encoded_row: dict[str, int]) -> float:
        """Average fraud rate of the row's categories (an index never trained scores 0)."""
        column_rates = [self.rates[column].get(index, 0.0) for column, index in encoded_row.items()]
        return sum(column_rates) / len(column_rates)


def score_batch(encoder: CategoryEncoder, model: FraudRateModel, rows: list[Row],
                threshold: float = DEFAULT_ALERT_UNKNOWN_RATE
                ) -> tuple[list[float], UnknownReport, list[str]]:
    """Score a whole batch; unknowns never fail it, they are counted and alerted on."""
    report = UnknownReport()
    encoded = encoder.encode(rows, report)
    scores = [model.score(encoded_row) for encoded_row in encoded]
    return scores, report, unknown_rate_alerts(report, threshold)
