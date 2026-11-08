"""Ingest payment events without silently skipping rows that fail to parse.

The original ingestion job wrapped each row in ``try/except: continue``. When
an upstream app release changed the ``amount`` field from a JSON number to a
formatted string ("1,299.00"), about 30% of rows started failing to parse and
simply disappeared, while the job still reported success. This module parses
each row into either an accepted event or a quarantined reject (raw text,
line number and reason), then applies a reject-rate gate that fails the whole
batch above a threshold. Small batches use an absolute limit instead, because
a rate over a handful of rows is mostly noise.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

MAX_REJECT_RATE = 0.01          # fail the batch if more than 1% of rows are rejected
MIN_ROWS_FOR_RATE_GATE = 200    # below this, a rate is too noisy to judge
MAX_REJECTS_IN_SMALL_BATCH = 5  # absolute limit used for small batches
REQUIRED_FIELDS = ("event_id", "amount", "currency", "created_at")


class RowRejected(Exception):
    """One row could not be parsed; ``reason`` is a short, countable code."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PaymentEvent:
    """A parsed, valid payment event."""

    event_id: str
    amount: float
    currency: str
    created_at: datetime


@dataclass(frozen=True)
class RejectedRow:
    """A quarantined row: kept, with enough context to debug and replay it."""

    line_number: int
    raw: str
    reason: str


@dataclass
class IngestResult:
    """Everything that came in, split into accepted events and rejects."""

    accepted: list[PaymentEvent] = field(default_factory=list)
    rejected: list[RejectedRow] = field(default_factory=list)

    def total_rows(self) -> int:
        """Rows read from the input; always accepted + rejected."""
        return len(self.accepted) + len(self.rejected)

    def reject_rate(self) -> float:
        """Share of input rows that were rejected (0.0 for an empty batch)."""
        if self.total_rows() == 0:
            return 0.0
        return len(self.rejected) / self.total_rows()

    def reasons(self) -> Counter:
        """How many rows were rejected for each reason."""
        return Counter(row.reason for row in self.rejected)


class BatchRejectedError(Exception):
    """The batch has too many rejects to publish; ``result`` holds the quarantine."""

    def __init__(self, message: str, result: IngestResult):
        super().__init__(message)
        self.result = result


def parse_event(line: str) -> PaymentEvent:
    """Parse one JSON line strictly, raising RowRejected with a reason code."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        raise RowRejected("invalid_json") from None
    if not isinstance(record, dict):
        raise RowRejected("not_an_object")
    for name in REQUIRED_FIELDS:
        if record.get(name) is None:
            raise RowRejected(f"missing_{name}")

    amount = record["amount"]
    # bool is a subclass of int in Python, so exclude it explicitly.
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        raise RowRejected("amount_not_a_number")
    # json.loads accepts NaN and Infinity, which would poison every sum.
    if not math.isfinite(amount):
        raise RowRejected("amount_not_finite")
    if amount < 0:
        raise RowRejected("negative_amount")
    try:
        created_at = datetime.fromisoformat(record["created_at"])
    except (TypeError, ValueError):
        raise RowRejected("bad_created_at") from None
    return PaymentEvent(str(record["event_id"]), float(amount),
                        str(record["currency"]), created_at)


# --- Original job --------------------------------------------------------------

def naive_ingest(lines: list[str]) -> list[PaymentEvent]:
    """What the old job did: skip anything that fails, tell nobody."""
    events = []
    for line in lines:
        try:
            events.append(parse_event(line))
        except Exception:
            continue
    return events


# --- Fixed job -----------------------------------------------------------------

def parse_batch(lines: list[str]) -> IngestResult:
    """Parse every line into accepted events or quarantined rejects."""
    result = IngestResult()
    for line_number, line in enumerate(lines, start=1):
        try:
            result.accepted.append(parse_event(line))
        except RowRejected as rejected:
            result.rejected.append(RejectedRow(line_number, line, rejected.reason))
    return result


def gate_failure(result: IngestResult, max_reject_rate: float,
                 min_rows_for_rate: int, max_small_batch_rejects: int) -> str | None:
    """Return why the batch must fail, or None if it may be published."""
    rejects = len(result.rejected)
    if rejects == 0:
        return None
    if rejects == result.total_rows():
        return f"all {rejects} rows rejected"
    if result.total_rows() < min_rows_for_rate:
        # One bad row in 20 is 5%; judge small batches by count instead.
        if rejects > max_small_batch_rejects:
            return f"{rejects} rejects in a batch of {result.total_rows()} rows"
        return None
    if result.reject_rate() > max_reject_rate:
        return f"reject rate {result.reject_rate():.1%} exceeds {max_reject_rate:.1%}"
    return None


def ingest_batch(lines: list[str], max_reject_rate: float = MAX_REJECT_RATE,
                 min_rows_for_rate: int = MIN_ROWS_FOR_RATE_GATE,
                 max_small_batch_rejects: int = MAX_REJECTS_IN_SMALL_BATCH) -> IngestResult:
    """Fixed job: parse, quarantine rejects, and fail the batch if the gate trips."""
    result = parse_batch(lines)
    failure = gate_failure(result, max_reject_rate, min_rows_for_rate,
                           max_small_batch_rejects)
    if failure is not None:
        top_reasons = result.reasons().most_common(3)
        raise BatchRejectedError(f"{failure}; top reasons {top_reasons}", result)
    return result
