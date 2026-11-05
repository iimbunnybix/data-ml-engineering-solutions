"""Training/serving skew: one feature definition, plus a parity check.

The original bug: the fraud model's features were computed twice, by two
teams, in two languages. The offline training job (SQL) and the online scoring
service (a hand-port in the API) disagreed on details nobody had written down:
whether refunded orders count, whether the 30-day window includes "now", and
what to return for a customer with no history. The model was trained on one
definition and served with another, so live scores drifted from offline
evaluation without any data change.

Fix:
1. ``compute_features`` is the single source of truth. Both the offline
   training-set builder and the online scorer call it.
2. ``ParityChecker`` compares features logged at serving time against an
   offline recomputation for the same (customer, timestamp) and reports the
   mismatch rate per feature, so any future divergence is caught before it
   reaches a model.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

WINDOW_DAYS = 30
REFUND_WINDOW_DAYS = 90
KNOWN_STATUSES = frozenset({"paid", "refunded"})


@dataclass(frozen=True)
class Order:
    ts: float          # days since epoch (fractional)
    amount: float
    status: str        # "paid" or "refunded" (the order's *current* status)
    refunded_at: float | None = None  # when the refund happened, if known

    def __post_init__(self) -> None:
        # WHY: an unexpected status ("PAID", "chargeback") would otherwise be
        # silently dropped from every feature on BOTH paths, so parity would
        # still report 0% mismatch. Fail loudly at the boundary instead.
        if self.status not in KNOWN_STATUSES:
            raise ValueError(f"unknown order status {self.status!r}; expected one of {sorted(KNOWN_STATUSES)}")


def status_as_of(order: Order, as_of: float) -> str:
    """The order's status as it was just before ``as_of``.

    ``status`` is the current value, so a refund that happens after ``as_of``
    must not be visible when rebuilding features for ``as_of`` (that would leak
    the future into training and break parity replays). When the refund time
    is known, it decides; otherwise fall back to the stored status.
    """
    if order.refunded_at is not None:
        return "refunded" if order.refunded_at < as_of else "paid"
    return order.status


FEATURES = ("order_count_30d", "avg_order_value_30d", "refund_rate_90d")


# --- the two legacy implementations that drifted apart --------------------
def offline_features_legacy(orders: list[Order], as_of: float) -> dict:
    """Port of the training SQL: half-open window [as_of-30, as_of), paid only, empty -> 0."""
    paid = [o for o in orders if as_of - WINDOW_DAYS <= o.ts < as_of and o.status == "paid"]
    recent = [o for o in orders if as_of - REFUND_WINDOW_DAYS <= o.ts < as_of]
    return {
        "order_count_30d": float(len(paid)),
        "avg_order_value_30d": sum(o.amount for o in paid) / len(paid) if paid else 0.0,
        "refund_rate_90d": (sum(o.status == "refunded" for o in recent) / len(recent)) if recent else 0.0,
    }


def online_features_legacy(orders: list[Order], now: float) -> dict:
    """The API's hand-port: closed window, refunds included, empty -> global mean."""
    window = [o for o in orders if now - WINDOW_DAYS <= o.ts <= now]
    recent = [o for o in orders if now - REFUND_WINDOW_DAYS <= o.ts <= now]
    return {
        "order_count_30d": float(len(window)),
        "avg_order_value_30d": sum(o.amount for o in window) / len(window) if window else 42.0,
        "refund_rate_90d": (sum(o.status == "refunded" for o in recent) / len(recent)) if recent else 0.0,
    }


# --- the fix: one definition used by both paths ----------------------------
def compute_features(orders: list[Order], as_of: float) -> dict:
    """Canonical feature definitions.

    * Only events strictly before ``as_of`` are visible (no peeking at the
      event being scored, which also keeps training point-in-time correct).
    * ``order_count_30d`` / ``avg_order_value_30d`` count *paid* orders.
    * Refunds count only if they happened before ``as_of`` (``refunded_at``).
    * No history -> 0.0 (the model also sees ``order_count_30d == 0``).
    """
    paid = [o for o in orders
            if as_of - WINDOW_DAYS <= o.ts < as_of and status_as_of(o, as_of) == "paid"]
    recent = [o for o in orders if as_of - REFUND_WINDOW_DAYS <= o.ts < as_of]
    refunded = [o for o in recent if status_as_of(o, as_of) == "refunded"]
    return {
        "order_count_30d": float(len(paid)),
        "avg_order_value_30d": round(sum(o.amount for o in paid) / len(paid), 6) if paid else 0.0,
        "refund_rate_90d": round(len(refunded) / len(recent), 6) if recent else 0.0,
    }


def build_training_rows(history: dict[str, list[Order]], labels: list[tuple[str, float, int]]) -> list[dict]:
    """Offline path: one row per labelled (customer, as_of, label)."""
    return [{"customer": c, "as_of": t, "label": y, **compute_features(history.get(c, []), t)}
            for c, t, y in labels]


def serve_features(history: dict[str, list[Order]], customer: str, now: float) -> dict:
    """Online path: identical function, identical semantics."""
    return compute_features(history.get(customer, []), now)


# --- guard rail --------------------------------------------------------------
class ParityChecker:
    """Accumulates online-vs-offline comparisons and reports per-feature mismatch rates."""

    def __init__(self, rel_tol: float = 1e-6, abs_tol: float = 1e-9):
        self.rel_tol, self.abs_tol = rel_tol, abs_tol
        self.checked: dict[str, int] = defaultdict(int)
        self.mismatched: dict[str, int] = defaultdict(int)
        self.examples: dict[str, list] = defaultdict(list)

    def _equal(self, a, b) -> bool:
        if a is None or b is None:
            return a is b
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        try:
            return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        except TypeError:
            return a == b

    def compare(self, key, online: dict, offline: dict) -> None:
        for name in sorted(set(online) | set(offline)):
            self.checked[name] += 1
            if name not in online or name not in offline or not self._equal(online[name], offline[name]):
                self.mismatched[name] += 1
                if len(self.examples[name]) < 5:
                    self.examples[name].append((key, online.get(name), offline.get(name)))

    def report(self) -> dict[str, float]:
        return {n: self.mismatched[n] / self.checked[n] for n in sorted(self.checked)}

    def failing(self, max_rate: float = 0.001) -> list[str]:
        """Features whose mismatch rate exceeds ``max_rate``: block the model release."""
        return [n for n, r in self.report().items() if r > max_rate]
