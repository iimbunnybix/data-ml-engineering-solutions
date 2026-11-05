"""Stop an online model from silently scoring with hours-old features.

A delivery-ETA model reads two live features per zone (pending orders and
active couriers) from an online feature store that a streaming job refreshes
every minute. When that job stalled before the dinner peak, the store kept
returning the last values it had, from the quiet afternoon, so the model
promised short ETAs all evening and nothing alerted.

Fix: every feature has a freshness policy (maximum age plus a fallback
value). A read checks each value's write timestamp against a clock; a value
that is too old (or missing) is replaced by the fallback and counted in a
staleness metric, so the model degrades predictably and on-call is paged.
"""
from __future__ import annotations

from dataclasses import dataclass

SECONDS_PER_MINUTE = 60.0
PENDING_ORDERS = "pending_orders_5m"
ACTIVE_COURIERS = "active_couriers_5m"
ETA_FEATURES = (PENDING_ORDERS, ACTIVE_COURIERS)
BASE_ETA_MINUTES = 15.0
MINUTES_PER_ORDER_PER_COURIER = 10.0
STALE_RATE_ALERT_THRESHOLD = 0.05


@dataclass(frozen=True)
class FeatureValue:
    value: float
    written_at: float  # seconds on the store's clock when the pipeline wrote it


@dataclass(frozen=True)
class FreshnessPolicy:
    max_age_seconds: float
    fallback: float  # safe value used when the stored one is too old or missing


class FeatureStore:
    """A minimal online feature store: latest value per (entity, feature)."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], FeatureValue] = {}

    def write(self, entity: str, feature: str, value: float, written_at: float) -> None:
        """Store the latest value together with the time it was computed."""
        self._values[(entity, feature)] = FeatureValue(value, written_at)

    def get(self, entity: str, feature: str) -> FeatureValue | None:
        """Return the latest stored value, or None if it was never written."""
        return self._values.get((entity, feature))


class StalenessMetrics:
    """Counts reads and stale reads per feature, like a pair of Prometheus counters."""

    def __init__(self) -> None:
        self.reads: dict[str, int] = {}
        self.stale_reads: dict[str, int] = {}

    def record(self, feature: str, was_stale: bool) -> None:
        """Count one read of ``feature``."""
        self.reads[feature] = self.reads.get(feature, 0) + 1
        if was_stale:
            self.stale_reads[feature] = self.stale_reads.get(feature, 0) + 1

    def stale_rate(self, feature: str) -> float:
        """Share of reads of ``feature`` that had to use the fallback."""
        reads = self.reads.get(feature, 0)
        if reads == 0:
            return 0.0
        return self.stale_reads.get(feature, 0) / reads

    def alerting_features(self, threshold: float = STALE_RATE_ALERT_THRESHOLD) -> list[str]:
        """Features whose stale rate is above ``threshold``: page the pipeline owner."""
        alerting = []
        for feature in sorted(self.reads):
            if self.stale_rate(feature) > threshold:
                alerting.append(feature)
        return alerting


# --- the original read path ---------------------------------------------------
def naive_read_features(store: FeatureStore, entity: str, features: tuple[str, ...]) -> dict[str, float]:
    """Original: return whatever is stored, however old it is (missing -> 0.0)."""
    values = {}
    for feature in features:
        stored = store.get(entity, feature)
        values[feature] = stored.value if stored is not None else 0.0
    return values


# --- the fix -------------------------------------------------------------------
def is_fresh(stored: FeatureValue | None, policy: FreshnessPolicy, now: float) -> bool:
    """True if a value exists and is no older than the policy allows."""
    if stored is None:
        return False
    age = now - stored.written_at
    return age <= policy.max_age_seconds


def read_fresh_features(store: FeatureStore, entity: str, policies: dict[str, FreshnessPolicy],
                        now: float, metrics: StalenessMetrics) -> dict[str, float]:
    """Read every feature in ``policies``, replacing stale or missing values by their fallback.

    WHY: the store cannot know that the job feeding it stopped, so freshness
    must be checked at read time against the write timestamp.
    """
    values = {}
    for feature in sorted(policies):
        policy = policies[feature]
        stored = store.get(entity, feature)
        fresh = is_fresh(stored, policy, now)
        metrics.record(feature, was_stale=not fresh)
        values[feature] = stored.value if fresh else policy.fallback
    return values


def predict_eta_minutes(features: dict[str, float]) -> float:
    """The ETA model: base time plus a load term (pending orders per courier)."""
    couriers = max(features[ACTIVE_COURIERS], 1.0)  # avoid division by zero
    load = features[PENDING_ORDERS] / couriers
    return BASE_ETA_MINUTES + MINUTES_PER_ORDER_PER_COURIER * load
