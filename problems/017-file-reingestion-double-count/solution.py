"""Exactly-once loading of landed files with a checksum manifest.

A nightly job loads CSV files from a landing folder into a warehouse table and
then moves each file to an archive folder. When the job crashed after loading a
file but before archiving it, the retry saw the file again and loaded its rows
a second time, so the table double-counted sales. This module keeps a manifest
of loaded files keyed by (path, SHA-256 checksum) and writes the rows and the
manifest entry in ONE warehouse transaction, so a retry can always tell which
files are already in the table.
"""
from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field

# Outcomes reported for each file, so operators can see why a file was skipped.
LOADED = "loaded"
ALREADY_LOADED = "already_loaded"
DUPLICATE_CONTENT = "duplicate_content"
CHANGED_AFTER_LOAD = "changed_after_load"


class SimulatedCrash(Exception):
    """Raised by the fake landing zone to imitate the job dying mid-run."""


@dataclass
class LandingZone:
    """A folder of landed files. `crash_on_archive` imitates a crash after loading."""

    files: dict[str, bytes]
    archived: list[str] = field(default_factory=list)
    crash_on_archive: str | None = None

    def list_pending(self) -> list[str]:
        """Return file paths still waiting in the landing folder, in name order."""
        return sorted(self.files)

    def archive(self, path: str) -> None:
        """Move a file out of the landing folder (may crash once, on purpose)."""
        if path == self.crash_on_archive:
            # Crash only once, like a pod being killed during the move.
            self.crash_on_archive = None
            raise SimulatedCrash(f"killed while archiving {path}")
        del self.files[path]
        self.archived.append(path)


@dataclass
class Warehouse:
    """Fake warehouse with a sales table and a manifest table."""

    sales_rows: list[dict] = field(default_factory=list)
    # (path, checksum) -> number of rows loaded from that file
    manifest: dict[tuple[str, str], int] = field(default_factory=dict)

    def append_rows(self, rows: list[dict]) -> None:
        """Plain append with no record of where the rows came from."""
        self.sales_rows.extend(rows)

    def commit_file(self, path: str, checksum: str, rows: list[dict]) -> None:
        """Insert rows and the manifest entry together (one transaction).

        Real warehouses do this with BEGIN ... COMMIT or a MERGE; either both
        writes become visible or neither does.
        """
        self.sales_rows.extend(rows)
        self.manifest[(path, checksum)] = len(rows)

    def total_amount(self) -> int:
        """Sum of the amount column, the number finance reads."""
        total = 0
        for row in self.sales_rows:
            total += row["amount"]
        return total


def parse_rows(content: bytes) -> list[dict]:
    """Parse a sales CSV (order_id, amount) into typed rows."""
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    rows = []
    for record in reader:
        rows.append({"order_id": record["order_id"], "amount": int(record["amount"])})
    return rows


def file_checksum(content: bytes) -> str:
    """SHA-256 of the file bytes; identifies the exact content that was loaded."""
    return hashlib.sha256(content).hexdigest()


def naive_ingest(zone: LandingZone, warehouse: Warehouse) -> None:
    """Original job: load, then archive. 'Still in the folder' meant 'not loaded'."""
    for path in zone.list_pending():
        warehouse.append_rows(parse_rows(zone.files[path]))
        zone.archive(path)


def classify_file(path: str, checksum: str, warehouse: Warehouse) -> str:
    """Decide whether a file should be loaded, using only the manifest."""
    if (path, checksum) in warehouse.manifest:
        return ALREADY_LOADED
    for loaded_path, loaded_checksum in warehouse.manifest:
        # Same bytes under a new name: the vendor re-uploaded the same file.
        if loaded_checksum == checksum:
            return DUPLICATE_CONTENT
        # Same name, different bytes: a correction. Appending would double
        # count, so a human must decide whether to replace the old rows.
        if loaded_path == path:
            return CHANGED_AFTER_LOAD
    return LOADED


def exactly_once_ingest(zone: LandingZone, warehouse: Warehouse) -> dict[str, str]:
    """Load each new file once; safe to re-run after a crash at any point.

    Returns path -> outcome for every file seen in this run.
    """
    outcomes: dict[str, str] = {}
    for path in zone.list_pending():
        content = zone.files[path]
        checksum = file_checksum(content)
        outcome = classify_file(path, checksum, warehouse)
        if outcome == LOADED:
            warehouse.commit_file(path, checksum, parse_rows(content))
        outcomes[path] = outcome
        # Changed files stay in the landing folder for a human to review.
        if outcome != CHANGED_AFTER_LOAD:
            zone.archive(path)
    return outcomes


def run_until_success(job, zone: LandingZone, warehouse: Warehouse, max_attempts: int = 3):
    """Imitate the scheduler: retry the whole job when it crashes."""
    for attempt in range(1, max_attempts + 1):
        try:
            return job(zone, warehouse)
        except SimulatedCrash:
            if attempt == max_attempts:
                raise
    return None
