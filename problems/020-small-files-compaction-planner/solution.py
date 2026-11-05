"""Plan compaction of the tiny files a streaming job writes into each partition.

A streaming job flushes one small Parquet file per partition every minute, so
a day partition ends up with ~1,440 files of a few MB. Query engines pay a
fixed cost per file (listing, opening, reading footers), so queries slowed
down. The first fix, "rewrite each partition into one file", re-wrote large,
already-healthy files on every run and produced single files far bigger than
the target size. This planner only picks files that are too small, bin-packs
them into outputs close to a target size, and skips groups where rewriting
would not reduce the file count.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

MB = 1024 * 1024
TARGET_FILE_BYTES = 128 * MB
# Files at least this share of the target are "healthy" and never rewritten,
# otherwise every run would rewrite yesterday's output again.
SMALL_FILE_RATIO = 0.75
# Rewriting fewer files than this per partition costs more than it saves.
MIN_FILES_TO_COMPACT = 5


@dataclass(frozen=True)
class DataFile:
    partition: str
    path: str
    size_bytes: int


@dataclass
class CompactionTask:
    """One output file: these inputs are read and written as one file."""
    partition: str
    inputs: list[DataFile] = field(default_factory=list)

    @property
    def output_bytes(self) -> int:
        return sum(data_file.size_bytes for data_file in self.inputs)


def naive_compact_partition(files: list[DataFile]) -> list[CompactionTask]:
    """First attempt: rewrite every file of every partition into a single file."""
    tasks: dict[str, CompactionTask] = {}
    for data_file in files:
        task = tasks.setdefault(data_file.partition, CompactionTask(data_file.partition))
        task.inputs.append(data_file)
    return list(tasks.values())


def _small_files_by_partition(files: list[DataFile], target_bytes: int) -> dict[str, list[DataFile]]:
    """Group files below the small-file threshold by partition."""
    threshold = target_bytes * SMALL_FILE_RATIO
    grouped: dict[str, list[DataFile]] = {}
    for data_file in files:
        if data_file.size_bytes < threshold:
            grouped.setdefault(data_file.partition, []).append(data_file)
    return grouped


def _bin_pack(partition: str, files: list[DataFile], target_bytes: int) -> list[CompactionTask]:
    """First-fit decreasing: place each file (largest first) in the first bin it fits.

    Largest-first leaves small files to fill the gaps, which keeps outputs
    close to the target and the number of outputs low.
    """
    bins: list[CompactionTask] = []
    for data_file in sorted(files, key=lambda item: (-item.size_bytes, item.path)):
        placed = False
        for task in bins:
            if task.output_bytes + data_file.size_bytes <= target_bytes:
                task.inputs.append(data_file)
                placed = True
                break
        if not placed:
            bins.append(CompactionTask(partition, [data_file]))
    return bins


def plan_compaction(files: list[DataFile], target_bytes: int = TARGET_FILE_BYTES) -> list[CompactionTask]:
    """Return rewrite tasks that reduce small files without touching healthy ones."""
    tasks: list[CompactionTask] = []
    for partition, small_files in sorted(_small_files_by_partition(files, target_bytes).items()):
        if len(small_files) < MIN_FILES_TO_COMPACT:
            continue
        for task in _bin_pack(partition, small_files, target_bytes):
            # A one-file "group" would just copy the file: all cost, no benefit.
            if len(task.inputs) > 1:
                tasks.append(task)
    return tasks


def files_after(files: list[DataFile], tasks: list[CompactionTask]) -> int:
    """How many files the table has once the tasks have run."""
    rewritten = sum(len(task.inputs) for task in tasks)
    return len(files) - rewritten + len(tasks)


def bytes_rewritten(tasks: list[CompactionTask]) -> int:
    """Total bytes read and written again: the cost of running the plan."""
    return sum(task.output_bytes for task in tasks)


def simulate_streaming_partition(partition: str, minutes: int, rng: random.Random) -> list[DataFile]:
    """One small file per minute, 1-3 MB each, like a micro-batch sink."""
    files = []
    for minute in range(minutes):
        size = rng.randint(1 * MB, 3 * MB)
        files.append(DataFile(partition, f"{partition}/part-{minute:05d}.parquet", size))
    return files
