import random
import unittest

from solution import (
    MB,
    MIN_FILES_TO_COMPACT,
    TARGET_FILE_BYTES,
    DataFile,
    bytes_rewritten,
    files_after,
    naive_compact_partition,
    plan_compaction,
    simulate_streaming_partition,
)

MINUTES_PER_DAY = 24 * 60


def one_day_of_streaming() -> list[DataFile]:
    return simulate_streaming_partition("dt=2026-11-04", MINUTES_PER_DAY, random.Random(7))


def compacted_history_plus_new_small_files() -> list[DataFile]:
    """Ten healthy 128 MB files from earlier runs plus an hour of new tiny files."""
    healthy = [DataFile("dt=2026-11-04", f"dt=2026-11-04/compacted-{i}.parquet", TARGET_FILE_BYTES)
               for i in range(10)]
    fresh = simulate_streaming_partition("dt=2026-11-04", 60, random.Random(3))
    return healthy + fresh


class ReproduceProblem(unittest.TestCase):
    def test_streaming_sink_leaves_one_file_per_minute(self):
        self.assertEqual(len(one_day_of_streaming()), MINUTES_PER_DAY)

    def test_naive_compaction_rewrites_healthy_files_every_run(self):
        files = compacted_history_plus_new_small_files()
        tasks = naive_compact_partition(files)
        self.assertGreater(bytes_rewritten(tasks), 10 * TARGET_FILE_BYTES)

    def test_naive_compaction_produces_one_oversized_file(self):
        tasks = naive_compact_partition(one_day_of_streaming())
        self.assertEqual(len(tasks), 1)
        self.assertGreater(tasks[0].output_bytes, 20 * TARGET_FILE_BYTES)


class FixCompactionPlanner(unittest.TestCase):
    def test_day_of_tiny_files_becomes_a_few_target_sized_files(self):
        files = one_day_of_streaming()
        tasks = plan_compaction(files)
        self.assertLess(files_after(files, tasks), 30)

    def test_no_output_exceeds_the_target_size(self):
        for task in plan_compaction(one_day_of_streaming()):
            self.assertLessEqual(task.output_bytes, TARGET_FILE_BYTES)

    def test_every_small_file_is_rewritten_exactly_once(self):
        files = one_day_of_streaming()
        inputs = [f.path for task in plan_compaction(files) for f in task.inputs]
        self.assertEqual(sorted(inputs), sorted(f.path for f in files))

    def test_healthy_files_are_never_rewritten(self):
        files = compacted_history_plus_new_small_files()
        rewritten = {f.path for task in plan_compaction(files) for f in task.inputs}
        self.assertFalse(any("compacted" in path for path in rewritten))

    def test_rewrite_cost_is_only_the_new_small_files(self):
        files = compacted_history_plus_new_small_files()
        small_bytes = sum(f.size_bytes for f in files if "compacted" not in f.path)
        self.assertEqual(bytes_rewritten(plan_compaction(files)), small_bytes)

    def test_partition_with_few_small_files_is_skipped(self):
        files = simulate_streaming_partition("dt=x", MIN_FILES_TO_COMPACT - 1, random.Random(1))
        self.assertEqual(plan_compaction(files), [])

    def test_partitions_are_never_mixed_in_one_output(self):
        files = (simulate_streaming_partition("dt=a", 50, random.Random(1))
                 + simulate_streaming_partition("dt=b", 50, random.Random(2)))
        for task in plan_compaction(files):
            self.assertEqual({f.partition for f in task.inputs}, {task.partition})

    def test_group_that_would_be_a_single_file_is_not_copied(self):
        files = [DataFile("dt=a", f"dt=a/{i}", 90 * MB) for i in range(MIN_FILES_TO_COMPACT)]
        # 90 MB files are "small" (< 96 MB) but no two fit together under 128 MB.
        self.assertEqual(plan_compaction(files), [])

    def test_plan_is_deterministic(self):
        files = one_day_of_streaming()
        first = [[f.path for f in t.inputs] for t in plan_compaction(files)]
        second = [[f.path for f in t.inputs] for t in plan_compaction(list(reversed(files)))]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
