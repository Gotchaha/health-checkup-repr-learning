"""
Profiled version of PersonBatchSampler with timing instrumentation.

Goals:
- Keep sampling semantics IDENTICAL to the original samplers.
- Add minimal, low-overhead instrumentation for research profiling.
- Unify conventions with dataset/collate: timers vs. counters, and warmup gating.

Notes:
- Timers are recorded in SECONDS internally (perf_counter deltas).
  Summaries convert to MILLISECONDS for human readability.
- Counters remain in their natural units (dimensionless).
"""

import random
import time
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Iterator, Union
from collections import defaultdict


class ProfiledPersonBatchSampler:
    """
    Person-aware batch sampler with profiling.
    Tracks: manifest read, index map build, grouping, shuffle, batch generation, etc.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        enable_profiling: bool = True,
        start_logging_after_batches: int = 0,  # unified warmup gate (in batches), like collate
    ):
        self.manifest_path = Path(manifest_path)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.enable_profiling = bool(enable_profiling)

        # --- unified warmup gate (batch-based) ---
        self.start_after_batches = int(start_logging_after_batches or 0)
        self._batches_seen = 0  # increases across iterations; used for gating

        # --- profiling containers ---
        # timers: seconds; counters: native units
        self.timers = defaultdict(list)    # e.g., 'read_manifest', 'group_persons', 'batch_generation', ...
        self.counters = defaultdict(list)  # e.g., 'batches_per_iteration'

        # --- build person index map (one-time costs) ---
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        t0_all = time.perf_counter()
        t0 = time.perf_counter()
        df = pd.read_parquet(self.manifest_path)
        if self.enable_profiling:
            self.timers['read_manifest'].append(time.perf_counter() - t0)

        # Assumes manifest is already sorted by person_id (same assumption as original)
        df = df.reset_index(drop=True)
        if 'person_id' not in df.columns:
            raise ValueError(f"Manifest must contain 'person_id' column: {self.manifest_path}")

        t0 = time.perf_counter()
        self.persons = self._build_person_index_map(df)
        if self.enable_profiling:
            self.timers['group_persons'].append(time.perf_counter() - t0)
            self.timers['build_index_map'].append(time.perf_counter() - t0_all)

        # Cache total exams
        self.total_exams = sum(len(exam_indices) for _, exam_indices in self.persons)

    @staticmethod
    def _build_person_index_map(df) -> List[Tuple[str, List[int]]]:
        """
        Build person -> list[exam_idx] mapping by scanning (linear pass).
        The manifest is expected to be grouped by person_id already.
        """
        person_groups = []
        current_person = None
        current_indices = []

        for idx, row in df.iterrows():
            person_id = row['person_id']
            if person_id != current_person:
                if current_indices:
                    person_groups.append((current_person, current_indices))
                current_person = person_id
                current_indices = [idx]
            else:
                current_indices.append(idx)

        if current_indices:
            person_groups.append((current_person, current_indices))
        return person_groups

    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate batches. Profiling timers for iteration are recorded only after
        warmup gate (self._batches_seen >= self.start_after_batches).
        One-time costs (manifest read / index build) are already recorded in __init__.
        """
        iteration_start = time.perf_counter()
        effective_epoch = False  # mark if this iteration produced any post-gate batches

        # Shuffle persons if requested
        persons = self.persons.copy()
        if self.shuffle:
            t0 = time.perf_counter()
            random.shuffle(persons)
            if self.enable_profiling and (self._batches_seen >= self.start_after_batches):
                self.timers['shuffle_persons'].append(time.perf_counter() - t0)

        person_idx = 0
        person_exam_offset = 0
        batch_count = 0

        while person_idx < len(persons):
            batch_start = time.perf_counter()
            current_batch = []

            # Fill current batch by taking contiguous exams per person
            while len(current_batch) < self.batch_size and person_idx < len(persons):
                person_id, exam_indices = persons[person_idx]
                remaining_exams = exam_indices[person_exam_offset:]

                slots_available = self.batch_size - len(current_batch)
                if len(remaining_exams) <= slots_available:
                    current_batch.extend(remaining_exams)
                    person_idx += 1
                    person_exam_offset = 0
                else:
                    current_batch.extend(remaining_exams[:slots_available])
                    person_exam_offset += slots_available

            # Yield batch (full or partial depending on drop_last)
            if len(current_batch) == self.batch_size:
                # timers for batch generation after warmup gate
                if self.enable_profiling and (self._batches_seen >= self.start_after_batches):
                    self.timers['batch_generation'].append(time.perf_counter() - batch_start)
                    effective_epoch = True
                batch_count += 1
                self._batches_seen += 1
                yield current_batch

            elif len(current_batch) > 0 and not self.drop_last:
                if self.enable_profiling and (self._batches_seen >= self.start_after_batches):
                    self.timers['batch_generation'].append(time.perf_counter() - batch_start)
                    effective_epoch = True
                batch_count += 1
                self._batches_seen += 1
                yield current_batch

        # End-of-iteration metrics (only when the iteration included post-gate batches)
        if self.enable_profiling and effective_epoch:
            self.timers['full_iteration'].append(time.perf_counter() - iteration_start)
            self.counters['batches_per_iteration'].append(float(batch_count))

    def __len__(self) -> int:
        """Number of batches in a finite pass (human used by logs/UI)."""
        if self.drop_last:
            return self.total_exams // self.batch_size
        else:
            return (self.total_exams + self.batch_size - 1) // self.batch_size

    def __repr__(self) -> str:
        """String representation with key parameters (kept identical to original)."""
        return (f"ProfiledPersonBatchSampler(persons={len(self.persons)}, "
                f"total_exams={self.total_exams}, batch_size={self.batch_size}, "
                f"shuffle={self.shuffle}, drop_last={self.drop_last})")

    def get_stats(self) -> dict:
        """
        Get sampler statistics (finite sampler) including profiling summaries.
        - Structural fields match the original sampler.
        - Profiling fields: timers summarized in milliseconds; counters kept in native units.
        """
        import numpy as np

        stats = {
            'total_persons': len(self.persons),
            'total_exams': self.total_exams,
            'avg_exams_per_person': (self.total_exams / len(self.persons)) if self.persons else 0.0,
            'batch_size': self.batch_size,
            'batches_per_epoch': len(self),
            'shuffle': self.shuffle,
            'drop_last': self.drop_last,
        }

        if self.enable_profiling:
            profiling = {}

            # Timers → ms summary
            if self.timers:
                tstats = {}
                for key, values in self.timers.items():
                    if not values:
                        continue
                    arr = np.asarray(values, dtype=float)
                    tstats[f'{key}_mean_ms'] = float(arr.mean() * 1000.0)
                    tstats[f'{key}_std_ms']  = float(arr.std()  * 1000.0)
                    tstats[f'{key}_p50_ms']  = float(np.percentile(arr, 50) * 1000.0)
                    tstats[f'{key}_p95_ms']  = float(np.percentile(arr, 95) * 1000.0)
                    tstats[f'{key}_p99_ms']  = float(np.percentile(arr, 99) * 1000.0)
                    tstats[f'{key}_count']   = int(arr.size)
                profiling['timers'] = tstats

            # Counters → native units summary
            if self.counters:
                cstats = {}
                for key, values in self.counters.items():
                    arr = np.asarray(values, dtype=float)
                    if arr.size == 0:
                        continue
                    cstats[f'{key}_sum']   = float(arr.sum())
                    cstats[f'{key}_mean']  = float(arr.mean())
                    cstats[f'{key}_p50']   = float(np.percentile(arr, 50))
                    cstats[f'{key}_p95']   = float(np.percentile(arr, 95))
                    cstats[f'{key}_p99']   = float(np.percentile(arr, 99))
                    cstats[f'{key}_max']   = float(arr.max())
                    cstats[f'{key}_count'] = int(arr.size)
                profiling['counters'] = cstats

            if profiling:
                stats['profiling'] = profiling

        return stats


class ProfiledInfinitePersonBatchSampler(ProfiledPersonBatchSampler):
    """
    Infinite sampler variant with profiling.
    Extends finite sampler and preserves identical batching semantics.
    """

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield batches forever. Epoch-level timers/counters are recorded
        only if at least one post-gate batch occurs in that epoch.
        """
        self.infinite = True
        iteration_count = 0

        while True:
            epoch_start = time.perf_counter()
            batch_count_in_epoch = 0
            effective_epoch = False

            for batch in super().__iter__():
                batch_count_in_epoch += 1
                # super().__iter__ already updates _batches_seen and gates timers
                if (not effective_epoch) and (self.enable_profiling and self._batches_seen > self.start_after_batches):
                    effective_epoch = True
                yield batch

            if self.enable_profiling and effective_epoch:
                self.timers['epoch_time'].append(time.perf_counter() - epoch_start)
                self.counters['batches_per_epoch_actual'].append(float(batch_count_in_epoch))
                iteration_count += 1
                self.counters['total_epochs'].append(float(iteration_count))

    def __len__(self) -> int:
        """Return a very large number for infinite sampler (not practically used)."""
        return 2**63 - 1

    def __repr__(self) -> str:
        """String representation (explicitly show profiling flag for clarity)."""
        return (f"ProfiledInfinitePersonBatchSampler(persons={len(self.persons)}, "
                f"total_exams={self.total_exams}, batch_size={self.batch_size}, "
                f"shuffle={self.shuffle}, drop_last={self.drop_last}, "
                f"profiling={self.enable_profiling})")

    def get_stats(self) -> dict:
        """
        Return sampler statistics with human-friendly infinity and a flag.

        - Keeps parent stats & profiling summaries.
        - Replaces 'batches_per_epoch' with '∞' for readability.
        - Adds 'infinite': True.
        """
        stats = super().get_stats()
        stats['batches_per_epoch'] = '∞'
        stats['infinite'] = True
        return stats
