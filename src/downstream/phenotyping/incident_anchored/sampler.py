# src/downstream/phenotyping/incident_anchored/sampler.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Tuple, Union

import pandas as pd

from src.models import PersonBatchSampler

logger = logging.getLogger(__name__)


class IncidentAnchoredPersonBatchSampler(PersonBatchSampler):
    """
    Person-aware sampler for incident-anchored phenotyping.

    - Filters out index visits (is_index==1) before building batches.
    - Keeps complete individuals within a batch (no truncation).
    - Allows variable batch sizes and stops when the next person does not fit.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        batch_size: int,
        shuffle: bool = False,
        index_col: str = "is_index",
    ) -> None:
        self.index_col = index_col
        self._filtered_index_exams = 0
        self._skipped_oversize_persons = 0
        self._skipped_oversize_exams = 0
        self._skip_summary_logged = False
        drop_last = False
        super().__init__(manifest_path, batch_size, shuffle, drop_last)

    def _build_person_index_map(self) -> List[Tuple[str, List[int]]]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        df = pd.read_parquet(self.manifest_path)
        df = df.reset_index(drop=True)

        if "person_id" not in df.columns:
            raise ValueError(f"Manifest file must contain 'person_id' column: {self.manifest_path}")
        if self.index_col not in df.columns:
            raise ValueError(
                f"Manifest file must contain '{self.index_col}' column: {self.manifest_path}"
            )

        index_mask = df[self.index_col].fillna(0).astype(int) != 0
        self._filtered_index_exams = int(index_mask.sum())
        df = df.loc[~index_mask].copy()

        person_groups: List[Tuple[str, List[int]]] = []
        for person_id, person_group in df.groupby("person_id", sort=False):
            exam_indices = person_group.index.tolist()
            if exam_indices:
                person_groups.append((person_id, exam_indices))

        return person_groups

    def __iter__(self) -> Iterator[List[int]]:
        persons = self.persons.copy()
        if self.shuffle:
            import random

            random.shuffle(persons)

        person_idx = 0
        skipped_persons = 0
        skipped_total_exams = 0
        skipped_max_exams = 0

        while person_idx < len(persons):
            current_batch: List[int] = []

            while person_idx < len(persons):
                person_id, exam_indices = persons[person_idx]

                if len(exam_indices) > self.batch_size:
                    skipped_persons += 1
                    skipped_total_exams += len(exam_indices)
                    skipped_max_exams = max(skipped_max_exams, len(exam_indices))
                    person_idx += 1
                    if current_batch:
                        break
                    continue

                slots_available = self.batch_size - len(current_batch)
                if len(exam_indices) <= slots_available:
                    current_batch.extend(exam_indices)
                    person_idx += 1
                else:
                    break

            if not current_batch:
                continue

            yield current_batch

        if skipped_persons > 0 and not self._skip_summary_logged:
            logger.warning(
                "Skipping %d persons due to oversized sequences (total_exams=%d, max_exams=%d, "
                "batch_size=%d)",
                skipped_persons,
                skipped_total_exams,
                skipped_max_exams,
                self.batch_size,
            )
            self._skipped_oversize_persons = skipped_persons
            self._skipped_oversize_exams = skipped_total_exams
            self._skip_summary_logged = True

    def __len__(self) -> int:
        count = 0
        current = 0
        for _, exam_indices in self.persons:
            n_exams = len(exam_indices)
            if n_exams > self.batch_size:
                continue
            if current + n_exams > self.batch_size:
                count += 1
                current = 0
            current += n_exams
        if current > 0:
            count += 1
        return count

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats.update(
            {
                "filtered_index_exams": self._filtered_index_exams,
                "skipped_oversize_persons": self._skipped_oversize_persons,
                "skipped_oversize_exams": self._skipped_oversize_exams,
                "index_col": self.index_col,
            }
        )
        return stats

    def __repr__(self) -> str:
        return (
            "IncidentAnchoredPersonBatchSampler("
            f"persons={len(self.persons)}, total_exams={self.total_exams}, "
            f"batch_size={self.batch_size}, shuffle={self.shuffle}, "
            f"index_col='{self.index_col}')"
        )
