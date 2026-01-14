# src/downstream/grade5/repr_datamodule.py

"""
Datamodule utilities for Grade5 training using exported representations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch


class ReprGrade5Dataset:
    """
    Dataset for Grade5 training from exported post_causal embeddings.
    """

    def __init__(self, repr_path: Union[str, Path]) -> None:
        self.repr_path = Path(repr_path)
        if not self.repr_path.exists():
            raise FileNotFoundError(f"Representation file not found: {self.repr_path}")
        self.df = pd.read_parquet(self.repr_path)
        self._validate_schema()

    def _validate_schema(self) -> None:
        required = {
            "exam_id",
            "person_id",
            "ExamDate",
            "post_causal_emb",
            "split",
            "grade5",
            "is_grade5_valid",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Representation file missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "exam_id": row["exam_id"],
            "person_id": row["person_id"],
            "ExamDate": row["ExamDate"],
            "split": row["split"],
            "grade5": row["grade5"],
            "is_grade5_valid": bool(row["is_grade5_valid"]),
            "post_causal_emb": row["post_causal_emb"],
        }


class ReprGrade5PersonBatchSampler:
    """
    Person-aware sampler for representation datasets with split filtering.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        mode: str = "train",
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got '{mode}'")
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._skip_summary_logged = False
        self.persons = self._build_person_index_map()

    def _build_person_index_map(self) -> List[Tuple[str, List[int]]]:
        if "person_id" not in self.df.columns or "split" not in self.df.columns:
            raise ValueError("Representation file must contain person_id and split columns")

        person_groups: List[Tuple[str, List[int]]] = []
        for person_id, person_group in self.df.groupby("person_id", sort=False):
            splits = person_group["split"].astype(str).str.lower().unique().tolist()
            if len(splits) != 1:
                raise ValueError(f"Person {person_id} has multiple splits: {sorted(splits)}")
            if splits[0] != self.mode:
                continue
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
                _, exam_indices = persons[person_idx]
                slots_available = self.batch_size - len(current_batch)

                if len(exam_indices) <= slots_available:
                    current_batch.extend(exam_indices)
                    person_idx += 1
                else:
                    break

            if len(current_batch) == 0:
                if not self._skip_summary_logged:
                    exams = len(persons[person_idx][1])
                    skipped_persons += 1
                    skipped_total_exams += exams
                    skipped_max_exams = max(skipped_max_exams, exams)
                person_idx += 1
                continue

            if self.drop_last and len(current_batch) < self.batch_size:
                break

            yield current_batch

        if (not self._skip_summary_logged) and skipped_persons > 0:
            print(
                f"Skipping {skipped_persons} persons in mode='{self.mode}' "
                f"(total_exams={skipped_total_exams}, max_exams={skipped_max_exams}, "
                f"batch_size={self.batch_size})"
            )
            self._skip_summary_logged = True


def _to_tensor_embeddings(values: Sequence[Any]) -> torch.Tensor:
    arr = np.stack([np.asarray(v) for v in values])
    return torch.from_numpy(arr)


def _pack_by_segment(
    values: torch.Tensor,
    segment_lengths: List[int],
    pad_value: Union[int, float],
) -> torch.Tensor:
    if not segment_lengths:
        return values.new_empty((0, 0, values.size(-1)))
    total = sum(segment_lengths)
    if total != values.size(0):
        raise ValueError(
            f"Segment lengths sum ({total}) does not match values ({values.size(0)})"
        )
    e_max = max(segment_lengths)
    d = values.size(-1)
    packed = values.new_full((len(segment_lengths), e_max, d), pad_value)
    start = 0
    for i, length in enumerate(segment_lengths):
        end = start + length
        packed[i, :length] = values[start:end]
        start = end
    return packed


def _pack_mask(segment_lengths: List[int]) -> torch.Tensor:
    if not segment_lengths:
        return torch.zeros((0, 0), dtype=torch.bool)
    e_max = max(segment_lengths)
    mask = torch.zeros((len(segment_lengths), e_max), dtype=torch.bool)
    for i, length in enumerate(segment_lengths):
        mask[i, :length] = True
    return mask


def repr_grade5_collate_fn(
    batch: List[Dict[str, Any]],
    label_order: Sequence[str],
    ignore_index: int,
) -> Dict[str, Any]:
    if not batch:
        return {}

    label_to_id = {label: idx for idx, label in enumerate(label_order)}

    embeddings = _to_tensor_embeddings([b["post_causal_emb"] for b in batch]).float()
    grade5_values = [b.get("grade5") for b in batch]
    grade5_valid = [bool(b.get("is_grade5_valid", False)) for b in batch]

    targets_flat = torch.full((len(batch),), ignore_index, dtype=torch.long)
    label_mask_flat = torch.zeros((len(batch),), dtype=torch.bool)
    for i, (label, is_valid) in enumerate(zip(grade5_values, grade5_valid)):
        if is_valid and label in label_to_id:
            targets_flat[i] = label_to_id[label]
            label_mask_flat[i] = True

    person_ids = [b["person_id"] for b in batch]
    segment_lengths: List[int] = []
    current_person = None
    count = 0
    for pid in person_ids:
        if pid != current_person:
            if count > 0:
                segment_lengths.append(count)
            current_person = pid
            count = 1
        else:
            count += 1
    if count > 0:
        segment_lengths.append(count)

    embeddings_packed = _pack_by_segment(embeddings, segment_lengths, pad_value=0.0)
    attention_mask = _pack_mask(segment_lengths)

    targets_packed = _pack_by_segment(
        targets_flat.unsqueeze(-1).float(),
        segment_lengths,
        pad_value=float(ignore_index),
    ).squeeze(-1).long()
    label_mask_packed = _pack_by_segment(
        label_mask_flat.unsqueeze(-1).float(),
        segment_lengths,
        pad_value=0.0,
    ).squeeze(-1).bool()

    return {
        "embeddings": embeddings_packed,
        "attention_mask": attention_mask,
        "grade5_targets": targets_packed,
        "grade5_label_mask": label_mask_packed,
        "segment_lengths": segment_lengths,
    }


def create_repr_grade5_data_loaders(
    config: Dict[str, Any],
    debug: bool = False,
):
    data_cfg = config["data"]
    datamodule_cfg = config["datamodule"]
    label_cfg = datamodule_cfg["label_processing"]
    label_order = label_cfg["label_order"]
    ignore_index = label_cfg.get("ignore_index", -100)

    repr_path = data_cfg.get("repr_path")
    if not repr_path:
        raise ValueError("data.repr_path must be set for repr training")

    dataset = ReprGrade5Dataset(repr_path)

    batch_size = datamodule_cfg["batch_size"]
    drop_last = datamodule_cfg.get("drop_last", False)

    train_sampler = ReprGrade5PersonBatchSampler(
        df=dataset.df,
        batch_size=batch_size,
        mode="train",
        shuffle=datamodule_cfg["sampler"]["shuffle_train"],
        drop_last=drop_last,
    )
    val_sampler = ReprGrade5PersonBatchSampler(
        df=dataset.df,
        batch_size=batch_size,
        mode="val",
        shuffle=datamodule_cfg["sampler"]["shuffle_val"],
        drop_last=False,
    )
    test_sampler = ReprGrade5PersonBatchSampler(
        df=dataset.df,
        batch_size=batch_size,
        mode="test",
        shuffle=datamodule_cfg["sampler"]["shuffle_test"],
        drop_last=False,
    )

    def _collate(batch: List[Dict[str, Any]]):
        return repr_grade5_collate_fn(batch, label_order=label_order, ignore_index=ignore_index)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_sampler,
        collate_fn=_collate,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=val_sampler,
        collate_fn=_collate,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=test_sampler,
        collate_fn=_collate,
        num_workers=0,
        pin_memory=False,
    )

    if debug:
        from itertools import islice

        debug_cfg = config.get("debug", {})
        train_loader = list(islice(train_loader, debug_cfg.get("limit_train_batches", 20)))
        val_loader = list(islice(val_loader, debug_cfg.get("limit_val_batches", 10)))
        test_loader = list(islice(test_loader, debug_cfg.get("limit_test_batches", 5)))

    return train_loader, val_loader, test_loader
