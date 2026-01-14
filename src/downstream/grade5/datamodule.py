# src/downstream/grade5/datamodule.py

"""
Datamodule utilities for Grade5 downstream classification.

Provides a dataset wrapper, person-level sampler, and collate function that
augment the SSL inputs with grade5 labels and masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union
from itertools import islice

import pandas as pd
import torch

from src.models import HealthExamDataset, PersonBatchSampler, collate_exams
from src.models.embedders import CategoricalEmbedder, TextEmbedder


class Grade5Dataset(HealthExamDataset):
    """
    Dataset for Grade5 downstream classification.

    Extends HealthExamDataset and attaches manifest label fields
    (split, grade5, is_grade5_valid) to each sample.
    """

    def __init__(self, manifest_path: Union[str, Path], **kwargs) -> None:
        manifest_path = Path(manifest_path)
        super().__init__(
            split_name="grade5",
            manifest_path=str(manifest_path),
            **kwargs,
        )
        self.manifest_path = manifest_path
        self._validate_manifest_schema()

    def _validate_manifest_schema(self) -> None:
        required = {"exam_id", "person_id", "ExamDate", "split", "grade5", "is_grade5_valid"}
        schema_names = set(self.manifest.schema.names)
        missing = required - schema_names
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        result = super().__getitem__(idx)

        manifest_row = self.manifest.slice(idx, 1).to_pydict()
        result["split"] = manifest_row["split"][0]
        result["grade5"] = manifest_row["grade5"][0]
        result["is_grade5_valid"] = bool(manifest_row["is_grade5_valid"][0])

        return result


class Grade5PersonBatchSampler(PersonBatchSampler):
    """
    Person-aware sampler that keeps complete individuals in a batch.

    Filters persons by split and yields variable-size batches without splitting
    any individual's exam sequence across batches.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        batch_size: int,
        mode: str = "train",
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got '{mode}'")
        self.mode = mode
        super().__init__(manifest_path, batch_size, shuffle, drop_last)
        self._skip_summary_logged = False

    def _build_person_index_map(self) -> List[Tuple[str, List[int]]]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        df = pd.read_parquet(self.manifest_path).reset_index(drop=True)

        required_cols = {"person_id", "split"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        person_groups: List[Tuple[str, List[int]]] = []

        for person_id, person_group in df.groupby("person_id", sort=False):
            exam_indices = person_group.index.tolist()
            splits = person_group["split"].astype(str).tolist()

            unique_splits = {s.lower() for s in splits}
            if len(unique_splits) != 1:
                raise ValueError(
                    f"Person {person_id} has multiple splits: {sorted(unique_splits)}"
                )

            person_split = unique_splits.pop()
            if person_split != self.mode:
                continue

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
                    if exams > skipped_max_exams:
                        skipped_max_exams = exams
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


def _pack_by_segment(
    values: torch.Tensor,
    segment_lengths: List[int],
    pad_value: Union[int, bool],
) -> torch.Tensor:
    if not segment_lengths:
        return values.new_empty((0, 0))

    total = sum(segment_lengths)
    if total != values.size(0):
        raise ValueError(
            f"Segment lengths sum ({total}) does not match values ({values.size(0)})"
        )

    e_max = max(segment_lengths)
    packed = values.new_full((len(segment_lengths), e_max), pad_value)

    start = 0
    for i, length in enumerate(segment_lengths):
        end = start + length
        packed[i, :length] = values[start:end]
        start = end

    return packed


def grade5_collate_fn(
    batch: List[Dict[str, Any]],
    code_embedder: CategoricalEmbedder,
    text_embedder: TextEmbedder,
    config: Dict[str, Any],
    mode: str = "train",
    device: str = "cpu",
) -> Dict[str, Any]:
    if not batch:
        return {}

    outputs = collate_exams(
        batch=batch,
        code_embedder=code_embedder,
        text_embedder=text_embedder,
        config=config,
        device=device,
    )

    label_cfg = config.get("datamodule", {}).get("label_processing", {})
    label_order = label_cfg.get("label_order", [])
    if not label_order:
        raise ValueError("label_order must be provided in config['datamodule']['label_processing']")
    label_to_id = {label: idx for idx, label in enumerate(label_order)}
    ignore_index = label_cfg.get("ignore_index", -100)

    grade5_values = [sample.get("grade5") for sample in batch]
    grade5_valid = [bool(sample.get("is_grade5_valid", False)) for sample in batch]

    targets_flat = torch.full(
        (len(batch),), ignore_index, dtype=torch.long, device=device
    )
    label_mask_flat = torch.zeros((len(batch),), dtype=torch.bool, device=device)

    for i, (label, is_valid) in enumerate(zip(grade5_values, grade5_valid)):
        if is_valid and label in label_to_id:
            targets_flat[i] = label_to_id[label]
            label_mask_flat[i] = True

    segment_lengths = outputs["segment_lengths"]
    targets_packed = _pack_by_segment(targets_flat, segment_lengths, ignore_index)
    label_mask_packed = _pack_by_segment(label_mask_flat, segment_lengths, False)

    outputs["grade5_targets"] = targets_packed
    outputs["grade5_label_mask"] = label_mask_packed
    outputs["grade5_targets_flat"] = targets_flat
    outputs["grade5_label_mask_flat"] = label_mask_flat

    return outputs


def create_grade5_data_loaders(
    config: Dict[str, Any],
    embedders,
    debug: bool = False,
):
    data_config = config["data"]
    datamodule_config = config["datamodule"]

    dataset = Grade5Dataset(
        manifest_path=data_config["manifest_path"],
        mcinfo_dir=data_config["mcinfo_dir"],
        demographics_path=data_config["demographics_path"],
        use_result=data_config.get("use_result", True),
        result_path=data_config.get("result_path"),
        use_interview=data_config.get("use_interview", False),
        interview_path=data_config.get("interview_path"),
        use_pretokenized_result=data_config.get("use_pretokenized_result", False),
        result_tokenized_path=data_config.get("result_tokenized_path", None),
        mcinfo_materialized_path=data_config.get("mcinfo_materialized_path"),
        mcinfo_rg_cache_size=data_config.get("mcinfo_rg_cache_size", 2),
        mcinfo_validate_footer=data_config.get("mcinfo_validate_footer", True),
    )

    batch_size = datamodule_config["batch_size"]
    drop_last = datamodule_config.get("drop_last", False)

    train_sampler = Grade5PersonBatchSampler(
        manifest_path=data_config["manifest_path"],
        batch_size=batch_size,
        mode="train",
        shuffle=datamodule_config["sampler"]["shuffle_train"],
        drop_last=drop_last,
    )
    val_sampler = Grade5PersonBatchSampler(
        manifest_path=data_config["manifest_path"],
        batch_size=batch_size,
        mode="val",
        shuffle=datamodule_config["sampler"]["shuffle_val"],
        drop_last=False,
    )
    test_sampler = Grade5PersonBatchSampler(
        manifest_path=data_config["manifest_path"],
        batch_size=batch_size,
        mode="test",
        shuffle=datamodule_config["sampler"]["shuffle_test"],
        drop_last=False,
    )

    def _collate(mode: str):
        return lambda batch: grade5_collate_fn(
            batch=batch,
            code_embedder=embedders.categorical,
            text_embedder=embedders.text,
            config=config,
            mode=mode,
            device="cpu",
        )

    train_collate_fn = _collate("train")
    val_collate_fn = _collate("val")
    test_collate_fn = _collate("test")

    num_workers = data_config.get("num_workers", 4)
    pin_memory = data_config.get("pin_memory", True)
    prefetch_factor = data_config.get("prefetch_factor", 2)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_sampler,
        collate_fn=train_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=val_sampler,
        collate_fn=val_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=test_sampler,
        collate_fn=test_collate_fn,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
        prefetch_factor=max(1, prefetch_factor // 2),
        persistent_workers=False,
    )

    if debug:
        debug_cfg = config.get("debug", {})
        train_loader = list(islice(train_loader, debug_cfg.get("limit_train_batches", 20)))
        val_loader = list(islice(val_loader, debug_cfg.get("limit_val_batches", 10)))
        test_loader = list(islice(test_loader, debug_cfg.get("limit_test_batches", 5)))

    return train_loader, val_loader, test_loader
