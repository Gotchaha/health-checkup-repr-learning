"""
Unit tests for Grade5 representation datamodule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.downstream.grade5.repr_datamodule import (
    ReprGrade5Dataset,
    ReprGrade5PersonBatchSampler,
    repr_grade5_collate_fn,
)


def _write_repr_parquet(path: Path) -> None:
    rows = [
        {
            "exam_id": "e1",
            "person_id": "p1",
            "ExamDate": "2020-01-01",
            "split": "train",
            "grade5": "Normal",
            "is_grade5_valid": True,
            "post_causal_emb": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        },
        {
            "exam_id": "e2",
            "person_id": "p1",
            "ExamDate": "2021-01-01",
            "split": "train",
            "grade5": "Watch",
            "is_grade5_valid": True,
            "post_causal_emb": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        },
        {
            "exam_id": "e3",
            "person_id": "p2",
            "ExamDate": "2020-06-01",
            "split": "val",
            "grade5": "Normal",
            "is_grade5_valid": True,
            "post_causal_emb": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        },
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_repr_sampler_and_collate(tmp_path: Path) -> None:
    repr_path = tmp_path / "repr.parquet"
    _write_repr_parquet(repr_path)

    dataset = ReprGrade5Dataset(repr_path)
    sampler = ReprGrade5PersonBatchSampler(dataset.df, batch_size=10, mode="train", shuffle=False)
    batch_indices = next(iter(sampler))
    batch = [dataset[idx] for idx in batch_indices]

    outputs = repr_grade5_collate_fn(
        batch=batch,
        label_order=["Normal", "Mild", "Watch", "Treat", "UnderTreatment"],
        ignore_index=-100,
    )

    assert outputs["embeddings"].shape == (1, 2, 3)
    assert outputs["attention_mask"].shape == (1, 2)
    assert outputs["grade5_targets"].shape == (1, 2)
    assert outputs["grade5_label_mask"].shape == (1, 2)
