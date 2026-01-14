# tests/downstream/grade/test_datamodule.py
"""
Unit tests for Grade5 downstream datamodule.

Focuses on sampler split filtering and label packing in grade5_collate_fn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.downstream.grade5.datamodule import Grade5PersonBatchSampler, grade5_collate_fn


def _write_manifest(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def test_sampler_filters_by_split(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    rows = [
        {"exam_id": "e1", "person_id": "p1", "ExamDate": "2020-01-01", "split": "train"},
        {"exam_id": "e2", "person_id": "p1", "ExamDate": "2021-01-01", "split": "train"},
        {"exam_id": "e3", "person_id": "p2", "ExamDate": "2020-03-01", "split": "val"},
        {"exam_id": "e4", "person_id": "p3", "ExamDate": "2019-06-01", "split": "train"},
    ]
    _write_manifest(manifest_path, rows)

    sampler = Grade5PersonBatchSampler(
        manifest_path=manifest_path,
        batch_size=10,
        mode="train",
        shuffle=False,
        drop_last=False,
    )

    persons = [pid for pid, _ in sampler.persons]
    assert persons == ["p1", "p3"]


def test_sampler_rejects_mixed_splits(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    rows = [
        {"exam_id": "e1", "person_id": "p1", "ExamDate": "2020-01-01", "split": "train"},
        {"exam_id": "e2", "person_id": "p1", "ExamDate": "2021-01-01", "split": "val"},
    ]
    _write_manifest(manifest_path, rows)

    with pytest.raises(ValueError, match="multiple splits"):
        Grade5PersonBatchSampler(
            manifest_path=manifest_path,
            batch_size=10,
            mode="train",
            shuffle=False,
            drop_last=False,
        )


def test_collate_packs_grade5_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_collate(*args, **kwargs):
        return {"segment_lengths": [2, 1]}

    monkeypatch.setattr(
        "src.downstream.grade5.datamodule.collate_exams",
        _fake_collate,
    )

    batch = [
        {"grade5": "Normal", "is_grade5_valid": True},
        {"grade5": "Watch", "is_grade5_valid": True},
        {"grade5": "Unknown", "is_grade5_valid": False},
    ]
    config = {
        "datamodule": {
            "label_processing": {
                "label_order": ["Normal", "Mild", "Watch", "Treat", "UnderTreatment"],
                "ignore_index": -100,
            }
        }
    }

    outputs = grade5_collate_fn(
        batch=batch,
        code_embedder=None,
        text_embedder=None,
        config=config,
        mode="train",
        device="cpu",
    )

    targets = outputs["grade5_targets"]
    mask = outputs["grade5_label_mask"]

    assert targets.shape == (2, 2)
    assert mask.shape == (2, 2)

    expected_targets = torch.tensor([[0, 2], [-100, -100]])
    expected_mask = torch.tensor([[True, True], [False, False]])

    assert torch.equal(targets, expected_targets)
    assert torch.equal(mask, expected_mask)
