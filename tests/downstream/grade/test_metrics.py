# tests/downstream/grade/test_metrics.py
"""
Unit tests for Grade5 metrics utilities.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.downstream.grade5.metrics import (
    compute_confusion_matrix,
    compute_per_exam_metrics,
    compute_per_patient_macro_f1,
    compute_per_patient_macro_f1_observed,
)


def test_confusion_matrix_counts():
    preds = torch.tensor([0, 1, 1, 2])
    targets = torch.tensor([0, 0, 1, 2])
    valid = torch.tensor([True, True, True, True])
    cm = compute_confusion_matrix(preds, targets, valid, num_classes=3)

    expected = torch.tensor([
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    assert torch.equal(cm, expected)


def test_per_exam_metrics_masking():
    logits = torch.tensor([
        [[2.0, 0.1], [0.1, 2.0]],
        [[0.1, 2.0], [2.0, 0.1]],
    ])
    targets = torch.tensor([[0, 1], [1, 0]])
    label_mask = torch.tensor([[True, False], [True, True]])
    attention_mask = torch.tensor([[True, True], [True, True]])

    metrics = compute_per_exam_metrics(
        logits=logits,
        targets=targets,
        label_mask=label_mask,
        attention_mask=attention_mask,
        num_classes=2,
    )

    assert metrics["per_exam_accuracy"] == 1.0


def test_per_patient_macro_f1():
    logits = torch.tensor([
        [[2.0, 0.1], [0.1, 2.0]],
        [[2.0, 0.1], [2.0, 0.1]],
    ])
    targets = torch.tensor([[0, 1], [0, 1]])
    label_mask = torch.tensor([[True, True], [True, True]])
    attention_mask = torch.tensor([[True, True], [True, True]])

    f1, n_valid = compute_per_patient_macro_f1(
        logits=logits,
        targets=targets,
        label_mask=label_mask,
        attention_mask=attention_mask,
        num_classes=2,
    )

    assert n_valid == 2
    assert f1 < 1.0


def test_per_patient_macro_f1_observed():
    logits = torch.tensor([
        [[2.0, 0.1, 0.1], [2.0, 0.1, 0.1]],
        [[0.1, 2.0, 0.1], [0.1, 0.1, 2.0]],
    ])
    targets = torch.tensor([[0, 0], [1, 2]])
    label_mask = torch.tensor([[True, True], [True, True]])
    attention_mask = torch.tensor([[True, True], [True, True]])

    f1, n_valid = compute_per_patient_macro_f1_observed(
        logits=logits,
        targets=targets,
        label_mask=label_mask,
        attention_mask=attention_mask,
        num_classes=3,
    )

    assert n_valid == 2
    assert 0.0 <= f1 <= 1.0
