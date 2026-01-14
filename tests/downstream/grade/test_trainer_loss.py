"""
Unit tests for Grade5Trainer loss reduction logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.downstream.grade5.trainer import Grade5Trainer


class _DummyTrainer:
    def __init__(self) -> None:
        self.num_classes = 3
        self.ignore_index = -100
        self.class_weights = torch.tensor([1.0, 1.0, 1.0])

    _compute_loss = Grade5Trainer._compute_loss


def test_per_patient_loss_reduction():
    trainer = _DummyTrainer()

    logits = torch.tensor(
        [
            [[5.0, 0.1, 0.1], [0.1, 5.0, 0.1]],
            [[0.1, 5.0, 0.1], [0.1, 5.0, 0.1]],
        ]
    )
    targets = torch.tensor([[0, 1], [1, -100]])
    label_mask = torch.tensor([[True, True], [True, False]])
    attention_mask = torch.tensor([[True, True], [True, True]])

    loss, valid_patients, valid_steps = trainer._compute_loss(
        logits=logits,
        targets=targets,
        label_mask=label_mask,
        attention_mask=attention_mask,
    )

    assert valid_patients == 2
    assert valid_steps == 3
    assert loss.item() >= 0.0
