# tests/downstream/grade/test_model.py
"""
Unit tests for Grade5LinearHead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.downstream.grade5.model import Grade5LinearHead


def test_linear_head_shapes() -> None:
    config = {
        "ssl_backbone": {"encoder_dim": 8},
        "datamodule": {"label_processing": {"num_classes": 5, "label_order": ["a", "b", "c", "d", "e"]}},
    }
    head = Grade5LinearHead.from_config(config)
    x = torch.randn(2, 3, 8)
    logits = head(x)
    assert logits.shape == (2, 3, 5)
