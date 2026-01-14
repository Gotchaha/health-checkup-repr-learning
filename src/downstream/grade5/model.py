# src/downstream/grade5/model.py

"""
Minimal prediction head for Grade5 downstream classification.
"""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn


class Grade5LinearHead(nn.Module):
    """Linear classification head over SSL representations."""

    def __init__(self, encoder_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(encoder_dim, num_classes)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Grade5LinearHead":
        encoder_dim = config["ssl_backbone"]["encoder_dim"]
        label_cfg = config["datamodule"]["label_processing"]
        num_classes = label_cfg["num_classes"]
        label_order = label_cfg.get("label_order", [])
        if label_order and len(label_order) != num_classes:
            raise ValueError(
                f"label_order length ({len(label_order)}) does not match num_classes ({num_classes})"
            )
        return cls(encoder_dim=encoder_dim, num_classes=num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)
