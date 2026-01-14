# src/downstream/grade5/metrics.py

"""
Metrics for Grade5 downstream classification.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute confusion matrix for valid positions.

    Args:
        preds: Predicted class indices [N]
        targets: True class indices [N]
        valid_mask: Boolean mask [N]
        num_classes: Number of classes
    """
    preds = preds.view(-1)
    targets = targets.view(-1)
    valid = valid_mask.view(-1)

    if preds.numel() != targets.numel():
        raise ValueError("Preds and targets must have the same number of elements.")

    preds = preds[valid]
    targets = targets[valid]

    if preds.numel() == 0:
        return torch.zeros(num_classes, num_classes, dtype=torch.long)

    indices = targets * num_classes + preds
    indices = indices.to(dtype=torch.int64, device="cpu")
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    return cm.view(num_classes, num_classes)


def normalize_confusion_matrix_rows(cm: torch.Tensor) -> torch.Tensor:
    """Row-normalize a confusion matrix."""
    row_sums = cm.sum(dim=1, keepdim=True)
    row_sums = row_sums.clamp(min=1)
    return cm.float() / row_sums


def macro_f1_from_confusion(cm: torch.Tensor, zero_division: float = 0.0) -> float:
    """
    Compute macro-F1 from a confusion matrix with zero-division handling.
    """
    num_classes = cm.size(0)
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp

        denom_p = tp + fp
        denom_r = tp + fn

        if denom_p == 0:
            precision = zero_division
        else:
            precision = tp / denom_p

        if denom_r == 0:
            recall = zero_division
        else:
            recall = tp / denom_r

        denom_f1 = precision + recall
        if denom_f1 == 0:
            f1 = zero_division
        else:
            f1 = 2 * precision * recall / denom_f1

        f1s.append(f1)

    return float(sum(f1s) / num_classes)


def accuracy_from_confusion(cm: torch.Tensor) -> float:
    """Compute accuracy from a confusion matrix."""
    total = cm.sum().item()
    if total == 0:
        return 0.0
    correct = cm.diag().sum().item()
    return float(correct / total)


def macro_recall_from_confusion(cm: torch.Tensor, zero_division: float = 0.0) -> float:
    """Compute macro-recall from a confusion matrix."""
    num_classes = cm.size(0)
    recalls = []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fn = cm[c, :].sum().item() - tp
        denom = tp + fn
        if denom == 0:
            recall = zero_division
        else:
            recall = tp / denom
        recalls.append(recall)
    return float(sum(recalls) / num_classes)


def compute_per_exam_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    num_classes: int,
    zero_division: float = 0.0,
) -> Dict[str, float]:
    """
    Compute per-exam metrics aggregated across all valid positions.
    """
    preds = torch.argmax(logits, dim=-1)
    valid = label_mask & attention_mask

    cm = compute_confusion_matrix(preds, targets, valid, num_classes)
    return {
        "per_exam_macro_f1": macro_f1_from_confusion(cm, zero_division=zero_division),
        "per_exam_accuracy": accuracy_from_confusion(cm),
        "per_exam_macro_recall": macro_recall_from_confusion(cm, zero_division=zero_division),
        "per_exam_confusion_matrix": cm,
    }


def compute_per_patient_macro_f1(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    num_classes: int,
    zero_division: float = 0.0,
) -> Tuple[float, int]:
    """
    Compute per-patient macro-F1 and return (metric, num_valid_patients).
    """
    preds = torch.argmax(logits, dim=-1)
    valid = label_mask & attention_mask

    total_f1 = 0.0
    valid_patients = 0

    for i in range(targets.size(0)):
        row_valid = valid[i]
        if row_valid.sum().item() == 0:
            continue
        cm = compute_confusion_matrix(
            preds[i], targets[i], row_valid, num_classes
        )
        total_f1 += macro_f1_from_confusion(cm, zero_division=zero_division)
        valid_patients += 1

    if valid_patients == 0:
        return 0.0, 0
    return float(total_f1 / valid_patients), valid_patients


def compute_per_patient_macro_f1_observed(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    num_classes: int,
    zero_division: float = 0.0,
) -> Tuple[float, int]:
    """
    Compute per-patient macro-F1 over observed labels only.

    For each patient i, define C_i as labels appearing in y_{i,t} over valid steps.
    Macro-F1 is averaged over C_i rather than all classes.
    """
    preds = torch.argmax(logits, dim=-1)
    valid = label_mask & attention_mask

    total_f1 = 0.0
    valid_patients = 0

    for i in range(targets.size(0)):
        row_valid = valid[i]
        if row_valid.sum().item() == 0:
            continue
        row_targets = targets[i][row_valid]
        observed = torch.unique(row_targets)
        if observed.numel() == 0:
            continue

        cm = compute_confusion_matrix(
            preds[i], targets[i], row_valid, num_classes
        )

        f1s = []
        for cls in observed.tolist():
            tp = cm[cls, cls].item()
            fp = cm[:, cls].sum().item() - tp
            fn = cm[cls, :].sum().item() - tp

            denom_p = tp + fp
            denom_r = tp + fn

            precision = zero_division if denom_p == 0 else tp / denom_p
            recall = zero_division if denom_r == 0 else tp / denom_r
            denom_f1 = precision + recall
            f1 = zero_division if denom_f1 == 0 else 2 * precision * recall / denom_f1
            f1s.append(f1)

        total_f1 += float(sum(f1s) / len(f1s))
        valid_patients += 1

    if valid_patients == 0:
        return 0.0, 0
    return float(total_f1 / valid_patients), valid_patients
