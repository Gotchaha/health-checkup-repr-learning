# src/downstream/grade5/__init__.py

"""
Grade5 downstream package.

Avoid importing heavyweight trainers at module import time so that lightweight
utilities (e.g., metrics) can be imported in isolation during test collection.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    # Datamodule (backbone path)
    "Grade5Dataset",
    "Grade5PersonBatchSampler",
    "grade5_collate_fn",
    # Model
    "Grade5LinearHead",
    # Datamodule (repr path)
    "ReprGrade5Dataset",
    "ReprGrade5PersonBatchSampler",
    "repr_grade5_collate_fn",
    "create_repr_grade5_data_loaders",
    # Trainers
    "Grade5Trainer",
    "ReprGrade5Trainer",
    # Metrics
    "compute_confusion_matrix",
    "normalize_confusion_matrix_rows",
    "macro_f1_from_confusion",
    "accuracy_from_confusion",
    "macro_recall_from_confusion",
    "compute_per_exam_metrics",
    "compute_per_patient_macro_f1",
    "compute_per_patient_macro_f1_observed",
]


_LAZY_IMPORTS = {
    # Datamodule (backbone path)
    "Grade5Dataset": (".datamodule", "Grade5Dataset"),
    "Grade5PersonBatchSampler": (".datamodule", "Grade5PersonBatchSampler"),
    "grade5_collate_fn": (".datamodule", "grade5_collate_fn"),
    # Model
    "Grade5LinearHead": (".model", "Grade5LinearHead"),
    # Datamodule (repr path)
    "ReprGrade5Dataset": (".repr_datamodule", "ReprGrade5Dataset"),
    "ReprGrade5PersonBatchSampler": (".repr_datamodule", "ReprGrade5PersonBatchSampler"),
    "repr_grade5_collate_fn": (".repr_datamodule", "repr_grade5_collate_fn"),
    "create_repr_grade5_data_loaders": (".repr_datamodule", "create_repr_grade5_data_loaders"),
    # Trainers
    "Grade5Trainer": (".trainer", "Grade5Trainer"),
    "ReprGrade5Trainer": (".repr_trainer", "ReprGrade5Trainer"),
    # Metrics
    "compute_confusion_matrix": (".metrics", "compute_confusion_matrix"),
    "normalize_confusion_matrix_rows": (".metrics", "normalize_confusion_matrix_rows"),
    "macro_f1_from_confusion": (".metrics", "macro_f1_from_confusion"),
    "accuracy_from_confusion": (".metrics", "accuracy_from_confusion"),
    "macro_recall_from_confusion": (".metrics", "macro_recall_from_confusion"),
    "compute_per_exam_metrics": (".metrics", "compute_per_exam_metrics"),
    "compute_per_patient_macro_f1": (".metrics", "compute_per_patient_macro_f1"),
    "compute_per_patient_macro_f1_observed": (".metrics", "compute_per_patient_macro_f1_observed"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
