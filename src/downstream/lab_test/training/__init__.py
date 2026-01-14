# src/downstream/lab_test/training/__init__.py

"""
Downstream lab test training components.

This module provides training infrastructure for downstream lab test prediction tasks.
Includes simplified trainer, logging utilities, and training script support.
"""

from .trainer import LabTestTrainer
from .utils import Logger, WandbLogger, DualMetricEarlyStopping, create_experiment_dirs, setup_reproducibility, unpack_individual_sequences
from .metrics import compute_task_metrics, compute_aggregate_metrics

__all__ = [
    'LabTestTrainer',
    'Logger', 
    'WandbLogger',
    'DualMetricEarlyStopping',
    'compute_task_metrics',
    'compute_aggregate_metrics',
    'create_experiment_dirs',
    'setup_reproducibility',
    'unpack_individual_sequences'
]