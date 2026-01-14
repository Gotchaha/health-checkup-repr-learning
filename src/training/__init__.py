# src/training/__init__.py

"""
Training infrastructure for self-supervised medical learning.

This module provides the complete training pipeline for our multi-modal 
self-supervised learning approach:

Core Training Infrastructure:
- SSLTrainer: Main training orchestrator with monitoring and checkpointing
- MultiTaskLoss: Learnable combination of multiple training objectives

Training Objectives:
- MLM: Masked Language Modeling for result text narratives
- MCM: Masked Category Modeling for categorical test values  
- CVR: Cell Value Retrieval for text test values
- MCC: Multiple-Choice Cloze for numerical test values
- CPC: Contrastive Predictive Coding for individual-level temporal modeling

Training Control & Monitoring:
- ArchitectureMonitor: Monitor attention patterns, fusion weights, loss dynamics
- Callbacks: EarlyStopping, ModelCheckpoint, LearningRateScheduler, etc.

Usage:
    # Main training workflow
    from src.training import SSLTrainer, MultiTaskLoss, ArchitectureMonitor
    
    # Training objectives
    from src.training import MLMHead, MCMHead, CVRHead, MCCHead, CPCHead
    
    # Training control
    from src.training import EarlyStopping, ModelCheckpoint
"""

# Core training infrastructure
from .trainer import SSLTrainer
from .multi_task_loss import MultiTaskLoss

# Training objective heads
from .heads import (
    MLMHead,      # Masked Language Modeling
    MCMHead,      # Masked Category Modeling  
    CVRHead,      # Cell Value Retrieval
    MCCHead,      # Multiple-Choice Cloze
    CPCHead       # Contrastive Predictive Coding
)

# Loss functions
from .losses import InfoNCE

# Training control and monitoring
from .monitoring import ArchitectureMonitor
from .callbacks import (
    # Base classes
    Callback,
    CallbackList,
    
    # Specific callbacks
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    GradientMonitor,
    NaNDetector,

    # Custom exception class
    CriticalTrainingError,
)

# Define what gets imported with "from src.training import *"
__all__ = [
    # Core training infrastructure
    'SSLTrainer',
    'MultiTaskLoss',
    
    # Training heads
    'MLMHead',
    'MCMHead', 
    'CVRHead',
    'MCCHead',
    'CPCHead',

    
    # Loss functions
    'InfoNCE',
    
    # Training control and monitoring
    'ArchitectureMonitor',
    'Callback',
    'CallbackList',
    'EarlyStopping',
    'ModelCheckpoint',
    'LearningRateScheduler',
    'GradientMonitor',
    'NaNDetector',

    # Custom exception class
    'CriticalTrainingError',
]