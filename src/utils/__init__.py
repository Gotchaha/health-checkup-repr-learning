# src/utils/__init__.py

"""
Utilities for experiment management, configuration, logging, and reproducibility.

This module provides essential infrastructure for NeurIPS-level research:
- Configuration management with inheritance and validation
- Two-tier experiment logging (master CSV + detailed JSON)
- Reproducibility controls for deterministic experiments
- Optional Weights & Biases integration
- Environment information capture
- Multiprocessing utilities for DataLoader worker management

Usage:
    # Configuration and experiment setup
    from src.utils import load_experiment_config, create_experiment_dirs
    config = load_experiment_config("config/experiments/my_experiment.yaml")
    dirs = create_experiment_dirs(config)
    
    # Reproducibility
    from src.utils import setup_reproducibility
    env_info = setup_reproducibility(seed=42, strict=False)
    
    # Experiment logging
    from src.utils import ExperimentLogger
    logger = ExperimentLogger()
    experiment = logger.start_experiment(config)
    
    # Optional wandb integration
    from src.utils import WandbLogger
    with WandbLogger(config, enabled=True) as wandb_logger:
        wandb_logger.log_step_metrics(step, train_metrics, val_metrics)
        
    # DataLoader multiprocessing support
    from src.utils import worker_init_with_cleanup
    train_loader = torch.utils.data.DataLoader(
        dataset, worker_init_fn=worker_init_with_cleanup, num_workers=4
    )
"""

# Configuration management
from .config import (
    ConfigManager,
    load_experiment_config,
    create_experiment_dirs,
    get_config_summary
)

# Experiment logging
from .logging import (
    ExperimentLogger,
    ExperimentRun,
    create_experiment_logger,
    load_experiment_results
)

# Reproducibility
from .reproducibility import (
    set_all_seeds,
    make_deterministic,
    get_environment_info,
    with_reproducible_context,
    setup_reproducibility
)

# Weights & Biases integration
from .wandb_logger import (
    WandbLogger,
    create_wandb_logger,
    get_wandb_url_from_experiment_name
)

# Multiprocessing utilities
from .multiprocessing import (
    worker_init_with_cleanup
)

# Define what gets imported with "from src.utils import *"
__all__ = [
    # Configuration management
    'ConfigManager',
    'load_experiment_config',
    'create_experiment_dirs', 
    'get_config_summary',
    
    # Experiment logging
    'ExperimentLogger',
    'ExperimentRun',
    'create_experiment_logger',
    'load_experiment_results',
    
    # Reproducibility
    'set_all_seeds',
    'make_deterministic',
    'get_environment_info',
    'with_reproducible_context',
    'setup_reproducibility',
    
    # Weights & Biases integration
    'WandbLogger',
    'create_wandb_logger',
    'get_wandb_url_from_experiment_name',
    
    # Multiprocessing utilities
    'worker_init_with_cleanup'
]