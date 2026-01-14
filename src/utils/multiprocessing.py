# src/utils/multiprocessing.py

"""
Multiprocessing utilities for PyTorch DataLoader worker management.

This module provides utilities for managing DataLoader worker processes,
particularly for datasets using PyArrow objects that need careful cleanup
to prevent process hanging and resource leaks.

Usage:
    from src.utils import worker_init_with_cleanup
    
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        worker_init_fn=worker_init_with_cleanup,
        pin_memory=True,
        persistent_workers=True
    )
"""

import os
import signal
import atexit
import logging
import torch
import pyarrow as pa
import resource


def worker_init_with_cleanup(worker_id: int) -> None:
    """
    Initialize DataLoader worker with proper cleanup handling for PyArrow datasets.
    
    This function should be used as the worker_init_fn in PyTorch DataLoaders
    when working with datasets that use PyArrow objects. It ensures proper
    resource cleanup and prevents hanging worker processes.
    
    Features:
    - Logs worker initialization with PID and PyArrow version
    - Ignores SIGINT to allow graceful main process termination
    - Registers cleanup function to run on worker exit
    - Handles PyArrow C++ resource cleanup to prevent hanging
    
    Args:
        worker_id: Worker ID assigned by DataLoader (0 to num_workers-1)
        
    Usage:
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            worker_init_fn=worker_init_with_cleanup,
            num_workers=4
        )
        
    Note:
        This function assumes the dataset has a 'close()' method for cleanup.
        See HealthExamDataset.close() for the expected interface.
    """
    # ---------- ensure console logging in worker ----------
    root = logging.getLogger()
    if not root.handlers:                        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        root.addHandler(ch)
        root.setLevel(logging.INFO)
    
    # Log worker initialization once per worker
    logger = logging.getLogger(__name__)
    # rss  = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
    # arr  = pa.total_allocated_bytes() / 1024**2                      # MB
    # logger.info(f"[worker {os.getpid()}] init | pa={pa.__version__} | RSS={rss:.1f} MB | Arrow={arr:.1f} MB")


    # Ignore CTRL-C in worker processes - let main process handle termination
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Get worker info and dataset reference
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset if worker_info else None

    def _final_cleanup() -> None:
        """
        Cleanup function to run when worker process exits.
        
        This function is registered with atexit to ensure it runs even
        if the worker doesn't shut down cleanly. It calls the dataset's
        close() method if available to release PyArrow resources.
        """
        if dataset and hasattr(dataset, "close"):
            try:
                dataset.close()
            except Exception as e:
                # Log cleanup errors but don't raise to avoid masking other issues
                logger.warning(f"[worker {os.getpid()}] cleanup error: {e}")

        # peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        # arr  = pa.total_allocated_bytes() / 1024**2
        # logger.info(f"[worker {os.getpid()}] exit | peakRSS={peak:.1f} MB | Arrow={arr:.1f} MB")

    # Register cleanup to run on worker exit
    atexit.register(_final_cleanup)