"""
Simple profiling utilities for performance analysis.
"""

import time
import torch
from contextlib import contextmanager
from typing import Dict, List, Optional


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self, name: str = "Timer", cuda_sync: bool = False):
        """
        Initialize timer.
        
        Args:
            name: Name of the timer
            cuda_sync: Whether to synchronize CUDA before timing
        """
        self.name = name
        self.cuda_sync = cuda_sync
        self.times = []
        self.start_time = None
    
    def start(self):
        """Start timing."""
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
    
    def stop(self):
        """Stop timing and record elapsed time."""
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if self.start_time is None:
            raise RuntimeError(f"Timer {self.name} was not started")
        
        elapsed = time.perf_counter() - self.start_time
        self.times.append(elapsed)
        self.start_time = None
        return elapsed
    
    def reset(self):
        """Reset all recorded times."""
        self.times = []
        self.start_time = None
    
    def get_stats(self):
        """Get timing statistics."""
        import numpy as np
        
        if not self.times:
            return {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'total': 0,
                'count': 0
            }
        
        times_ms = [t * 1000 for t in self.times]  # Convert to milliseconds
        
        return {
            'mean': np.mean(times_ms),
            'std': np.std(times_ms),
            'min': np.min(times_ms),
            'max': np.max(times_ms),
            'total': np.sum(times_ms),
            'count': len(times_ms)
        }


class MemoryTracker:
    """Track GPU memory usage."""
    
    def __init__(self):
        """Initialize memory tracker."""
        self.checkpoints = {}
        self.enabled = torch.cuda.is_available()
    
    def checkpoint(self, name: str):
        """Record memory usage at a checkpoint."""
        if not self.enabled:
            return
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        self.checkpoints[name] = {
            'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
            'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2
        }
    
    def get_summary(self):
        """Get memory usage summary."""
        if not self.checkpoints:
            return {}
        
        summary = {}
        for name, stats in self.checkpoints.items():
            summary[name] = stats
        
        # Calculate differences between checkpoints
        checkpoint_names = list(self.checkpoints.keys())
        for i in range(1, len(checkpoint_names)):
            prev_name = checkpoint_names[i-1]
            curr_name = checkpoint_names[i]
            
            diff_name = f"delta_{prev_name}_to_{curr_name}"
            summary[diff_name] = {
                'allocated_mb': (self.checkpoints[curr_name]['allocated_mb'] - 
                                self.checkpoints[prev_name]['allocated_mb'])
            }
        
        return summary
    
    def reset(self):
        """Reset all checkpoints."""
        self.checkpoints = {}
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()


@contextmanager
def profile_block(name: str, timer_dict: Dict[str, Timer], cuda_sync: bool = False):
    """
    Context manager for profiling code blocks.
    
    Args:
        name: Name of the block being profiled
        timer_dict: Dictionary to store timers
        cuda_sync: Whether to synchronize CUDA
    
    Example:
        timers = {}
        with profile_block('forward_pass', timers):
            output = model(input)
    """
    if name not in timer_dict:
        timer_dict[name] = Timer(name, cuda_sync=cuda_sync)
    
    timer = timer_dict[name]
    timer.start()
    
    try:
        yield timer
    finally:
        timer.stop()


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f}μs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f}ms"
    else:
        return f"{seconds:.2f}s"


def format_memory(bytes_val: float) -> str:
    """Format memory in human-readable format."""
    mb = bytes_val / 1024**2
    if mb < 1024:
        return f"{mb:.2f}MB"
    else:
        gb = mb / 1024
        return f"{gb:.2f}GB"


def get_model_memory_usage(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate memory usage of a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with memory statistics
    """
    param_memory = 0
    buffer_memory = 0
    
    # Calculate parameter memory
    for param in model.parameters():
        param_memory += param.numel() * param.element_size()
    
    # Calculate buffer memory
    for buffer in model.buffers():
        buffer_memory += buffer.numel() * buffer.element_size()
    
    total_memory = param_memory + buffer_memory
    
    return {
        'param_mb': param_memory / 1024**2,
        'buffer_mb': buffer_memory / 1024**2,
        'total_mb': total_memory / 1024**2,
        'param_count': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }


def profile_data_loader(dataloader, num_batches: int = 10) -> Dict[str, float]:
    """
    Profile data loading speed.
    
    Args:
        dataloader: PyTorch DataLoader
        num_batches: Number of batches to profile
    
    Returns:
        Dictionary with timing statistics
    """
    times = []
    
    data_iter = iter(dataloader)
    
    for _ in range(num_batches):
        start = time.perf_counter()
        _ = next(data_iter)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    import numpy as np
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000
    }