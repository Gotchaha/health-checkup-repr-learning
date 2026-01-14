"""
Profiling module for SSL pretraining performance analysis.

This module provides independent profiling capabilities without
modifying the original training code.
"""

from .ssl_profiler import SSLProfiler, ProfileConfig, ProfileMetrics
from .profiling_utils import (
    Timer,
    MemoryTracker,
    profile_block,
    format_time,
    format_memory,
    get_model_memory_usage,
    profile_data_loader
)

__all__ = [
    'SSLProfiler',
    'ProfileConfig',
    'ProfileMetrics',
    'Timer',
    'MemoryTracker',
    'profile_block',
    'format_time',
    'format_memory',
    'get_model_memory_usage',
    'profile_data_loader'
]