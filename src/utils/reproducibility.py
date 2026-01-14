# src/utils/reproducibility.py

import random
import os
import platform
import warnings
from contextlib import contextmanager
from typing import Dict, Any, Optional

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """
    Set seeds for all random number generators for reproducible results.
    
    Note: PYTHONHASHSEED must be set BEFORE starting Python to affect the current process.
    To ensure hash reproducibility, set the environment variable before running your script:
        export PYTHONHASHSEED=42; python train.py
    
    Args:
        seed: Random seed value
    """
    # Check if PYTHONHASHSEED was set correctly before process start
    current_hash_seed = os.environ.get('PYTHONHASHSEED')
    if current_hash_seed != str(seed):
        warnings.warn(
            f"PYTHONHASHSEED is not set to {seed} (current: {current_hash_seed}). "
            f"For full hash reproducibility, set it before starting Python: "
            f"export PYTHONHASHSEED={seed}; python your_script.py"
        )
    
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # PyTorch CUDA (if available)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Handle CUDA errors gracefully - continue without CUDA seeding
        pass
    
    # Environment variable for Python hash seed (for future processes)
    os.environ['PYTHONHASHSEED'] = str(seed)


def make_deterministic(strict: bool = False) -> None:
    """
    Enable deterministic operations for reproducible results.
    
    Args:
        strict: If True, enable full determinism (slower performance)
               If False, enable basic determinism (balanced performance)
    """
    if strict:
        # Full determinism - slower but maximum reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Only call if the function exists and is callable (API compatibility)
        if hasattr(torch, 'use_deterministic_algorithms') and callable(torch.use_deterministic_algorithms):
            torch.use_deterministic_algorithms(True)
        
        # Suppress warnings about deterministic algorithms
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
    else:
        # Balanced determinism - reasonable reproducibility with better performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Don't force deterministic algorithms globally
        # Some operations may still be non-deterministic but with better performance


def get_environment_info(lightweight: bool = True) -> Dict[str, Any]:
    """
    Capture environment information for experiment logging.
    
    Args:
        lightweight: If True, capture core packages only
                    If False, capture more detailed information
        
    Returns:
        Dictionary with environment information
    """
    info = {}
    
    # Core system info with error handling
    try:
        info['python_version'] = platform.python_version()
    except Exception:
        info['python_version'] = 'unknown'
    
    try:
        info['platform'] = platform.platform()
    except Exception:
        info['platform'] = 'unknown'
    
    # PyTorch info with error handling
    try:
        info['torch_version'] = torch.__version__
    except Exception:
        info['torch_version'] = 'unknown'
    
    try:
        info['cuda_available'] = torch.cuda.is_available()
    except Exception:
        info['cuda_available'] = False
    
    # CUDA-specific info if available
    if info.get('cuda_available', False):
        try:
            info['cuda_version'] = torch.version.cuda
        except Exception:
            info['cuda_version'] = 'unknown'
        
        try:
            info['cudnn_version'] = torch.backends.cudnn.version()
        except Exception:
            info['cudnn_version'] = 'unknown'
        
        try:
            info['gpu_count'] = torch.cuda.device_count()
        except Exception:
            info['gpu_count'] = 0
        
        try:
            gpu_names = []
            for i in range(info.get('gpu_count', 0)):
                try:
                    gpu_names.append(torch.cuda.get_device_name(i))
                except Exception:
                    gpu_names.append(f'GPU {i} (unknown)')
            info['gpu_names'] = gpu_names
        except Exception:
            info['gpu_names'] = []
    
    # Core ML packages with error handling
    try:
        import numpy
        info['numpy_version'] = numpy.__version__
    except ImportError:
        info['numpy_version'] = 'not available'
    except Exception:
        info['numpy_version'] = 'unknown'
    
    try:
        import pandas
        info['pandas_version'] = pandas.__version__
    except ImportError:
        info['pandas_version'] = 'not available'
    except Exception:
        info['pandas_version'] = 'unknown'
    
    try:
        import transformers
        info['transformers_version'] = transformers.__version__
    except ImportError:
        info['transformers_version'] = 'not available'
    except Exception:
        info['transformers_version'] = 'unknown'
    
    if not lightweight:
        # Additional detailed info for debugging with error handling
        try:
            import pyarrow
            info['pyarrow_version'] = pyarrow.__version__
        except ImportError:
            info['pyarrow_version'] = 'not available'
        except Exception:
            info['pyarrow_version'] = 'unknown'
        
        # CPU info with error handling
        try:
            info['cpu_count'] = os.cpu_count()
        except Exception:
            info['cpu_count'] = 'unknown'
        
        # Memory info (basic) with error handling
        if info.get('cuda_available', False):
            try:
                gpu_memory = []
                for i in range(info.get('gpu_count', 0)):
                    try:
                        memory = torch.cuda.get_device_properties(i).total_memory
                        gpu_memory.append(memory)
                    except Exception:
                        gpu_memory.append(0)
                info['gpu_memory'] = gpu_memory
            except Exception:
                info['gpu_memory'] = []
    
    return info


@contextmanager
def with_reproducible_context(seed: int, strict: bool = False):
    """
    Context manager for temporary reproducible operations.
    
    Args:
        seed: Random seed to use within context
        strict: Whether to use strict determinism
        
    Example:
        with with_reproducible_context(42, strict=True):
            # Critical experiment code with full reproducibility
            model = create_model()
            results = train_model(model)
    """
    # Store original states
    original_python_state = random.getstate()
    original_numpy_state = np.random.get_state()
    original_torch_state = torch.get_rng_state()
    
    # Store original CUDA states if available
    original_cuda_states = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            original_cuda_states.append(torch.cuda.get_rng_state(i))
    
    # Store original deterministic settings
    original_deterministic = torch.backends.cudnn.deterministic
    original_benchmark = torch.backends.cudnn.benchmark
    original_use_deterministic = None
    
    # Store original environment variables
    original_cublas_config = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    
    # Check PyTorch API compatibility
    if hasattr(torch, 'are_deterministic_algorithms_enabled') and callable(torch.are_deterministic_algorithms_enabled):
        original_use_deterministic = torch.are_deterministic_algorithms_enabled()
    
    try:
        # Apply reproducible settings
        set_all_seeds(seed)
        make_deterministic(strict=strict)
        
        # Advance RNG states to ensure they differ from original
        random.random()
        np.random.rand()
        torch.rand(1)
        
        yield
        
    finally:
        # Restore original states
        random.setstate(original_python_state)
        np.random.set_state(original_numpy_state)
        torch.set_rng_state(original_torch_state)
        
        # Restore CUDA states
        if torch.cuda.is_available():
            for i, state in enumerate(original_cuda_states):
                torch.cuda.set_rng_state(state, i)
        
        # Restore deterministic settings
        torch.backends.cudnn.deterministic = original_deterministic
        torch.backends.cudnn.benchmark = original_benchmark
        
        # Restore deterministic algorithms setting (with API compatibility)
        if original_use_deterministic is not None and hasattr(torch, 'use_deterministic_algorithms') and callable(torch.use_deterministic_algorithms):
            torch.use_deterministic_algorithms(original_use_deterministic)
        
        # Restore environment variables
        if original_cublas_config is not None:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = original_cublas_config
        else:
            # Remove the environment variable if it wasn't set originally
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)


def setup_reproducibility(seed: int, strict: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to set up reproducibility and return environment info.
    
    Args:
        seed: Random seed value
        strict: Whether to use strict determinism
        verbose: Whether to print setup information
        
    Returns:
        Environment information dictionary
    """
    set_all_seeds(seed)
    make_deterministic(strict=strict)
    
    env_info = get_environment_info(lightweight=True)
    
    if verbose:
        print(f"Reproducibility setup complete:")
        print(f"  Seed: {seed}")
        print(f"  Strict mode: {strict}")
        print(f"  PyTorch: {env_info['torch_version']}")
        print(f"  CUDA available: {env_info['cuda_available']}")
        if env_info['cuda_available']:
            print(f"  GPU count: {env_info['gpu_count']}")
    
    return env_info