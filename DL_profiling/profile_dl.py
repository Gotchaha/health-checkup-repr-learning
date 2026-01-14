# DL_profiling/profile_dl.py
"""
Performance profiling script for SSL data loading pipeline.
Adapted from scripts/pre_profile_ssl.py

Usage:
    python DL_profiling/profile_dl.py config/experiments/dl_profiling.yaml
"""

import os
import sys
import torch
import logging
import gc
from pathlib import Path
from functools import partial
from itertools import islice
from time import perf_counter


# Calculate project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Change working directory to project root to ensure relative paths work
os.chdir(project_root)

from src.models import (
    HealthExamDataset, PersonBatchSampler, collate_exams,
    create_embedders_from_config
)
from src.utils import (
    load_experiment_config, setup_reproducibility
)
from DL_profiling.multiprocessing_utils import worker_init_with_memlog


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _move_batch_to_device_like_training(batch_data, device):
    """Exact copy of training-side device move (1-level nested dict)."""
    for key, value in batch_data.items():
        if isinstance(value, torch.Tensor):
            batch_data[key] = value.to(device)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    value[sub_key] = sub_value.to(device)
    return batch_data

# ---- helpers ----
def _bytes_to_mb(x: int) -> float:
    """Convert bytes to megabytes."""
    return x / (1024 ** 2)

def _count_h2d_bytes_one_level(batch):
    """Count bytes that will be moved from CPU to CUDA (one-level dict, match training move)."""
    total = 0
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.device.type == "cpu":
            total += v.numel() * v.element_size()
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, torch.Tensor) and sv.device.type == "cpu":
                    total += sv.numel() * sv.element_size()
    return total
# -------------------------------------------

def create_data_loaders(config, embedders, debug=False):
    """Create training and validation data loaders."""
    # Extract data configuration
    data_config = config['data']
    
    # Create datasets
    train_dataset = HealthExamDataset(
        split_name='train_ssl',
        mcinfo_dir=data_config['mcinfo_dir'],
        demographics_path=data_config['demographics_path'],
        use_result=data_config.get('use_result', True),
        result_path=data_config.get('result_path'),
        use_interview=data_config.get('use_interview', False),
        interview_path=data_config.get('interview_path'),
        use_pretokenized_result=data_config.get('use_pretokenized_result', False),
        result_tokenized_path=data_config.get('result_tokenized_path', None),
        mcinfo_materialized_path=data_config.get('mcinfo_materialized_path_train', None)
    )
    
    # Create samplers
    train_sampler = PersonBatchSampler(
        manifest_path="data/splits/core/sorted/train_ssl.parquet",
        batch_size=config['training']['batch_size'],
        shuffle=True,
        drop_last=True
    )
    
    # Debug mode: use only first few batches
    if debug:
        print("Debug mode: taking first 10 training")
        train_sampler = list(islice(train_sampler, 10))
    
    # Create collate function with embedders
    collate_fn = partial(
        collate_exams,
        code_embedder=embedders.categorical,
        text_embedder=embedders.text,
        config=config,
        device='cpu'
    )
    
    # Create data loaders with multiprocessing cleanup support
    num_workers = data_config.get('num_workers', 4)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=data_config.get('pin_memory', False),
        prefetch_factor=(data_config.get('prefetch_factor', 4) if num_workers > 0 else None),
        worker_init_fn=worker_init_with_memlog,
    )
    
    return train_loader


def profile_ssl_training(config_path: str, num_iterations: int = 100):
    """
    Profile SSL training pipeline by timing each component.
    
    Args:
        config_path: Path to configuration file
        num_iterations: Number of training iterations to profile
    """
    logger.info("="*60)
    logger.info("SSL TRAINING PERFORMANCE PROFILER")
    logger.info("="*60)
    
    # Load configuration
    logger.info(f"Loading config from: {config_path}")
    config = load_experiment_config(config_path)
    
    # Setup reproducibility
    setup_reproducibility(config.get('seed', 42))
    
    # Determine device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Ensure NVML and allocator stats point to the intended CUDA device
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(torch.device(device))
        except Exception:
            pass

    # Log current settings
    logger.info("\nCurrent Configuration:")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Num workers: {config['data'].get('num_workers', 'default')}")
    logger.info(f"  Pin memory: {config['data'].get('pin_memory', 'default')}")
    
    # Create embedders
    logger.info("\nCreating embedders...")
    start_time = perf_counter()
    embedders = create_embedders_from_config(
        config['model']['embedders'],
        device='cpu'  # Keep embedders on CPU during data loading
    )
    embedder_time = perf_counter() - start_time
    logger.info(f"Embedder creation time: {embedder_time:.3f}s")
    
    # Create data loaders
    logger.info("\nCreating data loaders...")
    start_time = perf_counter()
    
    train_loader = create_data_loaders(config, embedders, debug=False)
    dataloader_time = perf_counter() - start_time
    logger.info(f"DataLoader creation time: {dataloader_time:.3f}s")
    logger.info(f"Training batches per epoch: {len(train_loader)}")
    
    # ---- optional: NVML baseline ----
    nvml_handle = None
    if torch.cuda.is_available():
        try:
            import pynvml  # lazy import to avoid hard dep
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            logger.info(f"Baseline GPU Used (NVML): {_bytes_to_mb(info.used):.1f} MB / "
                        f"{_bytes_to_mb(info.total):.0f} MB")
        except Exception:
            nvml_handle = None
    # ------------------------------------
    
    # Profile data loading only
    logger.info(f"\nProfiling {num_iterations} data loading iterations...")
    logger.info("-" * 60)
    
    # Timing storage - separate first iteration from steady state
    first_iteration_times = {}
    steady_state_data_times = []
    steady_state_gpu_times = []
    steady_state_total_times = []
    
    # ---- steady-state GPU memory storages (summary only) ----
    steady_cuda_alloc = []
    steady_cuda_peak = []
    steady_nvml_used = []
    steady_cuda_reserved = []
    steady_cuda_peak_reserved = []

    # H2D bytes and bandwidth (steady-state; iterations 2..N)
    steady_h2d_bytes = []
    steady_bw_gbps = []
    # ---------------------------------------------------------
    
    # Get iterator
    data_iter = iter(train_loader)
    
    for iteration in range(num_iterations):
        iter_start = perf_counter()
        
        try:
            # 1) Data Loading (this includes collate function)
            data_start = perf_counter()
            batch = next(data_iter)
            data_time = perf_counter() - data_start

            # 2) GPU Transfer: measure bytes and bandwidth + use CUDA Events for accurate timing
            if torch.cuda.is_available():
                h2d_bytes = _count_h2d_bytes_one_level(batch)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                batch = _move_batch_to_device_like_training(batch, device=torch.device(device))
                end.record()
                torch.cuda.synchronize()  # wait for the copy to finish
                gpu_time = start.elapsed_time(end) / 1000.0  # seconds

                # Accumulate CUDA memory stats for end-of-run summary
                cuda_alloc_b = torch.cuda.memory_allocated()
                cuda_peak_b = torch.cuda.max_memory_allocated()
                steady_cuda_alloc.append(cuda_alloc_b)
                steady_cuda_peak.append(cuda_peak_b)

                cuda_res_b = torch.cuda.memory_reserved()
                cuda_peak_res_b = torch.cuda.max_memory_reserved()
                steady_cuda_reserved.append(cuda_res_b)
                steady_cuda_peak_reserved.append(cuda_peak_res_b)

                if nvml_handle is not None:
                    try:
                        import pynvml
                        info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
                        steady_nvml_used.append(info.used)
                    except Exception:
                        pass
            else:
                h2d_bytes = 0
                gpu_time = 0.0
            
            total_time = perf_counter() - iter_start
            
            # Per-iteration logging suffix: ONLY H2D bytes and bandwidth (no CUDA mem numbers)
            extra = []
            if h2d_bytes and gpu_time > 0:
                extra.append(f"H2D={h2d_bytes/1024/1024:.1f}MiB")
                extra.append(f"BW={h2d_bytes/gpu_time/1e9:.2f}GB/s")
            suffix = (" | " + ", ".join(extra)) if extra else ""
            
            # Log iteration details
            logger.info(f"Iteration {iteration + 1:2d}: data_load={data_time:.3f}s, "
                        f"gpu_transfer={gpu_time:.3f}s, total={total_time:.3f}s{suffix}")
            
            # Log batch info for first iteration
            if iteration == 0:
                batch_size = batch['code_ids'].shape[0]
                max_seq_len = batch['code_ids'].shape[1]
                result_text_len = batch['result_text'].shape[1] if 'result_text' in batch else 0
                logger.info(f"    Batch info: B={batch_size}, T_max={max_seq_len}")
                logger.info(f"    Code IDs shape: {batch['code_ids'].shape}")
                if 'result_text' in batch:
                    logger.info(f"    Result text shape: {batch['result_text'].shape}")
                
                first_iteration_times = {
                    'data': data_time,
                    'gpu': gpu_time,
                    'total': total_time
                }
            else:
                # Store steady-state iterations (2-N)
                steady_state_data_times.append(data_time)
                steady_state_gpu_times.append(gpu_time)
                steady_state_total_times.append(total_time)
                # Steady-state H2D bytes and bandwidth
                if torch.cuda.is_available() and (h2d_bytes > 0) and (gpu_time > 0):
                    steady_h2d_bytes.append(h2d_bytes)
                    steady_bw_gbps.append(h2d_bytes / gpu_time / 1e9)  # GB/s

        except StopIteration:
            logger.warning(f"DataLoader exhausted at iteration {iteration + 1}")
            break
        except Exception as e:
            logger.error(f"Error during iteration {iteration + 1}: {e}")
            break
    
    # Calculate steady-state averages (excluding first iteration)
    if steady_state_data_times:
        avg_data = sum(steady_state_data_times) / len(steady_state_data_times)
        avg_gpu = sum(steady_state_gpu_times) / len(steady_state_gpu_times)
        avg_total = sum(steady_state_total_times) / len(steady_state_total_times)
        n_steady = len(steady_state_data_times)
    else:
        logger.warning("No steady-state iterations completed!")
        return
    
    # Summary reporting
    logger.info("-" * 60)
    logger.info("PROFILING SUMMARY")
    logger.info("-" * 60)
    
    # First iteration (initialization cost)
    logger.info("FIRST ITERATION (Initialization Cost):")
    logger.info(f"  Data Loading (inc. collate): {first_iteration_times['data']:7.3f}s")
    logger.info(f"  GPU Transfer:               {first_iteration_times['gpu']:7.3f}s")
    logger.info(f"  TOTAL:                      {first_iteration_times['total']:7.3f}s")
    logger.info("")
    
    # Steady-state performance (iterations 2-N)
    logger.info(f"STEADY-STATE AVERAGE (Iterations 2-{n_steady+1}):")
    logger.info(f"  Data Loading (inc. collate): {avg_data:7.3f}s ({avg_data/avg_total*100:.1f}%)")
    logger.info(f"  GPU Transfer:               {avg_gpu:7.3f}s ({avg_gpu/avg_total*100:.1f}%)")
    logger.info(f"  TOTAL PER ITERATION:        {avg_total:7.3f}s (100.0%)")
    logger.info("")

    # H2D / Bandwidth summary (steady-state)
    if steady_h2d_bytes or steady_bw_gbps:
        logger.info("H2D / BANDWIDTH SUMMARY (steady-state):")
    if steady_h2d_bytes:
        h2d_mib = [x / (1024 * 1024) for x in steady_h2d_bytes]
        logger.info(f"  H2D per-iter:   avg={sum(h2d_mib)/len(h2d_mib):.1f}MiB "
                    f"(min={min(h2d_mib):.1f}, max={max(h2d_mib):.1f})")
    if steady_bw_gbps:
        logger.info(f"  H2D bandwidth:  avg={sum(steady_bw_gbps)/len(steady_bw_gbps):.2f}GB/s "
                    f"(min={min(steady_bw_gbps):.2f}, max={max(steady_bw_gbps):.2f})")
    if steady_h2d_bytes or steady_bw_gbps:
        logger.info("")

    # ---- GPU memory summary (no per-iter prints) ----
    if torch.cuda.is_available() and steady_cuda_alloc:
        ca = [_bytes_to_mb(x) for x in steady_cuda_alloc]
        cp = [_bytes_to_mb(x) for x in steady_cuda_peak]
        logger.info(f"  CUDA allocated (current tensors): avg={sum(ca)/len(ca):.1f}MB "
                    f"(min={min(ca):.1f}, max={max(ca):.1f})")
        logger.info(f"  CUDA per-iter peak (since reset): avg={sum(cp)/len(cp):.1f}MB "
                    f"(min={min(cp):.1f}, max={max(cp):.1f})")
        cr = [_bytes_to_mb(x) for x in steady_cuda_reserved] if steady_cuda_reserved else []
        cpr = [_bytes_to_mb(x) for x in steady_cuda_peak_reserved] if steady_cuda_peak_reserved else []
        if cr:
            logger.info(f"  CUDA reserved (allocator cache): avg={sum(cr)/len(cr):.1f}MB "
                        f"(min={min(cr):.1f}, max={max(cr):.1f})")
        if cpr:
            logger.info(f"  CUDA per-iter peak reserved:     avg={sum(cpr)/len(cpr):.1f}MB "
                        f"(min={min(cpr):.1f}, max={max(cpr):.1f})")
    # Optional (verbose): allocator report similar to nvidia-smi detail
    if torch.cuda.is_available():
        try:
            logger.info("\n" + torch.cuda.memory_summary())
        except Exception:
            pass

    if steady_nvml_used:
        nu = [_bytes_to_mb(x) for x in steady_nvml_used]
        logger.info(f"  NVML used (process-level): avg={sum(nu)/len(nu):.1f}MB "
                    f"(min={min(nu):.1f}, max={max(nu):.1f})")
    # --------------------------------
    
    # Estimated epoch time using steady-state performance
    estimated_epoch_hours = (avg_total * len(train_loader)) / 3600
    logger.info(f"Estimated time per epoch (steady-state): {estimated_epoch_hours:.2f} hours")

    # --- force workers to emit "exit" lines before shutdown ---
    try:
        import os, signal, time
        if 'data_iter' in locals() and hasattr(data_iter, "_workers") and data_iter._workers:
            for w in list(data_iter._workers):
                try:
                    os.kill(w.pid, signal.SIGTERM)
                except Exception:
                    pass
            time.sleep(0.3)
    except Exception:
        pass
    # ----------------------------------------------------------
    
    # Ensure DataLoader workers have terminated so exit logs are flushed.
    try:
        if 'data_iter' in locals() and hasattr(data_iter, "_shutdown_workers"):
            data_iter._shutdown_workers()  # internal but stable enough for profiling
    except Exception:
        pass
    # Drop references and force GC to hasten teardown on some platforms
    try:
        del data_iter
    except Exception:
        pass
    try:
        del train_loader
    except Exception:
        pass
    gc.collect()

    logger.info("="*60)
    logger.info("PROFILING COMPLETE")
    logger.info("="*60)

    # ---- NVML cleanup ----
    if 'nvml_handle' in locals() and nvml_handle is not None:
        try:
            import pynvml
            pynvml.nvmlShutdown()
        except Exception:
            pass
    # ---------------------------


def main():
    """Main profiling function."""
    if len(sys.argv) != 2:
        print("Usage: python DL_profiling/profile_dl.py <config_path>")
        print("Example: python DL_profiling/profile_dl.py config/experiments/dl_profiling.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        profile_ssl_training(config_path, 200)
    except KeyboardInterrupt:
        logger.info("\nProfiling interrupted by user")
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        raise


if __name__ == "__main__":
    main()
