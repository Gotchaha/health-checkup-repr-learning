# scripts/pre_profile_ssl.py
"""
Performance profiling script for SSL data loading pipeline.
Profiles data loading and GPU transfer to identify bottlenecks.

Usage:
    python scripts/pre_profile_ssl.py config/experiments/ssl_pretraining_test.yaml
"""

import os
os.environ["ARROW_PREBUFFER"] = "0"
os.environ["ARROW_NUM_THREADS"] = "1"
import sys
import time
import torch
import logging
import torch.multiprocessing as mp
from pathlib import Path
from functools import partial
from itertools import islice

# Calculate project root (script is in scripts/ directory, so go up one level)
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Change working directory to project root to ensure relative paths work
os.chdir(project_root)

# Verify we're in the correct directory by checking for expected files
if not (project_root / "src").exists() or not (project_root / "config").exists():
    raise RuntimeError(f"Not in project root directory. Current: {project_root}")

from src.models import (
    HealthExamDataset, PersonBatchSampler, InfinitePersonBatchSampler, collate_exams,
    create_embedders_from_config
)
from src.utils import (
    load_experiment_config, setup_reproducibility,
    worker_init_with_cleanup  # Only import the worker init function
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        result_tokenized_path=data_config.get('result_tokenized_path', None)
    )
    
    val_dataset = HealthExamDataset(
        split_name='val_ssl',
        mcinfo_dir=data_config['mcinfo_dir'],
        demographics_path=data_config['demographics_path'],
        use_result=data_config.get('use_result', True),
        result_path=data_config.get('result_path'),
        use_interview=data_config.get('use_interview', False),
        interview_path=data_config.get('interview_path'),
        use_pretokenized_result=data_config.get('use_pretokenized_result', False),
        result_tokenized_path=data_config.get('result_tokenized_path', None)
    )
    
    # Create samplers
    train_sampler = InfinitePersonBatchSampler(
        manifest_path="data/splits/core/sorted/train_ssl.parquet",
        batch_size=config['training']['batch_size'],
        shuffle=True,
        drop_last=True
    )
    
    val_sampler = PersonBatchSampler(
        manifest_path="data/splits/core/sorted/val_ssl.parquet",
        batch_size=config['training']['batch_size'],
        shuffle=False,
        drop_last=False
    )
    
    # Debug mode: use only first few batches
    if debug:
        print("Debug mode: taking first 10 training / 5 validation batches")
        train_sampler = list(islice(train_sampler, 10))
        val_sampler = list(islice(val_sampler, 5))
    
    # Create collate function with embedders
    collate_fn = partial(
        collate_exams,
        code_embedder=embedders.categorical,
        text_embedder=embedders.text,
        config=config,
        device='cpu'  # Collate on CPU, move to GPU in trainer
    )
    
    # Create data loaders with multiprocessing cleanup support
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', False),
        multiprocessing_context=mp.get_context('spawn'),
        persistent_workers=True,
        prefetch_factor=data_config.get('prefetch_factor', 4),
        worker_init_fn=worker_init_with_cleanup  # Use our cleanup function
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', False),
        multiprocessing_context=mp.get_context('spawn'),
        persistent_workers=True,
        prefetch_factor=data_config.get('prefetch_factor', 4),
        worker_init_fn=worker_init_with_cleanup  # Use our cleanup function
    )
    
    return train_loader, val_loader


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
    
    # Log current settings
    logger.info("\nCurrent Configuration:")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Num workers: {config['data'].get('num_workers', 'default')}")
    logger.info(f"  Pin memory: {config['data'].get('pin_memory', 'default')}")
    logger.info(f"  Model layers: {config['model']['n_cross_layers']}/{config['model']['n_uni_layers']}/{config['model']['n_ind_layers']}")
    
    # Create embedders
    logger.info("\nCreating embedders...")
    start_time = time.time()
    embedders = create_embedders_from_config(
        config['model']['embedders'],
        device='cpu'  # Keep embedders on CPU during data loading
    )
    embedder_time = time.time() - start_time
    logger.info(f"Embedder creation time: {embedder_time:.3f}s")
    
    # Create data loaders
    logger.info("\nCreating data loaders...")
    start_time = time.time()
    train_loader, val_loader = create_data_loaders(config, embedders, debug=False)
    dataloader_time = time.time() - start_time
    logger.info(f"DataLoader creation time: {dataloader_time:.3f}s")
    logger.info(f"Training batches per epoch: {len(train_loader)}")
    
    # Profile data loading only
    logger.info(f"\nProfiling {num_iterations} data loading iterations...")
    logger.info("-" * 60)
    
    # Timing storage - separate first iteration from steady state
    first_iteration_times = {}
    steady_state_data_times = []
    steady_state_gpu_times = []
    steady_state_total_times = []
    
    # Get iterator
    data_iter = iter(train_loader)
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        try:
            # 1. Data Loading (this includes collate function)
            data_start = time.time()
            batch = next(data_iter)
            data_time = time.time() - data_start

            # ADD HELD-OUT LOGGING HERE (after batch loading, before GPU timing)
            if 'held_out_cells_count' in batch:
                total_held_out = batch.get('held_out_cells_count', 0)
                if total_held_out > 0:
                    codes = batch.get('held_out_codes_in_batch', [])
                    logger.info(f"    Profile batch {iteration + 1}: {total_held_out} held-out cells "
                            f"(codes: {codes[:5]}{'...' if len(codes) > 5 else ''})")
                else:
                    logger.info(f"    Profile batch {iteration + 1}: No held-out codes in this batch")
            
            # 2. GPU Transfer (simulate moving to device)
            gpu_start = time.time()
            if torch.cuda.is_available():
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        batch[key] = value.cuda(non_blocking=True)
            gpu_time = time.time() - gpu_start
            
            total_time = time.time() - iter_start
            
            # Log iteration details
            logger.info(f"Iteration {iteration + 1:2d}: data_load={data_time:.3f}s, gpu_transfer={gpu_time:.3f}s, total={total_time:.3f}s")
            
            # Log batch info for first iteration
            if iteration == 0:
                batch_size = batch['code_ids'].shape[0]
                max_seq_len = batch['code_ids'].shape[1]
                result_text_len = batch['result_text'].shape[1] if 'result_text' in batch else 0
                logger.info(f"    Batch info: B={batch_size}, T_max={max_seq_len}")
                logger.info(f"    Code IDs shape: {batch['code_ids'].shape}")
                if 'result_text' in batch:
                    logger.info(f"    Result text shape: {batch['result_text'].shape}")
                
                # Store first iteration separately
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
    
    # Estimated epoch time using steady-state performance
    estimated_epoch_hours = (avg_total * len(train_loader)) / 3600
    logger.info(f"Estimated time per epoch (steady-state): {estimated_epoch_hours:.2f} hours")

    logger.info("="*60)
    logger.info("PROFILING COMPLETE")
    logger.info("="*60)


def main():
    """Main profiling function."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/pre_profile_ssl.py <config_path>")
        print("Example: python scripts/pre_profile_ssl.py config/experiments/ssl_pretraining_test.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        profile_ssl_training(config_path)
    except KeyboardInterrupt:
        logger.info("\nProfiling interrupted by user")
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        raise


if __name__ == "__main__":
    main()