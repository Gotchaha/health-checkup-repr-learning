# scripts/train_ssl.py

import os
os.environ["ARROW_PREBUFFER"] = "0"
os.environ["ARROW_NUM_THREADS"] = "1"
import signal
import sys
import argparse
import torch
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
from src.training import SSLTrainer
from src.utils import (
    load_experiment_config, setup_reproducibility,
    create_experiment_dirs, worker_init_with_cleanup
)


def setup_signal_handlers(trainer):
    """Register SIGINT / SIGTERM handlers that save a checkpoint before exit."""
    
    def _handler(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n[{sig_name}] received – saving checkpoint before exit...")
        try:
            # Use step-based naming for consistency
            checkpoint_path = os.path.join(
                trainer.dirs['checkpoint_dir'], 
                f'signal_checkpoint_step_{getattr(trainer, "global_step", 0)}.pt'
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"Signal checkpoint saved to: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to save signal checkpoint: {e}")
        # Flush logs before exit
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    # Common signals: Ctrl-C, qdel, SLURM preemption
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError):
            # Some signals might not be available on all platforms
            pass


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SSL model on medical examination data')
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment configuration file')
    
    # Optional arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (overrides config)')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode: use small subset of data')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run: check data loading without training')
    parser.add_argument('--no-timestamp', action='store_true',
                       help='Do not add timestamp to experiment name')
    
    return parser.parse_args()


def create_data_loaders(config, embedders, debug=False):
    """
    Create training and validation data loaders.
    
    Args:
        config: Experiment configuration
        embedders: EmbedderBundle for collate function
        debug: Whether to use debug mode (small subset)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Extract data configuration
    data_config = config['data']
    
    # Create datasets
    train_dataset = HealthExamDataset(
        split_name='train_ssl',
        manifest_path=data_config['manifest_path_train'],
        mcinfo_dir=data_config['mcinfo_dir'],
        demographics_path=data_config['demographics_path'],
        use_result=data_config.get('use_result', True),
        result_path=data_config.get('result_path'),
        use_interview=data_config.get('use_interview', False),
        interview_path=data_config.get('interview_path'),
        use_pretokenized_result=data_config.get('use_pretokenized_result', False),
        result_tokenized_path=data_config.get('result_tokenized_path', None),
        mcinfo_materialized_path=data_config.get('mcinfo_materialized_path_train', None),
        mcinfo_rg_cache_size=data_config.get('mcinfo_rg_cache_size', 2),
        mcinfo_validate_footer=data_config.get('mcinfo_validate_footer', True)
    )
    
    val_dataset = HealthExamDataset(
        split_name='val_ssl',
        manifest_path=data_config['manifest_path_val'],
        mcinfo_dir=data_config['mcinfo_dir'],
        demographics_path=data_config['demographics_path'],
        use_result=data_config.get('use_result', True),
        result_path=data_config.get('result_path'),
        use_interview=data_config.get('use_interview', False),
        interview_path=data_config.get('interview_path'),
        use_pretokenized_result=data_config.get('use_pretokenized_result', False),
        result_tokenized_path=data_config.get('result_tokenized_path', None),
        mcinfo_materialized_path=data_config.get('mcinfo_materialized_path_val', None),
        mcinfo_rg_cache_size=data_config.get('mcinfo_rg_cache_size', 2),
        mcinfo_validate_footer=data_config.get('mcinfo_validate_footer', True)
    )
    
    # Create samplers
    train_sampler = InfinitePersonBatchSampler(
        manifest_path=data_config['manifest_path_train'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
        drop_last=True
    )
    
    val_sampler = PersonBatchSampler(
        manifest_path=data_config['manifest_path_val'],
        batch_size=config['training']['batch_size'],
        shuffle=False,
        drop_last=False
    )
    
    # Create collate function with embedders
    collate_fn = partial(
        collate_exams,
        code_embedder=embedders.categorical,
        text_embedder=embedders.text,
        config=config,
        device='cpu'  # Collate on CPU, move to GPU in trainer
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', False),
        # multiprocessing_context=mp.get_context('spawn'),
        persistent_workers=False,
        prefetch_factor=config['data'].get('prefetch_factor', 4),
        # worker_init_fn=worker_init_with_cleanup  # Use our cleanup function
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=config['data'].get('pin_memory', False),
        # multiprocessing_context=mp.get_context('spawn'),
        persistent_workers=False,
        prefetch_factor=2,
        # worker_init_fn=worker_init_with_cleanup  # Use our cleanup function
    )

    # Debug mode: Convert to limited lists AFTER dataloader creation
    if debug:
        print("Debug mode: Converting to finite lists (20 train, 10 val batches)")
        train_loader = list(islice(train_loader, 20))
        val_loader = list(islice(val_loader, 10))
    
    return train_loader, val_loader

    
def report_data_statistics(train_loader, val_loader, train_sampler, val_sampler, config):
    """Report data statistics for verification."""
    print("\n" + "="*50)
    print("DATA STATISTICS")
    print("="*50)
    
    # Handle debug mode (when loaders are lists)
    if isinstance(train_loader, list):
        print(f"\nDEBUG MODE:")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Mode: Infinite cycling through limited batches")
    else:
        # Sampler statistics (normal mode)
        train_stats = train_sampler.get_stats() if hasattr(train_sampler, 'get_stats') else {}
        val_stats = val_sampler.get_stats() if hasattr(val_sampler, 'get_stats') else {}
        
        print(f"\nTraining data:")
        print(f"  Total persons: {train_stats.get('total_persons', 'N/A')}")
        print(f"  Total exams: {train_stats.get('total_exams', 'N/A')}")
        print(f"  Avg exams/person: {train_stats.get('avg_exams_per_person', 'N/A'):.2f}")
        print(f"  Batches per epoch: {train_stats.get('batches_per_epoch', 'N/A')}")
        
        print(f"\nValidation data:")
        print(f"  Total persons: {val_stats.get('total_persons', 'N/A')}")
        print(f"  Total exams: {val_stats.get('total_exams', 'N/A')}")
        print(f"  Avg exams/person: {val_stats.get('avg_exams_per_person', 'N/A'):.2f}")
        print(f"  Batches per epoch: {val_stats.get('batches_per_epoch', 'N/A')}")
    
    # Step-based training information
    max_steps = config['training']['max_steps']
    validation_freq = config['training']['validation_freq']
    early_stopping_cfg = config['training'].get('early_stopping', {})
    patience = early_stopping_cfg.get('patience')
    early_stopping_enabled = early_stopping_cfg.get('enabled', True)
    
    print(f"\nStep-based training configuration:")
    print(f"  Max steps: {max_steps:,}")
    print(f"  Validation frequency: {validation_freq:,} steps")
    if early_stopping_enabled and patience is not None:
        print(f"  Early stopping patience: {patience:,} steps ({patience // validation_freq} validation cycles)")
    else:
        print("  Early stopping: disabled")
    
    # Sample first batch to check shapes
    print("\nFirst batch shapes:")
    try:
        first_batch = next(iter(train_loader))
        print(f"  Batch size (B): {first_batch['B']}")
        print(f"  Max tests per exam (T_max): {first_batch['T_max']}")
        print(f"  Code IDs shape: {first_batch['code_ids'].shape}")
        print(f"  Result text shape: {first_batch['result_input_ids'].shape}")
        print(f"  Segment lengths: {first_batch['segment_lengths'][:5]}...")
    except Exception as e:
        print(f"  Error loading first batch: {e}")
    
    print("="*50 + "\n")


def main():
    """Main training function."""
    args = parse_args()
    
    # Print current working directory for verification
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_experiment_config(
        args.config,
        auto_timestamp=not args.no_timestamp,
        validate=True
    )
    
    # Override device if specified
    if args.device:
        config['device'] = args.device

    # Get device for training
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup experiment directories
    dirs = create_experiment_dirs(config)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Checkpoint directory: {dirs['checkpoint_dir']}")
    print(f"Log directory: {dirs['log_dir']}")
    
    # Setup reproducibility
    env_info = setup_reproducibility(
        seed=config.get('seed', 42),
        strict=config.get('deterministic', False),
        verbose=True
    )
    
    # Create embedders
    print("\nCreating embedders...")
    embedders = create_embedders_from_config(
        config['model']['embedders'],
        device='cpu'  # Keep embedders on CPU during data loading
    )
    
    # Report embedder information
    print(f"\nEmbedder information:")
    print(embedders)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(config, embedders, debug=args.debug)
    
    # Get the underlying samplers for statistics (None if debug mode lists)
    train_sampler = getattr(train_loader, "batch_sampler", None)
    val_sampler = getattr(val_loader, "batch_sampler", None)
    
    # Report data statistics
    report_data_statistics(train_loader, val_loader, train_sampler, val_sampler, config)
    
    # Dry run mode: exit after data verification
    if args.dry_run:
        print("Dry run complete. Exiting without training.")
        return
    
    # Move embedders to training device
    embedders.to(device)
    
    # Create trainer
    print(f"\nCreating trainer on device: {device}")
    trainer = SSLTrainer(
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        resume_from=args.resume
    )

    # Setup signal handlers for HPC environments
    setup_signal_handlers(trainer)
    
    # Start training
    try:
        print("\nStarting training...")
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        trainer.save_checkpoint()
        print("Checkpoint saved. Exiting.")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save emergency checkpoint
        try:
            emergency_path = os.path.join(
                trainer.dirs['checkpoint_dir'], 
                'emergency_checkpoint.pt'
            )
            trainer.save_checkpoint(emergency_path)
            print(f"Emergency checkpoint saved to: {emergency_path}")
        except:
            print("Failed to save emergency checkpoint")
        
        raise


if __name__ == "__main__":
    main()
