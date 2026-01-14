# scripts/train_lab_test.py

import os
import sys
import argparse
import torch
import yaml
from datetime import datetime
from pathlib import Path
from functools import partial
from itertools import islice

# Calculate project root (script is in scripts/ directory, so go up one level)
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Change working directory to project root to ensure relative paths work
os.chdir(project_root)

# Verify we're in the correct directory
if not (project_root / "src").exists() or not (project_root / "config").exists():
    raise RuntimeError(f"Not in project root directory. Current: {project_root}")

from src.downstream.lab_test.model.datamodule import LabTestDataset, LabTestPersonBatchSampler, lab_test_collate_fn
from src.downstream.lab_test.training.trainer import LabTestTrainer
from src.models import create_embedders_from_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train downstream lab test prediction model')
    
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
                       help='Dry run: check data loading and SSL backbone without training')
    parser.add_argument('--no-timestamp', action='store_true',
                       help='Do not add timestamp to experiment name')
    parser.add_argument('--test-only', action='store_true',
                       help='Run test evaluation only (requires --resume)')
    
    return parser.parse_args()


def create_downstream_data_loaders(config, embedders, debug=False):
    """
    Create downstream training and validation data loaders.
    
    Args:
        config: Experiment configuration
        embedders: EmbedderBundle for collate function
        debug: Whether to use debug mode (small subset)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Extract configuration sections
    data_config = config['data']
    datamodule_config = config['datamodule']
    
    print("\nCreating downstream datasets...")
    
    # Create datasets
    dataset = LabTestDataset(
        manifest_path=data_config['manifest_path'],
        labels_path=data_config['label_source'],
        label_order=datamodule_config['label_processing']['label_order'],
        # SSL base parameters (for HealthExamDataset inheritance)
        # split_name will be overridden by manifest_path
        mcinfo_dir=data_config['mcinfo_dir'],
        demographics_path=data_config['demographics_path'],
        use_result=data_config.get('use_result', True),
        result_path=data_config.get('result_path'),
        use_interview=data_config.get('use_interview', False),
        interview_path=data_config.get('interview_path'),
        use_pretokenized_result=data_config.get('use_pretokenized_result', False),
        result_tokenized_path=data_config.get('result_tokenized_path', None),
        mcinfo_materialized_path=data_config.get('mcinfo_materialized_path', None),
        mcinfo_rg_cache_size=data_config.get('mcinfo_rg_cache_size', 2),
        mcinfo_validate_footer=data_config.get('mcinfo_validate_footer', True)
    )

    print(f"Dataset: {len(dataset)} samples (all data loaded)")
    
    # Create samplers
    batch_size = datamodule_config['batch_size']
    drop_last = datamodule_config.get('drop_last', False)
    
    train_sampler = LabTestPersonBatchSampler(
        manifest_path=data_config['manifest_path'],
        batch_size=batch_size,
        mode='train',
        shuffle=datamodule_config['sampler']['shuffle_train'],
        drop_last=drop_last
    )
    
    val_sampler = LabTestPersonBatchSampler(
        manifest_path=data_config['manifest_path'],
        batch_size=batch_size,
        mode='val',
        shuffle=datamodule_config['sampler']['shuffle_val'],
        drop_last=False  # Never drop last for validation
    )
    
    test_sampler = LabTestPersonBatchSampler(
        manifest_path=data_config['manifest_path'],
        batch_size=batch_size,
        mode='test',
        shuffle=datamodule_config['sampler']['shuffle_test'],
        drop_last=False  # Never drop last for test
    )
    
    print(f"Train sampler: {len(train_sampler.persons)} persons")
    print(f"Val sampler: {len(val_sampler.persons)} persons")
    print(f"Test sampler: {len(test_sampler.persons)} persons")
        
    # Create mode-specific collate functions
    train_collate_fn = partial(
        lab_test_collate_fn,
        code_embedder=embedders.categorical,
        text_embedder=embedders.text,
        config=config,
        mode='train',
        device='cpu'
    )
    
    val_collate_fn = partial(
        lab_test_collate_fn,
        code_embedder=embedders.categorical,
        text_embedder=embedders.text,
        config=config,
        mode='val',
        device='cpu'
    )
    
    test_collate_fn = partial(
        lab_test_collate_fn,
        code_embedder=embedders.categorical,
        text_embedder=embedders.text,
        config=config,
        mode='test',
        device='cpu'
    )
    
    # Create data loaders
    num_workers = data_config.get('num_workers', 4)
    pin_memory = data_config.get('pin_memory', True)
    prefetch_factor = data_config.get('prefetch_factor', 2)
    
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_sampler,
        collate_fn=train_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=val_sampler,
        collate_fn=val_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=test_sampler,
        collate_fn=test_collate_fn,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
        prefetch_factor=max(1, prefetch_factor // 2),
        persistent_workers=False
    )
    
    # Debug mode: use smaller subsets
    if debug:
        debug_config = config['debug']
        print(f"Debug mode: Converting to finite lists ({debug_config['limit_train_batches']} train, {debug_config['limit_val_batches']} val, {debug_config['limit_test_batches']} test batches)")
        train_loader = list(islice(train_loader, debug_config['limit_train_batches']))
        val_loader = list(islice(val_loader, debug_config['limit_val_batches']))
        test_loader = list(islice(test_loader, debug_config['limit_test_batches']))
    
    return train_loader, val_loader, test_loader


def report_data_statistics(train_loader, val_loader, test_loader, config):
    """Report data loading statistics."""
    print("\n" + "="*60)
    print("DATA LOADING STATISTICS")
    print("="*60)
    
    # Check if loaders are lists (debug mode) or DataLoader objects
    if isinstance(train_loader, list):
        print(f"Train batches (debug): {len(train_loader)}")
        print(f"Val batches (debug): {len(val_loader)}")
        print(f"Test batches (debug): {len(test_loader)}")
    else:
        print(f"Train batches per epoch: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Training statistics
        num_epochs = config['training']['num_epochs']
        total_train_steps = len(train_loader) * num_epochs
        print(f"Total training steps ({num_epochs} epochs): {total_train_steps:,}")
    
    # First batch inspection (works for both debug and normal mode)
    print("\nFirst batch inspection:")
    try:
        batch = next(iter(train_loader))
        print(f"  Batch size (B): {batch['B']}")
        print(f"  Max tests per exam (T_max): {batch['T_max']}")
        if 'lab_targets' in batch:
            print(f"  Lab targets shape: {batch['lab_targets'].shape}")
            print(f"  Lab masks shape: {batch['lab_masks'].shape}")
        print(f"  Code IDs shape: {batch['code_ids'].shape}")
        print(f"  Result text shape: {batch['result_input_ids'].shape}")
    except Exception as e:
        print(f"  Error loading first batch: {e}")
    
    print("="*60)


def test_ssl_backbone_loading(config):
    """Test SSL backbone loading without training."""
    print("\n" + "="*60)
    print("SSL BACKBONE LOADING TEST")
    print("="*60)
    
    ssl_config = config['ssl_backbone']
    checkpoint_path = ssl_config['checkpoint_path']
    
    print(f"Loading SSL checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✓ SSL checkpoint loaded successfully")
        print(f"  - Checkpoint step: {checkpoint.get('global_step', 'unknown')}")
        print(f"  - Model config keys: {list(checkpoint.get('config', {}).keys())}")
        print(f"  - Model state dict size: {len(checkpoint.get('model_state_dict', {}))}")
        print(f"  - Embedders state dict: {list(checkpoint.get('embedders_state_dict', {}).keys())}")
        
    except Exception as e:
        print(f"✗ SSL checkpoint loading failed: {e}")
        raise
    
    print("="*60)


def dry_run_test(config, embedders):
    """Run dry test of data loading and SSL backbone."""
    print("\n" + "="*60)
    print("DRY RUN TEST")
    print("="*60)
    
    # Test SSL backbone loading
    test_ssl_backbone_loading(config)
    
    # Test data loading
    print("\nTesting data loading...")
    train_loader, val_loader, test_loader = create_downstream_data_loaders(
        config, embedders, debug=True
    )
    
    # Test one batch from each loader
    print("\nTesting batch processing...")
    
    for loader_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        try:
            batch = next(iter(loader))
            print(f"✓ {loader_name.capitalize()} batch processed successfully")
            print(f"  - Batch keys: {list(batch.keys())}")
            if 'lab_targets' in batch:
                print(f"  - Lab targets shape: {batch['lab_targets'].shape}")
                print(f"  - Lab masks shape: {batch['lab_masks'].shape}")
        except Exception as e:
            print(f"✗ {loader_name.capitalize()} batch processing failed: {e}")
            raise
    
    print("\n✓ Dry run completed successfully!")
    print("="*60)


def main():
    """Main training function."""
    args = parse_args()
    
    # Print current working directory
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Debug mode overrides
    if args.debug:
        debug_config = config['debug']
        print(f"Debug mode: Overriding config settings...")
        print(f"  - Epochs: {config['training']['num_epochs']} → {debug_config['num_epochs']}")
        config['training']['num_epochs'] = debug_config['num_epochs']
    
    # Handle timestamp if needed
    if not args.no_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['experiment_name'] = f"{config['experiment_name']}_{timestamp}"
    
    # Override device if specified
    if args.device:
        config['device'] = args.device

    # Get device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Experiment: {config['experiment_name']}")
    
    ssl_checkpoint = torch.load(config['ssl_backbone']['checkpoint_path'], map_location='cpu')
    embedders = create_embedders_from_config(
        ssl_checkpoint['config']['model']['embedders'],
        device='cpu'
    )
    print("✓ Embedders created from SSL checkpoint config")
    
    # Dry run mode: test data loading and SSL backbone
    if args.dry_run:
        dry_run_test(config, embedders)
        print("Dry run complete. Exiting.")
        return
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_downstream_data_loaders(
        config, embedders, debug=args.debug
    )
    
    # Report data statistics
    report_data_statistics(train_loader, val_loader, test_loader, config)
    
    # Move embedders to training device (will be handled by trainer)
    print(f"\nCreating trainer on device: {device}")
    trainer = LabTestTrainer(
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        resume_from=args.resume
    )
    
    # Test-only mode
    if args.test_only:
        if not args.resume:
            raise ValueError("--test-only requires --resume to specify model checkpoint")
        
        print("\nRunning test evaluation...")
        test_metrics = trainer.test()
        
        trainer.logger.log_test_results(test_metrics)
        
        return
    
    # Start training
    try:
        print("\nStarting training...")
        trainer.train()
        
        # Run test evaluation after training
        print("\nRunning final test evaluation...")
        test_metrics = trainer.test()
        
        trainer.logger.log_test_results(test_metrics)
            
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