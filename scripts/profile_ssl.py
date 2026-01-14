#!/usr/bin/env python
"""
Profiling script for SSL pretraining performance analysis.
Independent from training code - copies necessary components for isolation.
"""

import os
import sys
import argparse
import torch
import json
import time
from pathlib import Path
from datetime import datetime

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
os.chdir(project_root)

from src.profiling import SSLProfiler, ProfileConfig
from src.utils import load_experiment_config, setup_reproducibility


def parse_args():
    """Parse command line arguments for profiling."""
    parser = argparse.ArgumentParser(description='Profile SSL model performance')
    
    parser.add_argument('--config', type=str, default='config/experiments/ssl_profiling.yaml',
                       help='Path to profiling configuration file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for profiling results')
    parser.add_argument('--steps', type=int, default=50,
                       help='Number of steps to profile')
    parser.add_argument('--warmup-steps', type=int, default=5,
                       help='Number of warmup steps before profiling')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=None,
                       help='Batch sizes to profile (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (overrides config)')
    
    return parser.parse_args()


def main():
    """Main profiling function."""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_experiment_config(args.config, auto_timestamp=False)
    
    # Override with command line arguments
    if args.device:
        config['device'] = args.device
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'profile_results/profile_{timestamp}')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Setup device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Setup reproducibility for consistent profiling
    setup_reproducibility(seed=config.get('seed', 42), strict=False, verbose=True)
    
    # Create profile configuration
    profile_config = ProfileConfig(
        warmup_steps=args.warmup_steps,
        profile_steps=args.steps,
        output_dir=output_dir,
        profile_memory=config.get('profiling', {}).get('memory', True),
        profile_time=config.get('profiling', {}).get('time', True),
        profile_cuda=config.get('profiling', {}).get('cuda', torch.cuda.is_available()),
        breakdown_components=config.get('profiling', {}).get('breakdown', True)
    )
    
    # Determine batch sizes to profile
    if args.batch_sizes:
        batch_sizes = args.batch_sizes
    else:
        batch_sizes = config.get('profiling', {}).get('batch_sizes', [config['training']['batch_size']])
    
    print(f"\nProfiling configuration:")
    print(f"  Warmup steps: {profile_config.warmup_steps}")
    print(f"  Profile steps: {profile_config.profile_steps}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Profile memory: {profile_config.profile_memory}")
    print(f"  Profile CUDA: {profile_config.profile_cuda}")
    
    # Profile each batch size
    all_results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Profiling with batch size: {batch_size}")
        print(f"{'='*60}")
        
        # Update config with current batch size
        current_config = config.copy()
        current_config['training']['batch_size'] = batch_size
        
        # Create profiler
        profiler = SSLProfiler(
            config=current_config,
            profile_config=profile_config
        )
        
        # Run profiling
        results = profiler.profile()
        all_results[f'batch_{batch_size}'] = results
        
        # Print summary
        print(f"\n--- Summary for batch size {batch_size} ---")
        profiler.print_summary(results)
        
        # Save intermediate results
        with open(output_dir / f'profile_batch_{batch_size}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # Save combined results
    with open(output_dir / 'profile_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Profiling complete. Results saved to: {output_dir}")
    print(f"Run 'python scripts/analyze_profile.py {output_dir}' to generate visualizations")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()