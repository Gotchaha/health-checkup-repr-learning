# scripts/data_preparation/evaluation_slices_splits/make_fine_grained_splits.py

"""
Main script for generating fine-grained evaluation slices.

This script creates the following splits:
1. Cold-Start (CS-train, CS-test)
2. Sparse/Dense trajectories
3. IID sanity split
4. Test-Future with demographics

Usage:
    python make_fine_grained_splits.py --config config/splitting/evaluation_slices_config.yaml
"""

import os
import sys
import logging
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.data.evaluation_slices.utils.manifest_utils import (
    load_manifest, save_manifest, get_manifest_stats
)
from src.data.evaluation_slices.utils.validation_utils import (
    validate_no_record_overlap, validate_cold_start_integrity
)
from src.data.evaluation_slices.splits.cold_start_split import create_cold_start_split
from src.data.evaluation_slices.splits.density_split import create_density_split
from src.data.evaluation_slices.splits.iid_split import create_iid_split
from src.data.evaluation_slices.splits.demographic_utils import add_demographics

def setup_logging(log_dir):
    """Set up logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/fine_grained_splits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return log_file

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_metadata(metadata, output_path):
    """Save metadata to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    logger.info(f"Saved metadata to {output_path}")

def save_stats(stats, output_path):
    """Save statistics to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {output_path}")

def main(args):
    """Main execution function."""
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Create output directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['stats_dir'], exist_ok=True)
    
    # Load core split manifests
    logger.info("Loading core split manifests")
    train_ssl_df = load_manifest(config['paths']['train_ssl'])
    val_ssl_df = load_manifest(config['paths']['val_ssl'])
    test_future_df = load_manifest(config['paths']['test_future'])
    
    # Load all exams for density split
    logger.info("Loading all exams for density split")
    all_exams_df = pd.concat([train_ssl_df, val_ssl_df, test_future_df], ignore_index=True)
    
    # Load demographics data
    logger.info("Loading demographics data")
    demographics_df = pd.read_parquet(config['paths']['demographics'])
    
    # Load visit thresholds
    visit_thresholds = load_config(config['paths']['visit_thresholds'])
    
    # 1. Create Cold-Start split
    logger.info("Creating Cold-Start split")
    cold_start_k = visit_thresholds.get('cold_start_K', 3)
    cs_train_df, cs_test_df = create_cold_start_split(test_future_df, k=cold_start_k)
    
    # Validate Cold-Start split
    validate_cold_start_integrity(cs_train_df, cs_test_df)
    
    # Save Cold-Start manifests
    save_manifest(cs_train_df, config['paths']['output_dir'] + '/cold_start/cs_train.parquet')
    save_manifest(cs_test_df, config['paths']['output_dir'] + '/cold_start/cs_test.parquet')
    
    # Save Cold-Start statistics
    cs_stats = {
        "cs_train": get_manifest_stats(cs_train_df, "cs_train"),
        "cs_test": get_manifest_stats(cs_test_df, "cs_test"),
        "k_value": cold_start_k
    }
    save_stats(cs_stats, config['paths']['stats_dir'] + '/cold_start_stats.json')
    
    # 2. Create Sparse/Dense trajectory splits
    logger.info("Creating Sparse/Dense trajectory splits")
    sparse_max = visit_thresholds.get('sparse_max', 10)
    dense_min = visit_thresholds.get('dense_min', 30)
    sparse_df, dense_df = create_density_split(
        test_future_df, all_exams_df, sparse_max=sparse_max, dense_min=dense_min
    )
    
    # Save Density manifests
    save_manifest(sparse_df, config['paths']['output_dir'] + '/density/sparse.parquet')
    save_manifest(dense_df, config['paths']['output_dir'] + '/density/dense.parquet')
    
    # Save Density statistics
    density_stats = {
        "sparse": get_manifest_stats(sparse_df, "sparse"),
        "dense": get_manifest_stats(dense_df, "dense"),
        "thresholds": {
            "sparse_max": sparse_max,
            "dense_min": dense_min
        }
    }
    save_stats(density_stats, config['paths']['stats_dir'] + '/density_stats.json')
    
    # 3. Create IID sanity split
    logger.info("Creating IID sanity split")
    iid_train_df, iid_test_df = create_iid_split(
        train_ssl_df, test_ratio=0.2, random_seed=config.get('random_seed', 0)
    )
    
    # Save IID manifests
    save_manifest(iid_train_df, config['paths']['output_dir'] + '/iid/iid_train.parquet')
    save_manifest(iid_test_df, config['paths']['output_dir'] + '/iid/iid_test.parquet')
    
    # Save IID statistics
    iid_stats = {
        "iid_train": get_manifest_stats(iid_train_df, "iid_train"),
        "iid_test": get_manifest_stats(iid_test_df, "iid_test"),
        "test_ratio": 0.2,
        "random_seed": config.get('random_seed', 0)
    }
    save_stats(iid_stats, config['paths']['stats_dir'] + '/iid_stats.json')
    
    # 4. Add demographics to Test-Future
    logger.info("Adding demographics to Test-Future")
    test_future_with_demographics = add_demographics(test_future_df, demographics_df)
    
    # Save demographic-enhanced Test-Future
    save_manifest(
        test_future_with_demographics, 
        config['paths']['output_dir'] + '/demographic/test_future_with_demographics.parquet'
    )
    
    # Save demographic statistics
    demographic_stats = {
        "age_distribution": test_future_with_demographics['age_bucket'].value_counts().to_dict(),
        "sex_distribution": test_future_with_demographics['sex'].value_counts().to_dict(),
        "missing_values": {
            "sex": int(test_future_with_demographics['sex'].isna().sum()),
            "age": int(test_future_with_demographics['age_bucket'].isna().sum())
        }
    }
    save_stats(demographic_stats, config['paths']['stats_dir'] + '/demographic_stats.json')
    
    # 5. Create comprehensive evaluation slices metadata
    metadata = {
        "generation_timestamp": datetime.now().isoformat(),
        "source_data": {
            "train_ssl_count": len(train_ssl_df),
            "val_ssl_count": len(val_ssl_df),
            "test_future_count": len(test_future_df)
        },
        "cold_start": {
            "k_value": cold_start_k,
            "train_count": len(cs_train_df),
            "test_count": len(cs_test_df),
            "eligible_persons": cs_train_df['person_id'].nunique()
        },
        "density": {
            "sparse_max": sparse_max,
            "dense_min": dense_min,
            "sparse_count": len(sparse_df),
            "sparse_persons": sparse_df['person_id'].nunique(),
            "dense_count": len(dense_df),
            "dense_persons": dense_df['person_id'].nunique()
        },
        "iid": {
            "test_ratio": 0.2,
            "random_seed": config.get('random_seed', 0),
            "train_count": len(iid_train_df),
            "test_count": len(iid_test_df)
        },
        "demographic": {
            "enhanced_count": len(test_future_with_demographics),
            "age_buckets": list(map(str, test_future_with_demographics['age_bucket'].dropna().unique())),
            "sex_categories": list(test_future_with_demographics['sex'].dropna().unique())
        },
        "random_seed": config.get('random_seed', 0)
    }
    
    save_metadata(metadata, config['paths']['metadata_output'])
    
    logger.info("Fine-grained evaluation slices generation completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fine-grained evaluation slices")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/splitting/evaluation_slices_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Set up logging with correct path
    log_dir = "outputs/evaluation_slices/logs"
    log_file = setup_logging(log_dir)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logs will be saved to {log_file}")
    
    main(args)