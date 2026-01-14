# scripts/data_preparation/make_chronological_splits.py
"""
Split Generator for Chronological 70/15/15 Macro-Split

This script:
1. Loads exam data from year-partitioned Parquet files
2. Computes 70th and 85th percentiles of exam dates to define split boundaries
3. Creates train/val/test split manifest files
4. Generates metadata about the splits
5. Verifies the integrity of the splits
"""

import os
import sys
import argparse
import hashlib
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(description="Generate chronological 70/15/15 data splits")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed/mcinfo/exam_level",
        help="Directory containing year-partitioned exam data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits",
        help="Directory to write split manifests",
    )
    parser.add_argument(
        "--meta_dir",
        type=str,
        default="config/splitting",
        help="Directory to write split metadata",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def compute_source_hash(input_dir):
    """Compute SHA-256 hash of source data directories structure"""
    hasher = hashlib.sha256()
    
    # Get all year partition folders
    year_dirs = sorted([d for d in os.listdir(input_dir) if d.startswith("year=")])
    
    for year_dir in year_dirs:
        dir_path = os.path.join(input_dir, year_dir)
        parquet_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".parquet")])
        
        for parquet_file in parquet_files:
            file_path = os.path.join(dir_path, parquet_file)
            # Update hash with file name and modification time
            file_stat = os.stat(file_path)
            file_info = f"{file_path}:{file_stat.st_size}:{file_stat.st_mtime}"
            hasher.update(file_info.encode())
    
    return hasher.hexdigest()


def load_exam_data(input_dir):
    """
    Load all exam data from year-partitioned Parquet files
    
    Args:
        input_dir: Directory containing year=YYYY partitions
        
    Returns:
        pandas DataFrame with exam_id, person_id, and ExamDate columns
    """
    print(f"Loading exam data from {input_dir}...")
    
    # Use PyArrow dataset to load all partitioned data
    try:
        dataset = ds.dataset(
            input_dir,
            format="parquet",
            partitioning="hive"  # Use hive-style partitioning discovery
        )
        
        # Project only the needed columns and convert to pandas
        df = dataset.to_table(columns=["exam_id", "person_id", "ExamDate"]).to_pandas()
        
    except Exception as e:
        print(f"Error using PyArrow dataset: {e}")
        print("Falling back to direct file loading...")
        
        # Fallback: Load each partition manually
        dfs = []
        year_dirs = sorted([d for d in os.listdir(input_dir) if d.startswith("year=")])
        
        for year_dir in year_dirs:
            dir_path = os.path.join(input_dir, year_dir)
            parquet_files = [f for f in os.listdir(dir_path) if f.endswith(".parquet")]
            
            for parquet_file in parquet_files:
                file_path = os.path.join(dir_path, parquet_file)
                # Read only the columns we need
                part_df = pd.read_parquet(file_path, columns=["exam_id", "person_id", "ExamDate"])
                dfs.append(part_df)
        
        # Concatenate all DataFrames
        df = pd.concat(dfs, ignore_index=True)
    
    print(f"Loaded {len(df):,} exams")
    return df


def create_splits(df, train_val_cut_date, val_test_cut_date):
    """
    Split the data into train, validation, and test sets
    
    Args:
        df: pandas DataFrame with exam_id, person_id, and ExamDate columns
        train_val_cut_date: Cut date between train and validation
        val_test_cut_date: Cut date between validation and test
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("Creating splits...")
    
    # Split by date
    train_df = df[df["ExamDate"] < train_val_cut_date]
    val_df = df[(df["ExamDate"] >= train_val_cut_date) & (df["ExamDate"] < val_test_cut_date)]
    test_df = df[df["ExamDate"] >= val_test_cut_date]
    
    # Get statistics
    total_count = len(df)
    print(f"Train-SSL: {len(train_df):,} exams ({len(train_df)/total_count:.2%})")
    print(f"Val-SSL: {len(val_df):,} exams ({len(val_df)/total_count:.2%})")
    print(f"Test-Future: {len(test_df):,} exams ({len(test_df)/total_count:.2%})")
    
    return train_df, val_df, test_df


def verify_splits(train_df, val_df, test_df):
    """
    Verify the integrity of the splits
    
    Args:
        train_df: Train split DataFrame
        val_df: Validation split DataFrame
        test_df: Test split DataFrame
        
    Returns:
        bool: True if verification passes, False otherwise
    """
    print("Verifying splits...")
    
    # Check for overlapping exam_ids
    train_ids = set(train_df["exam_id"])
    val_ids = set(val_df["exam_id"])
    test_ids = set(test_df["exam_id"])
    
    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(f"ERROR: Overlapping exam_ids between splits")
        print(f"Train-Val overlap: {len(train_val_overlap)}")
        print(f"Train-Test overlap: {len(train_test_overlap)}")
        print(f"Val-Test overlap: {len(val_test_overlap)}")
        return False
    
    # Check chronological ordering
    if not train_df.empty and not val_df.empty:
        train_max_date = train_df["ExamDate"].max()
        val_min_date = val_df["ExamDate"].min()
        
        if train_max_date >= val_min_date:
            print(f"ERROR: Train max date ({train_max_date}) >= Val min date ({val_min_date})")
            return False
    
    if not val_df.empty and not test_df.empty:
        val_max_date = val_df["ExamDate"].max()
        test_min_date = test_df["ExamDate"].min()
        
        if val_max_date >= test_min_date:
            print(f"ERROR: Val max date ({val_max_date}) >= Test min date ({test_min_date})")
            return False
    
    print("Verification passed!")
    return True


def write_splits(train_df, val_df, test_df, output_dir):
    """
    Write split manifests to Parquet files
    
    Args:
        train_df: Train split DataFrame
        val_df: Validation split DataFrame
        test_df: Test split DataFrame
        output_dir: Directory to write the manifests
    """
    print("Writing split manifests...")
    
    # Ensure core directory exists
    core_dir = os.path.join(output_dir, "core")
    os.makedirs(core_dir, exist_ok=True)
    
    # Convert to PyArrow tables
    train_table = pa.Table.from_pandas(train_df)
    val_table = pa.Table.from_pandas(val_df)
    test_table = pa.Table.from_pandas(test_df)
    
    # Write train split
    train_path = os.path.join(core_dir, "train_ssl.parquet")
    pq.write_table(train_table, train_path, compression="snappy")
    print(f"Wrote train manifest to {train_path}")
    
    # Write validation split
    val_path = os.path.join(core_dir, "val_ssl.parquet")
    pq.write_table(val_table, val_path, compression="snappy")
    print(f"Wrote validation manifest to {val_path}")
    
    # Write test split
    test_path = os.path.join(core_dir, "test_future.parquet")
    pq.write_table(test_table, test_path, compression="snappy")
    print(f"Wrote test manifest to {test_path}")


def write_metadata(
    meta_dir, train_val_cut_date, val_test_cut_date, 
    train_count, val_count, test_count, total_count, 
    source_hash, seed
):
    """
    Write split metadata to YAML file
    
    Args:
        meta_dir: Directory to write the metadata
        train_val_cut_date: Cut date between train and validation
        val_test_cut_date: Cut date between validation and test
        train_count: Number of exams in train split
        val_count: Number of exams in validation split
        test_count: Number of exams in test split
        total_count: Total number of exams
        source_hash: Hash of source data
        seed: Random seed used
    """
    print("Writing metadata...")
    
    # Format dates as strings for YAML
    train_val_str = train_val_cut_date.strftime("%Y-%m-%d") if hasattr(train_val_cut_date, "strftime") else str(train_val_cut_date)
    val_test_str = val_test_cut_date.strftime("%Y-%m-%d") if hasattr(val_test_cut_date, "strftime") else str(val_test_cut_date)
    
    metadata = {
        "cut_dates": {
            "train_val": train_val_str,
            "val_test": val_test_str,
        },
        "statistics": {
            "total_exams": total_count,
            "train_ssl_count": train_count,
            "val_ssl_count": val_count,
            "test_future_count": test_count,
            "train_percentage": round(train_count / total_count * 100, 2),
            "val_percentage": round(val_count / total_count * 100, 2),
            "test_percentage": round(test_count / total_count * 100, 2),
        },
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_data_hash": source_hash,
        "random_seed": seed,
    }
    
    metadata_path = os.path.join(meta_dir, "split_metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    
    print(f"Wrote metadata to {metadata_path}")


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create metadata directory
    os.makedirs(args.meta_dir, exist_ok=True)
    
    # Compute hash of source data
    source_hash = compute_source_hash(args.input_dir)
    print(f"Source data hash: {source_hash}")
    
    # Load all exam data
    exams_df = load_exam_data(args.input_dir)
    
    # Calculate cut dates (70th and 85th percentiles)
    exams_df_sorted = exams_df.sort_values("ExamDate")
    train_val_cut_date = exams_df_sorted["ExamDate"].quantile(0.70, interpolation='nearest')
    val_test_cut_date = exams_df_sorted["ExamDate"].quantile(0.85, interpolation='nearest')
    print(f"70th percentile date: {train_val_cut_date}")
    print(f"85th percentile date: {val_test_cut_date}")
    
    # Create splits
    train_df, val_df, test_df = create_splits(exams_df, train_val_cut_date, val_test_cut_date)
    
    # Verify splits
    if not verify_splits(train_df, val_df, test_df):
        print("Split verification failed!")
        sys.exit(1)
    
    # Write splits
    write_splits(train_df, val_df, test_df, args.output_dir)
    
    # Write metadata
    write_metadata(
        args.meta_dir,
        train_val_cut_date,
        val_test_cut_date,
        len(train_df),
        len(val_df),
        len(test_df),
        len(exams_df),
        source_hash,
        args.seed,
    )
    
    print("Split generation completed successfully!")


if __name__ == "__main__":
    main()