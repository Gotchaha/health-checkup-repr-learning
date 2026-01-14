# scripts/data_preparation/generate_id_mapping.py
"""
Generate ID mapping and date jitter offsets for de-identification.
Loads demographic data and creates a mapping file with new IDs and date offsets.
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import uuid
import hashlib
from datetime import datetime
from pathlib import Path

# Set up logging
def setup_logging():
    """Configure logging to file and console"""
    os.makedirs("outputs/audit/phi_removal_logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("outputs/audit/phi_removal_logs/id_mapping_generation.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("id_mapping")

def create_id_mapping(demographics_path, min_age=20, max_age=89, seed=42):
    """
    Create mapping from original IDs to new random IDs with date offsets.
    
    Args:
        demographics_path: Path to demographic data file
        min_age: Minimum age to keep records (default: 20)
        max_age: Maximum age to keep records (default: 89)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        DataFrame with mapping information
    """
    logger = setup_logging()
    logger.info(f"Loading demographic data from {demographics_path}")
    
    try:
        demographics = pd.read_parquet(demographics_path)
        logger.info(f"Demographic data loaded: {len(demographics)} records")
    except Exception as e:
        logger.error(f"Failed to load demographic data: {str(e)}")
        raise
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    logger.info(f"Using random seed: {seed}")
    
    # Get unique person IDs
    unique_ids = demographics['AnonymousID'].unique()
    logger.info(f"Found {len(unique_ids)} unique person IDs")
    
    # Generate new random IDs
    logger.info("Generating new random IDs")
    id_mapping = {}
    for old_id in unique_ids:
        # Use deterministic method based on original ID + seed
        hash_input = f"{old_id}_{seed}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        new_id = f"pid_{hash_value}"
        id_mapping[old_id] = new_id
    
    # Generate date offsets (-180 to +180 days)
    logger.info("Generating date jitter offsets (-180 to +180 days)")
    date_offsets = {}
    for old_id in unique_ids:
        # Also make date offsets deterministic but different from ID hash
        hash_input = f"{old_id}_{seed}_offset"
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest()[:8], 16)
        # Map hash to range [-180, 180]
        offset = (hash_value % 361) - 180
        date_offsets[old_id] = offset
    
    # Apply age screening
    logger.info("Applying age screening")
    reference_year = datetime.now().year  # Current year
    demographics['Age'] = reference_year - demographics['BirthYear']
    keep_record = (demographics['Age'] >= min_age) & (demographics['Age'] <= max_age)
    
    # Create mapping DataFrame
    logger.info("Creating mapping DataFrame")
    mapping_df = pd.DataFrame({
        'original_id': demographics['AnonymousID'],
        'new_id': demographics['AnonymousID'].map(id_mapping),
        'date_offset': demographics['AnonymousID'].map(date_offsets),
        'keep_record': keep_record,
        'age': demographics['Age']
    })
    
    # Add gender if available
    if 'Gender' in demographics.columns:
        mapping_df['gender'] = demographics['Gender']
    
    # Remove duplicates to ensure one mapping per person
    mapping_df = mapping_df.drop_duplicates(subset=['original_id'])
    
    # Generate statistics for audit
    removed_count = (~keep_record).sum()
    logger.info(f"Age screening would remove {removed_count} records ({removed_count/len(mapping_df):.1%})")
    
    stats = {
        "total_persons": len(mapping_df),
        "persons_kept": keep_record.sum(),
        "persons_removed": removed_count,
        "age_range": {
            "min": int(demographics['Age'].min()),
            "max": int(demographics['Age'].max()),
            "median": int(demographics['Age'].median())
        },
        "date_offset_range": {
            "min": int(mapping_df['date_offset'].min()),
            "max": int(mapping_df['date_offset'].max()),
            "mean": float(mapping_df['date_offset'].mean())
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Gender distribution if available
    if 'Gender' in demographics.columns:
        gender_counts = demographics['Gender'].value_counts().to_dict()
        stats["gender_distribution"] = gender_counts
    
    # Save statistics
    os.makedirs("outputs/audit/phi_removal_logs", exist_ok=True)
    pd.DataFrame([stats]).to_json(
        "outputs/audit/phi_removal_logs/id_mapping_stats.json", 
        orient="records"
    )
    
    return mapping_df

def main():
    """Main entry point for ID mapping generation"""
    parser = argparse.ArgumentParser(description='Generate ID mapping for de-identification')
    parser.add_argument('--input', default='data/raw/person_cleaned.parquet',
                        help='Path to demographic data file')
    parser.add_argument('--output', default='data/private_backup/id_mapping.parquet',
                        help='Output path for mapping file')
    parser.add_argument('--min-age', type=int, default=20,
                        help='Minimum age to keep records')
    parser.add_argument('--max-age', type=int, default=89,
                        help='Maximum age to keep records')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("data/private_backup", exist_ok=True)
    
    # Generate mapping
    logger = setup_logging()
    logger.info("Starting ID mapping generation process")
    mapping_df = create_id_mapping(
        args.input,
        min_age=args.min_age,
        max_age=args.max_age,
        seed=args.seed
    )
    
    # Save mapping
    logger.info(f"Saving mapping to {args.output}")
    mapping_df.to_parquet(args.output, index=False)
    
    logger.info("ID mapping generation complete")
    
    # Print summary
    print("\nMapping Generation Summary:")
    print(f"Total persons: {len(mapping_df)}")
    print(f"Persons kept after age screening: {mapping_df['keep_record'].sum()}")
    print(f"Persons removed: {(~mapping_df['keep_record']).sum()}")
    print(f"Date offset range: {mapping_df['date_offset'].min()} to {mapping_df['date_offset'].max()} days")
    print(f"Mapping saved to: {args.output}")

if __name__ == "__main__":
    main()