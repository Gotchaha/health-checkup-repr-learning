# scripts/data_preparation/map_person_ids.py

"""
Script to map AnonymousID in person file to person_id using id_mapping.

This creates a processed person file with person_id instead of AnonymousID.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Create logs directory
log_dir = "outputs/audit/person_id_mapping"
os.makedirs(log_dir, exist_ok=True)

# Configure logging to both console and file
log_filename = f"{log_dir}/person_id_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main execution function."""
    logger.info("Starting person ID mapping process")
    
    # Define paths
    input_path = "data/raw/person_cleaned.parquet"
    mapping_path = "data/private_backup/id_mapping.parquet"
    output_path = "data/processed/person.parquet"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load person file
    logger.info(f"Loading person file from {input_path}")
    person_table = pq.read_table(input_path)
    person_df = person_table.to_pandas()
    logger.info(f"Loaded {len(person_df)} person records")
    
    # Load id mapping
    logger.info(f"Loading ID mapping from {mapping_path}")
    mapping_table = pq.read_table(mapping_path)
    mapping_df = mapping_table.to_pandas()
    logger.info(f"Loaded {len(mapping_df)} ID mappings")
    
    # Check that the id_mapping has the expected columns
    if 'original_id' not in mapping_df.columns or 'new_id' not in mapping_df.columns:
        logger.error(f"ID mapping does not have expected columns. Found: {mapping_df.columns}")
        raise ValueError("ID mapping has incorrect schema")
    
    # Join person data with ID mapping
    logger.info("Joining person data with ID mapping")
    merged_df = pd.merge(
        person_df,
        mapping_df[['original_id', 'new_id']],
        left_on='AnonymousID',
        right_on='original_id',
        how='left'
    )
    
    # Check for missing mappings
    missing_count = merged_df['new_id'].isna().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} persons without ID mapping")
        
        # Keep original ID as person_id for any missing mappings
        merged_df.loc[merged_df['new_id'].isna(), 'new_id'] = merged_df.loc[merged_df['new_id'].isna(), 'AnonymousID']
        logger.info("Used original AnonymousID as new_id for records without mapping")
    
    # Rename new_id to person_id
    merged_df.rename(columns={'new_id': 'person_id'}, inplace=True)
    
    # Select only the columns we want
    result_df = merged_df[['person_id', 'BirthYear', 'Gender']].copy()
    
    # Ensure proper types
    if not pd.api.types.is_string_dtype(result_df['person_id']):
        result_df['person_id'] = result_df['person_id'].astype(str)
        logger.info("Converted person_id column to string type")
    
    # Save to output path
    logger.info(f"Saving processed person data to {output_path}")
    result_table = pa.Table.from_pandas(result_df)
    pq.write_table(result_table, output_path)
    logger.info(f"Successfully saved {len(result_df)} person records to {output_path}")
    
    # Log detailed statistics
    logger.info("Person data statistics:")
    logger.info(f"  - Total records: {len(result_df)}")
    logger.info(f"  - Unique person_ids: {result_df['person_id'].nunique()}")
    logger.info(f"  - Gender distribution: {result_df['Gender'].value_counts().to_dict()}")
    logger.info(f"  - Birth year range: {result_df['BirthYear'].min()} - {result_df['BirthYear'].max()}")
    logger.info(f"  - Missing birth years: {result_df['BirthYear'].isna().sum()}")
    
    # Save basic stats to a separate JSON file for easy reference
    stats = {
        "total_records": len(result_df),
        "unique_person_ids": result_df['person_id'].nunique(),
        "gender_distribution": result_df['Gender'].value_counts().to_dict(),
        "birth_year_range": [int(result_df['BirthYear'].min()), int(result_df['BirthYear'].max())],
        "missing_birth_years": int(result_df['BirthYear'].isna().sum()),
        "mapped_from": input_path,
        "mapping_file": mapping_path,
        "timestamp": datetime.now().isoformat(),
        "log_file": log_filename
    }
    
    stats_file = f"{log_dir}/person_id_mapping_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_file}")
    
    logger.info("Person ID mapping process completed successfully")

if __name__ == "__main__":
    main()