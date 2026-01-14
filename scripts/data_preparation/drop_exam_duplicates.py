# scripts/data_preparation/drop_exam_duplicates.py

"""
Script to identify and remove duplicate (person_id, ExamDate) combinations from exam data.
This script preserves the original data structure and types exactly.
"""

import os
import logging
import json
from pathlib import Path
from datetime import datetime
import hashlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Configure logging
log_dir = "outputs/audit/drop_duplicates_logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/drop_duplicates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)

logger = logging.getLogger(__name__)

def compute_file_hash(file_path):
    """Compute SHA-256 hash of a file for provenance tracking."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def load_all_exam_data(input_dir):
    """
    Load all exam data from year-partitioned directories using PyArrow.
    
    Args:
        input_dir (str): Path to input directory containing year partitions
        
    Returns:
        pd.DataFrame: Combined DataFrame with all exam data
        dict: Mapping of input files and their hashes
        pa.Schema: Original schema
    """
    logger.info(f"Loading all exam data from {input_dir}")
    
    input_path = Path(input_dir)
    partition_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    # Track files processed for provenance
    file_hashes = {}
    tables = []
    schema = None
    
    for partition in partition_dirs:
        logger.info(f"Loading partition: {partition.name}")
        
        for parquet_file in partition.glob("*.parquet"):
            # Compute hash before loading
            file_hash = compute_file_hash(parquet_file)
            file_hashes[str(parquet_file)] = file_hash
            
            # Load data using PyArrow
            table = pq.read_table(parquet_file)
            
            # Capture schema from first file
            if schema is None:
                schema = table.schema
                logger.info(f"Captured schema: {schema}")
            
            logger.info(f"  Loaded {table.num_rows} rows from {parquet_file.name}")
            tables.append(table)
    
    # Combine all tables using PyArrow
    if tables:
        combined_table = pa.concat_tables(tables)
        logger.info(f"Loaded total of {combined_table.num_rows} rows from {len(tables)} files")
        
        # Convert to pandas for processing
        df = combined_table.to_pandas()
        logger.info(f"Converted to pandas DataFrame with columns: {df.columns.tolist()}")
        
        return df, file_hashes, schema
    else:
        logger.warning("No data found in the specified directory")
        return pd.DataFrame(), file_hashes, None

def deduplicate_data(df):
    """
    Remove duplicate (person_id, ExamDate) combinations from DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to deduplicate
        
    Returns:
        pd.DataFrame: Deduplicated DataFrame
        dict: Deduplication statistics
    """
    logger.info(f"Checking for duplicates in {len(df)} rows")
    
    # Find duplicate pairs
    dup_mask = df.duplicated(subset=['person_id', 'ExamDate'], keep=False)
    duplicates = df[dup_mask]
    unique_dup_pairs = duplicates[['person_id', 'ExamDate']].drop_duplicates()
    
    # Log detailed info about duplicates
    if len(unique_dup_pairs) > 0:
        logger.warning(f"Found {len(unique_dup_pairs)} unique (person_id, ExamDate) pairs with duplicates")
        logger.warning(f"Total duplicate rows: {len(duplicates)}")
        
        # Sample a few duplicates for inspection
        sample_size = min(5, len(unique_dup_pairs))
        sample_pairs = unique_dup_pairs.head(sample_size)
        
        for _, row in sample_pairs.iterrows():
            person = row['person_id']
            exam_date = row['ExamDate']
            instances = duplicates[(duplicates['person_id'] == person) & 
                                  (duplicates['ExamDate'] == exam_date)]
            logger.info(f"Sample duplicate - Person: {person}, Date: {exam_date}, "
                        f"Instances: {len(instances)}")
    else:
        logger.info("No duplicates found")
    
    # Remove duplicates (keep first occurrence)
    df_deduped = df.drop_duplicates(subset=['person_id', 'ExamDate'])
    
    # Compute deduplication stats
    stats = {
        "total_rows": len(df),
        "unique_rows": len(df_deduped),
        "duplicate_rows_removed": len(df) - len(df_deduped),
        "unique_duplicate_pairs": len(unique_dup_pairs) if len(unique_dup_pairs) > 0 else 0,
        "percentage_duplicates": ((len(df) - len(df_deduped)) / len(df)) * 100 if len(df) > 0 else 0
    }
    
    logger.info(f"Deduplication summary:")
    logger.info(f"  Total rows processed: {stats['total_rows']}")
    logger.info(f"  Unique rows after deduplication: {stats['unique_rows']}")
    logger.info(f"  Duplicate rows removed: {stats['duplicate_rows_removed']}")
    logger.info(f"  Percentage duplicates: {stats['percentage_duplicates']:.6f}%")
    
    return df_deduped, stats

def save_deduplicated_data(df, output_dir, schema):
    """
    Save deduplicated data preserving the original schema.
    
    Args:
        df (pd.DataFrame): Deduplicated DataFrame
        output_dir (str): Output directory
        schema (pa.Schema): Original PyArrow schema
        
    Returns:
        dict: Mapping of output files and their hashes
    """
    logger.info(f"Saving deduplicated data to {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Process each year-partition carefully
    output_hashes = {}
    
    # Group by year-partition without converting types
    # Extract the year from the partition name directly
    year_column = None
    year_groups = {}
    
    # First pass: determine which rows go to which year partition
    for idx, row in df.iterrows():
        # Extract year from ExamDate without type conversion
        try:
            exam_date = row['ExamDate']
            # Try different methods to extract year without changing types
            if hasattr(exam_date, 'year'):
                year = exam_date.year
            elif isinstance(exam_date, str):
                year = int(exam_date.split('-')[0])
            else:
                year = int(str(exam_date)[:4])
                
            if year not in year_groups:
                year_groups[year] = []
            year_groups[year].append(idx)
        except Exception as e:
            logger.error(f"Error extracting year from {exam_date}: {e}")
            raise
    
    # Second pass: process each year group
    for year, indices in year_groups.items():
        year_df = df.loc[indices]
        logger.info(f"Processing year={year} partition with {len(year_df)} rows")
        
        # Create year directory
        year_dir = output_path / f"year={year}"
        os.makedirs(year_dir, exist_ok=True)
        output_file = year_dir / "data.parquet"
        
        # Convert to PyArrow table preserving schema
        try:
            # FIXED: Remove physical year column to avoid conflict with virtual partition column
            if 'year' in year_df.columns:
                year_df = year_df.drop(columns=['year'])
                logger.info(f"  Removed physical year column (using virtual partitioning)")
            
            # FIXED: Create schema without year column  
            schema_without_year = pa.schema([
                field for field in schema 
                if field.name != 'year'
            ])
            
            # Use the schema without year column
            table = pa.Table.from_pandas(year_df, schema=schema_without_year)
            
            # Write with the same compression
            pq.write_table(table, output_file, compression='snappy')
            
            # Compute hash
            file_hash = compute_file_hash(output_file)
            output_hashes[str(output_file)] = file_hash
            
            logger.info(f"  Saved {table.num_rows} rows to {output_file}")
        except Exception as e:
            logger.error(f"Error saving year={year}: {e}")
            raise
    
    logger.info(f"Successfully saved deduplicated data to {len(output_hashes)} year partitions")
    return output_hashes

def main():
    """Main execution function."""
    logger.info("Starting exam data deduplication process")
    
    # Define paths
    input_dir = "data/deidentified/mcinfo/exam_level"
    output_dir = "data/processed/mcinfo/exam_level"
    
    # 1. Load all data at once using PyArrow and capture schema
    df, input_hashes, schema = load_all_exam_data(input_dir)
    
    # 2. Check column names
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    
    # 3. Deduplicate data
    df_deduped, stats = deduplicate_data(df)
    
    # 4. Save deduplicated data using original schema
    output_hashes = save_deduplicated_data(df_deduped, output_dir, schema)
    
    # 5. Save summary statistics and provenance info
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_directory": input_dir,
        "output_directory": output_dir,
        "deduplication_stats": stats,
        "input_file_hashes": input_hashes,
        "output_file_hashes": output_hashes,
        "log_file": log_filename
    }
    
    stats_file = f"{log_dir}/deduplication_summary.json"
    with open(stats_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved statistics to {stats_file}")
    logger.info("Deduplication process completed successfully")

if __name__ == "__main__":
    main()